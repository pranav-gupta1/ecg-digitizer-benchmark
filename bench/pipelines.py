"""Uniform wrapper around the two digitizers (V1 = Open-ECG-Digitizer, V2 = StenhedeV2).

Both pipelines are built from the SAME V1 YAML config so every stage receives identical
parameters; the only difference is the class path prefix (``src.model`` vs
``StenhendeV2.digitizer.model``). Device and weight paths are injected at runtime, so the
vendored third_party code is never modified.

Per-stage timing uses device synchronization (CUDA / MPS) before reading the clock, so
the numbers are comparable between devices and between pipelines.
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

# MPS lacks a few ops used by the pipelines (e.g. torch.quantile); fall back to CPU for those.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402
import yaml  # noqa: E402
from yacs.config import CfgNode as CN  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party"
V1_ROOT = THIRD_PARTY / "open_ecg_digitizer"
V1_CONFIG_DIR = V1_ROOT / "src" / "config"
DEFAULT_WEIGHTS_DIR = REPO_ROOT / "weights"

PIPELINE_NAMES = ("v1", "v2")
CLASS_PREFIX = {"v1": "src.model.", "v2": "StenhendeV2.digitizer.model."}
WRAPPER_MODULE = {"v1": "src.model.inference_wrapper", "v2": "StenhendeV2.digitizer.model.inference_wrapper"}

# Layout sets. "gm2024" is the 3x4+1R family (PTB-XL and Emory images); "ahus" is the two-page
# Cabrera 6x1 family used in the npj paper's Ahus validation.
LAYOUT_CONFIGS = {
    "gm2024": "inference_wrapper_george-moody-2024.yml",
    "ahus": "inference_wrapper_ahus_testset.yml",
}

STAGE_ORDER = [
    "Initial resampling",
    "Segmentation",
    "Perspective detection",
    "Cropping",
    "Feature map resampling",
    "Pixel size search",
    "Dewarping",
    "Signal extraction",
    "Layout identification",
]


def ensure_import_paths() -> None:
    """Make ``src`` (V1) and ``StenhendeV2`` (V2) importable.

    V1's ``src/utils.py`` imports ``ray.tune.Stopper`` for a training-only helper. Ray is a
    heavy dependency with no role in inference, so when it is absent a minimal stub is
    registered instead of installing it. The stub is never called.
    """
    for p in (str(V1_ROOT), str(THIRD_PARTY)):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import ray.tune  # noqa: F401
    except ImportError:
        import types

        ray = types.ModuleType("ray")
        tune = types.ModuleType("ray.tune")
        tune.Stopper = type("Stopper", (), {})  # type: ignore[attr-defined]
        ray.tune = tune  # type: ignore[attr-defined]
        sys.modules.setdefault("ray", ray)
        sys.modules.setdefault("ray.tune", tune)


def resolve_device(device: str) -> str:
    """Map 'auto' to the best available backend: cuda, then mps, then cpu."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def load_base_config(layout_set: str) -> dict[str, Any]:
    path = V1_CONFIG_DIR / LAYOUT_CONFIGS[layout_set]
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_config(
    name: str, layout_set: str, device: str, weights_dir: Path = DEFAULT_WEIGHTS_DIR
) -> CN:
    """Return the ``MODEL.KWARGS`` config node for pipeline ``name``.

    Starts from the V1 YAML, rewrites class paths for V2, and injects absolute weight /
    layout-config paths and the requested device.
    """
    raw = load_base_config(layout_set)["MODEL"]
    kwargs = raw["KWARGS"]
    cfg = kwargs["config"]
    prefix = CLASS_PREFIX[name]
    for key, node in cfg.items():
        if isinstance(node, dict) and "class_path" in node:
            node["class_path"] = node["class_path"].replace("src.model.", prefix)
    cfg["SEGMENTATION_MODEL"]["weight_path"] = str(weights_dir / Path(cfg["SEGMENTATION_MODEL"]["weight_path"]).name)
    li = cfg["LAYOUT_IDENTIFIER"]
    li["unet_weight_path"] = str(weights_dir / Path(li["unet_weight_path"]).name)
    li["config_path"] = str(V1_CONFIG_DIR / Path(li["config_path"]).name)
    li["unet_config_path"] = str(V1_CONFIG_DIR / Path(li["unet_config_path"]).name)
    li["KWARGS"]["device"] = device
    kwargs["device"] = device
    kwargs["enable_timing"] = False
    node = CN(kwargs)
    node.set_new_allowed(True)
    return node


def parse_pipeline_name(name: str, default_seed: int = 0) -> tuple[str, int]:
    """'v1' -> ('v1', default_seed); 'v1@3' -> ('v1', 3)."""
    if "@" in name:
        base, seed = name.split("@", 1)
        return base, int(seed)
    return name, default_seed


def wrapper_class_path(name: str) -> str:
    return WRAPPER_MODULE[name] + ".InferenceWrapper"


class Pipeline:
    """One digitizer (v1 or v2) on one device, with synchronized per-stage timers."""

    def __init__(
        self,
        name: str,
        device: str = "auto",
        layout_set: str = "gm2024",
        weights_dir: Path = DEFAULT_WEIGHTS_DIR,
        seed: int = 0,
    ) -> None:
        # "v1@1" = V1 with torch seed 1: a seed-control run that measures the algorithm's own
        # run-to-run variability (the perspective detector subsamples pixels at random).
        # "v2+orient" = V2 with orientation selection (see _forward_orientation_search).
        label = name
        self.orientation_search = "+orient" in name
        name, seed = parse_pipeline_name(name.replace("+orient", ""), seed)
        if name not in PIPELINE_NAMES:
            raise ValueError(f"unknown pipeline {label!r}; expected one of {PIPELINE_NAMES}, optionally with @seed")
        ensure_import_paths()
        self.name = name
        self.label = label
        self.device = resolve_device(device)
        self.layout_set = layout_set
        self.seed = seed
        self.config = build_model_config(name, layout_set, self.device, Path(weights_dir))

        import importlib

        self._module = importlib.import_module(WRAPPER_MODULE[name])
        # The lead-name U-Net config is patched for V2 so it instantiates V2's UNet class.
        self._patch_unet_config_for(name)
        self._install_synchronized_timer()
        # Both U-Nets are put in eval mode by the upstream loader; the wrapper itself must not
        # be .eval()'d because Dewarper is not a proper nn.Module child.
        self.wrapper = self._module.InferenceWrapper(**self.config)
        self._wrap_layout_identifier()
        self.times: dict[str, float] = {}

    # ----------------------------------------------------------------- setup
    def _patch_unet_config_for(self, name: str) -> None:
        """V1's lead_name_unet.yml names ``src.model.unet.UNet``. V2 must use its own copy of
        the (identical) class, so write a pipeline-specific copy next to the benchmark configs."""
        if name == "v1":
            return
        src = Path(self.config.config.LAYOUT_IDENTIFIER.unet_config_path)
        with open(src) as f:
            unet_cfg = yaml.safe_load(f)
        unet_cfg["MODEL"]["class_path"] = unet_cfg["MODEL"]["class_path"].replace("src.model.", CLASS_PREFIX[name])
        dst = REPO_ROOT / "configs" / "_generated" / f"lead_name_unet_{name}.yml"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with open(dst, "w") as f:
            yaml.safe_dump(unet_cfg, f)
        self.config.config.LAYOUT_IDENTIFIER.unet_config_path = str(dst)

    def _install_synchronized_timer(self) -> None:
        """Replace the module-level ``timed_section`` with one that synchronizes the device
        before reading the clock. Both V1 and V2 look the name up in their module globals."""
        device = self.device

        @contextmanager
        def timed_section(name: str, times_dict: dict[str, float]) -> Generator[None, None, None]:
            synchronize(device)
            start = time.perf_counter()
            yield
            synchronize(device)
            times_dict[name] = time.perf_counter() - start

        self._module.timed_section = timed_section

    def _wrap_layout_identifier(self) -> None:
        """The layout identifier is the one stage the upstream wrapper does not time."""
        identifier = self.wrapper.identifier
        wrapper = self.wrapper
        device = self.device

        class TimedIdentifier:
            def __call__(self, *args: Any, **kwargs: Any) -> Any:
                synchronize(device)
                start = time.perf_counter()
                out = identifier(*args, **kwargs)
                synchronize(device)
                wrapper.times["Layout identification"] = time.perf_counter() - start
                return out

            def __getattr__(self, item: str) -> Any:
                return getattr(identifier, item)

        self.wrapper.identifier = TimedIdentifier()

    # ------------------------------------------------------------------- run
    @torch.no_grad()
    def __call__(self, image: torch.Tensor, layout_substring: str | None = None) -> dict[str, Any]:
        """Digitize one image tensor (1, 3, H, W) uint8 or float.

        Returns the upstream output dict plus ``"stage_times"`` (seconds per stage, in
        STAGE_ORDER) and ``"wall_time"`` (seconds for the whole forward call).
        """
        if not isinstance(layout_substring, str):  # pandas turns None into NaN
            layout_substring = None
        if self.orientation_search:
            out = self._forward_orientation_search(image, layout_substring)
        else:
            out = self._forward(image, layout_substring)
        # Accelerator memory for this image, then release the cache between images (outside the
        # timed region). On a 16 GB Mac the MPS cache otherwise grows until the machine swaps.
        out["gpu_mem_gb"] = 0.0
        if self.device.startswith("mps"):
            out["gpu_mem_gb"] = torch.mps.driver_allocated_memory() / 1e9
            torch.mps.empty_cache()
        elif self.device.startswith("cuda"):
            out["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        return out

    def _forward(self, image: torch.Tensor, layout_substring: str | None) -> dict[str, Any]:
        torch.manual_seed(self.seed)
        synchronize(self.device)
        t0 = time.perf_counter()
        out = self.wrapper(image, layout_should_include_substring=layout_substring)
        synchronize(self.device)
        wall = time.perf_counter() - t0
        times = dict(self.wrapper.times)
        out["stage_times"] = {k: times.get(k, 0.0) for k in STAGE_ORDER}
        out["stage_times"]["Other"] = max(wall - sum(out["stage_times"].values(), 0.0), 0.0)
        out["wall_time"] = wall
        return out

    def _forward_orientation_search(self, image: torch.Tensor, layout_substring: str | None) -> dict[str, Any]:
        """Experimental: choose the post-warp rotation by layout-matching cost.

        Upstream ``_rotate_on_resample`` turns the warped tensor 90 degrees clockwise whenever it
        is taller than wide. That is wrong for portrait photos of a landscape ECG (monitor and
        phone photos) and for sideways paper photographed the other way round. When the default
        rule fires, this also runs the pipeline with a counter-clockwise turn and with no turn,
        and keeps the candidate with the lowest layout-matching cost (the algorithm's own fit
        score; no ground truth involved). Images where the rule does not fire run once and are
        identical to the base pipeline. Wall time is the sum over all candidates run.
        """
        fired = {"rotated": False}

        def rotation(mode: str) -> Any:
            def rotate(img: torch.Tensor, sig: torch.Tensor, grid: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, ...]:
                if img.shape[2] <= img.shape[3]:
                    return img, sig, grid, text
                fired["rotated"] = True
                if mode == "none":
                    return img, sig, grid, text
                k = 3 if mode == "cw" else 1  # k=3 is the upstream behaviour
                return tuple(torch.rot90(x, k=k, dims=(2, 3)) for x in (img, sig, grid, text))

            return rotate

        candidates: list[tuple[str, dict[str, Any]]] = []
        try:
            for mode in ("cw", "ccw", "none"):
                self.wrapper._rotate_on_resample = rotation(mode)  # instance attribute shadows the method
                fired["rotated"] = False
                candidates.append((mode, self._forward(image, layout_substring)))
                if mode == "cw" and not fired["rotated"]:
                    break
        finally:
            if "_rotate_on_resample" in self.wrapper.__dict__:
                del self.wrapper.__dict__["_rotate_on_resample"]

        def cost(o: dict[str, Any]) -> float:
            c = float(o["signal"]["layout_matching_cost"])
            return c if c == c else float("inf")  # NaN -> inf

        best_mode, best = min(candidates, key=lambda mc: cost(mc[1]))  # ties keep the upstream choice
        best["orientation"] = best_mode
        best["orientation_costs"] = {m: cost(o) for m, o in candidates}
        best["stage_times"] = {k: sum(o["stage_times"][k] for _, o in candidates) for k in best["stage_times"]}
        best["wall_time"] = sum(o["wall_time"] for _, o in candidates)
        return best

    def warmup(self, image: torch.Tensor, n: int = 1) -> None:
        for _ in range(n):
            try:
                self(image, None)
            except Exception:  # noqa: BLE001 - warmup only needs to touch the kernels
                pass


def decode_image(path: str | Path) -> torch.Tensor:
    """Decode an image file to a (1, 3, H, W) uint8 tensor, as the upstream scripts do."""
    from torchvision.io import decode_image as _decode

    image = _decode(str(path), mode="RGB")
    c, h, w = image.shape
    if c == 1:
        image = image.expand(3, h, w)
    elif c == 4:
        image = image[:3]
    return image.unsqueeze(0)


def canonical_numpy(out: dict[str, Any]) -> Any:
    """(12, N) float32 numpy array in microvolts, or None if no layout matched."""
    canonical = out.get("signal", {}).get("canonical_lines")
    if canonical is None:
        return None
    return canonical.squeeze().detach().cpu().numpy()
