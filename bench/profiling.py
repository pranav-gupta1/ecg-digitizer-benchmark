"""Environment info, peak memory, GPU utilization sampling and model FLOPs."""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import psutil
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return ""


def _read_pin(path: Path) -> str:
    try:
        for line in path.read_text().splitlines():
            if line.startswith("commit:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def env_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "cpu": platform.processor() or platform.machine(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "torch_num_threads": torch.get_num_threads(),
        "benchmark_commit": _git(["rev-parse", "HEAD"], REPO_ROOT),
        "v1_commit": _read_pin(REPO_ROOT / "third_party" / "open_ecg_digitizer" / "PIN.txt"),
        "v2_commit": _read_pin(REPO_ROOT / "third_party" / "StenhendeV2" / "PIN.txt"),
    }
    if platform.system() == "Darwin":
        try:
            info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:  # noqa: BLE001
            pass
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:  # noqa: BLE001
        pass
    if device.startswith("cuda") and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        info["gpu"] = torch.cuda.get_device_name(idx)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(idx).total_memory / 1e9, 1)
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
    elif device.startswith("mps"):
        info["gpu"] = "Apple GPU (MPS)"
    return info


class MemoryTracker:
    """Peak host RSS (always) and peak accelerator memory (CUDA / MPS) for a block of work."""

    def __init__(self, device: str) -> None:
        self.device = device
        self.process = psutil.Process(os.getpid())

    def reset(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        self._rss_peak = self.process.memory_info().rss

    def sample(self) -> None:
        self._rss_peak = max(self._rss_peak, self.process.memory_info().rss)

    def peak(self) -> dict[str, float]:
        self.sample()
        out = {"peak_rss_gb": round(self._rss_peak / 1e9, 3)}
        if self.device.startswith("cuda"):
            out["peak_gpu_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
            out["peak_gpu_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 3)
        elif self.device.startswith("mps"):
            out["gpu_driver_alloc_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 3)
        return out


class GpuUtilSampler:
    """Samples nvidia-smi utilization in a background thread (CUDA only; no-op elsewhere)."""

    def __init__(self, device: str, interval_s: float = 0.2) -> None:
        self.enabled = device.startswith("cuda") and shutil.which("nvidia-smi") is not None
        self.interval = interval_s
        self.samples: list[tuple[float, float]] = []  # (util %, mem used MB)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=5,
                )
                util, mem = out.strip().splitlines()[0].split(",")
                self.samples.append((float(util), float(mem)))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples:
            return {}
        utils = [s[0] for s in self.samples]
        mems = [s[1] for s in self.samples]
        return {
            "gpu_util_mean_pct": round(sum(utils) / len(utils), 1),
            "gpu_util_max_pct": max(utils),
            "gpu_mem_used_max_mb": max(mems),
            "gpu_util_samples": len(utils),
        }


def model_flops(pipeline: Any, image_hw: tuple[int, int]) -> dict[str, Any]:
    """FLOPs of the two U-Nets for one image at the pipeline's working resolution.

    Counted with fvcore (one multiply-add = one FLOP, fvcore convention). The networks are
    identical in V1 and V2, so this is reported once per run and shared by both pipelines.
    The lead-name U-Net sees the aligned text-probability map, whose size depends on the
    crop; the resampled input size is used as an upper bound.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return {"error": "fvcore not installed"}
    wrapper = pipeline.wrapper
    device = pipeline.device
    h, w = image_hw
    results: dict[str, Any] = {"input_hw": [h, w]}
    with torch.no_grad():
        x = torch.zeros(1, 3, h, w, device=device)
        seg = FlopCountAnalysis(wrapper.segmentation_model, x)
        seg.unsupported_ops_warnings(False)
        seg.uncalled_modules_warnings(False)
        results["segmentation_unet_gflops"] = round(seg.total() / 1e9, 2)
        results["segmentation_unet_params_m"] = round(sum(p.numel() for p in wrapper.segmentation_model.parameters()) / 1e6, 2)
        unet = pipeline.wrapper.identifier.unet
        x1 = torch.zeros(1, 1, h, w, device=device)
        lead = FlopCountAnalysis(unet, x1)
        lead.unsupported_ops_warnings(False)
        lead.uncalled_modules_warnings(False)
        results["lead_name_unet_gflops"] = round(lead.total() / 1e9, 2)
        results["lead_name_unet_params_m"] = round(sum(p.numel() for p in unet.parameters()) / 1e6, 2)
    results["total_gflops_per_image"] = round(results["segmentation_unet_gflops"] + results["lead_name_unet_gflops"], 2)
    return results


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
