"""Module-level equivalence: identical inputs into the V1 and V2 versions of each changed module.

Changed modules (see docs/audit.md): cropper, pixel_size_finder, perspective_detector.
Unchanged modules (unet, signal_extractor, lead_identifier, dewarper) are checked by file hash.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from bench.pipelines import THIRD_PARTY, V1_ROOT, ensure_import_paths

ensure_import_paths()

from src.model.cropper import Cropper as CropperV1  # noqa: E402
from src.model.perspective_detector import PerspectiveDetector as PerspV1  # noqa: E402
from src.model.pixel_size_finder import PixelSizeFinder as PsfV1  # noqa: E402
from StenhendeV2.digitizer.model.cropper import Cropper as CropperV2  # noqa: E402
from StenhendeV2.digitizer.model.perspective_detector import PerspectiveDetector as PerspV2  # noqa: E402
from StenhendeV2.digitizer.model.pixel_size_finder import PixelSizeFinder as PsfV2  # noqa: E402

CROPPER_KW = dict(granularity=80, percentiles=(0.02, 0.98), alpha=0.85)
PSF_KW = dict(min_number_of_grid_lines=30, max_number_of_grid_lines=70, lower_grid_line_factor=0.5)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("module", ["unet.py", "signal_extractor.py", "lead_identifier.py"])
def test_unchanged_modules_are_identical(module: str) -> None:
    v1 = V1_ROOT / "src" / "model" / module
    v2 = THIRD_PARTY / "StenhendeV2" / "digitizer" / "model" / module
    assert _sha(v1) == _sha(v2), f"{module} differs between V1 and V2"


def test_dewarper_differs_only_in_type_hints() -> None:
    """V1 is pinned at v1.9.3, whose commit 69d68c1 ("fix: typechecking") rewrote two lines of
    dewarper.py; V2 forked before that. Any other difference fails the test."""
    import difflib

    v1 = (V1_ROOT / "src" / "model" / "dewarper.py").read_text().splitlines()
    v2 = (THIRD_PARTY / "StenhendeV2" / "digitizer" / "model" / "dewarper.py").read_text().splitlines()
    changed = sorted(
        line[1:].strip()
        for line in difflib.unified_diff(v1, v2, lineterm="", n=0)
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    )
    assert changed == sorted([
        "from typing import Any, cast",
        "from typing import Any",
        "with cast(Any, torch.enable_grad)():",
        "with torch.enable_grad():  # type: ignore",
    ]), changed


def test_cropper_buffers_identical(intermediates: dict[str, torch.Tensor]) -> None:
    c1, c2 = CropperV1(**CROPPER_KW), CropperV2(**CROPPER_KW)
    n = c1._normalize_signal_probabilities(intermediates["signal_prob"])
    rhos, th, tv = c1._initialize_parameters(intermediates["params"])
    for mode, thetas in (("horizontal", th), ("vertical", tv)):
        b1 = c1._create_buffer(n, thetas, rhos, mode)
        b2 = c2._create_buffer(n, thetas, rhos, mode)
        assert torch.equal(b1.long(), b2.long()), f"{mode} buffer differs"
        # The V2 clamp(max=granularity-1) only matters if a bin index == granularity, which would
        # make V1's one_hot raise. Record whether it ever happens on this input.
        assert int((b2 == c2.granularity).sum()) == 0, "buffer reached granularity: V1 would have raised"
        v1 = c1._calculate_values(b1, n)
        v2 = c2._calculate_values(b2, n)
        assert torch.allclose(v1, v2, atol=1e-6), f"{mode} bin values differ"


def test_cropper_source_points_identical(intermediates: dict[str, torch.Tensor]) -> None:
    c1, c2 = CropperV1(**CROPPER_KW), CropperV2(**CROPPER_KW)
    s1 = c1(intermediates["signal_prob"], intermediates["params"])
    s2 = c2(intermediates["signal_prob"], intermediates["params"])
    assert torch.allclose(s1, s2, atol=1e-4), f"source points differ: {s1} vs {s2}"


def test_pixel_size_finder_close(intermediates: dict[str, torch.Tensor]) -> None:
    p1, p2 = PsfV1(**PSF_KW), PsfV2(**PSF_KW)
    x1, y1 = p1(intermediates["aligned_grid"])
    x2, y2 = p2(intermediates["aligned_grid"])
    # FFT autocorrelation vs np.correlate: float roundoff only; the grid search picks the same
    # candidate unless two candidates tie to within roundoff.
    assert abs(float(x1) - float(x2)) < 1e-4 * float(x1), (x1, x2)
    assert abs(float(y1) - float(y2)) < 1e-4 * float(y1), (y1, y2)


def test_pixel_size_autocorrelation_close(intermediates: dict[str, torch.Tensor]) -> None:
    import numpy as np

    img = intermediates["aligned_grid"].squeeze()
    col_sum = img.sum(dim=-1)
    col_sum = col_sum - col_sum.mean()
    ref = np.correlate(col_sum.numpy(), col_sum.numpy(), mode="full")
    n = col_sum.shape[0]
    spec = torch.fft.rfft(col_sum, n=2 * n - 1)
    ac = torch.fft.irfft(spec * spec.conj(), n=2 * n - 1)
    ac = torch.cat([ac[n:], ac[:n]]).numpy()
    rel = np.abs(ac - ref).max() / np.abs(ref).max()
    assert rel < 1e-4, f"FFT autocorrelation relative error {rel}"


def test_perspective_detector_identical_on_cpu_with_seed(intermediates: dict[str, torch.Tensor]) -> None:
    d1, d2 = PerspV1(num_thetas=250), PerspV2(num_thetas=250)
    torch.manual_seed(123)
    o1 = d1(intermediates["grid_prob"])
    torch.manual_seed(123)
    o2 = d2(intermediates["grid_prob"])
    assert set(o1) == set(o2)
    for k in o1:
        assert torch.allclose(o1[k].float(), o2[k].float(), atol=1e-6), f"{k}: {o1[k]} vs {o2[k]}"
