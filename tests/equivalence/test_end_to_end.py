"""End-to-end equivalence of V1 and V2 on the same image, plus the audit of 'no skipped steps'."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from bench.pipelines import STAGE_ORDER, Pipeline, build_model_config, canonical_numpy, resolve_device


def test_both_pipelines_get_identical_config() -> None:
    for layout in ("gm2024", "ahus"):
        c1 = build_model_config("v1", layout, "cpu")
        c2 = build_model_config("v2", layout, "cpu")

        def strip(node):  # drop the class path prefix and compare everything else
            if hasattr(node, "items"):
                return {k: strip(v) for k, v in node.items()}
            if isinstance(node, str):
                return node.replace("src.model.", "X.").replace("StenhendeV2.digitizer.model.", "X.")
            return node

        d1, d2 = strip(c1), strip(c2)
        d1["config"]["LAYOUT_IDENTIFIER"].pop("unet_config_path")
        d2["config"]["LAYOUT_IDENTIFIER"].pop("unet_config_path")
        assert d1 == d2


def test_v2_runs_every_stage(v1_cpu: Pipeline, v2_cpu: Pipeline, test_image: torch.Tensor) -> None:
    o1 = v1_cpu(test_image)
    o2 = v2_cpu(test_image)
    for stage in STAGE_ORDER:
        if stage == "Dewarping":  # disabled by config in both (apply_dewarping: false)
            continue
        assert stage in o1["stage_times"] and stage in o2["stage_times"]
    assert o1["layout_name"] == o2["layout_name"]
    assert tuple(o1["input_image"].shape) == tuple(o2["input_image"].shape), "resample size differs"
    assert v1_cpu.wrapper.resample_size == v2_cpu.wrapper.resample_size == 3000
    assert v1_cpu.wrapper.apply_dewarping == v2_cpu.wrapper.apply_dewarping


def test_waveforms_identical_on_cpu(v1_cpu: Pipeline, v2_cpu: Pipeline, test_image: torch.Tensor) -> None:
    a = canonical_numpy(v1_cpu(test_image))
    b = canonical_numpy(v2_cpu(test_image))
    assert a is not None and b is not None
    assert np.array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a)
    assert np.abs(a[m] - b[m]).max() < 1e-3, "waveforms differ on CPU"


@pytest.mark.skipif(resolve_device("auto") == "cpu", reason="no accelerator")
def test_accelerator_difference_is_only_randperm(test_image: torch.Tensor) -> None:
    """On GPU, V2's on-device randperm draws from a different RNG stream than V1's CPU randperm.
    Forcing V2's randperm onto the CPU generator must remove the logic difference; what remains
    is float32 roundoff from V2 doing the FFT and scatter_add on the device (measured: max
    2.1 uV on a D1 scan, against signal amplitudes of ~1000 uV; 68 uV without the fix)."""
    dev = resolve_device("auto")
    p1 = Pipeline("v1", dev, "gm2024")
    p2 = Pipeline("v2", dev, "gm2024")
    a = canonical_numpy(p1(test_image))
    real_randperm = torch.randperm

    def cpu_randperm(n, *args, device=None, **kwargs):
        return real_randperm(n, *args, **kwargs).to(device or "cpu")

    torch.randperm = cpu_randperm
    try:
        b = canonical_numpy(p2(test_image))
    finally:
        torch.randperm = real_randperm
    assert np.array_equal(np.isnan(a), np.isnan(b))
    m = ~np.isnan(a)
    max_diff = float(np.abs(a[m] - b[m]).max())
    print(f"V1 vs V2 on {dev} with CPU randperm: max |diff| = {max_diff:.3f} uV")
    assert max_diff < 10.0, f"{max_diff:.2f} uV disagreement, beyond float32 roundoff"
