"""Shared fixtures: real intermediate tensors captured from one V1 forward pass.

Set ECG_BENCH_TEST_IMAGE to any ECG image to use it; otherwise the first image found under
the benchmark data_dir in configs/benchmark.yaml is used.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench.pipelines import Pipeline, decode_image, ensure_import_paths  # noqa: E402


def _find_test_image() -> Path:
    env = os.environ.get("ECG_BENCH_TEST_IMAGE")
    if env:
        return Path(env)
    with open(REPO_ROOT / "configs" / "benchmark.yaml") as f:
        data_dir = Path(yaml.safe_load(f)["data_dir"]).expanduser()
    for pattern in ("D1/*/*_04.jpeg", "D0/*/*_04.jpeg", "D2/*/*_l01.jpg", "**/*.jpeg", "**/*.png", "**/*.jpg"):
        hits = sorted(data_dir.glob(pattern))
        if hits:
            return hits[0]
    pytest.skip("no test image found; set ECG_BENCH_TEST_IMAGE")


@pytest.fixture(scope="session")
def test_image() -> torch.Tensor:
    return decode_image(_find_test_image())


@pytest.fixture(scope="session")
def v1_cpu() -> Pipeline:
    ensure_import_paths()
    return Pipeline("v1", "cpu", "gm2024")


@pytest.fixture(scope="session")
def v2_cpu() -> Pipeline:
    ensure_import_paths()
    return Pipeline("v2", "cpu", "gm2024")


@pytest.fixture(scope="session")
def intermediates(v1_cpu: Pipeline, test_image: torch.Tensor) -> dict[str, torch.Tensor]:
    """Feature maps and alignment params from V1 on CPU, fed identically to both versions."""
    w = v1_cpu.wrapper
    with torch.no_grad():
        image = w.min_max_normalize(test_image).to(w.device)
        image = w._resample_image(image)
        signal_prob, grid_prob, text_prob = w._get_feature_maps(image)
        torch.manual_seed(0)
        params = w.perspective_detector(grid_prob)
        source_points = w.cropper(signal_prob, params)
        aligned_grid = w.cropper.apply_perspective(grid_prob, source_points, fill_value=0)
    return {
        "image": image,
        "signal_prob": signal_prob,
        "grid_prob": grid_prob,
        "text_prob": text_prob,
        "params": params,
        "source_points": source_points,
        "aligned_grid": aligned_grid,
    }
