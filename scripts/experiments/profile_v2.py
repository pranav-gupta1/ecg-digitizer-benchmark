"""torch.profiler pass over one V2 forward, to list further optimisation candidates.

Prints the top operators by self CPU time and saves the full table. On MPS, GPU kernel time is
not broken out per op by the profiler, so per-stage wall time (bench.pipelines timers) is printed
alongside to point at the heavy stages.

Usage: python scripts/experiments/profile_v2.py <out_dir> <image> [device]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

from bench.pipelines import Pipeline, decode_image  # noqa: E402


def main() -> None:
    out_dir, image = Path(sys.argv[1]), sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else "mps"
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = Pipeline("v2", device, "gm2024")
    img = decode_image(image)
    pipe(img)  # warm-up
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.startswith("cuda") else [])
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("v2_forward"):
            out = pipe(img)
    sort_key = "self_cuda_time_total" if device.startswith("cuda") else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=40)
    (out_dir / f"profile_v2_{device}_{Path(image).stem}.txt").write_text(table)
    by_shape = prof.key_averages(group_by_input_shape=True).table(sort_by=sort_key, row_limit=30)
    (out_dir / f"profile_v2_{device}_{Path(image).stem}_by_shape.txt").write_text(by_shape)
    print(table)
    print("stage times:", {k: round(v, 3) for k, v in out["stage_times"].items()})
    print("wall:", round(out["wall_time"], 2))


if __name__ == "__main__":
    main()
