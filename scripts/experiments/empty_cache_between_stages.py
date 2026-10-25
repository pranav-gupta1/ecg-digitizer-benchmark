"""Does releasing the MPS cache after every stage speed V2 up on memory-limited machines?

The variant changes no arithmetic: after each timed stage it synchronizes and calls
torch.mps.empty_cache(). On a 16 GB Mac, 3000 px photos push the pipeline into swap, and the
layout-identification experiment showed that cached blocks left by one stage slow the next.
Outputs are compared to the plain run to confirm they are identical.

Usage: python scripts/experiments/empty_cache_between_stages.py <out.csv> <image> [<image> ...]
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from bench.pipelines import Pipeline, canonical_numpy, decode_image  # noqa: E402

REPS = 2


def install_empty_cache(pipe: Pipeline) -> None:
    @contextmanager
    def timed_section(name, times_dict):
        torch.mps.synchronize()
        start = time.perf_counter()
        yield
        torch.mps.synchronize()
        times_dict[name] = time.perf_counter() - start
        torch.mps.empty_cache()

    pipe._module.timed_section = timed_section


def main() -> None:
    out_csv, images = sys.argv[1], sys.argv[2:]
    rows, outputs = [], {}
    for variant in ["v2", "v2_empty_cache_each_stage"]:
        pipe = Pipeline("v2", "mps", "gm2024")
        if variant != "v2":
            install_empty_cache(pipe)
        for path in images:
            img = decode_image(path)
            pipe(img)  # warm-up
            for rep in range(REPS):
                out = pipe(img)
                outputs[(variant, Path(path).stem, rep)] = canonical_numpy(out)
                st = out["stage_times"]
                rows.append({"variant": variant, "image": Path(path).stem, "rep": rep, "wall_s": out["wall_time"],
                             **{k.lower().replace(" ", "_") + "_s": v for k, v in st.items()}})
                print(f"{variant:28s} {Path(path).stem:16s} rep{rep} wall={out['wall_time']:6.1f}s "
                      + " ".join(f"{k[:8]}={v:.1f}" for k, v in sorted(st.items(), key=lambda kv: -kv[1])[:4]), flush=True)
        del pipe
        torch.mps.empty_cache()
    for path in images:
        a, b = outputs[("v2", Path(path).stem, 0)], outputs[("v2_empty_cache_each_stage", Path(path).stem, 0)]
        same = a is not None and b is not None and np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) < 1e-3
        print(f"outputs identical for {Path(path).stem}: {same}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(df.groupby(["image", "variant"])["wall_s"].median().round(1))


if __name__ == "__main__":
    main()
