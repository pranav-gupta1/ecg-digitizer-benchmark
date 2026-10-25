"""Why is layout identification slower inside V1 than inside V2? (identical code in both)

Hypotheses, tested on scans (which fit in memory on a 16 GB Mac):
  A. type: V1's PixelSizeFinder returns 0-dim tensors, V2's returns Python floats. The value
     feeds the layout identifier, where a tensor may trigger many small device syncs.
  B. memory: V1's cropper leaves a large MPS cache behind, slowing the later stages.

Variants: v1, v1 with PixelSizeFinder outputs cast to float (A), v1 with the MPS cache emptied
right after cropping (B), and v2. Each runs REPS times per image after one warm-up pass.

Usage: python scripts/experiments/layout_id_attribution.py <out.csv> <image> [<image> ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from bench.pipelines import Pipeline, decode_image  # noqa: E402

REPS = 3


def make(variant: str) -> Pipeline:
    pipe = Pipeline("v2" if variant == "v2" else "v1", "mps", "gm2024")
    w = pipe.wrapper
    if variant == "v1_float_pixel_size":
        psf = w.pixel_size_finder
        orig = psf.forward

        def forward_float(*args, **kwargs):
            x, y = orig(*args, **kwargs)
            return float(x), float(y)

        psf.forward = forward_float
    elif variant == "v1_empty_cache_after_crop":
        cropper = w.cropper
        orig = cropper.forward

        def forward_then_empty(*args, **kwargs):
            out = orig(*args, **kwargs)
            torch.mps.synchronize()
            torch.mps.empty_cache()
            return out

        cropper.forward = forward_then_empty
    return pipe


def main() -> None:
    out_csv, images = sys.argv[1], sys.argv[2:]
    rows = []
    for variant in ["v1", "v1_float_pixel_size", "v1_empty_cache_after_crop", "v2"]:
        pipe = make(variant)
        for path in images:
            img = decode_image(path)
            pipe(img)  # warm-up
            for rep in range(REPS):
                out = pipe(img)
                st = out["stage_times"]
                rows.append({"variant": variant, "image": Path(path).stem, "rep": rep, "wall_s": out["wall_time"],
                             "layout_id_s": st["Layout identification"], "cropping_s": st["Cropping"],
                             "segmentation_s": st["Segmentation"], "gpu_mem_gb": out.get("gpu_mem_gb", 0.0),
                             "pixel_size_type": type(out["pixel_spacing_mm"]["x"]).__name__})
                print(f"{variant:28s} {Path(path).stem:16s} rep{rep} wall={out['wall_time']:6.2f}s "
                      f"layout_id={st['Layout identification']:6.2f}s px_type={rows[-1]['pixel_size_type']}", flush=True)
        del pipe
        torch.mps.empty_cache()
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(df.groupby("variant")[["wall_s", "layout_id_s", "cropping_s", "segmentation_s"]].median().round(2))


if __name__ == "__main__":
    main()
