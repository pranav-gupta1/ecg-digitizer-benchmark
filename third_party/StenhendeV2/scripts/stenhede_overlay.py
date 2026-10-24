# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "networkx",
#     "numpy",
#     "opencv-python",
#     "python-dotenv",
#     "pyyaml",
#     "scikit-image",
#     "scikit-learn",
#     "scipy",
#     "torch",
#     "torch-tps",
#     "torchvision",
#     "yacs",
# ]
# ///
"""Render Stenhede's recovered ECG traces back onto the original photo.

Usage:
    uv run src/scripts/stenhede_overlay.py --image photo.jpg --out overlay.png

Environment:
    Expects STENHEDE_WEIGHTS_PATH in data.env, pointing to the directory
    containing unet_weights_07072025.pt and lead_name_unet_weights_07072025.pt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from dotenv import load_dotenv

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / "data.env")
# `uv run script.py` with a PEP 723 block uses an ephemeral env that does
# not include the project root on sys.path; add it so `src.StenhendeV2`
# imports resolve under both `uv run script.py` and `uv run python script.py`.
sys.path.insert(0, str(_PROJECT_ROOT))

from src.StenhendeV2.overlay import build_wrapper, instrument_wrapper, overlay_one


def parse_args() -> argparse.Namespace:
    """Parse the overlay CLI arguments."""
    p = argparse.ArgumentParser(description="Stenhede overlay on original photo.")
    p.add_argument("--image", required=True, help="Input ECG image (jpg/png).")
    p.add_argument("--out", required=True, help="Output overlay path.")
    p.add_argument(
        "--config",
        default="inference_wrapper_george-moody-2024.yml",
        help="Config filename in src/StenhendeV2/digitizer/configs/.",
    )
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--thickness", type=int, default=3)
    p.add_argument(
        "--no-per-lead",
        action="store_true",
        help="Draw all lines in a single color instead of the 3x4+1R per-lead palette.",
    )
    p.add_argument(
        "--layout-filter",
        default=None,
        help="Optional Stenhede layout-name substring (e.g. '3x4').",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    print(f"image: {img.shape}")

    print("loading Stenhede ...")
    wrapper = build_wrapper(args.config)
    holder = instrument_wrapper(wrapper)

    blended, layout, cost = overlay_one(
        wrapper,
        holder,
        img,
        alpha=args.alpha,
        thickness=args.thickness,
        per_lead=not args.no_per_lead,
        layout_filter=args.layout_filter,
    )
    print(
        f"  rotated={holder['rotated']}  "
        f"y0={holder['y0']} y1={holder['y1']} pre_h={holder['pre_h']}  "
        f"x0={holder['x0']} x1={holder['x1']} pre_w={holder['pre_w']}"
    )
    print(f"  layout: {layout}  cost={cost:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)
    print(f"wrote {out_path}")
