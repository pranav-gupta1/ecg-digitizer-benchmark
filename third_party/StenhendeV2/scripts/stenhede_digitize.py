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
#     "tqdm",
#     "yacs",
# ]
# ///
"""Batch-digitize a folder of ECG photographs to canonical CSV + 4-panel PNG.

Usage:
    uv run src/scripts/stenhede_digitize.py \
        --config inference_wrapper_george-moody-2024.yml \
        DATA.images_path=/tmp/in \
        DATA.output_path=/tmp/out \
        DATA.save_mode=all

Environment:
    Expects STENHEDE_WEIGHTS_PATH in data.env, pointing to the directory
    containing unet_weights_07072025.pt and lead_name_unet_weights_07072025.pt.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / "data.env")
# `uv run script.py` with a PEP 723 block uses an ephemeral env that does
# not include the project root on sys.path; add it so `src.StenhendeV2`
# imports resolve under both `uv run script.py` and `uv run python script.py`.
sys.path.insert(0, str(_PROJECT_ROOT))

from src.StenhendeV2.digitizer.config_loader import get_cfg
from src.StenhendeV2.digitizer.pipeline import main
from src.StenhendeV2.digitizer.utils import find_config_path


def parse_args() -> argparse.Namespace:
    """Parse the --config option and any positional KEY=VALUE overrides."""
    parser = argparse.ArgumentParser(description="Digitize ECGs and save output.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file name or path (searched in . and the bundled configs/).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Override config options like A.B.C=123 or DATA.save_mode='png_only'.",
    )
    return parser.parse_args()


def apply_overrides(cfg, overrides: list[str]) -> None:
    """Merge KEY=VALUE strings into cfg, warning on unknown keys."""
    if not overrides:
        return
    kv_list: list[str] = []
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(
                f"Malformed override '{ov}'. Use KEY=VALUE (e.g., MODEL.device='cuda:0')."
            )
        k, v = ov.split("=", 1)
        node = cfg
        key_parts = k.split(".")
        exists = True
        for part in key_parts[:-1]:
            if part in node:
                node = node[part]
            else:
                exists = False
                break
        if not exists or key_parts[-1] not in node:
            warnings.warn(f"Config key '{k}' not found in loaded config. Override skipped.")
            continue
        kv_list.extend([k, v])
    if kv_list:
        cfg.merge_from_list(kv_list)


if __name__ == "__main__":
    args = parse_args()
    config_path = find_config_path(args.config)
    cfg = get_cfg(config_path)
    apply_overrides(cfg, args.overrides)
    main(cfg)
