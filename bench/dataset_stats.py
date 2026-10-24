"""Dataset characteristics table for ECG-Image-Database (Kaggle release v2).

Usage:
    python bench/dataset_stats.py --data-dir ~/Documents/ecg-image-database --out outputs/dataset_stats

Counts come from the files on disk. Paper speed, gain, layout and capture metadata come from
the dataset paper (Reyna et al., Physiol. Meas. 2026, Table 1 and Section 5) and the WFDB
headers. Image resolution is read from the files (fast header read, no decoding).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.dataset import SOURCE_NAMES, build_index  # noqa: E402

SOURCE_META = {
    "D0": {"name": "PTB-XL (Germany)", "paper_speed_mm_s": 25, "gain_mm_mV": 10, "layout": "3x4 + lead II rhythm strip (10 s)", "gt_fs_hz": 500, "gt_seconds": 10},
    "D1": {"name": "Emory Healthcare (USA)", "paper_speed_mm_s": 25, "gain_mm_mV": 10, "layout": "3x4 + lead II rhythm strip (10 s)", "gt_fs_hz": 500, "gt_seconds": 10},
    "D2": {"name": "Akershus University Hospital (Norway)", "paper_speed_mm_s": 50, "gain_mm_mV": 10, "layout": "two-page Cabrera 6x1 (limb page, precordial page), 5 s", "gt_fs_hz": 1000, "gt_seconds": 5},
}

# Published totals for the full database (Reyna et al. 2026); the Kaggle release is a subset.
PAPER_TOTALS = {"records": 2243, "images": 37191, "ahus_images": 1596, "ahus_patients": 266, "ptbxl_records": 977, "emory_records": 1000}


def image_size(path: str) -> tuple[int, int]:
    with Image.open(path) as im:
        return im.size  # (w, h)


def build_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    idx = build_index(data_dir)
    pdfs = Counter(p.parts[-3] for p in data_dir.rglob("*.pdf"))
    rows = []
    for src, name in SOURCE_NAMES.items():
        sub = idx[idx["source"] == src]
        if sub.empty:
            continue
        meta = SOURCE_META[src]
        rows.append({
            "source": f"{src} {meta['name']}",
            "records (= unique ECGs)": sub["record_id"].nunique(),
            "unique patients": sub["record_id"].nunique() if src == "D2" else "n/a (not released)",
            "raster images": len(sub),
            "PDF images (not digitized)": pdfs.get(src, 0),
            "images per record": round(len(sub) / sub["record_id"].nunique(), 1),
            "paper speed (mm/s)": meta["paper_speed_mm_s"],
            "gain (mm/mV)": meta["gain_mm_mV"],
            "layout": meta["layout"],
            "ground truth": f"WFDB, {meta['gt_fs_hz']} Hz, {meta['gt_seconds']} s, 12 leads",
        })
    overview = pd.DataFrame(rows)

    # image types with resolution (one sample per variant, plus min/max over all files)
    var_rows = []
    for (src, variant), grp in idx.groupby(["source", "variant"]):
        sizes = [image_size(p) for p in grp["image_path"].head(50)]
        ws = sorted(s[0] for s in sizes)
        hs = sorted(s[1] for s in sizes)
        var_rows.append({
            "source": SOURCE_NAMES[src],
            "variant": variant,
            "description": grp["variant_label"].iloc[0],
            "category": grp["category"].iloc[0],
            "n_images": len(grp),
            "n_records": grp["record_id"].nunique(),
            "typical_resolution": f"{ws[len(ws)//2]}x{hs[len(hs)//2]}",
            "resolution_range": f"{ws[0]}x{hs[0]} .. {ws[-1]}x{hs[-1]}",
            "extension": Path(grp["image_path"].iloc[0]).suffix,
        })
    variants = pd.DataFrame(var_rows).sort_values(["source", "variant"])

    cat = idx.groupby(["source_name", "category"]).agg(n_images=("image_id", "size"), n_records=("record_id", "nunique")).reset_index()
    return {"overview": overview, "variants": variants, "categories": cat}


def write_markdown(tables: dict[str, pd.DataFrame], path: Path, data_dir: Path) -> None:
    lines = ["# ECG-Image-Database characteristics", "", f"Source: `{data_dir}` (Kaggle physionet/ecg-image-database v2, CC BY-ND 4.0).", "",
             f"Published database: {PAPER_TOTALS['records']} records, {PAPER_TOTALS['images']} images across 28 imaging variants "
             f"(Reyna et al. 2026). The Kaggle release contains a subset of the variants; counts below are what is on disk.", "",
             "## Overview per source", "", tables["overview"].to_markdown(index=False), "",
             "## Image types (imaging variants)", "", tables["variants"].to_markdown(index=False), "",
             "## Capture category summary", "", tables["categories"].to_markdown(index=False), "",
             "Ahus devices: scanner = RICOH IM C3500 flatbed (600 dpi, 7016x4964), iphone = iPhone 16 Pro Max (5712x4284), "
             "oneplus = OnePlus Nord CE 2 (4624x3468). PTB-XL/Emory prints: HP LaserJet Pro M501 at 600 dpi; scans: Brother MFC-L8900CDW; "
             "phone photos: Samsung Galaxy S20 FE, iPhone 12, iPhone 13 Mini, Samsung S10+, iPhone 11, Samsung S22 Ultra; "
             "monitor photos of a Lenovo LT2252PWA. Degradations (variants 08-13): water/coffee/soy-sauce stains, ironing marks, mold.", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default=str(REPO_ROOT / "outputs" / "dataset_stats"))
    args = p.parse_args()
    data_dir = Path(args.data_dir).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = build_tables(data_dir)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    write_markdown(tables, out / "dataset_stats.md", data_dir)
    print(tables["overview"].to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
