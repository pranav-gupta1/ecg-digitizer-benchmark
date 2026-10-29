"""Index of ECG-Image-Database images with ground-truth links and category labels.

Dataset layout (Kaggle physionet/ecg-image-database, v2):
    <data_dir>/Dn/Dn_RECORD/Dn_RECORD.hea, .dat        WFDB ground truth (12 leads)
    <data_dir>/Dn/Dn_RECORD/Dn_RECORD_<variant>.<ext>    images (png/jpeg/jpg) and PDFs

Sources: D0 = PTB-XL (Germany), D1 = Emory (USA), D2 = Ahus (Norway; the npj paper calls it D3).
Variant IDs follow Table 1 of Reyna et al. (Physiol. Meas. 2026). Ahus images are named
``_l<dev>`` (limb page) / ``_p<dev>`` (precordial page) with dev 00 scanner, 01 iPhone 16 Pro
Max, 02 OnePlus Nord CE 2 (identified by image resolution: 7016x4964, 5712x4284, 4624x3468).
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

SOURCE_NAMES = {"D0": "ptbxl", "D1": "emory", "D2": "ahus"}
SUBSET_TO_SOURCE = {v: k for k, v in SOURCE_NAMES.items()}

# Variant ID -> (short label, group). Groups follow the npj paper's reporting categories.
# Only the variants present in the Kaggle v2 release are listed with certainty; the rest of
# Table 1 is kept for completeness of the dataset-characteristics table.
VARIANTS: dict[tuple[str, str], tuple[str, str]] = {
    # PTB-XL (D0)
    ("D0", "01"): ("synthetic PDF, red grid", "synthetic"),
    ("D0", "02"): ("synthetic PNG, red grid", "synthetic"),
    ("D0", "03"): ("color laser printout", "printout"),
    ("D0", "04"): ("clean color scan", "scan_color"),
    ("D0", "05"): ("clean grayscale scan", "scan_gray"),
    ("D0", "06"): ("clean print, phone photo", "photo_phone"),
    ("D0", "07"): ("monitor photo", "photo_monitor"),
    ("D0", "08"): ("primary damage (stains, ironing)", "degraded"),
    ("D0", "09"): ("secondary damage (mold)", "degraded"),
    ("D0", "10"): ("phone photo after primary damage", "photo_degraded"),
    ("D0", "11"): ("phone photo after secondary damage", "photo_degraded"),
    ("D0", "12"): ("color scan after secondary damage", "scan_degraded"),
    ("D0", "13"): ("b/w scan after secondary damage", "scan_degraded"),
    ("D0", "14"): ("synthetic PDF, green grid", "synthetic"),
    ("D0", "15"): ("synthetic PNG, green grid", "synthetic"),
    ("D0", "16"): ("synthetic, handwritten artifacts", "synthetic"),
    ("D0", "17"): ("synthetic, creases and wrinkles", "synthetic"),
    ("D0", "18"): ("synthetic, rotations", "synthetic"),
    # Emory (D1); Kaggle uses the same two-digit codes as PTB-XL for the matching variants
    ("D1", "01"): ("synthetic PDF, red grid", "synthetic"),
    ("D1", "02"): ("synthetic PNG, red grid", "synthetic"),
    ("D1", "04"): ("clean color scan", "scan_color"),
    ("D1", "05"): ("clean grayscale scan", "scan_gray"),
    ("D1", "06"): ("clean print, phone photo", "photo_phone"),
    ("D1", "07"): ("monitor photo", "photo_monitor"),
}

AHUS_DEVICES = {"00": "scanner", "01": "iphone", "02": "oneplus"}
AHUS_PAGES = {"l": "limb", "p": "precordial"}

_AHUS_RE = re.compile(r"^(D2_\d+)_([lp])(\d\d)$")
_STD_RE = re.compile(r"^(D[01]_\d+)_(\d\d)$")


@dataclass
class ImageEntry:
    image_id: str  # file stem, unique across the dataset
    image_path: str
    source: str  # D0 / D1 / D2
    source_name: str  # ptbxl / emory / ahus
    record_id: str
    variant: str  # two-digit code; for Ahus "l00" style page+device
    variant_label: str
    category: str  # scan_color, scan_gray, photo_phone, ... or scanner/iphone/oneplus for Ahus
    page: str  # "" for 12-lead pages, limb/precordial for Ahus
    layout_set: str  # gm2024 or ahus (which inference config to use)
    layout_substring: str | None  # layout-name filter passed to the lead identifier
    gt_hea: str  # path to the WFDB header
    seconds: float  # duration shown on the page (10 for 3x4+1R, 5 for Ahus)


def _entry_for(path: Path, data_dir: Path) -> ImageEntry | None:
    stem = path.stem
    source = path.parts[-3] if len(path.parts) >= 3 else ""
    m = _AHUS_RE.match(stem)
    if m and source == "D2":
        record, page, dev = m.groups()
        device = AHUS_DEVICES.get(dev, f"dev{dev}")
        return ImageEntry(
            image_id=stem,
            image_path=str(path),
            source="D2",
            source_name="ahus",
            record_id=record,
            variant=f"{page}{dev}",
            variant_label=f"{AHUS_PAGES[page]} page, {device}",
            category=device,
            page=AHUS_PAGES[page],
            layout_set="ahus",
            layout_substring=AHUS_PAGES[page],
            gt_hea=str(path.parent / f"{record}.hea"),
            seconds=5.0,
        )
    m = _STD_RE.match(stem)
    if m and source in ("D0", "D1"):
        record, var = m.groups()
        label, group = VARIANTS.get((source, var), (f"variant {var}", "unknown"))
        return ImageEntry(
            image_id=stem,
            image_path=str(path),
            source=source,
            source_name=SOURCE_NAMES[source],
            record_id=record,
            variant=var,
            variant_label=label,
            category=group,
            page="",
            layout_set="gm2024",
            layout_substring=None,
            gt_hea=str(path.parent / f"{record}.hea"),
            seconds=10.0,
        )
    return None


def build_index(data_dir: str | Path, custom_layout_set: str = "gm2024") -> pd.DataFrame:
    """Walk ``data_dir`` and return one row per raster image.

    In the ECG-Image-Database layout (D0/D1/D2 folders) only images with ground truth are kept.
    Any other folder is treated as a custom image set: every image found (recursively) is
    included with no ground truth, so it is digitized, timed and compared across pipelines but
    not scored. ``custom_layout_set`` picks the layout family for such images.
    """
    data_dir = Path(data_dir)
    entries: list[ImageEntry] = []
    for src in sorted(SOURCE_NAMES):
        src_dir = data_dir / src
        if not src_dir.is_dir():
            continue
        for path in sorted(src_dir.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            entry = _entry_for(path, data_dir)
            if entry is not None and Path(entry.gt_hea).is_file():
                entries.append(entry)
    if not entries:
        entries = _custom_entries(data_dir, custom_layout_set)
    if not entries:
        raise FileNotFoundError(f"No images ({', '.join(IMAGE_EXTENSIONS)}) found under {data_dir}")
    return pd.DataFrame([asdict(e) for e in entries])


def _custom_entries(data_dir: Path, layout_set: str) -> list[ImageEntry]:
    """Every image under ``data_dir``, without ground truth. Ahus-style limb/precordial pages
    are recognised from 'limb' or 'precordial' in the file name when layout_set is 'ahus'."""
    entries = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS or path.name.startswith("."):
            continue
        rel = path.relative_to(data_dir)
        substring = None
        if layout_set == "ahus":
            name = path.name.lower()
            substring = "limb" if "limb" in name else ("precordial" if "precordial" in name else None)
        entries.append(ImageEntry(
            image_id="__".join(rel.with_suffix("").parts),
            image_path=str(path),
            source="custom",
            source_name="custom",
            record_id=path.stem,
            variant="",
            variant_label="custom image",
            category=rel.parts[0] if len(rel.parts) > 1 else "custom",
            page="",
            layout_set=layout_set,
            layout_substring=substring,
            gt_hea="",
            seconds=10.0 if layout_set == "gm2024" else 5.0,
        ))
    return entries


def select_images(
    index: pd.DataFrame,
    subset: str = "all",
    n_images: int | None = None,
    fraction: float | None = None,
    seed: int = 0,
    categories: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Deterministic sample. ``subset`` is all|ahus|emory|ptbxl (comma-separated allowed).

    Sampling is stratified by source so that a small ``n_images`` still covers every source
    in proportion, with at least one image per source present in the subset.
    """
    df = index
    if subset != "all":
        wanted = {SUBSET_TO_SOURCE[s.strip()] for s in subset.split(",")}
        df = df[df["source"].isin(wanted)]
    if categories:
        df = df[df["category"].isin(set(categories))]
    df = df.sort_values("image_id").reset_index(drop=True)
    if n_images is None and fraction is None:
        return df
    total = len(df)
    target = n_images if n_images is not None else max(1, round(total * float(fraction)))
    target = min(target, total)
    rng = random.Random(seed)
    parts = []
    sources = sorted(df["source"].unique())
    # proportional allocation with at least one per source
    alloc = {s: max(1, round(target * (df["source"] == s).sum() / total)) for s in sources}
    while sum(alloc.values()) > target:
        biggest = max(alloc, key=lambda s: alloc[s])
        alloc[biggest] -= 1
    for s in sources:
        ids = df.loc[df["source"] == s, "image_id"].tolist()
        rng.shuffle(ids)
        parts.extend(ids[: alloc[s]])
    out = df[df["image_id"].isin(set(parts))].sort_values("image_id").reset_index(drop=True)
    return out
