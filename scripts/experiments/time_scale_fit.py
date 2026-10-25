"""How much of the remaining error is a horizontal (time-scale) error?

For each Ahus image, stretch the digitized time axis by a factor s in [0.97, 1.03] and re-align
(shift up to +/- 40 ms); report the SNR at s = 1 and at the best s. A best s far from 1 on
photos but at 1 on scans means the pixel-spacing estimate, not trace extraction, limits photos.
This is a diagnostic only: it uses the ground truth to choose s.

Usage: python scripts/experiments/time_scale_fit.py <out.csv> <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bench.evaluate import LEAD_NAMES, load_ground_truth  # noqa: E402

SCALES = np.round(np.arange(0.97, 1.0301, 0.005), 3)
SHIFTS = np.arange(-0.04, 0.0401, 0.004)  # fraction of the 5 s page, i.e. +/- 200 ms grid at 20 ms steps


def snr(g: np.ndarray, p: np.ndarray) -> float:
    m = ~np.isnan(p)
    if m.sum() < 1000:
        return -np.inf
    gv, pv = g[m] - g[m].mean(), p[m] - p[m].mean()
    return float(10 * np.log10((gv ** 2).mean() / ((gv - pv) ** 2).mean()))


def fit_image(gt: np.ndarray, raw: np.ndarray) -> dict[str, float]:
    n = gt.shape[1]
    t = np.linspace(0, 1, n)
    per_scale = []
    for s in SCALES:
        lead_best = []
        for i in range(12):
            src = raw[i]
            v = ~np.isnan(src)
            if v.sum() < 100:
                continue
            x = np.linspace(0, 1, len(src))[v] * s
            lead_best.append(max(snr(gt[i], np.interp(t, x + sh, src[v], left=np.nan, right=np.nan)) for sh in SHIFTS))
        per_scale.append(float(np.mean(lead_best)) if lead_best else np.nan)
    per_scale = np.array(per_scale)
    k = int(np.nanargmax(per_scale))
    k1 = int(np.where(SCALES == 1.0)[0][0])
    return {"snr_scale1": per_scale[k1], "best_scale": float(SCALES[k]), "snr_best_scale": per_scale[k]}


def main() -> None:
    out_csv, runs = sys.argv[1], sys.argv[2:]
    rows, seen = [], set()
    for run in runs:
        run = Path(run)
        images = pd.read_csv(run / "images.csv")
        for _, e in images[images.source == "D2"].iterrows():
            path = run / "waveforms" / "v2" / f"{e.image_id}.csv"
            if e.image_id in seen or not path.is_file():
                continue
            seen.add(e.image_id)
            raw = pd.read_csv(path)
            if raw.empty:
                continue
            gt, _ = load_ground_truth(e.gt_hea, e.seconds)
            r = fit_image(gt, raw[LEAD_NAMES].to_numpy(dtype=float).T)
            rows.append({"image_id": e.image_id, "device": e.category, "page": e.page, **r})
            print(f"{e.image_id} {e.category:8s} s=1: {r['snr_scale1']:6.2f}  best s={r['best_scale']:.3f}: {r['snr_best_scale']:6.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    df["gain_db"] = df.snr_best_scale - df.snr_scale1
    df["abs_scale_err_pct"] = (df.best_scale - 1).abs() * 100
    print(df.groupby("device").agg(n=("image_id", "size"), snr_scale1=("snr_scale1", "mean"), snr_best=("snr_best_scale", "mean"),
                                   gain_db=("gain_db", "mean"), median_abs_scale_err_pct=("abs_scale_err_pct", "median")).round(2))


if __name__ == "__main__":
    main()
