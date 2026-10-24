"""Quality metrics, ported from Open-ECG-Digitizer ``src/evaluate.py`` (the npj paper's method).

Per lead, for every horizontal shift within +/- ``max_shift_ms``:
    zero-center both signals on their jointly non-NaN samples, then
    SNR_dB = 10 log10( sum(gt^2) / sum((gt - pred)^2) )
and keep the shift with the best SNR. Pearson r, RMSE, MSE and the NaN fraction are reported at
that shift. Units are microvolts. Ground truth is resampled to 1000 Hz (linear) so that the
shift grid is 1 ms, as in the paper (shift_steps = 200 -> +/- 100 ms).

Additions over the upstream evaluator: MSE, NaN fraction inside the lead's expected window
(the 3x4+1R layout only shows each limb/chest lead for 2.5 s), and a "rhythm" row for the
full-length lead II strip.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import wfdb
from scipy.stats import pearsonr

LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TARGET_FS = 1000

# Column (quarter) of each lead in the 3x4+1R layout; lead II is also the full-width rhythm strip.
LAYOUT_3X4_COLUMN = {
    "I": 0, "II": 0, "III": 0,
    "aVR": 1, "aVL": 1, "aVF": 1,
    "V1": 2, "V2": 2, "V3": 2,
    "V4": 3, "V5": 3, "V6": 3,
}


def load_ground_truth(hea_path: str | Path, seconds: float | None = None) -> tuple[npt.NDArray[np.float64], int]:
    """Return (12, N) ground truth in microvolts at 1000 Hz, in LEAD_NAMES order."""
    record = wfdb.rdrecord(str(Path(hea_path).with_suffix("")))
    sig = np.asarray(record.p_signal, dtype=np.float64)  # (N, leads) in physical units
    units = [u.lower() for u in record.units]
    names = [n.strip() for n in record.sig_name]
    order = [names.index(n) for n in LEAD_NAMES] if all(n in names for n in LEAD_NAMES) else list(range(12))
    sig = sig[:, order]
    scale = np.array([1000.0 if u == "mv" else 1.0 for u in (units[i] for i in order)])
    sig = sig * scale  # -> microvolts
    fs = int(record.fs)
    if fs != TARGET_FS:
        n_new = int(round(sig.shape[0] * TARGET_FS / fs))
        t_old = np.arange(sig.shape[0]) / fs
        t_new = np.arange(n_new) / TARGET_FS
        sig = np.stack([np.interp(t_new, t_old, sig[:, i]) for i in range(sig.shape[1])], axis=1)
    if seconds is not None:
        n_keep = int(round(seconds * TARGET_FS))
        sig = sig[:n_keep]  # the npj evaluator crops Ahus ground truth to the first 5 s
    return sig.T, TARGET_FS


def resample_prediction(pred: npt.NDArray[np.float64], n_target: int) -> npt.NDArray[np.float64]:
    """Linear, NaN-preserving resample of a (leads, n) array to n_target samples."""
    if pred.shape[1] == n_target:
        return pred
    x_old = np.linspace(0, 1, pred.shape[1])
    x_new = np.linspace(0, 1, n_target)
    out = np.full((pred.shape[0], n_target), np.nan)
    for i in range(pred.shape[0]):
        valid = ~np.isnan(pred[i])
        if valid.sum() < 2:
            continue
        out[i] = np.interp(x_new, x_old[valid], pred[i][valid], left=np.nan, right=np.nan)
        # keep gaps as NaN: mark target samples whose nearest source neighbours are NaN
        nan_src = np.interp(x_new, x_old, (~valid).astype(float))
        out[i][nan_src > 0.5] = np.nan
    return out


def _lead_metrics(gt: npt.NDArray[np.float64], pred: npt.NDArray[np.float64], shifts: list[int]) -> dict[str, float]:
    best = {"snr_db": np.nan, "pearson": np.nan, "rmse": np.nan, "mse": np.nan, "shift_ms": 0, "n_valid": 0}
    best_snr = -np.inf
    for shift in shifts:
        if shift == 0:
            g, p = gt, pred
        elif shift > 0:
            g, p = gt[shift:], pred[:-shift]
        else:
            g, p = gt[:shift], pred[-shift:]
        mask = ~(np.isnan(g) | np.isnan(p))
        if mask.sum() < 2:
            continue
        gv = g[mask] - g[mask].mean()
        pv = p[mask] - p[mask].mean()
        noise = float(np.mean((gv - pv) ** 2))
        signal = float(np.mean(gv**2))
        snr = 10 * np.log10(signal / noise) if noise > 0 else np.inf
        if snr > best_snr:
            try:
                r = float(pearsonr(gv, pv)[0])
            except Exception:  # noqa: BLE001
                r = np.nan
            best_snr = snr
            best = {
                "snr_db": float(snr),
                "pearson": r,
                "rmse": float(np.sqrt(noise)),
                "mse": noise,
                "shift_ms": int(shift),
                "n_valid": int(mask.sum()),
            }
    return best


def evaluate_image(
    gt: npt.NDArray[np.float64],
    pred: npt.NDArray[np.float64] | None,
    layout_set: str,
    max_shift_ms: int = 100,
) -> list[dict[str, Any]]:
    """Per-lead metric rows for one image. ``pred`` is (12, n) in microvolts or None.

    Returns one row per lead that the pipeline produced (non-NaN somewhere), plus a
    ``"II_rhythm"`` row for the full-length lead II strip in 3x4+1R layouts.
    """
    n = gt.shape[1]
    rows: list[dict[str, Any]] = []
    if pred is None:
        return rows
    pred = resample_prediction(np.asarray(pred, dtype=np.float64), n)
    shifts = list(range(-max_shift_ms, max_shift_ms + 1))
    for i, lead in enumerate(LEAD_NAMES):
        p = pred[i]
        if np.all(np.isnan(p)):
            continue
        if layout_set == "gm2024":
            if lead == "II":
                windows = [("II_rhythm", 0, n)]
                col = LAYOUT_3X4_COLUMN[lead]
                windows.append(("II", col * n // 4, (col + 1) * n // 4))
            else:
                col = LAYOUT_3X4_COLUMN[lead]
                windows = [(lead, col * n // 4, (col + 1) * n // 4)]
        else:
            windows = [(lead, 0, n)]
        for name, a, b in windows:
            g_w, p_w = gt[i, a:b], p[a:b]
            m = _lead_metrics(g_w, p_w, shifts)
            m.update(
                {
                    "lead": name,
                    "nan_fraction": float(np.isnan(p_w).mean()),
                    "window_start_ms": a,
                    "window_end_ms": b,
                }
            )
            rows.append(m)
    return rows
