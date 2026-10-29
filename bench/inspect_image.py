"""Run several pipelines on ONE image and show every stage side by side, with scores.

Usage:
    python bench/inspect_image.py --image ~/Documents/ecg-image-database/D1/D1_000354/D1_000354_04.jpeg
    python bench/inspect_image.py --image <path> --pipelines v1,v2,v2+cache,v2+orient --device mps

Writes to outputs/inspect/<image_id>/:
    <pipeline>_stages.png   input with detected paper corners, straightened page, signal
                            probability map, extracted traces, overlay on the original, and
                            all 12 leads against the ground truth
    comparison.png          one column per pipeline: overlay and lead II against ground truth
    metrics.md / .csv       per-lead scores, mean scores, time per stage, difference from v1

Ground truth is found from the dataset's naming convention (Dn/Dn_RECORD/Dn_RECORD_VV.ext).
For an image outside the dataset the scores are skipped and only the pictures are made.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from bench.dataset import _entry_for  # noqa: E402
from bench.evaluate import LEAD_NAMES, evaluate_image, load_ground_truth, resample_prediction  # noqa: E402
from bench.pipelines import ROTATION_K, STAGE_ORDER, Pipeline, canonical_numpy, decode_image, ensure_import_paths, make_rotation  # noqa: E402
from bench.report import INK, MUTED, color  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--pipelines", default="v1,v2,v2+cache,v2+orient,v2+cache+orient")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="output folder (default outputs/inspect/<image_id>)")
    return p.parse_args()


def run_one(name: str, device: str, layout_set: str, layout_substring: str | None, img: torch.Tensor) -> dict[str, Any]:
    """Warm up, then one timed run. Also captures the geometry the overlay needs to undo the
    resampling, perspective warp, rotation and crops (out["_geo"], out["_holder"], out["_k"])."""
    ensure_import_paths()
    from StenhendeV2.overlay.overlay import instrument_wrapper

    pipe = Pipeline(name, device, layout_set)
    pipe._forward(img, layout_substring)  # warm-up, untimed (one plain pass is enough to warm the kernels)
    if not pipe.orientation_search:
        holder = instrument_wrapper(pipe.wrapper)  # records the crop offsets and the rotation flag
        out = pipe(img, layout_substring)
        out["_geo"], out["_holder"], out["_k"] = out, holder, (ROTATION_K["cw"] if holder["rotated"] else 0)
    else:
        out = pipe(img, layout_substring)  # timed: the orientation search over all quarter turns
        # Re-run once with the chosen turn forced, instrumented, to get that candidate's geometry.
        holder = instrument_wrapper(pipe.wrapper)
        fired = {"rotated": False}
        pipe.wrapper._rotate_on_resample = make_rotation(out["orientation"], fired)
        try:
            geo = pipe._forward(img, layout_substring)
        finally:
            del pipe.wrapper.__dict__["_rotate_on_resample"]
        out["_geo"], out["_holder"], out["_k"] = geo, holder, (ROTATION_K[out["orientation"]] if fired["rotated"] else 0)
    out["_pipe"] = pipe
    return out


def inverse_mapper(src: np.ndarray, dst: np.ndarray, h_res: int, w_res: int, y0: int, x0: int, k: int, h_orig: int, w_orig: int) -> Any:
    """Map trace coordinates (post-crop, post-rotation space) back to the original image.
    Same chain as StenhendeV2.overlay.build_inverse_mapper, generalised to all four quarter
    turns (that module handles only the upstream clockwise turn)."""
    h_inv = cv2.getPerspectiveTransform(dst.astype(np.float32), src.astype(np.float32))
    sx, sy = w_orig / w_res, h_orig / h_res

    def map_pts(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xp = np.asarray(xs, dtype=np.float32) + x0
        yp = np.asarray(ys, dtype=np.float32) + y0
        if k == 0:
            xw, yw = xp, yp
        elif k == 3:  # 90 degrees clockwise (upstream)
            xw, yw = yp, (h_res - 1) - xp
        elif k == 1:  # 90 degrees counter-clockwise
            xw, yw = (w_res - 1) - yp, xp
        else:  # 180 degrees
            xw, yw = (w_res - 1) - xp, (h_res - 1) - yp
        pts = np.column_stack([xw, yw]).astype(np.float32).reshape(-1, 1, 2)
        src_pts = cv2.perspectiveTransform(pts, h_inv).reshape(-1, 2)
        return src_pts[:, 0] * sx, src_pts[:, 1] * sy

    return map_pts


def overlay_bgr(out: dict[str, Any], img_path: Path) -> np.ndarray:
    """Draw the extracted traces on the original image (V2's renderer, with the general mapper)."""
    from StenhendeV2.overlay.overlay import render_overlay

    geo, holder, k = out["_geo"], out["_holder"], out["_k"]
    bgr = cv2.imread(str(img_path))
    h_orig, w_orig = bgr.shape[:2]
    src = geo["source_points"].cpu().numpy()
    h_res, w_res = (int(v) for v in geo["input_image"].shape[-2:])
    dst = out["_pipe"].wrapper.cropper._calculate_destination_points(h_res, w_res, torch.tensor(src, dtype=torch.float32)).cpu().numpy()
    map_pts = inverse_mapper(src, dst, h_res, w_res, holder["y0"] or 0, holder["x0"] or 0, k, h_orig, w_orig)
    src_orig = src.copy()
    src_orig[:, 0] *= w_orig / w_res
    src_orig[:, 1] *= h_orig / h_res
    return render_overlay(bgr, geo["signal"]["raw_lines"], map_pts, src_orig, per_lead=geo["layout_name"] == "3x4+1R")


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def plot_leads(ax_list: list[Any], gt: np.ndarray | None, pred: np.ndarray | None, rows: list[dict[str, Any]], name: str) -> None:
    by_lead = {r["lead"]: r for r in rows}
    for ax, lead in zip(ax_list, LEAD_NAMES):
        ax.set_yticks([])
        ax.tick_params(labelsize=6)
        i = LEAD_NAMES.index(lead)
        if gt is not None:
            t = np.arange(gt.shape[1]) / 1000
            ax.plot(t, gt[i] - np.nanmean(gt[i]), color=MUTED, linewidth=0.8)
        if pred is not None and gt is not None:
            p = resample_prediction(pred, gt.shape[1])[i]
            shift = by_lead.get(lead, {}).get("shift_ms", 0)
            p = np.roll(p, shift)
            ax.plot(np.arange(len(p)) / 1000, p - np.nanmean(p), color=color(name), linewidth=0.8)
        snr = by_lead.get(lead, {}).get("snr_db")
        label = f"{lead}" + (f"  {snr:.1f} dB" if snr is not None and np.isfinite(snr) else "")
        ax.set_title(label, fontsize=7, loc="left", pad=1)


def stages_figure(name: str, out: dict[str, Any], img_path: Path, gt: np.ndarray | None, rows: list[dict[str, Any]], path: Path) -> None:
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.2, 1.2, 1.0], hspace=0.35, wspace=0.15)

    geo = out["_geo"]
    inp = geo["input_image"].squeeze().permute(1, 2, 0).numpy()
    pts = geo["source_points"].numpy()
    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(np.clip(inp, 0, 1))
    ax.plot(*np.vstack([pts, pts[:1]]).T, color="#1baf7a", linewidth=1.5)
    ax.scatter(pts[:, 0], pts[:, 1], s=18, color="#1baf7a")
    ax.set_title("1. Input (resampled) and detected paper corners", fontsize=9, loc="left")
    ax.axis("off")

    aligned = geo["aligned"]["image"].squeeze().permute(1, 2, 0).numpy()
    ax = fig.add_subplot(gs[0, 2:4])
    ax.imshow(np.clip(aligned, 0, 1))
    turn = {0: "no turn", 3: "turned 90 deg clockwise", 1: "turned 90 deg counter-clockwise", 2: "turned 180 deg"}[out["_k"]]
    ax.set_title(f"2. Straightened page, {turn}  (pixel size {geo['pixel_spacing_mm']['x']:.3f} x {geo['pixel_spacing_mm']['y']:.3f} mm)", fontsize=9, loc="left")
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 4:6])
    ax.imshow(geo["aligned"]["signal_prob"].squeeze().numpy(), cmap="gray_r", vmin=0, vmax=1, interpolation="none")
    ax.set_title("3. Trace probability from the segmentation U-Net", fontsize=9, loc="left")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0:3])
    ax.imshow(np.clip(aligned, 0, 1), alpha=0.5)
    raw = geo["signal"]["raw_lines"].numpy()
    x0 = out["_holder"].get("x0") or 0
    for k, line in enumerate(raw):
        ax.plot(np.arange(len(line)) + x0, line, linewidth=0.8)
    ax.set_title(f"4. Extracted traces ({len(raw)} rows), layout matched: {out['layout_name']}", fontsize=9, loc="left")
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 3:6])
    ax.imshow(to_rgb(overlay_bgr(out, img_path)))
    ax.set_title("5. Overlay: digitized traces drawn back on the original", fontsize=9, loc="left")
    ax.axis("off")

    sub = gs[2, :].subgridspec(2, 6, hspace=0.6, wspace=0.08)
    lead_axes = [fig.add_subplot(sub[r, c]) for r in range(2) for c in range(6)]
    plot_leads(lead_axes, gt, canonical_numpy(out), rows, name)

    wall = out["wall_time"]
    fig.suptitle(f"{name}: {img_path.name}   total {wall:.1f} s   (grey = ground truth, colour = digitized; dB = SNR per lead)",
                 fontsize=11, x=0.01, ha="left")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def comparison_figure(results: dict[str, dict[str, Any]], img_path: Path, gt: np.ndarray | None, rows_by: dict[str, list[dict[str, Any]]], path: Path) -> None:
    names = list(results)
    fig, axes = plt.subplots(2, len(names), figsize=(6.5 * len(names), 9), gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    for j, name in enumerate(names):
        out = results[name]
        ax = axes[0, j]
        ax.imshow(to_rgb(overlay_bgr(out, img_path)))
        ax.axis("off")
        mean_snr = np.nanmean([r["snr_db"] for r in rows_by[name] if r["lead"] != "II_rhythm"]) if rows_by[name] else np.nan
        turn = f", rotation {out['orientation']}" if "orientation" in out else ""
        ax.set_title(f"{name}\nmean SNR {mean_snr:.2f} dB, {out['wall_time']:.1f} s{turn}", fontsize=11, color=INK)
        ax = axes[1, j]
        pred = canonical_numpy(out)
        i = LEAD_NAMES.index("II")
        if gt is not None:
            ax.plot(np.arange(gt.shape[1]) / 1000, gt[i] - np.nanmean(gt[i]), color=MUTED, linewidth=0.9, label="ground truth")
        if pred is not None and gt is not None:
            rhythm = [r for r in rows_by[name] if r["lead"] in ("II_rhythm", "II")]
            shift = rhythm[0]["shift_ms"] if rhythm else 0
            p = np.roll(resample_prediction(pred, gt.shape[1])[i], shift)
            ax.plot(np.arange(len(p)) / 1000, p - np.nanmean(p), color=color(name), linewidth=0.9, label="digitized")
        ax.set_title("lead II (rhythm strip where present)", fontsize=9, loc="left")
        ax.set_xlabel("seconds", fontsize=8)
        ax.set_yticks([])
        ax.legend(fontsize=7, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    img_path = Path(args.image).expanduser().resolve()
    names = [s.strip() for s in args.pipelines.split(",") if s.strip()]
    entry = _entry_for(img_path, img_path.parents[2]) if len(img_path.parents) > 2 else None
    layout_set = entry.layout_set if entry else "gm2024"
    substring = entry.layout_substring if entry else None
    out_dir = Path(args.out) if args.out else REPO_ROOT / "outputs" / "inspect" / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth(entry.gt_hea, entry.seconds)[0] if entry and Path(entry.gt_hea).is_file() else None
    print(f"image {img_path.name}: layout set {layout_set}, ground truth {'found' if gt is not None else 'not found'}")

    img = decode_image(img_path)
    results: dict[str, dict[str, Any]] = {}
    rows_by: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        print(f"running {name} ...", flush=True)
        out = run_one(name, args.device, layout_set, substring, img)
        results[name] = out
        rows_by[name] = evaluate_image(gt, canonical_numpy(out), layout_set) if gt is not None else []
        stages_figure(name, out, img_path, gt, rows_by[name], out_dir / f"{name}_stages.png")
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    comparison_figure(results, img_path, gt, rows_by, out_dir / "comparison.png")

    # scores and timings
    ref = canonical_numpy(results[names[0]])
    summary, per_lead = [], []
    for name in names:
        out, rows = results[name], rows_by[name]
        std = [r for r in rows if r["lead"] != "II_rhythm"]
        pred = canonical_numpy(out)
        diff = np.nan
        if ref is not None and pred is not None and ref.shape == pred.shape:
            m = ~(np.isnan(ref) | np.isnan(pred))
            diff = float(np.abs(ref[m] - pred[m]).max()) if m.any() else 0.0
        summary.append({
            "pipeline": name, "layout": out["layout_name"] + (f" [rot={out['orientation']}]" if "orientation" in out else ""),
            "mean_snr_db": np.nanmean([r["snr_db"] for r in std]) if std else np.nan,
            "mean_pearson": np.nanmean([r["pearson"] for r in std]) if std else np.nan,
            "mean_rmse_uv": np.nanmean([r["rmse"] for r in std]) if std else np.nan,
            "nan_fraction": np.nanmean([r["nan_fraction"] for r in std]) if std else np.nan,
            f"max_diff_vs_{names[0]}_uv": diff,
            "total_s": out["wall_time"],
            **{k.lower().replace(" ", "_") + "_s": out["stage_times"].get(k, 0.0) for k in STAGE_ORDER},
        })
        per_lead += [{"pipeline": name, **{k: r[k] for k in ("lead", "snr_db", "pearson", "rmse", "shift_ms", "nan_fraction")}} for r in rows]
    s = pd.DataFrame(summary)
    s.to_csv(out_dir / "metrics_summary.csv", index=False)
    pl = pd.DataFrame(per_lead)
    pl.to_csv(out_dir / "metrics_per_lead.csv", index=False)
    stage_cols = [c for c in s.columns if c.endswith("_s") and c != "total_s" and s[c].max() > 0.005]
    md = [f"# {img_path.name}", "",
          "## Scores (mean over leads; higher SNR and Pearson are better)", "",
          s[["pipeline", "layout", "mean_snr_db", "mean_pearson", "mean_rmse_uv", "nan_fraction", f"max_diff_vs_{names[0]}_uv"]].round(3).to_markdown(index=False), "",
          "## Time per stage (seconds)", "", s[["pipeline", "total_s"] + stage_cols].round(2).to_markdown(index=False), ""]
    if len(pl):
        md += ["## SNR per lead (dB)", "", pl.pivot_table(index="lead", columns="pipeline", values="snr_db").round(2).to_markdown(), ""]
    md += ["Figures: comparison.png, " + ", ".join(f"{n}_stages.png" for n in names), ""]
    (out_dir / "metrics.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
