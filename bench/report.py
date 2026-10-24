"""Summary tables, figures and summary.md / summary.json for one benchmark run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bench.evaluate import LEAD_NAMES, load_ground_truth  # noqa: E402
from bench.pipelines import STAGE_ORDER  # noqa: E402

# Validated categorical palette (dataviz reference): slot 1 blue = V1, slot 2 orange = V2.
COLORS = {"v1": "#2a78d6", "v2": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
STAGE_COLS = [s.lower().replace(" ", "_") + "_s" for s in STAGE_ORDER] + ["other_s"]

# Published numbers from Stenhede et al., npj Digit. Med. 2026 (validation section).
PAPER = {
    "ahus_snr_db": {"scanner": 19.65, "oneplus": 12.19, "iphone": 10.47},
    "ahus_nan_fraction": 0.0021,
    "emory_snr_db": {"scan_color": 7.34, "scan_gray": 7.24, "photos_and_degraded": "1.05 to 2.03"},
    "seconds_per_image": {"gpu": {"mean": 5.88, "min": 5.43, "max": 7.45}, "cpu": {"mean": 16.78, "min": 15.53, "max": 23.39}},
}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "#fcfcfb",
    "legend.frameon": False,
})


# ------------------------------------------------------------------ tables
def _agg(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by)
    out = g.agg(
        n_images=("image_id", "nunique"), n_leads=("snr_db", "size"),
        snr_mean=("snr_db", "mean"), snr_median=("snr_db", "median"), snr_std=("snr_db", "std"),
        pearson_mean=("pearson", "mean"), rmse_uv_mean=("rmse", "mean"), mse_uv2_mean=("mse", "mean"),
        nan_fraction_mean=("nan_fraction", "mean"), shift_ms_mean=("shift_ms", lambda s: float(np.mean(np.abs(s)))),
    ).reset_index()
    return out.round(3)


def per_image(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per (image, pipeline): mean of the standard 12 leads (rhythm row excluded)."""
    m = metrics[metrics["lead"] != "II_rhythm"]
    return m.groupby(["image_id", "pipeline", "source_name", "category", "layout_set"]).agg(
        snr_db=("snr_db", "mean"), pearson=("pearson", "mean"), rmse=("rmse", "mean"), nan_fraction=("nan_fraction", "mean")
    ).reset_index()


def agreement(out_dir: Path, images: pd.DataFrame, pipelines: list[str]) -> pd.DataFrame:
    """Per-image V1 vs V2 waveform difference, read back from the saved CSVs."""
    if "v1" not in pipelines or "v2" not in pipelines:
        return pd.DataFrame()
    rows = []
    for image_id in images["image_id"]:
        p1 = out_dir / "waveforms" / "v1" / f"{image_id}.csv"
        p2 = out_dir / "waveforms" / "v2" / f"{image_id}.csv"
        if not (p1.is_file() and p2.is_file()):
            continue
        a, b = pd.read_csv(p1), pd.read_csv(p2)
        if a.empty or b.empty or a.shape != b.shape:
            rows.append({"image_id": image_id, "comparable": False, "max_abs_diff_uv": np.nan, "nan_mask_mismatch": np.nan})
            continue
        a, b = a[LEAD_NAMES].to_numpy(), b[LEAD_NAMES].to_numpy()
        m = ~(np.isnan(a) | np.isnan(b))
        rows.append({
            "image_id": image_id, "comparable": True,
            "max_abs_diff_uv": float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0,
            "mean_abs_diff_uv": float(np.abs(a[m] - b[m]).mean()) if m.any() else 0.0,
            "nan_mask_mismatch": float((np.isnan(a) != np.isnan(b)).mean()),
        })
    return pd.DataFrame(rows)


def speed_table(timing: pd.DataFrame) -> pd.DataFrame:
    ok = timing[timing["status"] == "ok"]
    rows = []
    for name, g in ok.groupby("pipeline"):
        row: dict[str, Any] = {
            "pipeline": name, "n_images": len(g), "n_failed": int((timing[timing["pipeline"] == name]["status"] != "ok").sum()),
            "wall_mean_s": g["wall_s"].mean(), "wall_median_s": g["wall_s"].median(), "wall_min_s": g["wall_s"].min(),
            "wall_max_s": g["wall_s"].max(), "throughput_img_per_min": 60 / g["wall_s"].mean(), "decode_mean_s": g["decode_s"].mean(),
        }
        if "gpu_mem_gb" in g and g["gpu_mem_gb"].fillna(0).max() > 0:
            row["gpu_mem_mean_gb"] = g["gpu_mem_gb"].mean()
            row["gpu_mem_max_gb"] = g["gpu_mem_gb"].max()
        for c in STAGE_COLS:
            row[c[:-2] + "_mean_s"] = g[c].mean()
        rows.append(row)
    df = pd.DataFrame(rows).round(4)
    if {"v1", "v2"} <= set(df["pipeline"]):
        v1 = df[df.pipeline == "v1"].iloc[0]
        v2 = df[df.pipeline == "v2"].iloc[0]
        speedup = {"pipeline": "speedup v1/v2", "n_images": "", "n_failed": "", "wall_mean_s": v1.wall_mean_s / v2.wall_mean_s,
                   "wall_median_s": v1.wall_median_s / v2.wall_median_s, "wall_min_s": "", "wall_max_s": "",
                   "throughput_img_per_min": v2.throughput_img_per_min / v1.throughput_img_per_min, "decode_mean_s": ""}
        if "gpu_mem_max_gb" in df:
            speedup["gpu_mem_mean_gb"] = v1.gpu_mem_mean_gb / v2.gpu_mem_mean_gb if v2.gpu_mem_mean_gb else np.nan
            speedup["gpu_mem_max_gb"] = v1.gpu_mem_max_gb / v2.gpu_mem_max_gb if v2.gpu_mem_max_gb else np.nan
        for c in STAGE_COLS:
            k = c[:-2] + "_mean_s"
            speedup[k] = (v1[k] / v2[k]) if v2[k] > 0 else np.nan
        df = pd.concat([df, pd.DataFrame([speedup])], ignore_index=True)
    return df.round(3)


def stage_attribution(timing: pd.DataFrame) -> pd.DataFrame:
    """Which stages account for the saved time (mean seconds per image, V1 minus V2)."""
    ok = timing[timing["status"] == "ok"]
    if not {"v1", "v2"} <= set(ok["pipeline"]):
        return pd.DataFrame()
    common = set(ok[ok.pipeline == "v1"].image_id) & set(ok[ok.pipeline == "v2"].image_id)
    ok = ok[ok.image_id.isin(common)]
    m = ok.groupby("pipeline")[STAGE_COLS].mean()
    saved = (m.loc["v1"] - m.loc["v2"])
    total = saved.sum()
    df = pd.DataFrame({"stage": [c[:-2].replace("_", " ") for c in STAGE_COLS], "v1_mean_s": m.loc["v1"].values,
                       "v2_mean_s": m.loc["v2"].values, "saved_s": saved.values,
                       "share_of_total_saving_pct": 100 * saved.values / total if total else np.nan})
    return df.round(4)


# ------------------------------------------------------------------ figures
def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_snr_by_category(pi: pd.DataFrame, path: Path) -> None:
    cats = pi.groupby(["source_name", "category"]).size().reset_index()[["source_name", "category"]].values.tolist()
    labels = [f"{s}\n{c}" for s, c in cats]
    pipelines = sorted(pi["pipeline"].unique())
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(cats) + 2), 4))
    width = 0.8 / len(pipelines)
    for i, name in enumerate(pipelines):
        data = [pi[(pi.source_name == s) & (pi.category == c) & (pi.pipeline == name)]["snr_db"].dropna().values for s, c in cats]
        pos = np.arange(len(cats)) + (i - (len(pipelines) - 1) / 2) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True, showfliers=False,
                        medianprops={"color": "white", "linewidth": 1.5}, whiskerprops={"color": COLORS[name]}, capprops={"color": COLORS[name]})
        for b in bp["boxes"]:
            b.set(facecolor=COLORS[name], edgecolor="white", linewidth=1)
        ax.plot([], [], color=COLORS[name], linewidth=6, label=name.upper())
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("per-image mean SNR (dB)")
    ax.set_title("Digitization SNR by source and capture category")
    ax.legend(loc="upper right")
    _save(fig, path)


def fig_v1_vs_v2(pi: pd.DataFrame, path: Path) -> None:
    w = pi.pivot_table(index="image_id", columns="pipeline", values="snr_db")
    if not {"v1", "v2"} <= set(w.columns):
        return
    w = w.dropna()
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    lo, hi = float(min(w.v1.min(), w.v2.min())) - 1, float(max(w.v1.max(), w.v2.max())) + 1
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--")
    ax.scatter(w.v1, w.v2, s=14, color=COLORS["v2"], edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.set_xlabel("V1 mean SNR (dB)")
    ax.set_ylabel("V2 mean SNR (dB)")
    ax.set_title(f"Per-image SNR, V1 vs V2 (n={len(w)})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    _save(fig, path)


def fig_speed_stages(timing: pd.DataFrame, path: Path) -> None:
    ok = timing[timing["status"] == "ok"]
    means = ok.groupby("pipeline")[STAGE_COLS].mean()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    names = list(means.index)
    left = np.zeros(len(names))
    shades = plt.cm.Blues(np.linspace(0.35, 0.9, len(STAGE_COLS)))
    for j, c in enumerate(STAGE_COLS):
        vals = means[c].values
        ax.barh(names, vals, left=left, color=shades[j], edgecolor="white", linewidth=1, label=c[:-2].replace("_", " "))
        left += vals
    for i, name in enumerate(names):
        ax.text(left[i] + 0.02 * left.max(), i, f"{left[i]:.2f} s", va="center", color=INK, fontsize=9)
    ax.set_xlabel("mean seconds per image, by stage")
    ax.set_title("Wall time per image (device-synchronized)")
    ax.legend(ncol=3, fontsize=7, loc="lower right")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def fig_shift_hist(metrics: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    bins = np.arange(-100, 101, 5)
    for name, g in metrics.groupby("pipeline"):
        ax.hist(g["shift_ms"].dropna(), bins=bins, histtype="step", linewidth=1.8, color=COLORS[name], label=name.upper())
    ax.set_xlabel("best alignment shift (ms)")
    ax.set_ylabel("lead count")
    ax.set_title("Time shift at best SNR (paper Fig. 2)")
    ax.legend()
    _save(fig, path)


def fig_snr_distribution_ahus(metrics: pd.DataFrame, path: Path, pipeline: str = "v2") -> None:
    m = metrics[(metrics.source_name == "ahus") & (metrics.pipeline == pipeline)]
    if m.empty:
        return
    devices = ["scanner", "iphone", "oneplus"]
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    for ax, dev in zip(axes, devices):
        d = m[m.category == dev]["snr_db"].dropna()
        ax.hist(d, bins=np.arange(-10, 41, 2), color=COLORS[pipeline], edgecolor="white")
        ax.axvline(d.mean(), color=INK, linewidth=1)
        ax.set_title(f"{dev}: mean {d.mean():.2f} dB (paper {PAPER['ahus_snr_db'][dev]})", fontsize=9)
        ax.set_xlabel("per-lead SNR (dB)")
    axes[0].set_ylabel("lead count")
    fig.suptitle(f"Ahus per-lead SNR by device ({pipeline.upper()}, paper Fig. 3)", fontsize=10)
    _save(fig, path)


def fig_lead_ii_percentiles(out_dir: Path, images: pd.DataFrame, metrics: pd.DataFrame, path_dir: Path, pipeline: str = "v2") -> None:
    """Lead II at the 5/25/50/75/95th SNR percentiles per Ahus device (paper Figs 4 to 6)."""
    m = metrics[(metrics.source_name == "ahus") & (metrics.pipeline == pipeline) & (metrics.lead == "II")]
    if m.empty:
        # fall back to the rhythm strip for 3x4+1R sources
        m = metrics[(metrics.pipeline == pipeline) & (metrics.lead == "II_rhythm")]
        if m.empty:
            return
    pct = [5, 25, 50, 75, 95]
    for cat, g in m.groupby("category"):
        g = g.sort_values("snr_db").reset_index(drop=True)
        if len(g) < 2:
            continue
        idxs = [min(len(g) - 1, int(round(p / 100 * (len(g) - 1)))) for p in pct]
        fig, axes = plt.subplots(len(pct), 1, figsize=(9, 1.6 * len(pct)), sharex=True)
        for ax, p, i in zip(axes, pct, idxs):
            row = g.loc[i]
            e = images[images.image_id == row.image_id].iloc[0]
            gt, fs = load_ground_truth(e.gt_hea, e.seconds)
            pred = pd.read_csv(out_dir / "waveforms" / pipeline / f"{row.image_id}.csv")[LEAD_NAMES].to_numpy().T
            from bench.evaluate import resample_prediction

            pred = resample_prediction(pred, gt.shape[1])
            t = np.arange(gt.shape[1]) / fs
            lead = LEAD_NAMES.index("II")
            g_ii = gt[lead] - np.nanmean(gt[lead])
            p_ii = pred[lead] - np.nanmean(pred[lead])
            ax.plot(t, g_ii, color=MUTED, linewidth=1, label="ground truth")
            ax.plot(t, p_ii, color=COLORS[pipeline], linewidth=1, label="digitized")
            ax.set_ylabel("µV", fontsize=8)
            ax.set_title(f"{p}th percentile: {row.image_id}, SNR {row.snr_db:.1f} dB", fontsize=8, loc="left")
        axes[-1].set_xlabel("time (s)")
        axes[0].legend(loc="upper right", fontsize=7)
        fig.suptitle(f"Lead II, {cat} ({pipeline.upper()})", fontsize=10)
        _save(fig, path_dir / f"lead_ii_percentiles_{cat}_{pipeline}.png")


# ------------------------------------------------------------------ summary
def paper_comparison(pi: pd.DataFrame, metrics: pd.DataFrame, speed: pd.DataFrame, env: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(pi["pipeline"].unique()):
        for dev, ref in PAPER["ahus_snr_db"].items():
            sel = metrics[(metrics.pipeline == name) & (metrics.source_name == "ahus") & (metrics.category == dev) & (metrics.lead != "II_rhythm")]
            if len(sel):
                rows.append({"pipeline": name, "metric": f"Ahus {dev} mean per-lead SNR (dB)", "ours": round(sel.snr_db.mean(), 2), "paper": ref, "n_leads": len(sel)})
        sel = metrics[(metrics.pipeline == name) & (metrics.source_name == "ahus")]
        if len(sel):
            rows.append({"pipeline": name, "metric": "Ahus NaN fraction", "ours": round(sel.nan_fraction.mean(), 4), "paper": PAPER["ahus_nan_fraction"], "n_leads": len(sel)})
        for cat, ref in PAPER["emory_snr_db"].items():
            cats = [cat] if cat != "photos_and_degraded" else ["photo_phone", "photo_monitor", "photo_degraded", "scan_degraded"]
            sel = metrics[(metrics.pipeline == name) & (metrics.source_name == "emory") & (metrics.category.isin(cats)) & (metrics.lead != "II_rhythm")]
            if len(sel):
                rows.append({"pipeline": name, "metric": f"Emory {cat} mean per-lead SNR (dB)", "ours": round(sel.snr_db.mean(), 2), "paper": ref, "n_leads": len(sel)})
        sp = speed[speed.pipeline == name]
        if len(sp):
            dev = "gpu" if env.get("device", "cpu") != "cpu" else "cpu"
            rows.append({"pipeline": name, "metric": f"seconds per image ({env.get('device')})", "ours": float(sp.wall_mean_s.iloc[0]),
                         "paper": f"{PAPER['seconds_per_image'][dev]['mean']} ({dev}, paper hardware)", "n_leads": ""})
    return rows


def build(out_dir: Path, images: pd.DataFrame, metrics: pd.DataFrame, timing: pd.DataFrame, env: dict[str, Any]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pipelines = sorted(timing["pipeline"].unique())
    pi = per_image(metrics) if len(metrics) else pd.DataFrame()

    tables: dict[str, pd.DataFrame] = {}
    if len(metrics):
        tables["by_source"] = _agg(metrics[metrics.lead != "II_rhythm"], ["pipeline", "source_name"])
        tables["by_category"] = _agg(metrics[metrics.lead != "II_rhythm"], ["pipeline", "source_name", "category"])
        tables["by_lead"] = _agg(metrics, ["pipeline", "source_name", "lead"])
        tables["rhythm_strip"] = _agg(metrics[metrics.lead == "II_rhythm"], ["pipeline", "source_name", "category"])
        tables["per_image"] = pi
    tables["speed"] = speed_table(timing)
    tables["stage_attribution"] = stage_attribution(timing)
    tables["agreement"] = agreement(out_dir, images, pipelines)
    for name, df in tables.items():
        df.to_csv(out_dir / f"table_{name}.csv", index=False)

    if len(pi):
        fig_snr_by_category(pi, fig_dir / "snr_by_category.png")
        fig_v1_vs_v2(pi, fig_dir / "v1_vs_v2_scatter.png")
        fig_shift_hist(metrics, fig_dir / "shift_histogram.png")
        for name in pipelines:
            fig_snr_distribution_ahus(metrics, fig_dir / f"ahus_snr_distribution_{name}.png", name)
            try:
                fig_lead_ii_percentiles(out_dir, images, metrics, fig_dir, name)
            except Exception as ex:  # noqa: BLE001
                print(f"lead II percentile figure failed for {name}: {ex}")
    fig_speed_stages(timing, fig_dir / "speed_by_stage.png")

    summary: dict[str, Any] = {
        "env": {k: v for k, v in env.items() if k != "settings"}, "settings": env.get("settings", {}),
        "n_images": int(len(images)), "images_per_source": images["source_name"].value_counts().to_dict(),
        "paper_comparison": paper_comparison(pi, metrics, tables["speed"], env) if len(pi) else [],
        "speed": tables["speed"].to_dict(orient="records"),
        "stage_attribution": tables["stage_attribution"].to_dict(orient="records"),
    }
    if len(pi):
        summary["quality_by_source"] = tables["by_source"].to_dict(orient="records")
        summary["quality_by_category"] = tables["by_category"].to_dict(orient="records")
    agr = tables["agreement"]
    if len(agr):
        comp = agr[agr.comparable]
        summary["v1_v2_agreement"] = {
            "n_compared": int(len(comp)), "n_not_comparable": int((~agr.comparable).sum()),
            "images_identical_within_1uV": int((comp.max_abs_diff_uv < 1).sum()),
            "max_abs_diff_uv_median": float(comp.max_abs_diff_uv.median()) if len(comp) else None,
            "max_abs_diff_uv_max": float(comp.max_abs_diff_uv.max()) if len(comp) else None,
            "nan_mask_mismatch_mean": float(comp.nan_mask_mismatch.mean()) if len(comp) else None,
        }
        if len(pi) and {"v1", "v2"} <= set(pipelines):
            w = pi.pivot_table(index="image_id", columns="pipeline", values="snr_db").dropna()
            d = w.v2 - w.v1
            summary["v1_v2_agreement"].update({"snr_diff_db_mean": float(d.mean()), "snr_diff_db_median": float(d.median()),
                                               "snr_diff_db_min": float(d.min()), "snr_diff_db_max": float(d.max()),
                                               "images_v2_worse_by_more_than_0.5dB": int((d < -0.5).sum()),
                                               "images_v2_better_by_more_than_0.5dB": int((d > 0.5).sum())})
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    write_markdown(out_dir, summary, tables, env)


def write_markdown(out_dir: Path, summary: dict[str, Any], tables: dict[str, pd.DataFrame], env: dict[str, Any]) -> None:
    md = [f"# Benchmark summary: {out_dir.name}", ""]
    e = summary["env"]
    md += [f"Device: **{e.get('device')}** ({e.get('gpu', e.get('cpu'))}), torch {e.get('torch')}, {e.get('os')}, "
           f"{summary['n_images']} images ({summary['images_per_source']}).", f"V1 commit {e.get('v1_commit', '')[:12]}, V2 commit {e.get('v2_commit', '')[:12]}.", ""]
    if "flops" in e and e["flops"]:
        f = e["flops"]
        md += [f"Model FLOPs per image at {f.get('input_hw')}: segmentation U-Net {f.get('segmentation_unet_gflops')} GFLOPs "
               f"({f.get('segmentation_unet_params_m')} M params), lead-name U-Net {f.get('lead_name_unet_gflops')} GFLOPs "
               f"({f.get('lead_name_unet_params_m')} M params). Identical for V1 and V2.", ""]
    md += ["## Speed", "", tables["speed"].to_markdown(index=False), ""]
    sp = tables["speed"]
    fl = e.get("flops", {}).get("total_gflops_per_image")
    if fl:
        for _, r in sp[sp.pipeline.isin(["v1", "v2"])].iterrows():
            md.append(f"- {r.pipeline}: achieved {fl / r.wall_mean_s:.1f} GFLOP/s over the whole pipeline "
                      f"({fl / (r.segmentation_mean_s + r.layout_identification_mean_s):.1f} GFLOP/s during the network stages).")
        md.append("")
    if len(tables["stage_attribution"]):
        md += ["### Where the speedup comes from (mean seconds per image on common images)", "", tables["stage_attribution"].to_markdown(index=False), ""]
    if e.get("memory"):
        md += ["### Memory", "", "```", json.dumps(e["memory"], indent=1), "```", ""]
    if e.get("gpu_util") and any(e["gpu_util"].values()):
        md += ["### GPU utilization (nvidia-smi sampling)", "", "```", json.dumps(e["gpu_util"], indent=1), "```", ""]
    if "quality_by_source" in summary:
        md += ["## Quality (per-lead metrics, mean over the 12 standard leads; rhythm strip reported separately)", "",
               tables["by_source"].to_markdown(index=False), "", "### By capture category", "", tables["by_category"].to_markdown(index=False), "",
               "### Lead II rhythm strip", "", tables["rhythm_strip"].to_markdown(index=False), ""]
    if summary.get("v1_v2_agreement"):
        md += ["## V1 vs V2 agreement", "", "```", json.dumps(summary["v1_v2_agreement"], indent=1), "```", ""]
    if summary.get("paper_comparison"):
        md += ["## Comparison with the npj paper", "", pd.DataFrame(summary["paper_comparison"]).to_markdown(index=False), ""]
    md += ["## Figures", ""] + [f"- figures/{p.name}" for p in sorted((out_dir / "figures").glob("*.png"))] + [""]
    (out_dir / "summary.md").write_text("\n".join(md))
