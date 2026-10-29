"""Run V1 and V2 on the same images, save waveforms, timings and metrics, then build the report.

Usage (all flags override configs/benchmark.yaml):
    python bench/run_benchmark.py --data-dir ~/Documents/ecg-image-database --n-images 100 --device auto

Runs are resumable: an image is skipped for a pipeline when its waveform CSV and timing row
already exist in the output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench import profiling  # noqa: E402
from bench.dataset import build_index, select_images  # noqa: E402
from bench.evaluate import LEAD_NAMES, evaluate_image, load_ground_truth  # noqa: E402
from bench.pipelines import STAGE_ORDER, Pipeline, canonical_numpy, decode_image, resolve_device  # noqa: E402

TIMING_FIELDS = [
    "image_id",
    "pipeline",
    "device",
    "status",
    "error",
    "layout",
    "layout_cost",
    "decode_s",
    "wall_s",
    *[s.lower().replace(" ", "_") + "_s" for s in STAGE_ORDER],
    "other_s",
    "image_h",
    "image_w",
    "peak_rss_gb",
    "gpu_mem_gb",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "benchmark.yaml"))
    p.add_argument("--data-dir")
    p.add_argument("--output-dir")
    p.add_argument("--subset", help="all | ahus | emory | ptbxl (comma-separated ok)")
    p.add_argument("--layout", help="layout family for a custom image folder: gm2024 (3x4 + rhythm strip, default) or ahus (Cabrera 6x1 pages)")
    p.add_argument("--n-images", type=int)
    p.add_argument("--fraction", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", help="auto | cpu | cuda | mps")
    p.add_argument("--pipelines", help="comma-separated, e.g. v1,v2")
    p.add_argument("--weights-dir")
    p.add_argument("--warmup-images", type=int)
    p.add_argument("--save-overlays", type=int)
    p.add_argument("--max-shift-ms", type=int)
    p.add_argument("--image-list", help="text file with one image_id per line; overrides --subset/--n-images/--fraction")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--skip-report", action="store_true", help="only digitize and evaluate")
    return p.parse_args()


def load_settings(args: argparse.Namespace) -> dict[str, Any]:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for key in ["data_dir", "output_dir", "subset", "n_images", "fraction", "seed", "device", "weights_dir",
                "warmup_images", "save_overlays", "max_shift_ms", "layout"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    if args.pipelines:
        cfg["pipelines"] = [s.strip() for s in args.pipelines.split(",")]
    if args.no_resume:
        cfg["resume"] = False
    cfg["data_dir"] = str(Path(cfg["data_dir"]).expanduser())
    cfg["output_dir"] = str(Path(cfg["output_dir"]).expanduser())
    cfg["weights_dir"] = str((REPO_ROOT / cfg["weights_dir"]).resolve() if not Path(cfg["weights_dir"]).is_absolute() else cfg["weights_dir"])
    cfg["device"] = resolve_device(cfg["device"])
    return cfg


def waveform_path(out_dir: Path, pipeline: str, image_id: str) -> Path:
    return out_dir / "waveforms" / pipeline / f"{image_id}.csv"


def save_waveform(path: Path, pred: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pred is None:
        pd.DataFrame(columns=LEAD_NAMES).to_csv(path, index=False)
        return
    pd.DataFrame(pred.T, columns=LEAD_NAMES).to_csv(path, index=False, float_format="%.3f")


def load_waveform(path: Path) -> np.ndarray | None:
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df[LEAD_NAMES].to_numpy(dtype=np.float64).T


def read_done(timing_csv: Path) -> set[tuple[str, str]]:
    if not timing_csv.is_file():
        return set()
    df = pd.read_csv(timing_csv, usecols=["image_id", "pipeline"])
    return set(zip(df["image_id"], df["pipeline"]))


def digitize(cfg: dict[str, Any], images: pd.DataFrame, out_dir: Path) -> dict[str, Any]:
    device = cfg["device"]
    timing_csv = out_dir / "timing.csv"
    done = read_done(timing_csv) if cfg.get("resume", True) else set()
    if not cfg.get("resume", True) and timing_csv.exists():
        timing_csv.unlink()
    write_header = not timing_csv.exists()
    run_info: dict[str, Any] = {"flops": {}, "memory": {}, "gpu_util": {}}
    mem = profiling.MemoryTracker(device)
    names: list[str] = list(cfg["pipelines"])

    # Images are the outer loop and pipelines the inner loop, with the order alternating per
    # image. Both pipelines then see the same thermal/load conditions, and a run interrupted
    # part-way still holds a paired V1/V2 comparison on every finished image.
    todo = images[[any((i, n) not in done for n in names) for i in images["image_id"]]]
    print(f"{len(todo)} images to run on {device} with {names} ({len(images) - len(todo)} already done)")
    if todo.empty:
        return run_info
    pipes: dict[tuple[str, str], Pipeline] = {}
    warmed: set[tuple[str, str]] = set()
    util = profiling.GpuUtilSampler(device)
    util.start()
    mem.reset()
    t_start = time.time()
    with open(timing_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        if write_header:
            writer.writeheader()
        for k, (_, e) in enumerate(todo.iterrows(), 1):
            img = None
            decode_s = 0.0
            order = names if k % 2 else names[::-1]
            for name in order:
                if (e.image_id, name) in done:
                    continue
                key = (name, e.layout_set)
                if key not in pipes:
                    pipes[key] = Pipeline(name, device, e.layout_set, Path(cfg["weights_dir"]), seed=cfg["seed"])
                pipe = pipes[key]
                row: dict[str, Any] = {"image_id": e.image_id, "pipeline": name, "device": device, "status": "ok", "error": ""}
                try:
                    if img is None:
                        t0 = time.perf_counter()
                        img = decode_image(e.image_path)
                        decode_s = time.perf_counter() - t0
                    row["decode_s"] = round(decode_s, 4)
                    row["image_h"], row["image_w"] = int(img.shape[2]), int(img.shape[3])
                    if key not in warmed:
                        pipe.warmup(img, n=int(cfg.get("warmup_images", 1)))
                        warmed.add(key)
                    out = pipe(img, e.layout_substring)
                    pred = canonical_numpy(out)
                    save_waveform(waveform_path(out_dir, name, e.image_id), pred)
                    row["layout"] = out["layout_name"] + (f" [rot={out['orientation']}]" if "orientation" in out else "")
                    row["layout_cost"] = round(float(out["signal"]["layout_matching_cost"]), 5)
                    row["wall_s"] = round(out["wall_time"], 4)
                    row["gpu_mem_gb"] = round(out.get("gpu_mem_gb", 0.0), 3)
                    for stage, secs in out["stage_times"].items():
                        row[stage.lower().replace(" ", "_") + "_s"] = round(secs, 4)
                except Exception as ex:  # noqa: BLE001 - keep going, record the failure
                    row["status"] = "error"
                    row["error"] = f"{type(ex).__name__}: {ex}"[:300]
                    save_waveform(waveform_path(out_dir, name, e.image_id), None)
                    print(f"  ! {name} {e.image_id}: {row['error']}")
                    traceback.print_exc(limit=2)
                mem.sample()
                row["peak_rss_gb"] = mem.peak()["peak_rss_gb"]
                writer.writerow({c: row.get(c, "") for c in TIMING_FIELDS})
                f.flush()
                if not run_info["flops"] and row["status"] == "ok":
                    try:  # once per run, outside the timed and error-recorded path
                        run_info["flops"] = profiling.model_flops(pipe, (row["image_h"], row["image_w"]))
                    except Exception as ex:  # noqa: BLE001
                        run_info["flops"] = {"error": f"{type(ex).__name__}: {ex}"[:200]}
            if k % 5 == 0 or k == len(todo):
                elapsed = time.time() - t_start
                eta_min = elapsed / k * (len(todo) - k) / 60
                print(f"  {k}/{len(todo)} images  elapsed {elapsed / 60:.1f} min  eta {eta_min:.0f} min", flush=True)
    for name in names:
        run_info["memory"][name] = mem.peak()
    run_info["gpu_util"]["all"] = util.stop()
    del pipes
    return run_info


def evaluate_all(cfg: dict[str, Any], images: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gt_cache: dict[str, tuple[np.ndarray, int]] = {}
    # every pipeline with saved waveforms in this output dir, not just this invocation's, so a
    # later seed-control run (e.g. --pipelines v1@1) re-evaluates the earlier ones too
    wave_dir = out_dir / "waveforms"
    names = sorted(set(cfg["pipelines"]) | ({p.name for p in wave_dir.iterdir() if p.is_dir()} if wave_dir.is_dir() else set()))
    for _, e in images.iterrows():
        if not isinstance(e.gt_hea, str) or not e.gt_hea:
            continue  # custom image without ground truth: nothing to score
        if e.gt_hea not in gt_cache:
            gt_cache[e.gt_hea] = load_ground_truth(e.gt_hea, e.seconds)
        gt, _ = gt_cache[e.gt_hea]
        for name in names:
            path = waveform_path(out_dir, name, e.image_id)
            if not path.is_file():
                continue
            pred = load_waveform(path)
            for m in evaluate_image(gt, pred, e.layout_set, int(cfg.get("max_shift_ms", 100))):
                rows.append({"image_id": e.image_id, "pipeline": name, **m})
    meta_cols = ["image_id", "source", "source_name", "record_id", "variant", "variant_label", "category", "page", "layout_set"]
    metric_cols = ["image_id", "pipeline", "lead", "snr_db", "pearson", "rmse", "mse", "shift_ms", "n_valid", "nan_fraction"]
    metrics = pd.DataFrame(rows) if rows else pd.DataFrame(columns=metric_cols)
    metrics = metrics.merge(images[meta_cols], on="image_id", how="left")
    metrics.to_csv(out_dir / "metrics_per_lead.csv", index=False)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_settings(args)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={cfg['device']}  data_dir={cfg['data_dir']}  output_dir={out_dir}")

    index = build_index(cfg["data_dir"], cfg.get("layout") or "gm2024")
    if (index["source"] == "custom").all():
        print(f"custom image folder: no ground truth, so images are digitized, timed and compared across pipelines but not scored (layout {cfg.get('layout') or 'gm2024'})")
    if args.image_list:
        wanted = {line.strip() for line in open(args.image_list) if line.strip()}
        images = index[index["image_id"].isin(wanted)].sort_values("image_id").reset_index(drop=True)
        missing = wanted - set(images["image_id"])
        if missing:
            print(f"warning: {len(missing)} listed images not found, e.g. {sorted(missing)[:3]}")
    else:
        images = select_images(index, cfg["subset"], cfg.get("n_images"), cfg.get("fraction"), cfg["seed"])
    images.to_csv(out_dir / "images.csv", index=False)
    print(f"{len(images)} images selected from {len(index)} indexed "
          f"({images['source_name'].value_counts().to_dict()})")

    env = profiling.env_info(cfg["device"])
    env["settings"] = cfg
    run_info = digitize(cfg, images, out_dir)
    env.update(run_info)
    profiling.save_json(env, out_dir / "env_info.json")

    metrics = evaluate_all(cfg, images, out_dir)
    print(f"metrics: {len(metrics)} lead rows")

    if not args.skip_report:
        from bench import report

        report.build(out_dir, images, metrics, pd.read_csv(out_dir / "timing.csv"), env)
        if int(cfg.get("save_overlays", 0)) > 0 and "v2" in cfg["pipelines"]:
            from bench import overlay

            overlay.make_overlay_figures(cfg, images, metrics, out_dir, n=int(cfg["save_overlays"]))
    print(f"done. summary: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
