"""Overlay figures: V2's digitized traces drawn back onto the original image.

Uses the StenhendeV2 ``overlay`` package unchanged. The wrapper is instrumented once
(``instrument_wrapper``) so the hidden geometric state (y-crop offset, rotation flag, dropped
empty columns) is captured during the forward pass and the chain can be inverted.

Representative cases are chosen from the evaluated metrics: for each source, the images at
the best, median and worst mean SNR, plus the worst "failure" cases overall.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from bench.pipelines import Pipeline, ensure_import_paths


def pick_cases(images: pd.DataFrame, metrics: pd.DataFrame, n: int) -> pd.DataFrame:
    """Return a frame of image rows with a ``case`` label (best/median/worst per source)."""
    m = metrics[metrics["pipeline"] == "v2"]
    per_image = m.groupby("image_id")["snr_db"].mean().rename("mean_snr").reset_index()
    per_image = per_image.merge(images, on="image_id")
    picks: list[pd.DataFrame] = []
    per_source = max(1, n // max(1, per_image["source"].nunique()))
    for _, grp in per_image.groupby("source"):
        grp = grp.sort_values("mean_snr", ascending=False).reset_index(drop=True)
        idx = sorted({0, len(grp) // 2, len(grp) - 1})
        labels = {0: "best", len(grp) // 2: "median", len(grp) - 1: "worst"}
        sel = grp.loc[idx].copy()
        sel["case"] = [labels[i] for i in idx]
        if per_source > 3:
            extra = grp.drop(index=idx).sample(min(per_source - 3, max(0, len(grp) - 3)), random_state=0).copy()
            extra["case"] = "random"
            sel = pd.concat([sel, extra])
        picks.append(sel)
    if not picks:
        return per_image.head(0)
    out = pd.concat(picks)
    # round-robin over sources by case priority, so a small n still covers every source
    priority = {"best": 0, "worst": 1, "median": 2, "random": 3}
    out = out.assign(_p=out["case"].map(priority)).sort_values(["_p", "source"], kind="stable").drop(columns="_p")
    return out.head(n)


def render(pipe: Pipeline, holder: dict[str, Any], image_path: str, layout_substring: str | None) -> tuple[np.ndarray, dict[str, Any]]:
    ensure_import_paths()
    from StenhendeV2.overlay.overlay import render_overlay_from_output

    img_bgr = cv2.imread(image_path)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    import torch

    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    out = pipe(tensor, layout_substring)
    # per-lead colouring is only defined for the 3x4+1R layout (exactly 4 extracted lines)
    per_lead = out["layout_name"] == "3x4+1R"
    blended = render_overlay_from_output(out, holder, pipe.wrapper, img_bgr, per_lead=per_lead)
    return blended, out


def make_overlay_figures(cfg: dict[str, Any], images: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path, n: int = 12) -> pd.DataFrame:
    ensure_import_paths()
    from StenhendeV2.overlay.overlay import instrument_wrapper

    cases = pick_cases(images, metrics, n)
    fig_dir = out_dir / "figures" / "overlays"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pipes: dict[str, tuple[Pipeline, dict[str, Any]]] = {}
    rows = []
    for _, e in cases.iterrows():
        if e.layout_set not in pipes:
            pipe = Pipeline("v2", cfg["device"], e.layout_set, Path(cfg["weights_dir"]), seed=cfg["seed"])
            pipes[e.layout_set] = (pipe, instrument_wrapper(pipe.wrapper))
        pipe, holder = pipes[e.layout_set]
        try:
            blended, out = render(pipe, holder, e.image_path, e.layout_substring)
            # downscale very large photos so the figure folder stays small
            h, w = blended.shape[:2]
            if max(h, w) > 2400:
                s = 2400 / max(h, w)
                blended = cv2.resize(blended, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            name = f"{e.source_name}_{e.case}_{e.image_id}_snr{e.mean_snr:.1f}.jpg"
            cv2.imwrite(str(fig_dir / name), blended, [cv2.IMWRITE_JPEG_QUALITY, 88])
            rows.append({"image_id": e.image_id, "source": e.source_name, "case": e.case, "mean_snr_db": round(e.mean_snr, 2),
                         "layout": out["layout_name"], "rotated": holder["rotated"], "y0": holder["y0"], "x0": holder["x0"], "file": name})
        except Exception as ex:  # noqa: BLE001
            rows.append({"image_id": e.image_id, "source": e.source_name, "case": e.case, "error": f"{type(ex).__name__}: {ex}"})
    df = pd.DataFrame(rows)
    df.to_csv(fig_dir / "overlay_cases.csv", index=False)
    return df
