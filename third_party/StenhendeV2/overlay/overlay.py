"""Project Stenhede's extracted lead traces back onto the *original* photo
as a transparent coloured overlay.

Stenhede emits its lines in the rectified+rotated+y-trimmed pixel space.
To draw them on the input photo we have to invert the full geometric
chain:

    original (W_orig, H_orig)
    -> _resample_image: longer-side scaled to resample_size (cfg = 3000)
                       (this also rotates portrait -> landscape *before*
                        segmentation, as a separate concern)
    -> apply_perspective: warps source_points (the 4 paper corners on the
                          resampled image) to a centered destination
                          rectangle, computed by Cropper._calculate_destination_points
                          using alpha-blending toward the image center
    -> _rotate_on_resample: torch.rot90(k=3) (= 90 deg CW) iff the warped
                            tensor has H > W
    -> _crop_y: trim rows where (signal_prob + grid_prob) row-sum is below
                its mean, leaving the lines' (x = col, y = row) coord space

Two pieces of hidden state are needed to invert: the y-trim offset y0,
and whether rotation fired. We monkey-patch the wrapper's _crop_y and
_rotate_on_resample for one call to capture both.

This file is self-contained: the CLI entry point uses :func:`build_wrapper`
from :mod:`stenhede_overlay.runner`, so no ``kaggle_sol`` dependency.
"""
from __future__ import annotations

import types

import cv2
import numpy as np
import torch


# ---------------------- forward-chain instrumentation ----------------

def instrument_wrapper(wrapper):
    """Monkey-patch wrapper to record:
        - y-trim offsets (from `_crop_y` on the post-rotate stack)
        - rotation flag (from `_rotate_on_resample`)
        - x-trim offset (from `signal_extractor.preprocess_lines`,
          which drops all-empty columns before line matching)

    Returns a dict that gets populated each forward call.
    """
    holder = {"y0": None, "y1": None, "pre_h": None, "rotated": False,
              "x0": None, "x1": None, "pre_w": None}

    def patched_crop_y(self, image, signal_prob, grid_prob, text_prob):
        pre_h = image.shape[2]
        sg = signal_prob + grid_prob
        prob = torch.clamp(
            sg.squeeze().sum(dim=sg.dim() - 3)
            - sg.squeeze().sum(dim=sg.dim() - 3).mean(),
            min=0,
        )
        nz = (prob > 0).nonzero(as_tuple=True)[0]
        if nz.numel() == 0:
            y0, y1 = 0, pre_h - 1
        else:
            y0 = int(nz[0].item()); y1 = int(nz[-1].item())
        holder["y0"] = y0; holder["y1"] = y1; holder["pre_h"] = pre_h
        slices = (slice(None), slice(None), slice(y0, y1 + 1), slice(None))
        return (image[slices], signal_prob[slices],
                grid_prob[slices], text_prob[slices])

    wrapper._crop_y = types.MethodType(patched_crop_y, wrapper)

    orig_rotate = wrapper._rotate_on_resample

    def patched_rotate(self, aligned_image, aligned_signal_prob,
                       aligned_grid_prob, aligned_text_prob):
        # Rotation only fires if H > W in the post-warp tensor.
        holder["rotated"] = bool(
            aligned_image.shape[2] > aligned_image.shape[3]
        )
        return orig_rotate(aligned_image, aligned_signal_prob,
                           aligned_grid_prob, aligned_text_prob)

    wrapper._rotate_on_resample = types.MethodType(patched_rotate, wrapper)

    # signal_extractor.preprocess_lines drops all-zero/NaN border columns
    # via `lines[:, first:last+1]`. Capture `first` so we can re-add it
    # when projecting raw_lines back to the rectified image.
    orig_preprocess = wrapper.signal_extractor.preprocess_lines

    def patched_preprocess(lines):
        pre_w = lines.shape[1]
        lines_c = lines.clone()
        lines_c[lines_c == 0] = float("nan")
        valid = lines_c.nan_to_num(0.0).abs().sum(0) > 0
        idx = torch.nonzero(valid, as_tuple=True)[0]
        if idx.numel() == 0:
            holder["x0"], holder["x1"] = 0, pre_w - 1
        else:
            holder["x0"] = int(idx[0].item())
            holder["x1"] = int(idx[-1].item())
        holder["pre_w"] = pre_w
        return orig_preprocess(lines)

    wrapper.signal_extractor.preprocess_lines = patched_preprocess
    return holder


# ---------------------- inverse mapping ------------------------------

def build_inverse_mapper(src_pts_resampled, dst_pts_resampled,
                         H_resampled, W_resampled,
                         y0, x0, rotated, H_orig, W_orig):
    """Return a vectorized function mapping (x_line, y_line) -> (x_orig, y_orig).

    `src_pts_resampled` and `dst_pts_resampled` are in the resampled-image
    coordinate system (the space `apply_perspective` warps within).
    `(x_line, y_line)` is in the post-rotate, post-y-trim, post-x-trim
    space (the space Stenhede's `signal/raw_lines` are emitted in).
    `x0` and `y0` are the trim offsets so that
        (x_warp_post_rot, y_warp_post_rot) = (x_line + x0, y_line + y0).
    """
    src = src_pts_resampled.astype(np.float32)
    dst = dst_pts_resampled.astype(np.float32)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    sx = W_orig / W_resampled
    sy = H_orig / H_resampled

    def map_pts(xs, ys):
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        # 1. Undo (x, y) trim
        x_post = xs + x0
        y_post = ys + y0
        # 2. Undo rotation (torch.rot90(k=3) = 90 deg CW)
        # Forward (pre-rot -> post-rot): (x_in, y_in) -> (x_out=H_pre-1-y_in, y_out=x_in)
        # Inverse: (x_in=y_out, y_in=H_pre-1-x_out) where H_pre = pre-rotation height
        # Pre-rotation height = H_resampled (apply_perspective preserves resampled dims).
        if rotated:
            x_warp = y_post
            y_warp = (H_resampled - 1) - x_post
        else:
            x_warp = x_post
            y_warp = y_post
        # 3. Inverse perspective in resampled space
        pts = np.column_stack([x_warp, y_warp]).astype(np.float32).reshape(-1, 1, 2)
        pts_src = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
        # 4. Scale resampled -> original
        x_orig = pts_src[:, 0] * sx
        y_orig = pts_src[:, 1] * sy
        return x_orig, y_orig

    return map_pts


# ---------------------- rendering ------------------------------------

# 3x4+1R layout: top 3 panel rows (split into 4 leads by x-quartile),
# bottom row is the rhythm strip (single lead = II).
LAYOUT_3x4_1R = [
    ["I",   "aVR", "V1", "V4"],
    ["II",  "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
RHYTHM_LEAD = "II_rhythm"

# Per-lead BGR colors (cv2 ordering). Limb leads = warm tones, augmented =
# distinct, precordial V1->V6 = cool gradient. Chosen for visual separation
# rather than any clinical convention.
LEAD_COLORS_BGR = {
    "I":          (50,  50,  220),    # red
    "II":         (50,  180,  50),    # green
    "III":        (220, 50,  50),     # blue
    "aVR":        (40,  165, 255),    # orange
    "aVL":        (200, 50,  200),    # magenta
    "aVF":        (200, 200,  40),    # cyan
    "V1":         (255, 100, 100),    # light blue
    "V2":         (180, 80,  255),    # violet
    "V3":         (80,  255, 200),    # mint
    "V4":         (130, 130, 255),    # salmon
    "V5":         (60,  220, 220),    # yellow-cyan
    "V6":         (180, 220, 60),     # teal-green
    "II_rhythm":  (50,  220, 50),     # green (same family as II)
}


def _draw_polylines_finite(overlay, line_y, x_lo, x_hi, map_pts,
                           color, thickness):
    """Draw polylines on `overlay` for the finite stretches of `line_y`
    restricted to columns [x_lo, x_hi)."""
    if hasattr(line_y, "cpu"):
        line_y = line_y.cpu().numpy()
    seg = line_y[x_lo:x_hi]
    finite = np.isfinite(seg)
    if not finite.any():
        return
    diff = np.diff(finite.astype(np.int8))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0] + 1)
    if finite[0]:
        starts = [0] + starts
    if finite[-1]:
        ends = ends + [seg.shape[0]]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        xs = np.arange(x_lo + s, x_lo + e, dtype=np.float32)
        ys = seg[s:e].astype(np.float32)
        x_orig, y_orig = map_pts(xs, ys)
        pts = np.stack([x_orig, y_orig], axis=1).round().astype(np.int32)
        cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], False,
                      color, thickness=thickness, lineType=cv2.LINE_AA)


def _draw_legend(img, lead_color_pairs, x=20, y=20,
                 swatch=24, gap=8, font_scale=0.7):
    """Draw a 2-column legend on `img` at top-left."""
    h_per_row = swatch + gap
    n = len(lead_color_pairs)
    cols = 2
    rows = (n + cols - 1) // cols
    box_w = 270
    box_h = rows * h_per_row + gap
    cv2.rectangle(img, (x - 8, y - 8), (x + box_w, y + box_h),
                  (255, 255, 255), thickness=-1)
    cv2.rectangle(img, (x - 8, y - 8), (x + box_w, y + box_h),
                  (60, 60, 60), thickness=2)
    for i, (lead, color) in enumerate(lead_color_pairs):
        col = i % cols
        row = i // cols
        cx = x + col * (box_w // cols)
        cy = y + row * h_per_row
        cv2.rectangle(img, (cx, cy), (cx + swatch, cy + swatch),
                      color, thickness=-1)
        cv2.putText(img, lead, (cx + swatch + 6, cy + swatch - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (40, 40, 40),
                    thickness=2, lineType=cv2.LINE_AA)


def render_overlay(img_orig, lines, map_pts, src_pts_orig,
                   line_color=(0, 0, 255), polygon_color=(0, 200, 0),
                   alpha=0.55, thickness=3, per_lead=False):
    """Build a (BGR) image: original + traces (alpha-blended) +
    paper-boundary polygon (green, alpha).

    If `per_lead=True` and `lines.shape[0] == 4`, the 4 raw lines are
    interpreted as a 3x4+1R layout: top 3 are panel rows split by
    x-quartile into 12 leads; bottom is the full-width rhythm strip
    (II). Each lead gets its own color and a legend is overlaid.
    Otherwise, all lines are drawn in `line_color` (this is the safe
    fallback for arbitrary layouts).
    """
    overlay = img_orig.copy()
    n_lines = lines.shape[0]
    W = lines.shape[1]

    use_per_lead = bool(per_lead and n_lines == 4)
    legend_pairs = []

    if use_per_lead:
        # Sort line indices by mean y, ascending = top-to-bottom on paper.
        if hasattr(lines, "cpu"):
            lines_np = lines.cpu().numpy()
        else:
            lines_np = np.asarray(lines)
        mean_y = np.array([np.nanmean(ln) if np.isfinite(ln).any() else np.inf
                           for ln in lines_np])
        order = np.argsort(mean_y)
        # First 3 = panel rows (3x4 quarters), last = rhythm strip.
        col_edges = np.linspace(0, W, 5).astype(int)
        for row_idx, line_idx in enumerate(order[:3]):
            row_leads = LAYOUT_3x4_1R[row_idx]
            for ci, lead in enumerate(row_leads):
                x_lo, x_hi = col_edges[ci], col_edges[ci + 1]
                color = LEAD_COLORS_BGR[lead]
                _draw_polylines_finite(overlay, lines_np[line_idx],
                                       x_lo, x_hi, map_pts, color, thickness)
                legend_pairs.append((lead, color))
        rhythm_idx = order[3]
        rhythm_color = LEAD_COLORS_BGR[RHYTHM_LEAD]
        _draw_polylines_finite(overlay, lines_np[rhythm_idx], 0, W, map_pts,
                               rhythm_color, thickness)
        legend_pairs.append(("II (rhythm)", rhythm_color))
    else:
        for li in range(n_lines):
            line_y = lines[li]
            _draw_polylines_finite(overlay, line_y, 0, W, map_pts,
                                   line_color, thickness)

    # Paper boundary polygon
    poly = src_pts_orig.round().astype(np.int32)
    cv2.polylines(overlay, [poly.reshape(-1, 1, 2)], True,
                  polygon_color, thickness=3, lineType=cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, alpha, img_orig, 1.0 - alpha, 0)
    if use_per_lead:
        _draw_legend(blended, legend_pairs)
    return blended


# ---------------------- end-to-end helpers ---------------------------

def render_overlay_from_output(out, holder, wrapper, img_bgr,
                               alpha=0.55, thickness=3, per_lead=True):
    """Render the overlay using an already-computed Stenhede output dict.
    Use this from a runner that needs the wrapper's other outputs in the
    same call (so the forward pass isn't repeated).
    """
    H_orig, W_orig = img_bgr.shape[:2]
    src_pts_resampled = out["source_points"].cpu().numpy()
    raw_lines = out["signal"]["raw_lines"]
    input_resampled = out["input_image"].squeeze().cpu()
    H_resampled = int(input_resampled.shape[-2])
    W_resampled = int(input_resampled.shape[-1])

    src_t = torch.tensor(src_pts_resampled, dtype=torch.float32)
    dst_t = wrapper.cropper._calculate_destination_points(
        H_resampled, W_resampled, src_t)
    dst_pts_resampled = dst_t.cpu().numpy()

    map_pts = build_inverse_mapper(
        src_pts_resampled, dst_pts_resampled,
        H_resampled, W_resampled,
        y0=holder["y0"] if holder["y0"] is not None else 0,
        x0=holder["x0"] if holder["x0"] is not None else 0,
        rotated=holder["rotated"],
        H_orig=H_orig, W_orig=W_orig,
    )

    src_pts_orig = src_pts_resampled.copy()
    src_pts_orig[:, 0] *= (W_orig / W_resampled)
    src_pts_orig[:, 1] *= (H_orig / H_resampled)

    return render_overlay(img_bgr, raw_lines, map_pts, src_pts_orig,
                          alpha=alpha, thickness=thickness,
                          per_lead=per_lead)


def overlay_one(wrapper, holder, img_bgr, alpha=0.55, thickness=3,
                per_lead=True, layout_filter: str | None = None):
    """Run an instrumented Stenhede wrapper on `img_bgr` and return a
    BGR overlay image. `holder` must be the dict returned by
    `instrument_wrapper(wrapper)`.

    Returns ``(overlay_image, layout_name, layout_cost)``.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    out = wrapper(t, layout_should_include_substring=layout_filter)
    blended = render_overlay_from_output(
        out, holder, wrapper, img_bgr,
        alpha=alpha, thickness=thickness, per_lead=per_lead,
    )
    return blended, out["layout_name"], float(
        out["signal"]["layout_matching_cost"])


