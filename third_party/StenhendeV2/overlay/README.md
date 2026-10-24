# overlay/

Project Stenhede's recovered ECG traces back onto the original photograph
as a transparent coloured overlay.

The overlay inverts the digitizer's geometric chain:

```
original (W_orig, H_orig)
  -> resample (longer side -> 3000 px)
  -> apply_perspective (4 paper corners -> centered rectangle)
  -> rotate_on_resample (90° CW iff portrait after warp)
  -> crop_y (drop empty rows)
  -> signal_extractor.preprocess_lines (drop empty cols)
```

Two pieces of hidden state — the y-trim offset and whether rotation fired —
are captured by monkey-patching `_crop_y` and `_rotate_on_resample` for
the duration of one forward pass (`instrument_wrapper`).

## Programmatic use

```python
import cv2
from src.StenhendeV2.overlay import build_wrapper, instrument_wrapper, overlay_one

wrapper = build_wrapper()                # uses bundled inference YAML
holder = instrument_wrapper(wrapper)     # call once per wrapper, then reuse

img = cv2.imread("photo.jpg")
overlay_bgr, layout, cost = overlay_one(
    wrapper, holder, img,
    alpha=0.55, thickness=3, per_lead=True,
)
cv2.imwrite("overlay.png", overlay_bgr)
print("layout:", layout, "  cost:", cost)
```

## Per-lead colouring

`per_lead=True` recognises the **3x4+1R** layout: when the signal extractor
returns exactly 4 polylines they are sorted top-to-bottom and the top three
are split by x-quartile into the 12 standard leads, with the bottom line
used as the rhythm strip. For other layouts (or `per_lead=False`) every
line is drawn in a single colour.
