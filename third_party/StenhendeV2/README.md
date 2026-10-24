# StenhendeV2/

End-to-end ECG photograph digitization. Adapted from the published
**Open-ECG-Digitizer** (Stenhede, *npj Digital Medicine*, 2026) plus a
small overlay package that projects recovered traces back onto the
original photo.

## Subpackages

- `digitizer/` — the inference pipeline: U-Net segmentation, GPU-vectorized
  perspective correction, signal extraction, layout matching.
- `overlay/` — wraps the digitizer and renders coloured per-lead polylines
  on top of the input photo.

## Runnable entry points

Both live in `src/scripts/` with PEP 723 inline metadata:

- `src/scripts/stenhede_digitize.py` — batch ECG photo → CSV + 4-panel PNG
- `src/scripts/stenhede_overlay.py` — single ECG photo → coloured overlay PNG

## Setup

1. Add `STENHEDE_WEIGHTS_PATH=/path/to/weights` to your local `data.env`.
   That directory must contain `unet_weights_07072025.pt` and
   `lead_name_unet_weights_07072025.pt`.
2. `uv sync` to install dependencies.
