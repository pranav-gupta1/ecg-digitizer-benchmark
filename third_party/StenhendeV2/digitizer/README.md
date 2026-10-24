# digitizer/

Inference pipeline for the Stenhede / Open-ECG-Digitizer model.

`InferenceWrapper` (`model/inference_wrapper.py`) chains:

1. U-Net semantic segmentation (`model/unet.py`) — signal / grid / text / background.
2. GPU Hough-based perspective detection (`model/perspective_detector.py`).
3. Vectorized cropping / perspective warp (`model/cropper.py`).
4. Pixel-spacing estimation (`model/pixel_size_finder.py`).
5. Optional dewarping (`model/dewarper.py`, disabled in the default config).
6. Iterative connected-components signal extraction (`model/signal_extractor.py`).
7. Layout identification via a second U-Net + bipartite matching
   (`model/lead_identifier.py`).

## Configs

`configs/` holds bundled YAMLs:

- `inference_wrapper_george-moody-2024.yml` — primary inference config.
- `lead_layouts_george-moody-2024.yml` — layout templates (3x4+1R, etc.).
- `lead_name_unet.yml` — config for the lead-identifier U-Net.

`config_loader.get_cfg()` expands `${STENHEDE_WEIGHTS_PATH}` and
resolves sibling YAML references to absolute paths automatically.

## Programmatic use

```python
from src.StenhendeV2.digitizer.config_loader import get_cfg
from src.StenhendeV2.digitizer.utils import find_config_path, import_class_from_path

cfg = get_cfg(find_config_path("inference_wrapper_george-moody-2024.yml"))
wrapper = import_class_from_path(cfg.MODEL.class_path)(**cfg.MODEL.KWARGS)
```

For batch inference over a folder, use `pipeline.main(cfg)` or run
`src/scripts/stenhede_digitize.py`.
