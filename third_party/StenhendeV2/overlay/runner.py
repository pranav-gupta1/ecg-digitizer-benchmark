"""Build a Stenhede ``InferenceWrapper`` from a bundled YAML config.

Weight paths are resolved via the ``${STENHEDE_WEIGHTS_PATH}`` env var
expanded by ``digitizer.config_loader``. Sibling YAML references inside
the config are resolved relative to the config's own directory. No
``sys.path`` munging or working-directory tricks are needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ..digitizer.config_loader import get_cfg
from ..digitizer.utils import import_class_from_path

DEFAULT_CONFIG_NAME = "inference_wrapper_george-moody-2024.yml"
_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "digitizer" / "configs"


def build_wrapper(config_name: str = DEFAULT_CONFIG_NAME) -> Any:
    """Build the Stenhede ``InferenceWrapper`` from a bundled YAML config.

    Args:
        config_name (str): Filename of a YAML in
            ``src/StenhendeV2/digitizer/configs/``.

    Returns:
        Any: An instantiated ``InferenceWrapper`` ready for inference.
    """
    cfg_path = _CONFIGS_DIR / config_name
    cfg = get_cfg(str(cfg_path))
    wrapper_cls = import_class_from_path(cfg.MODEL.class_path)
    return wrapper_cls(**cfg.MODEL.KWARGS)


def run_wrapper(
    wrapper: Any,
    img_bgr: np.ndarray,
    layout_filter: str | None = None,
) -> dict[str, Any]:
    """Run a wrapper on a BGR image and return its raw output dict.

    Args:
        wrapper (Any): A built ``InferenceWrapper``.
        img_bgr (np.ndarray): Input image in BGR (OpenCV) channel order.
        layout_filter (str | None): Optional substring restricting the
            layout matcher (e.g. ``'3x4'``).

    Returns:
        dict[str, Any]: The wrapper's full output dict.
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return wrapper(tensor, layout_should_include_substring=layout_filter)
