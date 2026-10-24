"""Project Stenhede's recovered ECG traces back onto the original photograph.

Build a wrapper, instrument it once, then call ``overlay_one`` per image.

Example:
    >>> from src.StenhendeV2.overlay import build_wrapper, instrument_wrapper, overlay_one
    >>> import cv2
    >>> wrapper = build_wrapper()
    >>> holder = instrument_wrapper(wrapper)
    >>> img = cv2.imread("photo.jpg")
    >>> overlay_bgr, layout, cost = overlay_one(wrapper, holder, img)
    >>> cv2.imwrite("overlay.png", overlay_bgr)
"""
from __future__ import annotations

from .overlay import (  # noqa: F401
    LAYOUT_3x4_1R,
    LEAD_COLORS_BGR,
    build_inverse_mapper,
    instrument_wrapper,
    overlay_one,
    render_overlay,
    render_overlay_from_output,
)
from .runner import build_wrapper, run_wrapper  # noqa: F401
