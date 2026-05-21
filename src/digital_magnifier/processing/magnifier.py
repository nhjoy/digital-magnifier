"""Digital zoom and pan for the magnifier.

Approach: crop a smaller region from the centre of the frame
(offset by pan), then upscale that crop back to the original
resolution. At ``zoom=1.0`` the frame is passed through unchanged.

Pan is in normalised coordinates: ``pan_x`` and ``pan_y`` range
from -1.0 to +1.0 and represent the fraction of the available
pan range to apply. The available pan range depends on the zoom
level — at higher zoom the crop is smaller so more of the original
frame is reachable by panning.

Mathematics
-----------
At zoom factor ``z``, the crop is sized ``(w/z, h/z)``. The crop
centre can move within ``±(w - w/z)/2`` horizontally without going
outside the source frame; ``pan_x = ±1.0`` corresponds to those
limits. ``pan_x = 0`` keeps the crop centred.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np


logger = logging.getLogger(__name__)


def apply_zoom(
    frame: np.ndarray,
    zoom: float,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Crop and upscale ``frame`` according to ``zoom`` and pan.

    Parameters
    ----------
    frame : np.ndarray
        BGR uint8 source frame, shape ``(H, W, 3)``.
    zoom : float
        Zoom factor. Values ``<= 1.0`` return the frame unchanged.
        Values are not clamped here; the caller (app controller)
        enforces the configured min/max.
    pan_x, pan_y : float
        Normalised pan offsets in ``[-1.0, 1.0]``. Values outside
        this range are clamped. ``(0, 0)`` keeps the crop centred.
    interpolation : int
        OpenCV interpolation flag. ``INTER_LINEAR`` is the default
        (fast, smooth). For sharper text, callers may pass
        ``INTER_CUBIC`` or ``INTER_LANCZOS4``.

    Returns
    -------
    np.ndarray
        A new BGR uint8 array at the same resolution as the input.
    """
    if zoom <= 1.0:
        return frame

    h, w = frame.shape[:2]

    # Crop dimensions. We use floor so a 1280-pixel-wide frame at
    # zoom=3.0 produces a crop of 426 px (rather than 426.66).
    crop_w = max(1, int(w / zoom))
    crop_h = max(1, int(h / zoom))

    # Clamp pan to its valid range.
    pan_x = max(-1.0, min(1.0, pan_x))
    pan_y = max(-1.0, min(1.0, pan_y))

    # Max distance the crop centre can move from the frame centre
    # before any part of the crop would fall outside the source.
    max_offset_x = (w - crop_w) // 2
    max_offset_y = (h - crop_h) // 2

    center_x = w // 2 + int(round(pan_x * max_offset_x))
    center_y = h // 2 + int(round(pan_y * max_offset_y))

    # Compute crop bounds. Clamp to frame extents to guarantee a
    # valid slice even if rounding nudges the centre off-frame.
    x1 = max(0, center_x - crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    x2 = min(w, x1 + crop_w)
    y2 = min(h, y1 + crop_h)
    # Re-clamp x1/y1 in case x2/y2 hit the edge — keeps crop size
    # consistent rather than shrinking near the borders.
    x1 = max(0, x2 - crop_w)
    y1 = max(0, y2 - crop_h)

    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h), interpolation=interpolation)