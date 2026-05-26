"""Vision filter implementations.

Each filter is a pure function that takes a BGR ``uint8`` frame and
returns one of the same shape and dtype. The dispatcher
:func:`apply_filter` selects a filter by name and forwards
per-filter configuration from the YAML so tunings (binary
threshold, CLAHE clip limit, colour) can be tweaked without code
changes.

New filters can be added by:
  1. Adding a ``FILTER_*`` name constant and entry in
     :data:`AVAILABLE_FILTERS`.
  2. Writing a ``_filter_name`` function with sensible
     keyword-arg defaults.
  3. Adding an ``elif`` branch in :func:`apply_filter`.
  4. Listing the filter in ``filters.available`` in
     ``config/app_config.yaml`` (comment it out to disable).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# Filter name constants — used both internally and in YAML.
FILTER_NORMAL: str = "normal"
FILTER_GRAYSCALE: str = "grayscale"
FILTER_HIGH_CONTRAST: str = "high_contrast"
FILTER_INVERTED: str = "inverted"
FILTER_BINARY: str = "binary"
FILTER_YELLOW_ON_BLACK: str = "yellow_on_black"
FILTER_WHITE_ON_BLACK: str = "white_on_black"


AVAILABLE_FILTERS: tuple[str, ...] = (
    FILTER_NORMAL,
    FILTER_GRAYSCALE,
    FILTER_HIGH_CONTRAST,
    FILTER_INVERTED,
    FILTER_BINARY,
    FILTER_YELLOW_ON_BLACK,
    FILTER_WHITE_ON_BLACK,
)


def apply_filter(
    frame: np.ndarray,
    filter_name: str,
    config: Optional[dict[str, dict[str, Any]]] = None,
) -> np.ndarray:
    """Apply the named filter to ``frame``.

    Unknown filter names are logged and fall through to ``normal``
    rather than raising — a typo in ``app_config.yaml`` must not
    crash the device during a reading session.

    Parameters
    ----------
    frame
        Source frame, BGR uint8, shape ``(H, W, 3)``.
    filter_name
        One of :data:`AVAILABLE_FILTERS`.
    config
        Optional mapping ``{filter_name: {param: value, ...}}``
        forwarded to the implementation as keyword arguments.
        Missing keys fall back to defaults defined per filter.

    Returns
    -------
    np.ndarray
        Filtered frame at the same shape and dtype as the input.
    """
    cfg = (config or {}).get(filter_name, {})

    if filter_name == FILTER_NORMAL:
        return frame
    if filter_name == FILTER_GRAYSCALE:
        return _grayscale(frame)
    if filter_name == FILTER_HIGH_CONTRAST:
        return _high_contrast(frame, **cfg)
    if filter_name == FILTER_INVERTED:
        return _inverted(frame)
    if filter_name == FILTER_BINARY:
        return _binary(frame, **cfg)
    if filter_name == FILTER_YELLOW_ON_BLACK:
        return _yellow_on_black(frame, **cfg)
    if filter_name == FILTER_WHITE_ON_BLACK:
        return _white_on_black(frame, **cfg)

    logger.warning(
        "unknown filter %r; falling through to 'normal'", filter_name
    )
    return frame


# ---------------------------------------------------------------------------
# Filter implementations
# ---------------------------------------------------------------------------


def _grayscale(frame: np.ndarray) -> np.ndarray:
    """Desaturate to grayscale. 3 channels are preserved so downstream
    rendering and saving don't need a special case for grayscale."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _high_contrast(
    frame: np.ndarray,
    clip_limit: float = 3.0,
    tile_size: int = 8,
) -> np.ndarray:
    """CLAHE on the L channel of LAB.

    Contrast-Limited Adaptive Histogram Equalization stretches the
    local intensity range, which makes faint text pop without
    blowing out highlights. Working in LAB and only touching L
    preserves colour fidelity.

    Parameters
    ----------
    clip_limit
        How aggressively contrast is stretched. 1.0 = gentle,
        3.0 = strong (default), 4.0+ may amplify noise.
    tile_size
        Side length of the tile grid (in pixels). Smaller = more
        local; 8 is a good general value for printed text.
    """
    try:
        clip = float(clip_limit)
        size = int(tile_size)
    except (TypeError, ValueError):
        clip, size = 3.0, 8
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(size, size))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _inverted(frame: np.ndarray) -> np.ndarray:
    """Photographic negative.

    For users sensitive to glare from white pages, inverting to a
    dark background with light text reduces eye strain.
    """
    return cv2.bitwise_not(frame)


def _binary(
    frame: np.ndarray,
    block_size: int = 21,
    offset: int = 10,
) -> np.ndarray:
    """Adaptive threshold to pure black-and-white text.

    Adaptive (rather than global) thresholding handles uneven
    lighting — important on a desk where the lamp creates one
    bright spot.

    Parameters
    ----------
    block_size
        Neighbourhood side length (must be odd ≥ 3). Larger blocks
        smooth out gradients but blur fine detail; smaller blocks
        react to every speck. 21–51 is the sweet spot for text.
    offset
        Constant subtracted from the mean — positive values shift
        the threshold toward black (more white pixels become
        white). Raise this if faint ink is being lost.
    """
    try:
        bs = int(block_size)
        if bs < 3:
            bs = 3
        if bs % 2 == 0:
            bs += 1                # opencv requires odd block_size
        c = int(offset)
    except (TypeError, ValueError):
        bs, c = 21, 10
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=bs,
        C=c,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def _yellow_on_black(
    frame: np.ndarray,
    block_size: int = 21,
    offset: int = 10,
) -> np.ndarray:
    """High-contrast yellow text on a pure black background.

    Many low-vision users (particularly those with macular
    degeneration or photophobia) report yellow-on-black is the
    most comfortable high-contrast palette: enough warmth to
    avoid the harshness of pure-white-on-black, while still
    delivering maximum perceived contrast.

    Internally this is an adaptive threshold followed by
    colourisation: ink → yellow, page → black.
    """
    try:
        bs = int(block_size)
        if bs < 3:
            bs = 3
        if bs % 2 == 0:
            bs += 1
        c = int(offset)
    except (TypeError, ValueError):
        bs, c = 21, 10
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Invert so "ink" pixels come out white (255), then we'll
    # colourise them yellow.
    mask = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=bs,
        C=c,
    )
    # BGR yellow = (0, 255, 255). Set G and R channels to the mask
    # to produce yellow ink on a black background.
    out = np.zeros_like(frame)
    out[:, :, 1] = mask        # G
    out[:, :, 2] = mask        # R
    return out


def _white_on_black(
    frame: np.ndarray,
    block_size: int = 21,
    offset: int = 10,
) -> np.ndarray:
    """Pure white text on a pure black background.

    The extreme contrast option — maximum luminance contrast,
    minimum colour information. Useful for severe low-vision
    cases or when reading at the edge of the user's effective
    range. Essentially ``inverted`` + ``binary`` combined into
    one step.
    """
    try:
        bs = int(block_size)
        if bs < 3:
            bs = 3
        if bs % 2 == 0:
            bs += 1
        c = int(offset)
    except (TypeError, ValueError):
        bs, c = 21, 10
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=bs,
        C=c,
    )
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)