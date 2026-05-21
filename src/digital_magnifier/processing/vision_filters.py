"""Vision filters for low-vision accessibility.

Each filter takes a BGR uint8 frame and returns a BGR uint8 frame
of the same shape, so the downstream display pipeline doesn't need
to care which filter is active.

Five filters ship in MVP 0.1:

- ``normal``: pass-through. The default; cheap.
- ``grayscale``: desaturate. Useful when colour is distracting.
- ``high_contrast``: CLAHE on the L channel of LAB. Boosts local
  contrast without losing colour. Best general-purpose filter for
  reading printed text.
- ``inverted``: photographic negative. Some children with high
  myopia or photophobia find white-on-black easier than
  black-on-white.
- ``binary``: adaptive threshold to pure black and white. Maximum
  contrast; ideal for clean printed text with even lighting.

Adding a new filter is two lines: a constant, a dispatch entry,
and a private function. The list in ``AVAILABLE_FILTERS`` is what
the config loader and overlay use to enumerate options.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np


logger = logging.getLogger(__name__)


# Filter name constants. Use these everywhere instead of bare
# strings so a typo fails at import time, not runtime.
FILTER_NORMAL: str = "normal"
FILTER_GRAYSCALE: str = "grayscale"
FILTER_HIGH_CONTRAST: str = "high_contrast"
FILTER_INVERTED: str = "inverted"
FILTER_BINARY: str = "binary"


AVAILABLE_FILTERS: tuple[str, ...] = (
    FILTER_NORMAL,
    FILTER_GRAYSCALE,
    FILTER_HIGH_CONTRAST,
    FILTER_INVERTED,
    FILTER_BINARY,
)


def apply_filter(frame: np.ndarray, filter_name: str) -> np.ndarray:
    """Apply the named filter to ``frame``.

    Unknown filter names are logged and fall through to ``normal``
    rather than raising — a typo in ``app_config.yaml`` must not
    crash the device during a reading session.

    Parameters
    ----------
    frame : np.ndarray
        Source frame, BGR uint8, shape ``(H, W, 3)``.
    filter_name : str
        One of :data:`AVAILABLE_FILTERS`.

    Returns
    -------
    np.ndarray
        Filtered frame at the same shape and dtype as the input.
    """
    if filter_name == FILTER_NORMAL:
        return frame
    if filter_name == FILTER_GRAYSCALE:
        return _grayscale(frame)
    if filter_name == FILTER_HIGH_CONTRAST:
        return _high_contrast(frame)
    if filter_name == FILTER_INVERTED:
        return _inverted(frame)
    if filter_name == FILTER_BINARY:
        return _binary(frame)

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


def _high_contrast(frame: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel of LAB.

    Contrast-Limited Adaptive Histogram Equalization stretches the
    local intensity range, which makes faint text pop without
    blowing out highlights. Working in LAB and only touching L
    preserves colour fidelity.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _inverted(frame: np.ndarray) -> np.ndarray:
    """Photographic negative.

    For users sensitive to glare from white pages, inverting to a
    dark background with light text reduces eye strain.
    """
    return cv2.bitwise_not(frame)


def _binary(frame: np.ndarray) -> np.ndarray:
    """Adaptive threshold to pure black-and-white text.

    Adaptive (rather than global) thresholding handles uneven
    lighting — important on a desk where the lamp creates one
    bright spot. ``blockSize=21`` is large enough to ignore
    individual letters and small enough to react to lighting
    gradients across the page. ``C=10`` shifts the threshold
    slightly toward black so faint ink isn't lost.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=21,
        C=10,
    )
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)