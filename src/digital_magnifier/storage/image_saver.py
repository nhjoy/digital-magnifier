"""Save captured frames to disk with timestamped filenames.

The image saver is intentionally small. Its only job is to take a
processed frame (post-zoom, post-filter, no UI chrome) and write it
to the configured directory. The directory is created on first use
so a missing ``captures/`` folder is never a failure mode.

Filename format: ``capture_YYYYMMDD_HHMMSS_mmm.png``

Millisecond suffix means rapid-fire captures (e.g., from a held
button) don't collide — important because a child may press the
capture button several times in quick succession.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


class ImageSaverError(Exception):
    """Raised when an image cannot be written to disk."""


class ImageSaver:
    """Writes frames to a configured directory."""

    def __init__(
        self,
        output_directory: str | Path,
        file_format: str = "png",
    ) -> None:
        """
        Parameters
        ----------
        output_directory : str or Path
            Where captures are written. Created if it does not exist.
        file_format : str
            File extension and OpenCV-supported format. ``"png"``
            is the default; ``"jpg"`` is also viable if storage
            space matters more than fidelity.
        """
        self._dir = Path(output_directory).expanduser().resolve()
        self._format = file_format.lstrip(".").lower()
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.debug("ImageSaver writing to %s", self._dir)

    @property
    def output_directory(self) -> Path:
        return self._dir

    def save(
        self,
        frame: np.ndarray,
        timestamp: datetime | None = None,
    ) -> Path:
        """Write ``frame`` to disk and return the resulting path.

        Parameters
        ----------
        frame : np.ndarray
            BGR uint8 array, as delivered by the camera HAL.
        timestamp : datetime, optional
            Used in the filename. Defaults to ``datetime.now()``;
            tests pass an explicit value for determinism.

        Raises
        ------
        ImageSaverError
            If ``cv2.imwrite`` reports failure (e.g. disk full,
            permissions, unsupported format).
        """
        # Lazy-import cv2 so this module imports cleanly in test
        # environments without OpenCV.
        import cv2

        ts = timestamp or datetime.now()
        # Strip the last 3 chars of microsecond suffix to keep
        # millisecond precision (cv2 filenames don't need ms but
        # rapid taps shouldn't collide).
        stamp = ts.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"capture_{stamp}.{self._format}"
        path = self._dir / filename

        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise ImageSaverError(f"cv2.imwrite returned false for {path}")

        logger.info("saved capture to %s", path)
        return path