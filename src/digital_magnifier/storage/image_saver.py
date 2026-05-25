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

    # ------------------------------------------------------------------
    # Gallery support (MVP 0.4)
    # ------------------------------------------------------------------

    # File extensions the gallery is willing to load. Kept small on
    # purpose — the saver only writes PNG by default, so anything
    # else in the directory is probably a stray file that should not
    # appear in the browse list.
    _IMAGE_EXTENSIONS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})

    def list_images(self, newest_first: bool = True) -> list[Path]:
        """Return all image files in the output directory.

        Filters by extension (PNG / JPG) so stray text files or
        thumbnails don't show up. Sorted by filesystem modification
        time so the order is meaningful even if files were copied in
        manually (rather than relying on the ``capture_YYYYMMDD…``
        filename pattern, which would miss user-imported images).

        Parameters
        ----------
        newest_first : bool
            True (default) — most recently modified file first.
            False — oldest first.

        Returns
        -------
        list[Path]
            Absolute paths to image files. Empty list if the directory
            is missing, empty, or has no images. Never raises for the
            "no captures yet" case — that's a legitimate state.
        """
        if not self._dir.exists():
            return []

        images: list[Path] = []
        try:
            for entry in self._dir.iterdir():
                if not entry.is_file():
                    continue
                if entry.suffix.lower() not in self._IMAGE_EXTENSIONS:
                    continue
                images.append(entry)
        except OSError:
            # Permission denied, disappeared mid-scan, etc. Treat as empty.
            logger.exception("could not scan %s", self._dir)
            return []

        # Sort by mtime so user-imported files appear in their natural
        # order, not interleaved by filename.
        try:
            images.sort(key=lambda p: p.stat().st_mtime, reverse=newest_first)
        except OSError:
            # A file vanished between iterdir and stat. Fall back to
            # whatever order we have.
            logger.exception("stat failed while sorting captures")

        return images

    def delete_image(self, path: Path) -> None:
        """Remove an image file from the output directory.

        Refuses to delete anything outside the configured directory —
        a tiny but important safety check, since the gallery passes
        whatever path it has and we don't want a path-traversal bug
        to wipe a user's home directory.

        Parameters
        ----------
        path : Path
            File to remove. Must be inside :attr:`output_directory`.

        Raises
        ------
        ImageSaverError
            If the path is outside the output directory, or the
            underlying ``unlink`` failed.
        """
        target = Path(path).resolve()

        # Containment check: target must be a descendant of our
        # output directory. ``relative_to`` raises if it isn't.
        try:
            target.relative_to(self._dir)
        except ValueError as exc:
            raise ImageSaverError(
                f"refusing to delete {target}: not inside output "
                f"directory {self._dir}"
            ) from exc

        try:
            target.unlink()
        except FileNotFoundError:
            # Already gone — treat as success; the caller's mental
            # model ("this image no longer exists") is satisfied.
            logger.warning("delete_image: %s already gone", target)
        except OSError as exc:
            raise ImageSaverError(f"could not delete {target}: {exc}") from exc
        else:
            logger.info("deleted capture %s", target)