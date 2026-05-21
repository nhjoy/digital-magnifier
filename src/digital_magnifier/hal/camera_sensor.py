"""Mock camera sensor — webcam if available, synthetic frame otherwise.

Used during development on laptops and in WSL. Strategy:

1. Try to open ``cv2.VideoCapture(index)``. If it returns a working
   frame on the first read, use that for the rest of the session.
2. Otherwise, if ``device.fallback_image`` is set in
   ``camera_config.yaml`` and the file exists, load it and serve
   that on every ``get_frame()`` call.
3. Otherwise, generate a synthetic frame that looks like simulated
   text on a page — enough structure that zoom/pan/filter actually
   show visible changes during development.

The fallback chain matters because WSL does not expose USB
webcams by default and the Pi Camera Module 3 isn't accessible
from a laptop. Without the synthetic frame, MVP 0.1 development
would be blocked on having physical hardware.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from digital_magnifier.hal.camera_base import CameraError, CameraSensor


logger = logging.getLogger(__name__)


# Sensor modes — used internally to track which fallback we are in.
_MODE_UNINITIALIZED = "uninitialized"
_MODE_WEBCAM = "webcam"
_MODE_FALLBACK_IMAGE = "fallback_image"
_MODE_SYNTHETIC = "synthetic"
_MODE_STOPPED = "stopped"


class MockCameraSensor(CameraSensor):
    """Webcam-or-synthetic-frame camera for development.

    Parameters
    ----------
    config : dict
        Parsed contents of ``config/camera_config.yaml``. Recognised
        sections:

        - ``device.source``: ``"webcam"`` (default; auto-falls back)
          or ``"test_image"`` (skip webcam, go straight to fallback).
        - ``device.index``: integer device index for the webcam.
          Default 0.
        - ``device.fallback_image``: optional path to a still image
          used when no webcam is available. Resolved relative to
          the project root.
        - ``resolution.width`` / ``resolution.height``: the frame
          size to request from the webcam and to use for the
          synthetic frame. Defaults 1280x720.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        device_cfg = config.get("device", {})
        res_cfg = config.get("resolution", {})

        self._source: str = str(device_cfg.get("source", "webcam")).lower()
        self._device_index: int = int(device_cfg.get("index", 0))
        self._fallback_image_path: str | None = device_cfg.get(
            "fallback_image"
        )

        self._width: int = int(res_cfg.get("width", 1280))
        self._height: int = int(res_cfg.get("height", 720))

        self._capture: Any = None  # cv2.VideoCapture, populated by start()
        self._static_frame: np.ndarray | None = None
        self._mode: str = _MODE_UNINITIALIZED

    # ------------------------------------------------------------------
    # CameraSensor interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the camera, falling through the chain on failure."""
        if self._source != "test_image":
            if self._try_open_webcam():
                return
        self._init_static_frame()

    def stop(self) -> None:
        """Release webcam if held; drop any static frame."""
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                logger.exception("error releasing webcam")
            self._capture = None
        self._static_frame = None
        self._mode = _MODE_STOPPED

    def get_frame(self) -> np.ndarray:
        """Return a fresh frame.

        For webcam mode each call reads a new frame. For static modes
        (``fallback_image`` and ``synthetic``) a copy of the cached
        frame is returned so downstream mutations cannot corrupt the
        cache.
        """
        if self._mode == _MODE_WEBCAM and self._capture is not None:
            ok, frame = self._capture.read()
            if not ok or frame is None:
                raise CameraError(
                    f"webcam read failed on device {self._device_index}"
                )
            return frame

        if self._static_frame is not None:
            return self._static_frame.copy()

        raise CameraError(
            "camera not started (call start() before get_frame())"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _try_open_webcam(self) -> bool:
        """Attempt to open the configured webcam. True on success."""
        try:
            import cv2
        except ImportError:
            logger.warning("cv2 not installed; cannot open webcam")
            return False

        try:
            cap = cv2.VideoCapture(self._device_index)
        except Exception:
            logger.exception(
                "VideoCapture raised while opening device %d",
                self._device_index,
            )
            return False

        if not cap.isOpened():
            logger.info(
                "webcam %d not available; will use fallback",
                self._device_index,
            )
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        # Sanity-check by reading one frame. Webcams sometimes
        # report opened==True but then refuse to deliver frames.
        ok, _ = cap.read()
        if not ok:
            logger.info(
                "webcam %d opened but first read failed; using fallback",
                self._device_index,
            )
            cap.release()
            return False

        self._capture = cap
        self._mode = _MODE_WEBCAM
        logger.info(
            "MockCameraSensor: webcam %d open at %dx%d",
            self._device_index, self._width, self._height,
        )
        return True

    def _init_static_frame(self) -> None:
        """Load fallback image, or generate synthetic frame as a last resort."""
        if self._fallback_image_path:
            loaded = self._try_load_fallback_image(self._fallback_image_path)
            if loaded is not None:
                self._static_frame = loaded
                self._mode = _MODE_FALLBACK_IMAGE
                logger.info(
                    "MockCameraSensor: using fallback image %s",
                    self._fallback_image_path,
                )
                return

        self._static_frame = self._generate_synthetic_frame()
        self._mode = _MODE_SYNTHETIC
        logger.info(
            "MockCameraSensor: using synthetic frame (%dx%d)",
            self._width, self._height,
        )

    def _try_load_fallback_image(self, path_str: str) -> np.ndarray | None:
        """Try to load and resize a fallback image. Returns None on failure."""
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            # Resolve relative to project root for robustness.
            from digital_magnifier.utils.config_loader import find_project_root
            path = find_project_root() / path

        if not path.exists():
            logger.warning("fallback image not found: %s", path)
            return None

        try:
            import cv2
            img = cv2.imread(str(path))
        except Exception:
            logger.exception("failed to load fallback image %s", path)
            return None

        if img is None:
            logger.warning("fallback image unreadable: %s", path)
            return None

        try:
            import cv2
            return cv2.resize(img, (self._width, self._height))
        except Exception:
            logger.exception(
                "failed to resize fallback image to %dx%d",
                self._width, self._height,
            )
            return None

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Build a frame resembling a page of printed text.

        Deterministic (seeded) so the same frame appears every run —
        makes visual regression testing easier. The watermark label
        in the bottom-left makes it obvious to the developer that
        no real camera is feeding the system.
        """
        # Near-white "page" background (240 rather than 255 avoids
        # blowing out highlights in the high-contrast filter).
        frame = np.full(
            (self._height, self._width, 3), 240, dtype=np.uint8
        )

        rng = np.random.default_rng(seed=42)
        margin_x = self._width // 12
        margin_y = self._height // 14
        line_height = max(6, self._height // 50)
        line_spacing = int(line_height * 2.2)

        y = margin_y
        line_index = 0
        while y + line_height < self._height - margin_y:
            # Paragraph break every 6 lines (skip a row).
            if line_index % 6 == 5:
                y += line_spacing
                line_index += 1
                continue

            # Random short last line per paragraph; full-width otherwise.
            end_short = (line_index + 1) % 6 == 0
            if end_short:
                line_end_x = self._width - margin_x - rng.integers(
                    0, self._width // 3
                )
            else:
                line_end_x = self._width - margin_x

            # "Words": dark blocks separated by small gaps.
            x = margin_x
            while x < line_end_x:
                word_len = int(rng.integers(8, 70))
                word_end = min(x + word_len, int(line_end_x))
                frame[y:y + line_height, x:word_end] = 30  # dark "ink"
                x = word_end + int(rng.integers(3, 14))

            y += line_spacing
            line_index += 1

        # Watermark so it's clear at a glance this isn't a real camera.
        try:
            import cv2
            cv2.putText(
                frame,
                "SYNTHETIC TEST FRAME (no camera detected)",
                (margin_x, self._height - margin_y // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 200),
                2,
            )
        except Exception:
            # cv2 unavailable in pure-test paths; the frame is still
            # useful without the watermark.
            pass

        return frame