"""Camera sensor implementations.

All concrete :class:`CameraSensor` subclasses live here:

- :class:`MockCameraSensor` — webcam via OpenCV's VideoCapture,
  with a fallback chain ending in a procedurally-generated test
  frame. Used on WSL and any other development machine.
- :class:`PiCameraSensor` — Pi Camera Module 3 via the
  ``picamera2`` library. Used on the Raspberry Pi CM5.

Both share the abstract base in ``camera_base.py``. The Pi-specific
imports (``picamera2``, ``libcamera``) are lazy-loaded inside
``PiCameraSensor.start()`` so this whole module imports cleanly on
WSL where those libraries are not installed — they only need to
be importable at runtime on the actual CM5.

``main.py`` picks the right implementation at startup based on
``hardware.platform`` in ``hardware_pins.yaml``; everything above
the HAL is unaware of which sensor is in use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
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


# ============================================================
# PiCameraSensor — Raspberry Pi Camera Module 3 (CM5)
# ============================================================
#
# Lazy-imports picamera2 and libcamera inside start() so that this
# whole module loads cleanly on WSL or any other environment without
# the Pi camera stack. The imports only need to succeed at runtime
# on the actual CM5.
#
# Configuration is read from the ``pi_camera`` section of
# ``camera_config.yaml``. See that file for documentation of each
# key (awb_mode, ae_mode, hdr, rotation, hflip, vflip, format).
#
# References:
# - https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
# ============================================================


# Maps YAML strings to the attribute names on libcamera's enum
# classes. Resolved at runtime since libcamera is a Pi-only import.
_AWB_MODE_NAMES: dict[str, str] = {
    "auto": "Auto",
    "incandescent": "Incandescent",
    "tungsten": "Tungsten",
    "fluorescent": "Fluorescent",
    "indoor": "Indoor",
    "daylight": "Daylight",
    "cloudy": "Cloudy",
}

_AE_MODE_NAMES: dict[str, str] = {
    "normal": "Normal",
    "short": "Short",
    "long": "Long",
}

_HDR_MODE_NAMES: dict[str, str] = {
    "off": "Off",
    "single_exposure": "SingleExposure",
    "night": "Night",
}

# Autofocus modes. The Pi Camera Module 3 has a motorised lens;
# without setting AfMode the lens sits at the driver default
# (usually infinity-ish), which is wrong for typical magnifier
# distances of 20-30 cm. "continuous" is the sensible default.
_AF_MODE_NAMES: dict[str, str] = {
    "manual": "Manual",
    "auto": "Auto",
    "continuous": "Continuous",
}

# cv2 rotation codes for software 90/270 rotation. 180 is achieved
# in hardware via Transform(hflip=True, vflip=True), so it's not
# listed here.
_SW_ROTATION_CODES: dict[int, int] = {
    90: cv2.ROTATE_90_CLOCKWISE,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class PiCameraSensor(CameraSensor):
    """Pi Camera Module 3 backed implementation of :class:`CameraSensor`.

    Parameters
    ----------
    config : dict
        Parsed contents of ``camera_config.yaml``. Reads:

        - ``resolution.width`` / ``resolution.height``
        - ``pi_camera.*`` (see camera_config.yaml for the full list)
    picamera2_module : module, optional
        Test injection point. Defaults to lazy-importing
        ``picamera2`` at start time.
    libcamera_module : module, optional
        Test injection point. Defaults to lazy-importing
        ``libcamera`` at start time.
    """

    def __init__(
        self,
        config: dict[str, Any],
        picamera2_module: Any = None,
        libcamera_module: Any = None,
    ) -> None:
        res_cfg = config.get("resolution", {})
        pi_cfg = config.get("pi_camera", {})

        self._width = int(res_cfg.get("width", 1280))
        self._height = int(res_cfg.get("height", 720))

        self._awb_mode = str(pi_cfg.get("awb_mode", "auto")).lower()
        self._ae_mode = str(pi_cfg.get("ae_mode", "normal")).lower()
        self._hdr_mode = str(pi_cfg.get("hdr", "off")).lower()
        self._af_mode = str(pi_cfg.get("af_mode", "continuous")).lower()
        # LensPosition is in diopters (1/distance_in_metres). Only
        # used when af_mode == "manual"; ignored otherwise.
        self._lens_position = pi_cfg.get("lens_position")
        self._rotation = int(pi_cfg.get("rotation", 0))
        self._hflip = bool(pi_cfg.get("hflip", False))
        self._vflip = bool(pi_cfg.get("vflip", False))
        self._format = str(pi_cfg.get("format", "RGB888"))

        # Software rotation code for 90/270, or None if rotation is
        # 0 (no rotation) or 180 (handled in hardware via Transform).
        self._sw_rotation_code: int | None = _SW_ROTATION_CODES.get(self._rotation)

        # Injection points for tests; resolved lazily at start time.
        self._picamera2_module = picamera2_module
        self._libcamera_module = libcamera_module

        # Picamera2 handle, populated by start().
        self._picam2: Any = None

    # ------------------------------------------------------------------
    # CameraSensor interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._resolve_dependencies()
        assert self._picamera2_module is not None
        assert self._libcamera_module is not None

        Picamera2 = self._picamera2_module.Picamera2
        Transform = self._libcamera_module.Transform
        lc_controls = self._libcamera_module.controls

        transform = self._build_transform(Transform)
        controls_dict = self._build_controls_dict(lc_controls)

        picam2 = Picamera2()
        try:
            cam_config = picam2.create_video_configuration(
                main={
                    "size": (self._width, self._height),
                    "format": self._format,
                },
                transform=transform,
                controls=controls_dict if controls_dict else None,
            )
            picam2.configure(cam_config)
            picam2.start()
        except Exception as e:
            # Best-effort cleanup on partial init.
            try:
                picam2.close()
            except Exception:
                pass
            raise CameraError(f"failed to start picamera2: {e}") from e

        self._picam2 = picam2
        logger.info(
            "PiCameraSensor started: %dx%d format=%s "
            "awb=%s ae=%s af=%s hdr=%s rotation=%d hflip=%s vflip=%s",
            self._width, self._height, self._format,
            self._awb_mode, self._ae_mode, self._af_mode, self._hdr_mode,
            self._rotation, self._hflip, self._vflip,
        )

    def stop(self) -> None:
        if self._picam2 is None:
            return
        try:
            self._picam2.stop()
        except Exception:
            logger.exception("error stopping picamera2")
        try:
            self._picam2.close()
        except Exception:
            logger.exception("error closing picamera2")
        self._picam2 = None

    def get_frame(self) -> np.ndarray:
        if self._picam2 is None:
            raise CameraError(
                "PiCameraSensor not started — call start() first"
            )

        try:
            arr = self._picam2.capture_array("main")
        except Exception as e:
            raise CameraError(f"capture_array failed: {e}") from e

        if arr is None:
            raise CameraError("capture_array returned None")

        # Some configurations (XBGR8888) return 4 channels; drop alpha.
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        # Software rotation for 90/270 (180 was applied via Transform).
        if self._sw_rotation_code is not None:
            arr = cv2.rotate(arr, self._sw_rotation_code)

        # picamera2 may return a view into a shared buffer that the
        # next capture overwrites. Copy so the caller can hold onto
        # the frame (the app controller does this for freeze).
        return arr.copy()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_dependencies(self) -> None:
        """Import picamera2 and libcamera if not already injected."""
        if self._picamera2_module is None:
            try:
                import picamera2  # type: ignore
            except ImportError as e:
                raise CameraError(
                    f"picamera2 not available: {e}. "
                    "Install with: sudo apt install -y python3-picamera2"
                ) from e
            self._picamera2_module = picamera2

        if self._libcamera_module is None:
            try:
                import libcamera  # type: ignore
            except ImportError as e:
                raise CameraError(
                    f"libcamera Python bindings not available: {e}. "
                    "Install with: sudo apt install -y python3-libcamera"
                ) from e
            self._libcamera_module = libcamera

    def _build_transform(self, Transform: Any) -> Any:
        """Compose Transform from rotation + hflip + vflip config."""
        hflip = self._hflip
        vflip = self._vflip

        # 180° rotation is equivalent to flipping both axes and is
        # free on the Pi ISP — much cheaper than rotating in software.
        if self._rotation == 180:
            hflip = not hflip
            vflip = not vflip

        return Transform(hflip=hflip, vflip=vflip)

    def _build_controls_dict(self, lc_controls: Any) -> dict[str, Any]:
        """Translate YAML config into a libcamera controls dict.

        Unknown or unsupported values log a warning and are dropped
        rather than raising — a typo or version-skew shouldn't keep
        the camera from starting.
        """
        controls_dict: dict[str, Any] = {}

        controls_dict.update(
            self._resolve_enum_control(
                lc_controls, "AwbMode", "AwbModeEnum",
                self._awb_mode, _AWB_MODE_NAMES, "AWB",
            )
        )
        controls_dict.update(
            self._resolve_enum_control(
                lc_controls, "AeExposureMode", "AeExposureModeEnum",
                self._ae_mode, _AE_MODE_NAMES, "AE",
            )
        )
        controls_dict.update(
            self._resolve_enum_control(
                lc_controls, "AfMode", "AfModeEnum",
                self._af_mode, _AF_MODE_NAMES, "AF",
            )
        )
        # Manual focus: lock the lens at a fixed distance. Only
        # emitted when af_mode is "manual" AND lens_position is set.
        if self._af_mode == "manual" and self._lens_position is not None:
            try:
                controls_dict["LensPosition"] = float(self._lens_position)
            except (TypeError, ValueError):
                logger.warning(
                    "invalid lens_position %r; ignored",
                    self._lens_position,
                )
        # HDR only emitted when not "off" — absent is the default.
        if self._hdr_mode != "off":
            controls_dict.update(
                self._resolve_enum_control(
                    lc_controls, "HdrMode", "HdrModeEnum",
                    self._hdr_mode, _HDR_MODE_NAMES, "HDR",
                )
            )

        return controls_dict

    @staticmethod
    def _resolve_enum_control(
        lc_controls: Any,
        control_key: str,
        enum_class_name: str,
        config_value: str,
        name_map: dict[str, str],
        log_label: str,
    ) -> dict[str, Any]:
        """Resolve one config-string-to-libcamera-enum mapping."""
        if config_value not in name_map:
            logger.warning(
                "%s mode %r not recognised; using camera default",
                log_label, config_value,
            )
            return {}

        enum_attr = name_map[config_value]
        enum_cls = getattr(lc_controls, enum_class_name, None)
        if enum_cls is None:
            logger.warning(
                "libcamera version does not expose %s; "
                "skipping %s configuration",
                enum_class_name, log_label,
            )
            return {}

        value = getattr(enum_cls, enum_attr, None)
        if value is None:
            logger.warning(
                "%s.%s not present in libcamera; using camera default",
                enum_class_name, enum_attr,
            )
            return {}

        return {control_key: value}