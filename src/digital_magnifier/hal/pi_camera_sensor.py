"""Pi Camera Module 3 sensor for the Raspberry Pi CM5.

Uses the ``picamera2`` library (https://github.com/raspberrypi/picamera2)
to capture frames from the IMX708 sensor in the Pi Camera Module 3.

The picamera2 and libcamera Python modules are **lazy-imported**
inside :meth:`PiCameraSensor.start`, so this module loads cleanly
on WSL or any other environment without the Pi camera stack. The
imports only need to succeed at runtime on the actual CM5.

Configuration
-------------
Read from the ``pi_camera`` section of ``camera_config.yaml``:

- ``awb_mode``: white balance — ``auto`` (default) | ``incandescent`` |
  ``tungsten`` | ``fluorescent`` | ``indoor`` | ``daylight`` | ``cloudy``
- ``ae_mode``: auto-exposure — ``normal`` | ``short`` | ``long``
- ``hdr``: HDR mode for the IMX708 — ``off`` | ``single_exposure`` |
  ``night``
- ``rotation``: 0 | 90 | 180 | 270 degrees. 0 and 180 are done by the
  ISP hardware (free); 90 and 270 use a cv2 software rotation (cheap
  but not free).
- ``hflip`` / ``vflip``: extra horizontal/vertical flip on top of
  whatever rotation does. Useful when mirroring for a self-view.
- ``format``: pixel format. Default ``"RGB888"``, which by Raspberry
  Pi convention delivers BGR-ordered data in numpy — exactly what
  cv2.imshow expects. Don't change unless you know why.

See https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
for the full library reference.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from digital_magnifier.hal.camera_base import CameraError, CameraSensor


logger = logging.getLogger(__name__)


# Maps YAML strings to the attribute names on libcamera's enum classes.
# Resolved at runtime since libcamera is also a Pi-only import.
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

# cv2 rotation codes for software 90/270. 180 is done by the ISP
# (hflip+vflip), so it's not listed here.
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
        - ``pi_camera.*`` (see module docstring)
    picamera2_module : module, optional
        Test injection point. Defaults to lazy-importing
        ``picamera2`` at start time.
    libcamera_module : module, optional
        Test injection point. Defaults to lazy-importing ``libcamera``
        at start time.
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
        self._rotation = int(pi_cfg.get("rotation", 0))
        self._hflip = bool(pi_cfg.get("hflip", False))
        self._vflip = bool(pi_cfg.get("vflip", False))
        self._format = str(pi_cfg.get("format", "RGB888"))

        # Software rotation code for 90/270, or None if rotation is
        # 0 or 180 (the latter being handled by Transform).
        self._sw_rotation_code: int | None = _SW_ROTATION_CODES.get(self._rotation)

        # Injection points for tests; resolved lazily at start time.
        self._picamera2_module = picamera2_module
        self._libcamera_module = libcamera_module

        # Picamera2 handle, populated by start()
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
            # Best-effort cleanup on partial init
            try:
                picam2.close()
            except Exception:
                pass
            raise CameraError(f"failed to start picamera2: {e}") from e

        self._picam2 = picam2
        logger.info(
            "PiCameraSensor started: %dx%d format=%s "
            "awb=%s ae=%s hdr=%s rotation=%d hflip=%s vflip=%s",
            self._width, self._height, self._format,
            self._awb_mode, self._ae_mode, self._hdr_mode,
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

        # Some configurations (XBGR8888) return a 4-channel array;
        # drop the alpha channel for cv2 compatibility.
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]

        # Software rotation for 90/270 (180 was applied in hardware
        # via the Transform).
        if self._sw_rotation_code is not None:
            arr = cv2.rotate(arr, self._sw_rotation_code)

        # picamera2 may return a view into a shared buffer that the
        # next capture overwrites. Copy so the caller can safely hold
        # onto the frame (the app controller does this for freeze).
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
        """Translate YAML config into libcamera controls dict.

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
        # HDR only set when not "off" — leaving it absent uses the
        # default (off), so we skip emitting it.
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