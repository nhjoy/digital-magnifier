"""Stdlib verification for MVP 0.2.

Covers:
  - PiCameraSensor: construction, start/stop, get_frame,
    config translation, rotation, error paths.
  - main.py: _resolve_platform, _detect_platform_auto,
    _build_camera lazy-import behaviour.
  - End-to-end: the same main() runs the app whether picamera2
    is available (uses PiCameraSensor) or not (uses Mock).
"""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str((__import__("pathlib").Path(__file__).resolve().parents[2] / "src")))

from digital_magnifier.hal.camera_base import CameraError
from digital_magnifier.hal.camera_sensor import PiCameraSensor


# ----- helpers -----------------------------------------------------
def make_fake_picam2(frame=None):
    if frame is None:
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    p = MagicMock(name="Picamera2")
    p.capture_array.return_value = frame
    return p


def make_fake_picamera2_module(picam2):
    return SimpleNamespace(Picamera2=MagicMock(return_value=picam2))


def make_fake_libcamera_module():
    AwbModeEnum = SimpleNamespace(
        Auto="awb_auto", Daylight="awb_daylight",
        Incandescent="awb_incandescent", Tungsten="awb_tungsten",
        Fluorescent="awb_fluorescent", Indoor="awb_indoor",
        Cloudy="awb_cloudy",
    )
    AeExposureModeEnum = SimpleNamespace(
        Normal="ae_normal", Short="ae_short", Long="ae_long",
    )
    HdrModeEnum = SimpleNamespace(
        Off="hdr_off", SingleExposure="hdr_single", Night="hdr_night",
    )
    AfModeEnum = SimpleNamespace(
        Manual="af_manual", Auto="af_auto", Continuous="af_continuous",
    )
    AfRangeEnum = SimpleNamespace(
        Normal="afr_normal", Macro="afr_macro", Full="afr_full",
    )
    AfSpeedEnum = SimpleNamespace(
        Normal="afs_normal", Fast="afs_fast",
    )
    controls = SimpleNamespace(
        AwbModeEnum=AwbModeEnum,
        AeExposureModeEnum=AeExposureModeEnum,
        HdrModeEnum=HdrModeEnum,
        AfModeEnum=AfModeEnum,
        AfRangeEnum=AfRangeEnum,
        AfSpeedEnum=AfSpeedEnum,
    )

    def Transform(hflip=False, vflip=False):
        return SimpleNamespace(hflip=hflip, vflip=vflip)

    return SimpleNamespace(Transform=Transform, controls=controls)


def make_config():
    return {
        "resolution": {"width": 1280, "height": 720},
        "pi_camera": {
            "awb_mode": "auto",
            "ae_mode": "normal",
            "hdr": "off",
            "rotation": 0,
            "hflip": False,
            "vflip": False,
            "format": "RGB888",
        },
    }


def make_sensor(cfg, picam2_mod, libcam_mod):
    return PiCameraSensor(
        cfg,
        picamera2_module=picam2_mod,
        libcamera_module=libcam_mod,
    )


# ===================================================================
# PiCameraSensor
# ===================================================================
class PiCameraSensorTests(unittest.TestCase):

    def test_minimal_config(self):
        s = PiCameraSensor({})
        self.assertEqual(s._width, 1280)
        self.assertEqual(s._height, 720)
        self.assertEqual(s._awb_mode, "auto")

    def test_rotation_90_sets_sw_code(self):
        cfg = make_config()
        cfg["pi_camera"]["rotation"] = 90
        s = PiCameraSensor(cfg)
        self.assertIsNotNone(s._sw_rotation_code)

    def test_rotation_180_no_sw_code(self):
        cfg = make_config()
        cfg["pi_camera"]["rotation"] = 180
        s = PiCameraSensor(cfg)
        self.assertIsNone(s._sw_rotation_code)

    def test_start_calls_picamera2(self):
        picam = make_fake_picam2()
        picam2_mod = make_fake_picamera2_module(picam)
        libcam_mod = make_fake_libcamera_module()
        s = make_sensor(make_config(), picam2_mod, libcam_mod)
        s.start()
        picam2_mod.Picamera2.assert_called_once_with()

    def test_start_configures_resolution(self):
        picam = make_fake_picam2()
        picam2_mod = make_fake_picamera2_module(picam)
        libcam_mod = make_fake_libcamera_module()
        s = make_sensor(make_config(), picam2_mod, libcam_mod)
        s.start()
        kwargs = picam.create_video_configuration.call_args.kwargs
        self.assertEqual(kwargs["main"]["size"], (1280, 720))
        self.assertEqual(kwargs["main"]["format"], "RGB888")

    def test_start_starts_camera(self):
        picam = make_fake_picam2()
        s = make_sensor(make_config(),
                        make_fake_picamera2_module(picam),
                        make_fake_libcamera_module())
        s.start()
        picam.configure.assert_called_once()
        picam.start.assert_called_once()

    def test_awb_mode_passed_through(self):
        cfg = make_config()
        cfg["pi_camera"]["awb_mode"] = "daylight"
        picam = make_fake_picam2()
        s = make_sensor(cfg,
                        make_fake_picamera2_module(picam),
                        make_fake_libcamera_module())
        s.start()
        controls = picam.create_video_configuration.call_args.kwargs["controls"]
        self.assertEqual(controls["AwbMode"], "awb_daylight")

    def test_hdr_off_skips_emission(self):
        picam = make_fake_picam2()
        s = make_sensor(make_config(),
                        make_fake_picamera2_module(picam),
                        make_fake_libcamera_module())
        s.start()
        kwargs = picam.create_video_configuration.call_args.kwargs
        controls = kwargs.get("controls") or {}
        self.assertNotIn("HdrMode", controls)

    def test_hdr_single_exposure_passed(self):
        cfg = make_config()
        cfg["pi_camera"]["hdr"] = "single_exposure"
        picam = make_fake_picam2()
        s = make_sensor(cfg,
                        make_fake_picamera2_module(picam),
                        make_fake_libcamera_module())
        s.start()
        controls = picam.create_video_configuration.call_args.kwargs["controls"]
        self.assertEqual(controls["HdrMode"], "hdr_single")

    def test_unknown_awb_logs_and_skips(self):
        cfg = make_config()
        cfg["pi_camera"]["awb_mode"] = "purple"
        picam = make_fake_picam2()
        s = make_sensor(cfg,
                        make_fake_picamera2_module(picam),
                        make_fake_libcamera_module())
        s.start()
        kwargs = picam.create_video_configuration.call_args.kwargs
        controls = kwargs.get("controls") or {}
        self.assertNotIn("AwbMode", controls)

    def test_rotation_180_sets_both_flips(self):
        cfg = make_config()
        cfg["pi_camera"]["rotation"] = 180
        picam = make_fake_picam2()
        s = make_sensor(cfg,
                        make_fake_picamera2_module(picam),
                        make_fake_libcamera_module())
        s.start()
        transform = picam.create_video_configuration.call_args.kwargs["transform"]
        self.assertTrue(transform.hflip)
        self.assertTrue(transform.vflip)

    def test_configure_failure_raises_camera_error(self):
        picam = make_fake_picam2()
        picam.configure.side_effect = RuntimeError("device busy")
        s = make_sensor(make_config(),
                        make_fake_picamera2_module(picam),
                        make_fake_libcamera_module())
        with self.assertRaises(CameraError):
            s.start()

    def test_stop_calls_close(self):
        picam = make_fake_picam2()
        s = make_sensor(make_config(),
                        make_fake_picamera2_module(picam),
                        make_fake_libcamera_module())
        s.start()
        s.stop()
        picam.stop.assert_called_once()
        picam.close.assert_called_once()

    def test_stop_without_start_no_op(self):
        s = make_sensor(make_config(),
                        make_fake_picamera2_module(make_fake_picam2()),
                        make_fake_libcamera_module())
        s.stop()  # no-op

    def test_get_frame_returns_3channel(self):
        s = make_sensor(make_config(),
                        make_fake_picamera2_module(make_fake_picam2()),
                        make_fake_libcamera_module())
        s.start()
        f = s.get_frame()
        self.assertEqual(f.shape, (240, 320, 3))

    def test_get_frame_strips_alpha(self):
        # 4-channel input -> 3-channel output
        rgba = np.zeros((240, 320, 4), dtype=np.uint8)
        s = make_sensor(make_config(),
                        make_fake_picamera2_module(make_fake_picam2(rgba)),
                        make_fake_libcamera_module())
        s.start()
        f = s.get_frame()
        self.assertEqual(f.shape, (240, 320, 3))

    def test_get_frame_before_start_raises(self):
        s = PiCameraSensor(make_config())
        with self.assertRaises(CameraError):
            s.get_frame()

    def test_capture_failure_raises_camera_error(self):
        picam = make_fake_picam2()
        picam.capture_array.side_effect = RuntimeError("dropped")
        s = make_sensor(make_config(),
                        make_fake_picamera2_module(picam),
                        make_fake_libcamera_module())
        s.start()
        with self.assertRaises(CameraError):
            s.get_frame()

    def test_rotation_90_swaps_shape(self):
        cfg = make_config()
        cfg["pi_camera"]["rotation"] = 90
        src = np.full((240, 320, 3), 128, dtype=np.uint8)
        s = make_sensor(cfg,
                        make_fake_picamera2_module(make_fake_picam2(src)),
                        make_fake_libcamera_module())
        s.start()
        f = s.get_frame()
        self.assertEqual(f.shape, (320, 240, 3))


# ===================================================================
# main.py platform detection
# ===================================================================
class MainPlatformTests(unittest.TestCase):

    def setUp(self):
        # Re-import main module fresh so module-level imports are
        # exercised — but in this case main has no top-level camera
        # imports, so we just import normally.
        from digital_magnifier import main as main_module
        self.main_module = main_module

    def test_resolve_platform_uses_yaml(self):
        configs = {"hardware_pins": {"hardware": {"platform": "raspberrypi_cm5"}}}
        result = self.main_module._resolve_platform(configs)
        self.assertEqual(result, "raspberrypi_cm5")

    def test_resolve_platform_cli_overrides_yaml(self):
        configs = {"hardware_pins": {"hardware": {"platform": "wsl_mock"}}}
        result = self.main_module._resolve_platform(
            configs, force_platform="raspberrypi_cm5"
        )
        self.assertEqual(result, "raspberrypi_cm5")

    def test_resolve_platform_unknown_falls_back_to_auto(self):
        configs = {"hardware_pins": {"hardware": {"platform": "weird_pi"}}}
        result = self.main_module._resolve_platform(configs)
        self.assertEqual(result, "auto")

    def test_resolve_platform_default_is_auto(self):
        result = self.main_module._resolve_platform({})
        self.assertEqual(result, "auto")

    def test_detect_platform_auto_with_picamera2(self):
        # Inject a fake picamera2/libcamera into sys.modules
        with patch.dict(sys.modules, {
            "picamera2": MagicMock(),
            "libcamera": MagicMock(),
        }):
            result = self.main_module._detect_platform_auto()
            self.assertEqual(result, "raspberrypi_cm5")

    def test_detect_platform_auto_without_picamera2(self):
        # In this sandbox picamera2 isn't installed; if it WERE we
        # would need to temporarily hide it. Try a clean test:
        try:
            import picamera2  # noqa
            self.skipTest("picamera2 is installed; can't test absence path")
        except ImportError:
            pass
        result = self.main_module._detect_platform_auto()
        self.assertEqual(result, "wsl_mock")

    def test_build_camera_mock_for_wsl(self):
        cam = self.main_module._build_camera("wsl_mock", {})
        # Must be a MockCameraSensor (lazy-imported)
        from digital_magnifier.hal.camera_sensor import MockCameraSensor
        self.assertIsInstance(cam, MockCameraSensor)

    def test_build_camera_pi_for_cm5(self):
        cam = self.main_module._build_camera("raspberrypi_cm5", {})
        from digital_magnifier.hal.camera_sensor import PiCameraSensor as P
        self.assertIsInstance(cam, P)


# ===================================================================
# main() end-to-end with mocked input
# ===================================================================
class MainEndToEndTests(unittest.TestCase):
    """Run main() on this sandbox — should fall back to Mock cleanly."""

    def setUp(self):
        import cv2
        self._patches = [
            patch.object(cv2, "imshow", lambda *a, **kw: None),
            patch.object(cv2, "namedWindow", lambda *a, **kw: None),
            patch.object(cv2, "destroyAllWindows", lambda *a, **kw: None),
        ]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def test_main_runs_to_quit(self):
        # Patch MockControls so the first poll returns QUIT
        from digital_magnifier.core.events import AppEvent
        from digital_magnifier.hal import mock_controls
        original_init = mock_controls.MockControls.__init__

        def patched_init(self, cfg, key_reader=None):
            quit_events = iter([ord("q"), -1, -1, -1])
            original_init(self, cfg, key_reader=lambda: next(quit_events, -1))

        with patch.object(mock_controls.MockControls, "__init__", patched_init):
            from digital_magnifier import main as main_module
            result = main_module.main(["--platform", "wsl_mock", "--log-level", "WARNING"])
            self.assertEqual(result, 0)


if __name__ == "__main__":
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    unittest.main(verbosity=2)