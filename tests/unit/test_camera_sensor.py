"""Unit tests for the camera sensor implementations.

Covers both ``MockCameraSensor`` and ``PiCameraSensor`` (in the
same module since both classes now live in ``camera_sensor.py``).

The Mock tests don't require a real webcam — they force
``source: test_image`` to take the fallback/synthetic path. The Pi
tests don't require ``picamera2`` to be installed — they use the
dependency-injection points on ``PiCameraSensor`` to supply fake
``picamera2`` and ``libcamera`` modules built with
``types.SimpleNamespace`` and ``unittest.mock.MagicMock``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from digital_magnifier.hal.camera_base import CameraError
from digital_magnifier.hal.camera_sensor import (
    MockCameraSensor,
    PiCameraSensor,
    _MODE_SYNTHETIC,
    _MODE_FALLBACK_IMAGE,
    _MODE_STOPPED,
)


# --------------------------------------------------------------------------- #
# Construction and config parsing
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_minimal_config(self):
        cam = MockCameraSensor({})
        assert cam._width == 1280
        assert cam._height == 720
        assert cam._device_index == 0

    def test_full_config(self):
        cfg = {
            "device": {
                "source": "test_image",
                "index": 2,
                "fallback_image": "some/path.png",
            },
            "resolution": {"width": 640, "height": 480},
        }
        cam = MockCameraSensor(cfg)
        assert cam._source == "test_image"
        assert cam._device_index == 2
        assert cam._fallback_image_path == "some/path.png"
        assert cam._width == 640
        assert cam._height == 480


# --------------------------------------------------------------------------- #
# Synthetic frame path
# --------------------------------------------------------------------------- #
class TestSyntheticFallback:
    @pytest.fixture
    def cam(self):
        # source=test_image skips the webcam attempt; no fallback
        # path provided, so it goes to synthetic.
        cfg = {
            "device": {"source": "test_image"},
            "resolution": {"width": 320, "height": 240},
        }
        return MockCameraSensor(cfg)

    def test_start_produces_synthetic_frame(self, cam):
        cam.start()
        assert cam._mode == _MODE_SYNTHETIC

    def test_get_frame_returns_correct_shape(self, cam):
        cam.start()
        frame = cam.get_frame()
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

    def test_get_frame_returns_independent_copies(self, cam):
        cam.start()
        f1 = cam.get_frame()
        f2 = cam.get_frame()
        # Mutating one must not affect the other.
        f1[0, 0] = [99, 99, 99]
        assert not np.array_equal(f1, f2)

    def test_synthetic_frame_is_deterministic(self):
        cfg = {
            "device": {"source": "test_image"},
            "resolution": {"width": 320, "height": 240},
        }
        cam1 = MockCameraSensor(cfg)
        cam2 = MockCameraSensor(cfg)
        cam1.start()
        cam2.start()
        # Seeded RNG -> identical synthetic content
        assert np.array_equal(cam1.get_frame(), cam2.get_frame())

    def test_stop_resets_state(self, cam):
        cam.start()
        cam.stop()
        assert cam._mode == _MODE_STOPPED
        with pytest.raises(CameraError):
            cam.get_frame()


# --------------------------------------------------------------------------- #
# Fallback image path
# --------------------------------------------------------------------------- #
class TestFallbackImage:
    def test_loads_valid_image(self, tmp_path: Path):
        # Create a real PNG to load.
        import cv2

        img_path = tmp_path / "fake.png"
        original = np.full((100, 200, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(img_path), original)

        cfg = {
            "device": {
                "source": "test_image",
                "fallback_image": str(img_path),  # absolute path
            },
            "resolution": {"width": 320, "height": 240},
        }
        cam = MockCameraSensor(cfg)
        cam.start()
        assert cam._mode == _MODE_FALLBACK_IMAGE

        # Should be resized to configured resolution
        frame = cam.get_frame()
        assert frame.shape == (240, 320, 3)

    def test_missing_image_falls_through_to_synthetic(self, tmp_path: Path):
        cfg = {
            "device": {
                "source": "test_image",
                "fallback_image": str(tmp_path / "does_not_exist.png"),
            },
            "resolution": {"width": 320, "height": 240},
        }
        cam = MockCameraSensor(cfg)
        cam.start()
        assert cam._mode == _MODE_SYNTHETIC


# --------------------------------------------------------------------------- #
# Context manager
# --------------------------------------------------------------------------- #
class TestContextManager:
    def test_context_manager(self):
        cfg = {
            "device": {"source": "test_image"},
            "resolution": {"width": 320, "height": 240},
        }
        with MockCameraSensor(cfg) as cam:
            frame = cam.get_frame()
            assert frame.shape == (240, 320, 3)
        # After exit, stop() has been called.
        assert cam._mode == _MODE_STOPPED


# --------------------------------------------------------------------------- #
# Webcam path (only when one is actually available)
# --------------------------------------------------------------------------- #
class TestWebcamPath:
    def test_unavailable_webcam_falls_back(self):
        # Use an obviously-invalid index. This may take a moment as
        # cv2 tries and fails; that's expected.
        cfg = {
            "device": {"source": "webcam", "index": 99},
            "resolution": {"width": 320, "height": 240},
        }
        cam = MockCameraSensor(cfg)
        cam.start()
        # We don't assert webcam vs fallback because some test
        # environments DO have a virtual webcam at high indexes.
        # Just verify start() didn't raise and a frame is available.
        frame = cam.get_frame()
        assert isinstance(frame, np.ndarray)
        assert frame.dtype == np.uint8
        cam.stop()


# ===========================================================================
# PiCameraSensor
# ===========================================================================
# Helpers to build fake picamera2 / libcamera modules so these tests run
# without those Pi-only libraries installed.

from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_fake_picam2(frame: np.ndarray | None = None) -> MagicMock:
    """Mock that behaves like a Picamera2 instance."""
    if frame is None:
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    picam2 = MagicMock(name="Picamera2 instance")
    picam2.capture_array.return_value = frame
    return picam2


def _make_fake_picamera2_module(picam2_instance: MagicMock) -> SimpleNamespace:
    return SimpleNamespace(
        Picamera2=MagicMock(return_value=picam2_instance, name="Picamera2 class"),
    )


def _make_fake_libcamera_module() -> SimpleNamespace:
    """Stand-in for libcamera with enum and Transform shapes."""
    AwbModeEnum = SimpleNamespace(
        Auto="awb_auto",
        Incandescent="awb_incandescent",
        Tungsten="awb_tungsten",
        Fluorescent="awb_fluorescent",
        Indoor="awb_indoor",
        Daylight="awb_daylight",
        Cloudy="awb_cloudy",
    )
    AeExposureModeEnum = SimpleNamespace(
        Normal="ae_normal",
        Short="ae_short",
        Long="ae_long",
    )
    HdrModeEnum = SimpleNamespace(
        Off="hdr_off",
        SingleExposure="hdr_single",
        Night="hdr_night",
    )
    AfModeEnum = SimpleNamespace(
        Manual="af_manual",
        Auto="af_auto",
        Continuous="af_continuous",
    )
    AfRangeEnum = SimpleNamespace(
        Normal="afr_normal",
        Macro="afr_macro",
        Full="afr_full",
    )
    AfSpeedEnum = SimpleNamespace(
        Normal="afs_normal",
        Fast="afs_fast",
    )
    controls = SimpleNamespace(
        AwbModeEnum=AwbModeEnum,
        AeExposureModeEnum=AeExposureModeEnum,
        HdrModeEnum=HdrModeEnum,
        AfModeEnum=AfModeEnum,
        AfRangeEnum=AfRangeEnum,
        AfSpeedEnum=AfSpeedEnum,
    )

    def Transform(hflip: bool = False, vflip: bool = False):
        return SimpleNamespace(hflip=hflip, vflip=vflip)

    return SimpleNamespace(Transform=Transform, controls=controls)


@pytest.fixture
def fake_picam2():
    return _make_fake_picam2()


@pytest.fixture
def fake_picamera2_module(fake_picam2):
    return _make_fake_picamera2_module(fake_picam2)


@pytest.fixture
def fake_libcamera_module():
    return _make_fake_libcamera_module()


@pytest.fixture
def pi_config():
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


def _make_pi_sensor(config, picam2_mod, libcam_mod):
    return PiCameraSensor(
        config,
        picamera2_module=picam2_mod,
        libcamera_module=libcam_mod,
    )


class TestPiConstruction:
    def test_minimal_config(self):
        sensor = PiCameraSensor({})
        assert sensor._width == 1280
        assert sensor._height == 720
        assert sensor._awb_mode == "auto"

    def test_reads_resolution(self):
        sensor = PiCameraSensor({"resolution": {"width": 640, "height": 480}})
        assert sensor._width == 640
        assert sensor._height == 480

    def test_rotation_90_sets_sw_code(self, pi_config):
        pi_config["pi_camera"]["rotation"] = 90
        sensor = PiCameraSensor(pi_config)
        assert sensor._sw_rotation_code is not None

    def test_rotation_180_no_sw_code(self, pi_config):
        pi_config["pi_camera"]["rotation"] = 180
        sensor = PiCameraSensor(pi_config)
        assert sensor._sw_rotation_code is None

    def test_rotation_270_sets_sw_code(self, pi_config):
        pi_config["pi_camera"]["rotation"] = 270
        sensor = PiCameraSensor(pi_config)
        assert sensor._sw_rotation_code is not None


class TestPiStart:
    def test_calls_picamera2_constructor(
        self, pi_config, fake_picamera2_module, fake_libcamera_module
    ):
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        fake_picamera2_module.Picamera2.assert_called_once_with()

    def test_configures_resolution(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        kwargs = fake_picam2.create_video_configuration.call_args.kwargs
        assert kwargs["main"]["size"] == (1280, 720)
        assert kwargs["main"]["format"] == "RGB888"

    def test_starts_camera(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        fake_picam2.configure.assert_called_once()
        fake_picam2.start.assert_called_once()

    def test_awb_mode_passed_through(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        pi_config["pi_camera"]["awb_mode"] = "daylight"
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        controls = fake_picam2.create_video_configuration.call_args.kwargs["controls"]
        assert controls["AwbMode"] == "awb_daylight"

    def test_hdr_off_not_emitted(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        pi_config["pi_camera"]["hdr"] = "off"
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        controls = (
            fake_picam2.create_video_configuration.call_args.kwargs.get("controls")
            or {}
        )
        assert "HdrMode" not in controls

    def test_hdr_single_exposure_passed(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        pi_config["pi_camera"]["hdr"] = "single_exposure"
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        controls = fake_picam2.create_video_configuration.call_args.kwargs["controls"]
        assert controls["HdrMode"] == "hdr_single"

    def test_unknown_awb_logs_and_skips(
        self, pi_config, fake_picamera2_module, fake_libcamera_module,
        fake_picam2, caplog,
    ):
        pi_config["pi_camera"]["awb_mode"] = "purple"
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        with caplog.at_level("WARNING"):
            sensor.start()
        controls = (
            fake_picam2.create_video_configuration.call_args.kwargs.get("controls")
            or {}
        )
        assert "AwbMode" not in controls
        assert any("AWB mode" in r.message for r in caplog.records)

    def test_rotation_180_sets_both_flips(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        pi_config["pi_camera"]["rotation"] = 180
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        transform = fake_picam2.create_video_configuration.call_args.kwargs["transform"]
        assert transform.hflip is True
        assert transform.vflip is True

    def test_configure_failure_raises_camera_error(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        fake_picam2.configure.side_effect = RuntimeError("device busy")
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        with pytest.raises(CameraError, match="failed to start picamera2"):
            sensor.start()

    def test_af_mode_continuous_by_default(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        # Default config has no af_mode set -> sensor should pick "continuous"
        pi_config["pi_camera"].pop("af_mode", None)
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        controls = fake_picam2.create_video_configuration.call_args.kwargs["controls"]
        assert controls["AfMode"] == "af_continuous"

    def test_af_mode_manual_with_lens_position(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        pi_config["pi_camera"]["af_mode"] = "manual"
        pi_config["pi_camera"]["lens_position"] = 4.0
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        controls = fake_picam2.create_video_configuration.call_args.kwargs["controls"]
        assert controls["AfMode"] == "af_manual"
        assert controls["LensPosition"] == 4.0

    def test_lens_position_ignored_when_not_manual(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        # lens_position set but af_mode is continuous -> position should be ignored
        pi_config["pi_camera"]["af_mode"] = "continuous"
        pi_config["pi_camera"]["lens_position"] = 4.0
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        controls = fake_picam2.create_video_configuration.call_args.kwargs["controls"]
        assert "LensPosition" not in controls

    def test_unknown_af_mode_logs_and_skips(
        self, pi_config, fake_picamera2_module, fake_libcamera_module,
        fake_picam2, caplog,
    ):
        pi_config["pi_camera"]["af_mode"] = "telescopic"
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        with caplog.at_level("WARNING"):
            sensor.start()
        controls = (
            fake_picam2.create_video_configuration.call_args.kwargs.get("controls")
            or {}
        )
        assert "AfMode" not in controls
        assert any("AF mode" in r.message for r in caplog.records)

    def test_af_range_macro_passed_through(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        pi_config["pi_camera"]["af_mode"] = "continuous"
        pi_config["pi_camera"]["af_range"] = "macro"
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        controls = fake_picam2.create_video_configuration.call_args.kwargs["controls"]
        assert controls["AfRange"] == "afr_macro"

    def test_af_speed_fast_passed_through(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        pi_config["pi_camera"]["af_mode"] = "continuous"
        pi_config["pi_camera"]["af_speed"] = "fast"
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        controls = fake_picam2.create_video_configuration.call_args.kwargs["controls"]
        assert controls["AfSpeed"] == "afs_fast"

    def test_af_range_and_speed_skipped_when_manual(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        # In manual mode the lens is locked and these controls have no
        # effect, so the driver shouldn't see them.
        pi_config["pi_camera"]["af_mode"] = "manual"
        pi_config["pi_camera"]["lens_position"] = 4.0
        pi_config["pi_camera"]["af_range"] = "macro"
        pi_config["pi_camera"]["af_speed"] = "fast"
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        controls = fake_picam2.create_video_configuration.call_args.kwargs["controls"]
        assert "AfRange" not in controls
        assert "AfSpeed" not in controls

    def test_unknown_af_range_logs_and_skips(
        self, pi_config, fake_picamera2_module, fake_libcamera_module,
        fake_picam2, caplog,
    ):
        pi_config["pi_camera"]["af_mode"] = "continuous"
        pi_config["pi_camera"]["af_range"] = "supermacro"
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        with caplog.at_level("WARNING"):
            sensor.start()
        controls = fake_picam2.create_video_configuration.call_args.kwargs["controls"]
        assert "AfRange" not in controls
        assert any("AF range" in r.message for r in caplog.records)


class TestPiStop:
    def test_stop_after_start(
        self, pi_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        sensor.stop()
        fake_picam2.stop.assert_called_once()
        fake_picam2.close.assert_called_once()

    def test_stop_without_start_is_noop(
        self, pi_config, fake_picamera2_module, fake_libcamera_module
    ):
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.stop()  # must not raise


class TestPiGetFrame:
    def test_returns_3channel(
        self, pi_config, fake_picamera2_module, fake_libcamera_module
    ):
        sensor = _make_pi_sensor(pi_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        frame = sensor.get_frame()
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

    def test_strips_alpha_from_4channel(self, pi_config, fake_libcamera_module):
        rgba_frame = np.zeros((240, 320, 4), dtype=np.uint8)
        picam2 = _make_fake_picam2(rgba_frame)
        sensor = _make_pi_sensor(
            pi_config,
            _make_fake_picamera2_module(picam2),
            fake_libcamera_module,
        )
        sensor.start()
        frame = sensor.get_frame()
        assert frame.shape == (240, 320, 3)

    def test_before_start_raises(self, pi_config):
        sensor = PiCameraSensor(pi_config)
        with pytest.raises(CameraError, match="not started"):
            sensor.get_frame()

    def test_capture_failure_raises_camera_error(self, pi_config, fake_libcamera_module):
        picam2 = _make_fake_picam2()
        picam2.capture_array.side_effect = RuntimeError("frame dropped")
        sensor = _make_pi_sensor(
            pi_config,
            _make_fake_picamera2_module(picam2),
            fake_libcamera_module,
        )
        sensor.start()
        with pytest.raises(CameraError, match="capture_array failed"):
            sensor.get_frame()


class TestPiSoftwareRotation:
    def test_rotation_90_swaps_shape(self, pi_config, fake_libcamera_module):
        pi_config["pi_camera"]["rotation"] = 90
        src = np.full((240, 320, 3), 128, dtype=np.uint8)
        picam2 = _make_fake_picam2(src)
        sensor = _make_pi_sensor(
            pi_config,
            _make_fake_picamera2_module(picam2),
            fake_libcamera_module,
        )
        sensor.start()
        frame = sensor.get_frame()
        # After 90° rotation, shape (H, W, 3) becomes (W, H, 3).
        assert frame.shape == (320, 240, 3)


class TestPiDependencyResolution:
    def test_missing_picamera2_raises_helpful_error(self, pi_config):
        try:
            import picamera2  # noqa: F401
            pytest.skip("picamera2 is installed in this env")
        except ImportError:
            pass

        sensor = PiCameraSensor(pi_config)
        with pytest.raises(CameraError, match="picamera2 not available"):
            sensor.start()