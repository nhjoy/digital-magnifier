"""Unit tests for ``PiCameraSensor``.

These tests do not require ``picamera2`` or ``libcamera`` to be
installed: the dependency-injection points in the sensor's
constructor accept fake modules. Tests construct a stub picamera2
module hierarchy that mimics the real API surface the sensor uses.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from digital_magnifier.hal.camera_base import CameraError
from digital_magnifier.hal.pi_camera_sensor import PiCameraSensor


# --------------------------------------------------------------------------- #
# Fake picamera2 / libcamera modules
# --------------------------------------------------------------------------- #
def make_fake_picam2(frame: np.ndarray | None = None) -> MagicMock:
    """Build a mock that behaves like a Picamera2 instance."""
    if frame is None:
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    picam2 = MagicMock(name="Picamera2 instance")
    picam2.capture_array.return_value = frame
    return picam2


def make_fake_picamera2_module(picam2_instance: MagicMock) -> SimpleNamespace:
    """Build a stand-in for the picamera2 module."""
    Picamera2 = MagicMock(return_value=picam2_instance, name="Picamera2 class")
    return SimpleNamespace(Picamera2=Picamera2)


def make_fake_libcamera_module() -> SimpleNamespace:
    """Build a stand-in for the libcamera module.

    Recreates the enum and Transform shapes the sensor reads. Each
    enum attribute is just a string sentinel; the test only checks
    that they're passed through correctly to controls dicts.
    """
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
    controls = SimpleNamespace(
        AwbModeEnum=AwbModeEnum,
        AeExposureModeEnum=AeExposureModeEnum,
        HdrModeEnum=HdrModeEnum,
    )

    def Transform(hflip: bool = False, vflip: bool = False):
        return SimpleNamespace(hflip=hflip, vflip=vflip)

    return SimpleNamespace(Transform=Transform, controls=controls)


@pytest.fixture
def fake_picam2():
    return make_fake_picam2()


@pytest.fixture
def fake_picamera2_module(fake_picam2):
    return make_fake_picamera2_module(fake_picam2)


@pytest.fixture
def fake_libcamera_module():
    return make_fake_libcamera_module()


@pytest.fixture
def default_config():
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


def make_sensor(
    config,
    fake_picam2_module,
    fake_libcam_module,
):
    return PiCameraSensor(
        config,
        picamera2_module=fake_picam2_module,
        libcamera_module=fake_libcam_module,
    )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_minimal_config(self):
        sensor = PiCameraSensor({})
        assert sensor._width == 1280
        assert sensor._height == 720
        assert sensor._awb_mode == "auto"
        assert sensor._ae_mode == "normal"

    def test_reads_resolution(self):
        sensor = PiCameraSensor({"resolution": {"width": 640, "height": 480}})
        assert sensor._width == 640
        assert sensor._height == 480

    def test_reads_pi_camera_section(self, default_config):
        sensor = PiCameraSensor(default_config)
        assert sensor._format == "RGB888"
        assert sensor._rotation == 0

    def test_rotation_90_sets_sw_code(self, default_config):
        default_config["pi_camera"]["rotation"] = 90
        sensor = PiCameraSensor(default_config)
        assert sensor._sw_rotation_code is not None

    def test_rotation_180_no_sw_code(self, default_config):
        default_config["pi_camera"]["rotation"] = 180
        sensor = PiCameraSensor(default_config)
        # 180 is done in hardware via Transform, no sw rotation
        assert sensor._sw_rotation_code is None

    def test_rotation_270_sets_sw_code(self, default_config):
        default_config["pi_camera"]["rotation"] = 270
        sensor = PiCameraSensor(default_config)
        assert sensor._sw_rotation_code is not None


# --------------------------------------------------------------------------- #
# start() with injected fakes
# --------------------------------------------------------------------------- #
class TestStart:
    def test_calls_picamera2_constructor(
        self, default_config, fake_picamera2_module, fake_libcamera_module
    ):
        sensor = make_sensor(default_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        fake_picamera2_module.Picamera2.assert_called_once_with()

    def test_configures_with_resolution(
        self, default_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        sensor = make_sensor(default_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        # create_video_configuration must have received our resolution
        kwargs = fake_picam2.create_video_configuration.call_args.kwargs
        assert kwargs["main"]["size"] == (1280, 720)
        assert kwargs["main"]["format"] == "RGB888"

    def test_starts_the_camera(
        self, default_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        sensor = make_sensor(default_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        fake_picam2.configure.assert_called_once()
        fake_picam2.start.assert_called_once()

    def test_awb_mode_passed_through(
        self, default_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        default_config["pi_camera"]["awb_mode"] = "daylight"
        sensor = make_sensor(default_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        kwargs = fake_picam2.create_video_configuration.call_args.kwargs
        controls = kwargs["controls"]
        assert controls["AwbMode"] == "awb_daylight"

    def test_hdr_off_not_passed(
        self, default_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        # hdr=off should mean we don't emit HdrMode at all
        default_config["pi_camera"]["hdr"] = "off"
        sensor = make_sensor(default_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        kwargs = fake_picam2.create_video_configuration.call_args.kwargs
        controls = kwargs.get("controls") or {}
        assert "HdrMode" not in controls

    def test_hdr_single_exposure_passed(
        self, default_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        default_config["pi_camera"]["hdr"] = "single_exposure"
        sensor = make_sensor(default_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        kwargs = fake_picam2.create_video_configuration.call_args.kwargs
        assert kwargs["controls"]["HdrMode"] == "hdr_single"

    def test_unknown_awb_mode_warns_and_skips(
        self, default_config, fake_picamera2_module, fake_libcamera_module,
        fake_picam2, caplog
    ):
        default_config["pi_camera"]["awb_mode"] = "nonsense"
        sensor = make_sensor(default_config, fake_picamera2_module, fake_libcamera_module)
        with caplog.at_level("WARNING"):
            sensor.start()
        kwargs = fake_picam2.create_video_configuration.call_args.kwargs
        controls = kwargs.get("controls") or {}
        assert "AwbMode" not in controls
        assert any("AWB mode" in r.message for r in caplog.records)

    def test_rotation_180_flips_in_hardware(
        self, default_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        default_config["pi_camera"]["rotation"] = 180
        sensor = make_sensor(default_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        kwargs = fake_picam2.create_video_configuration.call_args.kwargs
        transform = kwargs["transform"]
        # 180° = both flips
        assert transform.hflip is True
        assert transform.vflip is True

    def test_configure_failure_raises_camera_error(
        self, default_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        fake_picam2.configure.side_effect = RuntimeError("device busy")
        sensor = make_sensor(default_config, fake_picamera2_module, fake_libcamera_module)
        with pytest.raises(CameraError, match="failed to start picamera2"):
            sensor.start()


# --------------------------------------------------------------------------- #
# stop()
# --------------------------------------------------------------------------- #
class TestStop:
    def test_stop_after_start(
        self, default_config, fake_picamera2_module, fake_libcamera_module, fake_picam2
    ):
        sensor = make_sensor(default_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        sensor.stop()
        fake_picam2.stop.assert_called_once()
        fake_picam2.close.assert_called_once()

    def test_stop_without_start_is_noop(
        self, default_config, fake_picamera2_module, fake_libcamera_module
    ):
        sensor = make_sensor(default_config, fake_picamera2_module, fake_libcamera_module)
        sensor.stop()  # must not raise


# --------------------------------------------------------------------------- #
# get_frame()
# --------------------------------------------------------------------------- #
class TestGetFrame:
    def test_returns_3channel_frame(
        self, default_config, fake_picamera2_module, fake_libcamera_module
    ):
        sensor = make_sensor(default_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        frame = sensor.get_frame()
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

    def test_strips_alpha_from_4channel(
        self, default_config, fake_libcamera_module
    ):
        # Simulate XBGR8888 returning a 4-channel array
        rgba_frame = np.zeros((240, 320, 4), dtype=np.uint8)
        picam2 = make_fake_picam2(rgba_frame)
        sensor = make_sensor(
            default_config,
            make_fake_picamera2_module(picam2),
            fake_libcamera_module,
        )
        sensor.start()
        frame = sensor.get_frame()
        assert frame.shape == (240, 320, 3)

    def test_returns_a_copy(
        self, default_config, fake_picamera2_module, fake_libcamera_module
    ):
        # Mutating the returned frame must not affect the next get_frame
        sensor = make_sensor(default_config, fake_picamera2_module, fake_libcamera_module)
        sensor.start()
        f1 = sensor.get_frame()
        f1[0, 0] = [99, 99, 99]
        f2 = sensor.get_frame()
        # f2 came from the same source frame; if it's a copy, f1's
        # mutation didn't leak. The picam mock returns the same array
        # every time, but the sensor copies it.
        assert not np.array_equal(f1, f2) or f2[0, 0, 0] != 99

    def test_get_frame_before_start_raises(self, default_config):
        sensor = PiCameraSensor(default_config)
        with pytest.raises(CameraError, match="not started"):
            sensor.get_frame()

    def test_capture_failure_raises_camera_error(
        self, default_config, fake_libcamera_module
    ):
        picam2 = make_fake_picam2()
        picam2.capture_array.side_effect = RuntimeError("frame dropped")
        sensor = make_sensor(
            default_config,
            make_fake_picamera2_module(picam2),
            fake_libcamera_module,
        )
        sensor.start()
        with pytest.raises(CameraError, match="capture_array failed"):
            sensor.get_frame()


# --------------------------------------------------------------------------- #
# Software rotation
# --------------------------------------------------------------------------- #
class TestSoftwareRotation:
    def test_rotation_90_changes_shape(
        self, default_config, fake_libcamera_module
    ):
        # 90 rotation swaps width and height
        default_config["pi_camera"]["rotation"] = 90
        # Create a non-square frame so rotation is observable
        src = np.full((240, 320, 3), 128, dtype=np.uint8)
        picam2 = make_fake_picam2(src)
        sensor = make_sensor(
            default_config,
            make_fake_picamera2_module(picam2),
            fake_libcamera_module,
        )
        sensor.start()
        frame = sensor.get_frame()
        # After 90° rotate, shape is (320, 240, 3)
        assert frame.shape == (320, 240, 3)


# --------------------------------------------------------------------------- #
# Dependency resolution
# --------------------------------------------------------------------------- #
class TestDependencyResolution:
    def test_missing_picamera2_raises_helpful_error(self, default_config):
        # No injection; on a system without picamera2 the import
        # inside _resolve_dependencies fails and we expect a clear
        # CameraError. We can't easily simulate "module not present"
        # if it's actually installed, so this test runs only when
        # picamera2 is NOT importable.
        try:
            import picamera2  # noqa: F401
            pytest.skip("picamera2 is installed in this env; can't test missing-import path")
        except ImportError:
            pass

        sensor = PiCameraSensor(default_config)
        with pytest.raises(CameraError, match="picamera2 not available"):
            sensor.start()