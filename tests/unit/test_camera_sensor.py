"""Unit tests for ``MockCameraSensor``.

These tests don't require a real webcam. The fallback chain
(webcam → fallback_image → synthetic frame) is verified by either
forcing ``source: test_image`` (skips the webcam attempt) or by
configuring a non-existent webcam index and letting the chain run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from digital_magnifier.hal.camera_base import CameraError
from digital_magnifier.hal.camera_sensor import (
    MockCameraSensor,
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
