"""Unit tests for ``processing.magnifier.apply_zoom``."""

from __future__ import annotations

import numpy as np
import pytest

from digital_magnifier.processing.magnifier import apply_zoom


# A small frame for fast tests; pattern keeps the centre distinct
# from the edges so we can verify cropping works as expected.
@pytest.fixture
def frame() -> np.ndarray:
    f = np.zeros((100, 200, 3), dtype=np.uint8)
    # Centre block at (50, 100) ± 10 — bright red.
    f[40:60, 90:110] = [0, 0, 255]
    # Edges — dim green.
    f[0:5, :] = [0, 128, 0]
    f[-5:, :] = [0, 128, 0]
    f[:, 0:5] = [0, 128, 0]
    f[:, -5:] = [0, 128, 0]
    return f


# --------------------------------------------------------------------------- #
# Zoom = 1.0 (and below)
# --------------------------------------------------------------------------- #
class TestPassthrough:
    def test_zoom_one_returns_input(self, frame):
        result = apply_zoom(frame, zoom=1.0)
        assert np.array_equal(result, frame)

    def test_zoom_below_one_returns_input(self, frame):
        result = apply_zoom(frame, zoom=0.5)
        assert np.array_equal(result, frame)


# --------------------------------------------------------------------------- #
# Zoom > 1.0 — basic correctness
# --------------------------------------------------------------------------- #
class TestZoomDimensions:
    @pytest.mark.parametrize("zoom", [1.5, 2.0, 3.0, 4.0])
    def test_output_shape_matches_input(self, frame, zoom):
        result = apply_zoom(frame, zoom=zoom)
        assert result.shape == frame.shape

    def test_output_dtype_matches(self, frame):
        result = apply_zoom(frame, zoom=2.0)
        assert result.dtype == frame.dtype


# --------------------------------------------------------------------------- #
# Zoom focuses on the centre when pan=0
# --------------------------------------------------------------------------- #
class TestZoomCentered:
    def test_centered_zoom_picks_up_red_centre(self, frame):
        # At zoom=2.0 and pan=0, the crop is the centre 100x50, which
        # contains our red block. After upscaling, the centre of the
        # output should be very red.
        result = apply_zoom(frame, zoom=2.0, pan_x=0.0, pan_y=0.0)
        cx, cy = result.shape[1] // 2, result.shape[0] // 2
        pixel = result[cy, cx]
        # Blue channel is 0, green is 0, red is high.
        assert pixel[2] > 200, f"expected red centre, got {pixel}"


# --------------------------------------------------------------------------- #
# Pan
# --------------------------------------------------------------------------- #
class TestPan:
    def test_pan_clamped_to_unit(self, frame):
        # Out-of-range pan must not crash.
        result = apply_zoom(frame, zoom=2.0, pan_x=5.0, pan_y=-3.0)
        assert result.shape == frame.shape

    def test_pan_changes_result(self, frame):
        # Different pan offsets should produce visibly different frames.
        centred = apply_zoom(frame, zoom=2.0, pan_x=0.0, pan_y=0.0)
        right = apply_zoom(frame, zoom=2.0, pan_x=1.0, pan_y=0.0)
        # Not identical
        assert not np.array_equal(centred, right)

    def test_pan_at_one_finds_edge_green(self, frame):
        # At zoom=2 and pan_x=1, the crop is shifted hard right; the
        # right edge of the source (green) should dominate the right
        # side of the output.
        result = apply_zoom(frame, zoom=2.0, pan_x=1.0)
        # Sample a pixel on the right edge of the output
        h, w = result.shape[:2]
        right_pixel = result[h // 2, w - 5]
        # Should have some green (channel 1)
        assert right_pixel[1] > 50, f"expected green-ish edge, got {right_pixel}"