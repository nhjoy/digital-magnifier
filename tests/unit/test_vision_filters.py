"""Unit tests for ``processing.vision_filters``."""

from __future__ import annotations

import numpy as np
import pytest

from digital_magnifier.processing.vision_filters import (
    AVAILABLE_FILTERS,
    FILTER_BINARY,
    FILTER_GRAYSCALE,
    FILTER_HIGH_CONTRAST,
    FILTER_INVERTED,
    FILTER_NORMAL,
    FILTER_YELLOW_ON_BLACK,
    FILTER_WHITE_ON_BLACK,
    apply_filter,
)


@pytest.fixture
def frame() -> np.ndarray:
    """A frame with a colour gradient — exercises all filters meaningfully."""
    h, w = 100, 200
    f = np.zeros((h, w, 3), dtype=np.uint8)
    # Vertical red gradient (BGR so red is channel 2)
    for y in range(h):
        f[y, :, 2] = int(255 * y / h)
    # Horizontal blue gradient (channel 0)
    for x in range(w):
        f[:, x, 0] = int(255 * x / w)
    # Constant mid-grey green (channel 1)
    f[:, :, 1] = 128
    return f


# --------------------------------------------------------------------------- #
# Each filter preserves the basic invariants
# --------------------------------------------------------------------------- #
class TestFilterContract:
    @pytest.mark.parametrize("name", list(AVAILABLE_FILTERS))
    def test_shape_preserved(self, frame, name):
        result = apply_filter(frame, name)
        assert result.shape == frame.shape, f"{name} changed shape"

    @pytest.mark.parametrize("name", list(AVAILABLE_FILTERS))
    def test_dtype_preserved(self, frame, name):
        result = apply_filter(frame, name)
        assert result.dtype == frame.dtype, f"{name} changed dtype"


# --------------------------------------------------------------------------- #
# normal is pass-through
# --------------------------------------------------------------------------- #
class TestNormal:
    def test_normal_returns_input(self, frame):
        assert np.array_equal(apply_filter(frame, FILTER_NORMAL), frame)


# --------------------------------------------------------------------------- #
# grayscale: all 3 channels equal
# --------------------------------------------------------------------------- #
class TestGrayscale:
    def test_channels_equal(self, frame):
        result = apply_filter(frame, FILTER_GRAYSCALE)
        # B == G == R everywhere
        assert np.array_equal(result[..., 0], result[..., 1])
        assert np.array_equal(result[..., 1], result[..., 2])

    def test_changes_input(self, frame):
        result = apply_filter(frame, FILTER_GRAYSCALE)
        assert not np.array_equal(result, frame)


# --------------------------------------------------------------------------- #
# high_contrast
# --------------------------------------------------------------------------- #
class TestHighContrast:
    def test_modifies_frame(self, frame):
        result = apply_filter(frame, FILTER_HIGH_CONTRAST)
        # CLAHE redistributes intensities; result should differ
        assert not np.array_equal(result, frame)


# --------------------------------------------------------------------------- #
# inverted
# --------------------------------------------------------------------------- #
class TestInverted:
    def test_inverts(self, frame):
        result = apply_filter(frame, FILTER_INVERTED)
        assert np.array_equal(result, 255 - frame)


# --------------------------------------------------------------------------- #
# binary: only two values per channel
# --------------------------------------------------------------------------- #
class TestBinary:
    def test_only_extreme_values(self, frame):
        result = apply_filter(frame, FILTER_BINARY)
        # 3-channel output where each pixel is either fully black or
        # fully white (because we converted from a single-channel
        # binary image).
        unique_values = np.unique(result)
        # Should be at most {0, 255}
        assert set(unique_values.tolist()).issubset({0, 255}), (
            f"binary should only contain 0 or 255, got {unique_values}"
        )


# --------------------------------------------------------------------------- #
# Unknown filter is handled gracefully
# --------------------------------------------------------------------------- #
class TestUnknownFilter:
    def test_unknown_filter_logs_warning_and_passes_through(self, frame, caplog):
        with caplog.at_level("WARNING"):
            result = apply_filter(frame, "definitely_not_a_real_filter")
        assert np.array_equal(result, frame)
        assert any("unknown filter" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
# yellow_on_black: output is only black (0,0,0) or yellow (0,G,R)
# --------------------------------------------------------------------------- #
class TestYellowOnBlack:
    def test_only_yellow_and_black(self, frame):
        result = apply_filter(frame, FILTER_YELLOW_ON_BLACK)
        # Blue channel should be all zeros (no blue component).
        assert np.all(result[..., 0] == 0), "yellow_on_black should have no blue"
        # Green and red channels should be identical (both are the mask).
        assert np.array_equal(result[..., 1], result[..., 2])

    def test_only_extreme_values_in_mask(self, frame):
        result = apply_filter(frame, FILTER_YELLOW_ON_BLACK)
        unique_g = np.unique(result[..., 1])
        assert set(unique_g.tolist()).issubset({0, 255})


# --------------------------------------------------------------------------- #
# white_on_black: output is greyscale binary (only 0 or 255)
# --------------------------------------------------------------------------- #
class TestWhiteOnBlack:
    def test_only_extreme_values(self, frame):
        result = apply_filter(frame, FILTER_WHITE_ON_BLACK)
        unique_values = np.unique(result)
        assert set(unique_values.tolist()).issubset({0, 255})

    def test_channels_are_equal(self, frame):
        result = apply_filter(frame, FILTER_WHITE_ON_BLACK)
        assert np.array_equal(result[..., 0], result[..., 1])
        assert np.array_equal(result[..., 1], result[..., 2])


# --------------------------------------------------------------------------- #
# Config parameter passing
# --------------------------------------------------------------------------- #
class TestConfigPassthrough:
    def test_binary_with_custom_block_size(self, frame):
        cfg = {"binary": {"block_size": 51, "offset": 5}}
        result = apply_filter(frame, FILTER_BINARY, config=cfg)
        assert result.shape == frame.shape
        unique = np.unique(result)
        assert set(unique.tolist()).issubset({0, 255})

    def test_high_contrast_with_custom_clip_limit(self, frame):
        cfg = {"high_contrast": {"clip_limit": 1.0, "tile_size": 4}}
        result = apply_filter(frame, FILTER_HIGH_CONTRAST, config=cfg)
        assert result.shape == frame.shape
        assert not np.array_equal(result, frame)

    def test_config_for_wrong_filter_is_ignored(self, frame):
        # Config dict has binary params but we call high_contrast
        cfg = {"binary": {"block_size": 99}}
        result = apply_filter(frame, FILTER_HIGH_CONTRAST, config=cfg)
        assert result.shape == frame.shape

    def test_none_config_is_safe(self, frame):
        result = apply_filter(frame, FILTER_BINARY, config=None)
        assert result.shape == frame.shape