"""Unit tests for the Gallery class (MVP 0.4).

The gallery is rendered with real cv2 so tests check shape / dtype
rather than pixel content. Time is injected via a FakeClock so
delete-confirmation timing is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from digital_magnifier.core.gallery import Gallery
from digital_magnifier.storage.image_saver import ImageSaver


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_image(path: Path, colour: tuple[int, int, int] = (50, 100, 150)) -> Path:
    """Write a small real PNG with a recognisable solid colour."""
    frame = np.full((30, 40, 3), colour, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame)
    assert ok
    return path


def _set_mtime(path: Path, mtime: float) -> None:
    import os
    os.utime(path, (mtime, mtime))


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def filters() -> list[str]:
    return ["normal", "grayscale", "high_contrast", "inverted", "binary"]


@pytest.fixture
def saver(tmp_path: Path) -> ImageSaver:
    return ImageSaver(tmp_path)


@pytest.fixture
def populated_saver(tmp_path: Path) -> ImageSaver:
    """Saver with three captures: newest=c, oldest=a."""
    a = _make_image(tmp_path / "a.png", colour=(20, 20, 20))
    b = _make_image(tmp_path / "b.png", colour=(40, 40, 40))
    c = _make_image(tmp_path / "c.png", colour=(80, 80, 80))
    _set_mtime(a, 1000.0)
    _set_mtime(b, 2000.0)
    _set_mtime(c, 3000.0)
    return ImageSaver(tmp_path)


def _make_gallery(saver: ImageSaver, filters: list[str], clock=None) -> Gallery:
    return Gallery(
        image_saver=saver,
        display_width=320,
        display_height=240,
        filters_available=filters,
        initial_filter="normal",
        clock=clock if clock is not None else FakeClock(),
    )


# --------------------------------------------------------------------------- #
# Empty state
# --------------------------------------------------------------------------- #
class TestEmptyGallery:
    def test_is_empty_before_open(self, saver, filters):
        g = _make_gallery(saver, filters)
        assert g.is_empty
        assert g.count == 0

    def test_open_with_no_captures(self, saver, filters):
        g = _make_gallery(saver, filters)
        g.open()
        assert g.is_empty
        assert g.current_path is None

    def test_render_empty_returns_correct_shape(self, saver, filters):
        g = _make_gallery(saver, filters)
        g.open()
        frame = g.render()
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

    def test_navigation_is_noop_when_empty(self, saver, filters):
        g = _make_gallery(saver, filters)
        g.open()
        # Should not raise, should not throw IndexError
        g.next(); g.prev(); g.zoom_in(); g.zoom_out()
        g.pan_up(); g.pan_down(); g.filter_next(); g.reset_view()
        assert g.is_empty

    def test_request_delete_is_noop_when_empty(self, saver, filters):
        g = _make_gallery(saver, filters)
        g.open()
        assert g.request_delete() is None
        assert not g.delete_armed


# --------------------------------------------------------------------------- #
# Open / load / navigate
# --------------------------------------------------------------------------- #
class TestPopulatedGallery:
    def test_open_loads_newest_first(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        assert g.count == 3
        assert g.current_index == 0
        # Newest is c.png (mtime 3000)
        assert g.current_path.name == "c.png"

    def test_next_advances(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        g.next()
        assert g.current_path.name == "b.png"
        g.next()
        assert g.current_path.name == "a.png"

    def test_next_wraps_around(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        g.next(); g.next(); g.next()
        # Wrapped back to start (newest)
        assert g.current_index == 0
        assert g.current_path.name == "c.png"

    def test_prev_from_zero_wraps_to_end(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        g.prev()
        assert g.current_path.name == "a.png"

    def test_navigation_resets_zoom_and_pan(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        g.zoom_in()
        g.zoom_in()
        g.pan_down()
        # Sanity: state changed
        assert g._zoom > 1.0
        # Navigation should snap back to 1x, centred
        g.next()
        assert g._zoom == 1.0
        assert g._pan_x == 0.0
        assert g._pan_y == 0.0

    def test_navigation_preserves_filter(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        g.filter_next()
        assert g._current_filter() == "grayscale"
        g.next()
        # Filter should stick across image changes
        assert g._current_filter() == "grayscale"


# --------------------------------------------------------------------------- #
# Zoom / pan / filter
# --------------------------------------------------------------------------- #
class TestViewManipulation:
    def test_zoom_in_clamps_at_max(self, populated_saver, filters):
        g = Gallery(
            image_saver=populated_saver,
            display_width=320, display_height=240,
            filters_available=filters,
            zoom_max=3.0, zoom_step=1.0,
            clock=FakeClock(),
        )
        g.open()
        for _ in range(10):
            g.zoom_in()
        assert g._zoom == 3.0

    def test_zoom_out_clamps_at_min(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        for _ in range(10):
            g.zoom_out()
        assert g._zoom == 1.0

    def test_zoom_out_to_one_resets_pan(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        g.zoom_in(); g.zoom_in()
        g.pan_down()
        assert g._pan_y > 0
        # Walk all the way back to 1x
        for _ in range(10):
            g.zoom_out()
        assert g._zoom == 1.0
        assert g._pan_y == 0.0

    def test_pan_only_works_when_zoomed(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        # At 1x, pan should be a no-op (no point — image fills view)
        g.pan_down()
        g.pan_up()
        assert g._pan_y == 0.0

    def test_pan_clamps_to_unit_range(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        g.zoom_in(); g.zoom_in()   # > 1.0
        for _ in range(50):
            g.pan_down()
        assert g._pan_y == 1.0
        for _ in range(50):
            g.pan_up()
        assert g._pan_y == -1.0

    def test_filter_next_cycles_with_wraparound(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        for _ in range(len(filters)):
            g.filter_next()
        assert g._current_filter() == "normal"

    def test_reset_view_restores_defaults(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        g.zoom_in(); g.filter_next()
        g.reset_view()
        assert g._zoom == 1.0
        assert g._pan_x == 0.0
        assert g._pan_y == 0.0
        assert g._current_filter() == "normal"


# --------------------------------------------------------------------------- #
# Delete with confirmation
# --------------------------------------------------------------------------- #
class TestDeleteConfirmation:
    def test_first_press_arms_no_delete(self, populated_saver, filters):
        clock = FakeClock()
        g = _make_gallery(populated_saver, filters, clock=clock)
        g.open()
        result = g.request_delete()
        assert result is None
        assert g.delete_armed
        assert g.count == 3   # nothing deleted yet

    def test_second_press_confirms_and_deletes(self, populated_saver, filters):
        clock = FakeClock()
        g = _make_gallery(populated_saver, filters, clock=clock)
        g.open()
        target = g.current_path

        g.request_delete()             # arm
        deleted = g.request_delete()   # confirm
        assert deleted == target
        assert g.count == 2
        assert not target.exists()
        assert not g.delete_armed

    def test_confirmation_window_expires(self, populated_saver, filters):
        clock = FakeClock()
        g = _make_gallery(populated_saver, filters, clock=clock)
        g.open()
        g.request_delete()
        # Default window is 2 seconds; advance past it
        clock.advance(5.0)
        assert not g.delete_armed   # autoclears
        result = g.request_delete()
        # Treated as a fresh first press
        assert result is None
        assert g.delete_armed
        assert g.count == 3

    def test_navigation_cancels_arm(self, populated_saver, filters):
        clock = FakeClock()
        g = _make_gallery(populated_saver, filters, clock=clock)
        g.open()
        g.request_delete()
        assert g.delete_armed
        g.next()
        assert not g.delete_armed
        # Now next press is fresh arm, not confirm
        g.request_delete()
        assert g.count == 3

    def test_zoom_cancels_arm(self, populated_saver, filters):
        clock = FakeClock()
        g = _make_gallery(populated_saver, filters, clock=clock)
        g.open()
        g.request_delete()
        g.zoom_in()
        assert not g.delete_armed

    def test_filter_cycle_cancels_arm(self, populated_saver, filters):
        clock = FakeClock()
        g = _make_gallery(populated_saver, filters, clock=clock)
        g.open()
        g.request_delete()
        g.filter_next()
        assert not g.delete_armed

    def test_delete_clamps_index_to_last(self, populated_saver, filters):
        clock = FakeClock()
        g = _make_gallery(populated_saver, filters, clock=clock)
        g.open()
        # Go to last image (a.png)
        g.next(); g.next()
        assert g.current_path.name == "a.png"
        assert g.current_index == 2
        g.request_delete(); g.request_delete()
        # 2 images left, index should clamp to 1 (last available)
        assert g.count == 2
        assert g.current_index == 1

    def test_delete_last_image_leaves_empty(self, saver, filters):
        # Single image, delete it
        _make_image(saver.output_directory / "only.png")
        clock = FakeClock()
        g = _make_gallery(saver, filters, clock=clock)
        g.open()
        assert g.count == 1
        g.request_delete(); g.request_delete()
        assert g.is_empty
        # Render shouldn't crash on now-empty gallery
        frame = g.render()
        assert frame.shape == (240, 320, 3)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
class TestRendering:
    def test_render_returns_display_sized_frame(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        frame = g.render()
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8

    def test_render_consistent_after_navigation(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        first = g.render()
        g.next()
        second = g.render()
        # Different images, but both must be display-sized
        assert first.shape == second.shape == (240, 320, 3)

    def test_render_when_image_unreadable_does_not_crash(
        self, populated_saver, filters, tmp_path, monkeypatch
    ):
        g = _make_gallery(populated_saver, filters)
        g.open()
        # Corrupt all images by replacing them with empty files
        for img_path in populated_saver.list_images():
            img_path.write_bytes(b"")
        # Force a reload by clearing cache
        g._loaded_image = None
        g._loaded_index = None
        frame = g.render()
        # Should produce a message frame instead of crashing
        assert frame.shape == (240, 320, 3)

    def test_delete_armed_state_visible_in_render(self, populated_saver, filters):
        g = _make_gallery(populated_saver, filters)
        g.open()
        before = g.render().copy()
        g.request_delete()
        after = g.render()
        # Frames should differ because the confirmation banner is now drawn
        assert not np.array_equal(before, after)
