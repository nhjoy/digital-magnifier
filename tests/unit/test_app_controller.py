"""Unit tests for MagnifierApp.

Strategy
--------
All hardware-touching dependencies are injected, so tests construct
the controller with fakes and call methods directly. The tests focus
on the controller's behavioural surface — event dispatch, in-state
handlers, state transitions, frozen-frame cache, pan reset, capture
flash timing, and resilient error handling — not on the cv2/numpy
processing pipeline (those modules have their own tests).

Where the main loop or rendering are exercised, ``cv2.imshow`` and
``cv2.destroyAllWindows`` are monkey-patched to no-ops so the tests
run headless. Time-based logic uses an injectable clock via
``monkeypatch.setattr(time, 'monotonic', ...)``.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from digital_magnifier.core.app_controller import MagnifierApp
from digital_magnifier.core.events import AppEvent
from digital_magnifier.core.state_machine import AppState
from digital_magnifier.hal.camera_base import CameraSensor, CameraError
from digital_magnifier.hal.controls_base import ControlsHAL
from digital_magnifier.storage.image_saver import ImageSaver


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeCamera(CameraSensor):
    """A camera that yields a sequence of preset frames (or one frame
    forever). Tracks start/stop calls for lifecycle tests."""

    def __init__(self, frames: list[np.ndarray] | None = None) -> None:
        if frames is None:
            frames = [np.zeros((240, 320, 3), dtype=np.uint8)]
        self._frames = frames
        self._index = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.fail_next: int = 0  # number of upcoming get_frame() calls to fail

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def get_frame(self) -> np.ndarray:
        if self.fail_next > 0:
            self.fail_next -= 1
            raise CameraError("simulated camera failure")
        # Cycle through the frames so a long test doesn't run out.
        frame = self._frames[self._index % len(self._frames)]
        self._index += 1
        return frame.copy()


class FakeControls(ControlsHAL):
    """A controls HAL that returns a scripted sequence of events."""

    def __init__(self, events: list[AppEvent] | None = None) -> None:
        self._events = list(events or [])
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1

    def poll(self) -> AppEvent:
        if not self._events:
            return AppEvent.NONE
        return self._events.pop(0)


class FakeImageSaver:
    """Records saved frames in memory; pretends to write to /tmp paths."""

    def __init__(self) -> None:
        self.saved_frames: list[np.ndarray] = []

    @property
    def output_directory(self) -> Path:
        return Path("/tmp/fake-captures")

    def save(self, frame: np.ndarray, timestamp: datetime | None = None) -> Path:
        self.saved_frames.append(frame.copy())
        return Path(f"/tmp/fake-captures/capture_{len(self.saved_frames):04d}.png")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def config() -> dict:
    """Minimal but complete config covering every section the controller reads."""
    return {
        "app": {
            "target_fps": 60,    # high fps so tests aren't slowed
            "show_overlay": False,
        },
        "magnifier": {
            "min_zoom": 1.0,
            "max_zoom": 4.0,
            "zoom_step": 1.0,
            "default_zoom": 1.0,
            "pan_step": 0.5,
        },
        "filters": {
            "default": "normal",
            "available": ["normal", "grayscale", "high_contrast"],
        },
        "capture": {
            "flash_duration_ms": 100,
            "output_directory": "/tmp",
        },
        "state_machine": {
            "log_transitions": False,
        },
        "camera": {
            "width": 320,
            "height": 240,
        },
    }


@pytest.fixture
def camera() -> FakeCamera:
    return FakeCamera()


@pytest.fixture
def controls() -> FakeControls:
    return FakeControls()


@pytest.fixture
def saver() -> FakeImageSaver:
    return FakeImageSaver()


@pytest.fixture
def app(camera, controls, saver, config) -> MagnifierApp:
    """A controller wired with fakes, sitting in STARTUP."""
    return MagnifierApp(camera, controls, saver, config)


@pytest.fixture
def app_live(app) -> MagnifierApp:
    """A controller already transitioned into LIVE_VIEW."""
    app._state_machine.handle(AppEvent.STARTUP_COMPLETE)
    app._on_enter_live()
    return app


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_initial_state_is_startup(self, app):
        assert app._state_machine.current_state == AppState.STARTUP

    def test_initial_zoom_pan_filter(self, app):
        assert app._zoom == 1.0
        assert app._pan_x == 0.0
        assert app._pan_y == 0.0
        assert app._current_filter() == "normal"

    def test_config_values_loaded(self, app):
        assert app._zoom_min == 1.0
        assert app._zoom_max == 4.0
        assert app._zoom_step == 1.0
        assert app._capture_flash_ms == 100
        assert app._filters_available == ["normal", "grayscale", "high_contrast"]

    def test_unknown_default_filter_falls_back(self, camera, controls, saver):
        cfg = {"filters": {"default": "nonexistent", "available": ["normal", "x"]}}
        app = MagnifierApp(camera, controls, saver, cfg)
        # Should fall back to first available, not crash
        assert app._current_filter() == "normal"


# --------------------------------------------------------------------------- #
# Zoom and pan handlers (in-state)
# --------------------------------------------------------------------------- #
class TestZoomHandlers:
    def test_zoom_in_increments(self, app_live):
        app_live._dispatch(AppEvent.ZOOM_IN)
        assert app_live._zoom == 2.0

    def test_zoom_in_clamped_at_max(self, app_live):
        app_live._zoom = 4.0
        app_live._dispatch(AppEvent.ZOOM_IN)
        assert app_live._zoom == 4.0

    def test_zoom_out_decrements(self, app_live):
        app_live._zoom = 3.0
        app_live._dispatch(AppEvent.ZOOM_OUT)
        assert app_live._zoom == 2.0

    def test_zoom_out_clamped_at_min(self, app_live):
        assert app_live._zoom == 1.0
        app_live._dispatch(AppEvent.ZOOM_OUT)
        assert app_live._zoom == 1.0

    def test_zoom_returning_to_one_resets_pan(self, app_live):
        app_live._zoom = 2.0
        app_live._pan_x = 0.5
        app_live._pan_y = -0.3
        app_live._dispatch(AppEvent.ZOOM_OUT)
        assert app_live._zoom == 1.0
        assert app_live._pan_x == 0.0
        assert app_live._pan_y == 0.0

    def test_zoom_above_one_preserves_pan(self, app_live):
        app_live._zoom = 3.0
        app_live._pan_x = 0.5
        app_live._dispatch(AppEvent.ZOOM_OUT)
        assert app_live._zoom == 2.0
        assert app_live._pan_x == 0.5  # not reset

    def test_zoom_invalidates_cache(self, app_live):
        app_live._cache_dirty = False
        app_live._dispatch(AppEvent.ZOOM_IN)
        assert app_live._cache_dirty is True


class TestPanHandlers:
    def test_pan_up_decreases_y(self, app_live):
        app_live._dispatch(AppEvent.PAN_UP)
        assert app_live._pan_y == -0.5  # pan_step is 0.5

    def test_pan_down_increases_y(self, app_live):
        app_live._dispatch(AppEvent.PAN_DOWN)
        assert app_live._pan_y == 0.5

    def test_pan_left_decreases_x(self, app_live):
        app_live._dispatch(AppEvent.PAN_LEFT)
        assert app_live._pan_x == -0.5

    def test_pan_right_increases_x(self, app_live):
        app_live._dispatch(AppEvent.PAN_RIGHT)
        assert app_live._pan_x == 0.5

    @pytest.mark.parametrize("event,attr", [
        (AppEvent.PAN_UP, "_pan_y"),
        (AppEvent.PAN_DOWN, "_pan_y"),
        (AppEvent.PAN_LEFT, "_pan_x"),
        (AppEvent.PAN_RIGHT, "_pan_x"),
    ])
    def test_pan_clamped_to_unit_range(self, app_live, event, attr):
        # Hit the limit hard
        for _ in range(10):
            app_live._dispatch(event)
        value = getattr(app_live, attr)
        assert -1.0 <= value <= 1.0


class TestFilterHandler:
    def test_filter_next_cycles(self, app_live):
        assert app_live._current_filter() == "normal"
        app_live._dispatch(AppEvent.FILTER_NEXT)
        assert app_live._current_filter() == "grayscale"
        app_live._dispatch(AppEvent.FILTER_NEXT)
        assert app_live._current_filter() == "high_contrast"
        app_live._dispatch(AppEvent.FILTER_NEXT)
        assert app_live._current_filter() == "normal"  # wraps


class TestResetHandler:
    def test_reset_view_restores_defaults(self, app_live):
        app_live._zoom = 3.0
        app_live._pan_x = 0.5
        app_live._pan_y = -0.7
        app_live._filter_index = 2
        app_live._dispatch(AppEvent.RESET_VIEW)
        assert app_live._zoom == 1.0
        assert app_live._pan_x == 0.0
        assert app_live._pan_y == 0.0
        assert app_live._current_filter() == "normal"


# --------------------------------------------------------------------------- #
# In-state events should be ignored outside view states
# --------------------------------------------------------------------------- #
class TestStateGating:
    def test_zoom_ignored_in_menu(self, app_live):
        app_live._state_machine.force_transition(AppState.MENU_VIEW)
        z_before = app_live._zoom
        app_live._dispatch(AppEvent.ZOOM_IN)
        assert app_live._zoom == z_before

    def test_pan_ignored_in_gallery(self, app_live):
        app_live._state_machine.force_transition(AppState.GALLERY_VIEW)
        px_before = app_live._pan_x
        app_live._dispatch(AppEvent.PAN_LEFT)
        assert app_live._pan_x == px_before

    def test_filter_works_in_frozen_view(self, app_live, camera):
        # FROZEN_VIEW counts as a view state, filter cycling should work
        app_live._dispatch(AppEvent.FREEZE_TOGGLE)
        assert app_live._state_machine.is_in(AppState.FROZEN_VIEW)
        app_live._dispatch(AppEvent.FILTER_NEXT)
        assert app_live._current_filter() == "grayscale"

    def test_zoom_works_in_frozen_view(self, app_live):
        app_live._dispatch(AppEvent.FREEZE_TOGGLE)
        app_live._dispatch(AppEvent.ZOOM_IN)
        assert app_live._zoom == 2.0


# --------------------------------------------------------------------------- #
# State transitions
# --------------------------------------------------------------------------- #
class TestFreezeTransition:
    def test_freeze_captures_current_frame(self, app_live, camera):
        unique_frame = np.full((240, 320, 3), 42, dtype=np.uint8)
        camera._frames = [unique_frame]
        camera._index = 0
        app_live._dispatch(AppEvent.FREEZE_TOGGLE)
        assert app_live._state_machine.is_in(AppState.FROZEN_VIEW)
        assert app_live._frozen_raw_frame is not None
        assert np.array_equal(app_live._frozen_raw_frame, unique_frame)

    def test_unfreeze_clears_frozen_frame(self, app_live):
        app_live._dispatch(AppEvent.FREEZE_TOGGLE)
        assert app_live._frozen_raw_frame is not None
        app_live._dispatch(AppEvent.FREEZE_TOGGLE)
        assert app_live._state_machine.is_in(AppState.LIVE_VIEW)
        assert app_live._frozen_raw_frame is None
        assert app_live._cached_display_frame is None

    def test_freeze_failure_stays_live(self, app_live, camera):
        camera.fail_next = 1
        app_live._dispatch(AppEvent.FREEZE_TOGGLE)
        assert app_live._state_machine.is_in(AppState.LIVE_VIEW)


# --------------------------------------------------------------------------- #
# Frozen-frame cache
# --------------------------------------------------------------------------- #
class TestFrozenFrameCache:
    def test_cache_dirty_after_freeze(self, app_live):
        app_live._dispatch(AppEvent.FREEZE_TOGGLE)
        assert app_live._cache_dirty is True

    def test_cache_clean_after_first_render(self, app_live):
        app_live._dispatch(AppEvent.FREEZE_TOGGLE)
        frame = app_live._frozen_raw_frame
        _ = app_live._render_for_current_state(frame)
        assert app_live._cache_dirty is False
        assert app_live._cached_display_frame is not None

    def test_cache_reused_on_subsequent_render(self, app_live):
        app_live._dispatch(AppEvent.FREEZE_TOGGLE)
        frame = app_live._frozen_raw_frame
        first = app_live._render_for_current_state(frame)
        # Modify the cached object so we can detect reuse vs reprocess
        # (in real code processed frames are new arrays, but the cache
        # path returns the same object reference)
        second = app_live._render_for_current_state(frame)
        assert second is first

    def test_zoom_change_invalidates_cache(self, app_live):
        app_live._dispatch(AppEvent.FREEZE_TOGGLE)
        _ = app_live._render_for_current_state(app_live._frozen_raw_frame)
        assert app_live._cache_dirty is False
        app_live._dispatch(AppEvent.ZOOM_IN)
        assert app_live._cache_dirty is True

    def test_filter_change_invalidates_cache(self, app_live):
        app_live._dispatch(AppEvent.FREEZE_TOGGLE)
        _ = app_live._render_for_current_state(app_live._frozen_raw_frame)
        app_live._dispatch(AppEvent.FILTER_NEXT)
        assert app_live._cache_dirty is True


# --------------------------------------------------------------------------- #
# Capture flow
# --------------------------------------------------------------------------- #
class TestCapture:
    def test_capture_from_live_saves_image(self, app_live, saver):
        app_live._dispatch(AppEvent.CAPTURE_IMAGE)
        assert app_live._state_machine.is_in(AppState.CAPTURE_FLASH)
        assert len(saver.saved_frames) == 1

    def test_capture_from_frozen_saves_frozen_frame(self, app_live, saver, camera):
        unique = np.full((240, 320, 3), 99, dtype=np.uint8)
        camera._frames = [unique]
        camera._index = 0
        app_live._dispatch(AppEvent.FREEZE_TOGGLE)
        app_live._dispatch(AppEvent.CAPTURE_IMAGE)
        assert len(saver.saved_frames) == 1
        # Stub apply_zoom/apply_filter return frame unchanged
        assert np.array_equal(saver.saved_frames[0], unique)

    def test_capture_save_failure_does_not_crash(
        self, app_live, monkeypatch
    ):
        def boom(*a, **kw):
            raise IOError("disk full")
        monkeypatch.setattr(app_live._image_saver, "save", boom)
        app_live._dispatch(AppEvent.CAPTURE_IMAGE)
        # State machine still transitioned; we just logged the error
        assert app_live._state_machine.is_in(AppState.CAPTURE_FLASH)

    def test_flash_timeout_returns_to_live(self, app_live, monkeypatch):
        # Mock the clock so we can fast-forward
        now = [0.0]

        def fake_monotonic() -> float:
            return now[0]

        monkeypatch.setattr(
            "digital_magnifier.core.app_controller.time.monotonic",
            fake_monotonic,
        )

        app_live._dispatch(AppEvent.CAPTURE_IMAGE)
        assert app_live._state_machine.is_in(AppState.CAPTURE_FLASH)

        # Half the flash duration: still in flash
        now[0] += app_live._capture_flash_ms / 2 / 1000
        app_live._check_timed_transitions()
        assert app_live._state_machine.is_in(AppState.CAPTURE_FLASH)

        # Past the duration: should auto-return to live
        now[0] += app_live._capture_flash_ms / 1000
        app_live._check_timed_transitions()
        assert app_live._state_machine.is_in(AppState.LIVE_VIEW)


# --------------------------------------------------------------------------- #
# Mode stubs (gallery, menu)
# --------------------------------------------------------------------------- #
class TestModeStubs:
    def test_gallery_open_transitions(self, app_live):
        app_live._dispatch(AppEvent.GALLERY_OPEN)
        assert app_live._state_machine.is_in(AppState.GALLERY_VIEW)

    def test_gallery_back_returns_to_live(self, app_live):
        app_live._dispatch(AppEvent.GALLERY_OPEN)
        app_live._dispatch(AppEvent.BACK)
        assert app_live._state_machine.is_in(AppState.LIVE_VIEW)

    def test_menu_open_via_power_short(self, app_live):
        app_live._dispatch(AppEvent.POWER_SHORT_PRESS)
        assert app_live._state_machine.is_in(AppState.MENU_VIEW)

    def test_menu_back_returns_to_live(self, app_live):
        app_live._dispatch(AppEvent.POWER_SHORT_PRESS)
        app_live._dispatch(AppEvent.POWER_SHORT_PRESS)
        assert app_live._state_machine.is_in(AppState.LIVE_VIEW)

    def test_placeholder_frame_for_gallery(self, app_live):
        app_live._state_machine.force_transition(AppState.GALLERY_VIEW)
        frame = app_live._acquire_frame_for_current_state()
        assert frame.shape == (240, 320, 3)
        assert frame.dtype == np.uint8


# --------------------------------------------------------------------------- #
# Shutdown
# --------------------------------------------------------------------------- #
class TestShutdown:
    def test_power_long_press_shuts_down(self, app_live):
        app_live._dispatch(AppEvent.POWER_LONG_PRESS)
        assert app_live._state_machine.is_in(AppState.SHUTDOWN)

    def test_quit_event_shuts_down(self, app_live):
        app_live._dispatch(AppEvent.QUIT)
        assert app_live._state_machine.is_in(AppState.SHUTDOWN)


# --------------------------------------------------------------------------- #
# Main loop resilience (with cv2 stubbed)
# --------------------------------------------------------------------------- #
class TestMainLoopResilience:
    @pytest.fixture(autouse=True)
    def stub_cv2_display(self, monkeypatch):
        """Make cv2.imshow / namedWindow / destroyAllWindows no-ops."""
        import cv2
        monkeypatch.setattr(cv2, "imshow", lambda *a, **kw: None)
        monkeypatch.setattr(cv2, "namedWindow", lambda *a, **kw: None)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **kw: None)

    def test_loop_recovers_from_intermittent_camera_failure(
        self, camera, controls, saver, config
    ):
        """A few failed frames in a row do not shut the device down."""
        camera.fail_next = 3  # fewer than MAX_CONSECUTIVE_FAILURES
        controls._events = [AppEvent.NONE] * 10 + [AppEvent.QUIT]
        config["app"]["target_fps"] = 1000

        app = MagnifierApp(camera, controls, saver, config)
        app.run()

        # Reached SHUTDOWN via QUIT, not via the failure threshold.
        assert app._state_machine.is_in(AppState.SHUTDOWN)
        assert app._consecutive_failures < MagnifierApp.MAX_CONSECUTIVE_FAILURES

    def test_excessive_failures_triggers_shutdown(
        self, camera, controls, saver, config
    ):
        # Make camera fail more than MAX_CONSECUTIVE_FAILURES times
        camera.fail_next = MagnifierApp.MAX_CONSECUTIVE_FAILURES + 5
        config["app"]["target_fps"] = 1000   # essentially no sleep

        app = MagnifierApp(camera, controls, saver, config)
        app._state_machine.handle(AppEvent.STARTUP_COMPLETE)
        app._run_loop()

        assert app._state_machine.is_in(AppState.SHUTDOWN)

    def test_keyboard_interrupt_shuts_down(
        self, camera, controls, saver, config, monkeypatch
    ):
        # Make poll() raise KeyboardInterrupt on first call
        def boom() -> AppEvent:
            raise KeyboardInterrupt()
        controls.poll = boom  # type: ignore[method-assign]

        config["app"]["target_fps"] = 1000
        app = MagnifierApp(camera, controls, saver, config)
        app._state_machine.handle(AppEvent.STARTUP_COMPLETE)
        app._run_loop()

        assert app._state_machine.is_in(AppState.SHUTDOWN)


# --------------------------------------------------------------------------- #
# Lifecycle integration via run()
# --------------------------------------------------------------------------- #
class TestRunLifecycle:
    @pytest.fixture(autouse=True)
    def stub_cv2_display(self, monkeypatch):
        import cv2
        monkeypatch.setattr(cv2, "imshow", lambda *a, **kw: None)
        monkeypatch.setattr(cv2, "namedWindow", lambda *a, **kw: None)
        monkeypatch.setattr(cv2, "destroyAllWindows", lambda *a, **kw: None)

    def test_run_starts_and_stops_hals(self, camera, controls, saver, config):
        controls._events = [AppEvent.QUIT]  # immediate shutdown
        config["app"]["target_fps"] = 1000
        app = MagnifierApp(camera, controls, saver, config)
        app.run()
        assert camera.start_calls == 1
        assert camera.stop_calls == 1
        assert controls.start_calls == 1
        assert controls.stop_calls == 1
        assert app._state_machine.is_in(AppState.SHUTDOWN)

    def test_run_processes_scripted_session(self, camera, controls, saver, config):
        """Realistic mini-session: zoom in twice, freeze, capture, quit."""
        controls._events = [
            AppEvent.ZOOM_IN,
            AppEvent.ZOOM_IN,
            AppEvent.FREEZE_TOGGLE,
            AppEvent.CAPTURE_IMAGE,
            AppEvent.QUIT,
        ]
        config["app"]["target_fps"] = 1000
        app = MagnifierApp(camera, controls, saver, config)
        app.run()
        assert app._zoom == 3.0
        assert len(saver.saved_frames) == 1
        assert app._state_machine.is_in(AppState.SHUTDOWN)