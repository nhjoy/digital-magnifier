"""Stdlib-only verification of MagnifierApp.

Mirrors the assertions in tests/unit/test_app_controller.py using
plain unittest. The pytest version is what ships; this script just
confirms correctness in an environment without pytest.
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str((__import__("pathlib").Path(__file__).resolve().parents[2] / "src")))

from digital_magnifier.core.app_controller import MagnifierApp
from digital_magnifier.core.events import AppEvent
from digital_magnifier.core.state_machine import AppState
from digital_magnifier.hal.camera_base import CameraSensor, CameraError
from digital_magnifier.hal.controls_base import ControlsHAL


# ---- Fakes ----
class FakeCamera(CameraSensor):
    def __init__(self, frames=None):
        self._frames = frames or [np.zeros((240, 320, 3), dtype=np.uint8)]
        self._index = 0
        self.start_calls = 0
        self.stop_calls = 0
        self.fail_next = 0

    def start(self): self.start_calls += 1
    def stop(self): self.stop_calls += 1

    def get_frame(self):
        if self.fail_next > 0:
            self.fail_next -= 1
            raise CameraError("sim fail")
        f = self._frames[self._index % len(self._frames)]
        self._index += 1
        return f.copy()


class FakeControls(ControlsHAL):
    def __init__(self, events=None):
        self._events = list(events or [])
        self.start_calls = 0
        self.stop_calls = 0

    def start(self): self.start_calls += 1
    def stop(self): self.stop_calls += 1
    def poll(self):
        return self._events.pop(0) if self._events else AppEvent.NONE


class FakeImageSaver:
    def __init__(self):
        self.saved_frames = []

    @property
    def output_directory(self):
        return Path("/tmp/fake")

    def save(self, frame, timestamp=None):
        self.saved_frames.append(frame.copy())
        return Path(f"/tmp/fake/img_{len(self.saved_frames):04d}.png")

    def list_images(self, newest_first=True):
        return []

    def delete_image(self, path):
        pass


def make_config():
    return {
        "app": {"target_fps": 1000, "show_overlay": False, "splash_duration_s": 0},
        "magnifier": {
            "min_zoom": 1.0, "max_zoom": 4.0,
            "zoom_step": 1.0, "default_zoom": 1.0, "pan_step": 0.5,
        },
        "filters": {"default": "normal", "available": ["normal", "grayscale", "high_contrast"]},
        "capture": {"flash_duration_ms": 100, "output_directory": "/tmp"},
        "state_machine": {"log_transitions": False},
        "camera": {"width": 320, "height": 240},
    }


def make_app(events=None, frames=None):
    cam = FakeCamera(frames)
    ctrl = FakeControls(events)
    saver = FakeImageSaver()
    return MagnifierApp(cam, ctrl, saver, make_config()), cam, ctrl, saver


def make_live_app(events=None, frames=None):
    app, cam, ctrl, saver = make_app(events, frames)
    app._state_machine.handle(AppEvent.STARTUP_COMPLETE)
    app._on_enter_live()
    return app, cam, ctrl, saver


# ---- Tests ----
class AppVerification(unittest.TestCase):

    # ----- construction ----------
    def test_initial_state_is_startup(self):
        app, *_ = make_app()
        self.assertEqual(app._state_machine.current_state, AppState.STARTUP)

    def test_initial_values(self):
        app, *_ = make_app()
        self.assertEqual(app._zoom, 1.0)
        self.assertEqual(app._pan_x, 0.0)
        self.assertEqual(app._pan_y, 0.0)
        self.assertEqual(app._current_filter(), "normal")

    def test_unknown_default_filter_falls_back(self):
        cfg = {"filters": {"default": "nonexistent", "available": ["normal", "x"]}}
        app = MagnifierApp(FakeCamera(), FakeControls(), FakeImageSaver(), cfg)
        self.assertEqual(app._current_filter(), "normal")

    # ----- zoom ----------
    def test_zoom_in_increments(self):
        app, *_ = make_live_app()
        app._dispatch(AppEvent.ZOOM_IN)
        self.assertEqual(app._zoom, 2.0)

    def test_zoom_in_clamped_at_max(self):
        app, *_ = make_live_app()
        app._zoom = 4.0
        app._dispatch(AppEvent.ZOOM_IN)
        self.assertEqual(app._zoom, 4.0)

    def test_zoom_out_clamped_at_min(self):
        app, *_ = make_live_app()
        app._dispatch(AppEvent.ZOOM_OUT)
        self.assertEqual(app._zoom, 1.0)

    def test_zoom_returning_to_one_resets_pan(self):
        app, *_ = make_live_app()
        app._zoom = 2.0
        app._pan_x = 0.5
        app._pan_y = -0.3
        app._dispatch(AppEvent.ZOOM_OUT)
        self.assertEqual(app._zoom, 1.0)
        self.assertEqual(app._pan_x, 0.0)
        self.assertEqual(app._pan_y, 0.0)

    def test_zoom_above_one_preserves_pan(self):
        app, *_ = make_live_app()
        app._zoom = 3.0
        app._pan_x = 0.5
        app._dispatch(AppEvent.ZOOM_OUT)
        self.assertEqual(app._pan_x, 0.5)

    # ----- pan ----------
    def test_pan_handlers(self):
        cases = [
            (AppEvent.PAN_UP, "_pan_y", -0.5),
            (AppEvent.PAN_DOWN, "_pan_y", 0.5),
            (AppEvent.PAN_LEFT, "_pan_x", -0.5),
            (AppEvent.PAN_RIGHT, "_pan_x", 0.5),
        ]
        for ev, attr, expected in cases:
            app, *_ = make_live_app()
            app._dispatch(ev)
            self.assertEqual(getattr(app, attr), expected, f"{ev.name}")

    def test_pan_clamped(self):
        app, *_ = make_live_app()
        for _ in range(10):
            app._dispatch(AppEvent.PAN_RIGHT)
        self.assertLessEqual(app._pan_x, 1.0)
        self.assertGreaterEqual(app._pan_x, -1.0)

    # ----- filter ----------
    def test_filter_cycles(self):
        app, *_ = make_live_app()
        self.assertEqual(app._current_filter(), "normal")
        app._dispatch(AppEvent.FILTER_NEXT)
        self.assertEqual(app._current_filter(), "grayscale")
        app._dispatch(AppEvent.FILTER_NEXT)
        self.assertEqual(app._current_filter(), "high_contrast")
        app._dispatch(AppEvent.FILTER_NEXT)
        self.assertEqual(app._current_filter(), "normal")

    # ----- reset ----------
    def test_reset_view(self):
        app, *_ = make_live_app()
        app._zoom = 3.0
        app._pan_x = 0.7
        app._pan_y = -0.4
        app._filter_index = 2
        app._dispatch(AppEvent.RESET_VIEW)
        self.assertEqual(app._zoom, 1.0)
        self.assertEqual(app._pan_x, 0.0)
        self.assertEqual(app._pan_y, 0.0)
        self.assertEqual(app._current_filter(), "normal")

    # ----- state gating ----------
    def test_zoom_ignored_in_menu(self):
        app, *_ = make_live_app()
        app._state_machine.force_transition(AppState.MENU_VIEW)
        before = app._zoom
        app._dispatch(AppEvent.ZOOM_IN)
        self.assertEqual(app._zoom, before)

    def test_pan_ignored_in_gallery(self):
        app, *_ = make_live_app()
        app._state_machine.force_transition(AppState.GALLERY_VIEW)
        before = app._pan_x
        app._dispatch(AppEvent.PAN_LEFT)
        self.assertEqual(app._pan_x, before)

    def test_filter_works_in_frozen(self):
        app, *_ = make_live_app()
        app._dispatch(AppEvent.FREEZE_TOGGLE)
        self.assertTrue(app._state_machine.is_in(AppState.FROZEN_VIEW))
        app._dispatch(AppEvent.FILTER_NEXT)
        self.assertEqual(app._current_filter(), "grayscale")

    # ----- freeze ----------
    def test_freeze_captures_frame(self):
        unique = np.full((240, 320, 3), 42, dtype=np.uint8)
        app, *_ = make_live_app(frames=[unique])
        app._dispatch(AppEvent.FREEZE_TOGGLE)
        self.assertTrue(app._state_machine.is_in(AppState.FROZEN_VIEW))
        self.assertIsNotNone(app._frozen_raw_frame)
        self.assertTrue(np.array_equal(app._frozen_raw_frame, unique))

    def test_unfreeze_clears_frame(self):
        app, *_ = make_live_app()
        app._dispatch(AppEvent.FREEZE_TOGGLE)
        self.assertIsNotNone(app._frozen_raw_frame)
        app._dispatch(AppEvent.FREEZE_TOGGLE)
        self.assertIsNone(app._frozen_raw_frame)
        self.assertIsNone(app._cached_display_frame)

    def test_freeze_failure_stays_live(self):
        app, cam, *_ = make_live_app()
        cam.fail_next = 1
        app._dispatch(AppEvent.FREEZE_TOGGLE)
        self.assertTrue(app._state_machine.is_in(AppState.LIVE_VIEW))

    # ----- cache ----------
    def test_cache_dirty_after_freeze(self):
        app, *_ = make_live_app()
        app._dispatch(AppEvent.FREEZE_TOGGLE)
        self.assertTrue(app._cache_dirty)

    def test_cache_clean_after_render(self):
        app, *_ = make_live_app()
        app._dispatch(AppEvent.FREEZE_TOGGLE)
        app._render_for_current_state(app._frozen_raw_frame)
        self.assertFalse(app._cache_dirty)
        self.assertIsNotNone(app._cached_display_frame)

    def test_cache_reused(self):
        app, *_ = make_live_app()
        app._dispatch(AppEvent.FREEZE_TOGGLE)
        f1 = app._render_for_current_state(app._frozen_raw_frame)
        f2 = app._render_for_current_state(app._frozen_raw_frame)
        self.assertIs(f1, f2)

    def test_zoom_invalidates_cache(self):
        app, *_ = make_live_app()
        app._dispatch(AppEvent.FREEZE_TOGGLE)
        app._render_for_current_state(app._frozen_raw_frame)
        self.assertFalse(app._cache_dirty)
        app._dispatch(AppEvent.ZOOM_IN)
        self.assertTrue(app._cache_dirty)

    def test_filter_invalidates_cache(self):
        app, *_ = make_live_app()
        app._dispatch(AppEvent.FREEZE_TOGGLE)
        app._render_for_current_state(app._frozen_raw_frame)
        app._dispatch(AppEvent.FILTER_NEXT)
        self.assertTrue(app._cache_dirty)

    # ----- capture ----------
    def test_capture_from_live_saves(self):
        app, _, _, saver = make_live_app()
        app._dispatch(AppEvent.CAPTURE_IMAGE)
        self.assertTrue(app._state_machine.is_in(AppState.CAPTURE_FLASH))
        self.assertEqual(len(saver.saved_frames), 1)

    def test_capture_save_failure_does_not_crash(self):
        app, *_ = make_live_app()
        def boom(*a, **kw): raise IOError("disk full")
        app._image_saver.save = boom
        app._dispatch(AppEvent.CAPTURE_IMAGE)
        self.assertTrue(app._state_machine.is_in(AppState.CAPTURE_FLASH))

    def test_flash_timeout_returns_to_live(self):
        app, *_ = make_live_app()
        now = [0.0]
        with patch(
            "digital_magnifier.core.app_controller.time.monotonic",
            side_effect=lambda: now[0],
        ):
            app._dispatch(AppEvent.CAPTURE_IMAGE)
            self.assertTrue(app._state_machine.is_in(AppState.CAPTURE_FLASH))
            now[0] += app._capture_flash_ms / 2 / 1000
            app._check_timed_transitions()
            self.assertTrue(app._state_machine.is_in(AppState.CAPTURE_FLASH))
            now[0] += app._capture_flash_ms / 1000
            app._check_timed_transitions()
            self.assertTrue(app._state_machine.is_in(AppState.LIVE_VIEW))

    # ----- modes ----------
    def test_gallery_open_and_back(self):
        app, *_ = make_live_app()
        app._dispatch(AppEvent.GALLERY_OPEN)
        self.assertTrue(app._state_machine.is_in(AppState.GALLERY_VIEW))
        app._dispatch(AppEvent.BACK)
        self.assertTrue(app._state_machine.is_in(AppState.LIVE_VIEW))

    def test_menu_via_power_short(self):
        app, *_ = make_live_app()
        app._dispatch(AppEvent.POWER_SHORT_PRESS)
        self.assertTrue(app._state_machine.is_in(AppState.MENU_VIEW))
        app._dispatch(AppEvent.POWER_SHORT_PRESS)
        self.assertTrue(app._state_machine.is_in(AppState.LIVE_VIEW))

    def test_placeholder_frame_for_gallery(self):
        app, *_ = make_live_app()
        app._state_machine.force_transition(AppState.GALLERY_VIEW)
        frame = app._acquire_frame_for_current_state()
        self.assertEqual(frame.shape, (240, 320, 3))

    # ----- shutdown ----------
    def test_long_press_shuts_down(self):
        app, *_ = make_live_app()
        app._dispatch(AppEvent.POWER_LONG_PRESS)
        self.assertTrue(app._state_machine.is_in(AppState.SHUTDOWN))

    def test_quit_shuts_down(self):
        app, *_ = make_live_app()
        app._dispatch(AppEvent.QUIT)
        self.assertTrue(app._state_machine.is_in(AppState.SHUTDOWN))


class MainLoopVerification(unittest.TestCase):
    """Tests that exercise the main loop (cv2 stubbed for headless)."""

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

    def test_loop_recovers_from_intermittent_camera_failure(self):
        """A few failed frames in a row do not shut the device down."""
        cfg = make_config()
        cam = FakeCamera()
        cam.fail_next = 3  # fewer than MAX_CONSECUTIVE_FAILURES
        # Enough idle frames for the failures to pass, then quit.
        ctrl = FakeControls([AppEvent.NONE] * 10 + [AppEvent.QUIT])
        app = MagnifierApp(cam, ctrl, FakeImageSaver(), cfg)
        app.run()
        # Reached SHUTDOWN via QUIT, NOT via excessive failures.
        self.assertTrue(app._state_machine.is_in(AppState.SHUTDOWN))
        self.assertLess(app._consecutive_failures, MagnifierApp.MAX_CONSECUTIVE_FAILURES)

    def test_excessive_failures_shuts_down(self):
        cfg = make_config()
        cam = FakeCamera()
        cam.fail_next = MagnifierApp.MAX_CONSECUTIVE_FAILURES + 5
        ctrl = FakeControls()
        app = MagnifierApp(cam, ctrl, FakeImageSaver(), cfg)
        app._state_machine.handle(AppEvent.STARTUP_COMPLETE)
        app._run_loop()
        self.assertTrue(app._state_machine.is_in(AppState.SHUTDOWN))

    def test_keyboard_interrupt_shuts_down(self):
        cfg = make_config()
        ctrl = FakeControls()
        ctrl.poll = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
        app = MagnifierApp(FakeCamera(), ctrl, FakeImageSaver(), cfg)
        app._state_machine.handle(AppEvent.STARTUP_COMPLETE)
        app._run_loop()
        self.assertTrue(app._state_machine.is_in(AppState.SHUTDOWN))

    def test_run_starts_and_stops_hals(self):
        cfg = make_config()
        cam = FakeCamera()
        ctrl = FakeControls([AppEvent.QUIT])
        app = MagnifierApp(cam, ctrl, FakeImageSaver(), cfg)
        app.run()
        self.assertEqual(cam.start_calls, 1)
        self.assertEqual(cam.stop_calls, 1)
        self.assertEqual(ctrl.start_calls, 1)
        self.assertEqual(ctrl.stop_calls, 1)
        self.assertTrue(app._state_machine.is_in(AppState.SHUTDOWN))

    def test_run_processes_scripted_session(self):
        cfg = make_config()
        saver = FakeImageSaver()
        ctrl = FakeControls([
            AppEvent.ZOOM_IN,
            AppEvent.ZOOM_IN,
            AppEvent.FREEZE_TOGGLE,
            AppEvent.CAPTURE_IMAGE,
            AppEvent.QUIT,
        ])
        app = MagnifierApp(FakeCamera(), ctrl, saver, cfg)
        app.run()
        self.assertEqual(app._zoom, 3.0)
        self.assertEqual(len(saver.saved_frames), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)