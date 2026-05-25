"""Stdlib-only verification of state_machine.py.

This is NOT the shipped test file (that's tests/unit/test_state_machine.py
which uses pytest). This script just exercises the same assertions
using plain assert + unittest's assertLogs, so I can confirm the state
machine behaves correctly in an environment where pytest isn't
installed.
"""

import logging
import sys
import unittest

sys.path.insert(0, str((__import__("pathlib").Path(__file__).resolve().parents[2] / "src")))

from digital_magnifier.core.events import AppEvent
from digital_magnifier.core.state_machine import (
    AppState,
    IN_STATE_EVENTS,
    StateMachine,
    TRANSITION_TABLE,
    Transition,
)


def sm() -> StateMachine:
    return StateMachine(initial_state=AppState.LIVE_VIEW, log_transitions=False)


class StateMachineVerification(unittest.TestCase):
    # --- lifecycle ---
    def test_default_initial_state_is_startup(self):
        m = StateMachine(log_transitions=False)
        self.assertEqual(m.current_state, AppState.STARTUP)
        self.assertIsNone(m.previous_state)

    def test_startup_completes_to_live_view(self):
        m = StateMachine(initial_state=AppState.STARTUP, log_transitions=False)
        self.assertEqual(m.handle(AppEvent.STARTUP_COMPLETE), AppState.LIVE_VIEW)
        self.assertEqual(m.current_state, AppState.LIVE_VIEW)
        self.assertEqual(m.previous_state, AppState.STARTUP)

    def test_previous_state_tracked(self):
        m = sm()
        m.handle(AppEvent.FREEZE_TOGGLE)
        self.assertEqual(m.previous_state, AppState.LIVE_VIEW)
        self.assertEqual(m.current_state, AppState.FROZEN_VIEW)

    # --- freeze toggle ---
    def test_live_to_frozen(self):
        self.assertEqual(sm().handle(AppEvent.FREEZE_TOGGLE), AppState.FROZEN_VIEW)

    def test_frozen_back_to_live(self):
        m = sm()
        m.handle(AppEvent.FREEZE_TOGGLE)
        self.assertEqual(m.handle(AppEvent.FREEZE_TOGGLE), AppState.LIVE_VIEW)

    # --- capture flow ---
    def test_capture_from_live(self):
        self.assertEqual(sm().handle(AppEvent.CAPTURE_IMAGE), AppState.CAPTURE_FLASH)

    def test_capture_from_frozen(self):
        m = sm()
        m.handle(AppEvent.FREEZE_TOGGLE)
        self.assertEqual(m.handle(AppEvent.CAPTURE_IMAGE), AppState.CAPTURE_FLASH)

    def test_flash_ignores_user_events(self):
        m = sm()
        m.handle(AppEvent.CAPTURE_IMAGE)
        for ev in (AppEvent.ZOOM_IN, AppEvent.PAN_UP, AppEvent.FREEZE_TOGGLE):
            self.assertIsNone(m.handle(ev))
        self.assertEqual(m.current_state, AppState.CAPTURE_FLASH)

    def test_flash_exits_via_force_transition(self):
        m = sm()
        m.handle(AppEvent.CAPTURE_IMAGE)
        m.force_transition(AppState.LIVE_VIEW, reason="flash timeout")
        self.assertEqual(m.current_state, AppState.LIVE_VIEW)
        self.assertEqual(m.previous_state, AppState.CAPTURE_FLASH)

    # --- mode switches ---
    def test_gallery_toggle(self):
        m = sm()
        self.assertEqual(m.handle(AppEvent.GALLERY_OPEN), AppState.GALLERY_VIEW)
        self.assertEqual(m.handle(AppEvent.GALLERY_OPEN), AppState.LIVE_VIEW)

    def test_gallery_back(self):
        m = sm()
        m.handle(AppEvent.GALLERY_OPEN)
        self.assertEqual(m.handle(AppEvent.BACK), AppState.LIVE_VIEW)

    def test_menu_toggle(self):
        m = sm()
        self.assertEqual(m.handle(AppEvent.POWER_SHORT_PRESS), AppState.MENU_VIEW)
        self.assertEqual(m.handle(AppEvent.POWER_SHORT_PRESS), AppState.LIVE_VIEW)

    def test_menu_back(self):
        m = sm()
        m.handle(AppEvent.POWER_SHORT_PRESS)
        self.assertEqual(m.handle(AppEvent.BACK), AppState.LIVE_VIEW)

    # --- global shutdown ---
    def test_long_press_shuts_down_from_any_state(self):
        for from_state in (AppState.LIVE_VIEW, AppState.FROZEN_VIEW,
                           AppState.CAPTURE_FLASH, AppState.GALLERY_VIEW,
                           AppState.MENU_VIEW):
            m = StateMachine(initial_state=from_state, log_transitions=False)
            self.assertEqual(m.handle(AppEvent.POWER_LONG_PRESS), AppState.SHUTDOWN,
                             f"from {from_state}")

    def test_quit_also_shuts_down(self):
        self.assertEqual(sm().handle(AppEvent.QUIT), AppState.SHUTDOWN)

    # --- in-state events ---
    def test_in_state_events_in_live_view(self):
        m = sm()
        for ev in (AppEvent.ZOOM_IN, AppEvent.ZOOM_OUT, AppEvent.PAN_UP,
                   AppEvent.PAN_DOWN, AppEvent.PAN_LEFT, AppEvent.PAN_RIGHT,
                   AppEvent.FILTER_NEXT, AppEvent.RESET_VIEW):
            self.assertIsNone(m.handle(ev))
            self.assertEqual(m.current_state, AppState.LIVE_VIEW)

    def test_in_state_events_work_in_frozen_view_too(self):
        m = sm()
        m.handle(AppEvent.FREEZE_TOGGLE)
        for ev in (AppEvent.ZOOM_IN, AppEvent.FILTER_NEXT, AppEvent.RESET_VIEW):
            self.assertIsNone(m.handle(ev))
        self.assertEqual(m.current_state, AppState.FROZEN_VIEW)

    # --- invalid events ---
    def test_zoom_in_gallery_logs_warning(self):
        m = StateMachine(initial_state=AppState.GALLERY_VIEW, log_transitions=False)
        with self.assertLogs("digital_magnifier.core.state_machine", level="WARNING") as cm:
            self.assertIsNone(m.handle(AppEvent.ZOOM_IN))
        self.assertTrue(any("Ignored event" in msg for msg in cm.output))
        self.assertEqual(m.current_state, AppState.GALLERY_VIEW)

    def test_none_event_is_silent(self):
        m = sm()
        # assertNoLogs is 3.10+; use a manual handler instead
        logger = logging.getLogger("digital_magnifier.core.state_machine")
        records = []
        h = logging.Handler()
        h.emit = records.append
        h.setLevel(logging.WARNING)
        logger.addHandler(h)
        try:
            self.assertIsNone(m.handle(AppEvent.NONE))
        finally:
            logger.removeHandler(h)
        self.assertEqual(records, [])
        self.assertEqual(m.current_state, AppState.LIVE_VIEW)

    # --- queries ---
    def test_is_in(self):
        m = sm()
        self.assertTrue(m.is_in(AppState.LIVE_VIEW))
        self.assertFalse(m.is_in(AppState.FROZEN_VIEW))

    def test_can_handle(self):
        m = sm()
        self.assertTrue(m.can_handle(AppEvent.FREEZE_TOGGLE))
        self.assertTrue(m.can_handle(AppEvent.ZOOM_IN))
        self.assertTrue(m.can_handle(AppEvent.POWER_LONG_PRESS))
        self.assertTrue(m.can_handle(AppEvent.NONE))
        g = StateMachine(initial_state=AppState.GALLERY_VIEW, log_transitions=False)
        self.assertFalse(g.can_handle(AppEvent.FREEZE_TOGGLE))

    # --- history ---
    def test_history_records_transitions(self):
        m = sm()
        m.handle(AppEvent.FREEZE_TOGGLE)
        m.handle(AppEvent.CAPTURE_IMAGE)
        h = m.history
        self.assertEqual(len(h), 2)
        self.assertEqual(h[0], Transition(
            AppState.LIVE_VIEW, AppEvent.FREEZE_TOGGLE, AppState.FROZEN_VIEW))
        self.assertEqual(h[1], Transition(
            AppState.FROZEN_VIEW, AppEvent.CAPTURE_IMAGE, AppState.CAPTURE_FLASH))

    def test_history_excludes_in_state_events(self):
        m = sm()
        m.handle(AppEvent.ZOOM_IN)
        m.handle(AppEvent.PAN_LEFT)
        self.assertEqual(m.history, [])

    def test_history_excludes_invalid_events(self):
        m = StateMachine(initial_state=AppState.GALLERY_VIEW, log_transitions=False)
        m.handle(AppEvent.FREEZE_TOGGLE)
        self.assertEqual(m.history, [])

    def test_history_records_forced_transitions(self):
        m = sm()
        m.force_transition(AppState.MENU_VIEW, reason="test")
        self.assertEqual(len(m.history), 1)
        self.assertEqual(m.history[0].event, AppEvent.NONE)

    def test_history_is_a_copy(self):
        m = sm()
        m.handle(AppEvent.FREEZE_TOGGLE)
        h = m.history
        h.clear()
        self.assertEqual(len(m.history), 1)

    def test_reset_clears_history(self):
        m = sm()
        m.handle(AppEvent.FREEZE_TOGGLE)
        m.reset(to_state=AppState.LIVE_VIEW)
        self.assertEqual(m.history, [])
        self.assertEqual(m.current_state, AppState.LIVE_VIEW)
        self.assertIsNone(m.previous_state)

    # --- table integrity ---
    def test_every_state_has_in_state_entry(self):
        for state in AppState:
            self.assertIn(state, IN_STATE_EVENTS, f"{state} missing")

    def test_no_transition_lands_in_unknown_state(self):
        for target in TRANSITION_TABLE.values():
            self.assertIsInstance(target, AppState)

    def test_no_transition_uses_none_event(self):
        for (_, event) in TRANSITION_TABLE.keys():
            self.assertNotEqual(event, AppEvent.NONE)


if __name__ == "__main__":
    unittest.main(verbosity=2)