"""Stdlib-only verification of the controls HAL.

Mirrors the assertions in tests/unit/test_mock_controls.py using
plain unittest. The pytest version is what ships; this script is
just to confirm correctness in environments where pytest isn't
installable.
"""

from __future__ import annotations

import logging
import pathlib
import sys
import unittest

sys.path.insert(0, str((__import__("pathlib").Path(__file__).resolve().parents[2] / "src")))

import yaml

from digital_magnifier.core.events import AppEvent
from digital_magnifier.hal.controls_base import ControlsHAL
from digital_magnifier.hal.mock_controls import MockControls


class FakeKeyReader:
    def __init__(self, queue):
        self._queue = list(queue)

    def __call__(self) -> int:
        if not self._queue:
            return -1
        return self._queue.pop(0)


MINIMAL_CONFIG = {
    "mock_keyboard_map": {
        "space": "FREEZE_TOGGLE",
        "c": "CAPTURE_IMAGE",
        "f": "FILTER_NEXT",
        "p": "POWER_SHORT_PRESS",
        "P": "POWER_LONG_PRESS",
        "q": "QUIT",
    }
}


class ControlsVerification(unittest.TestCase):

    # ----- key mapping -------------------------------------------
    def test_known_key_returns_correct_event(self):
        c = MockControls(MINIMAL_CONFIG, key_reader=FakeKeyReader([ord("c")]))
        self.assertEqual(c.poll(), AppEvent.CAPTURE_IMAGE)

    def test_unknown_key_returns_none(self):
        c = MockControls(MINIMAL_CONFIG, key_reader=FakeKeyReader([ord("z")]))
        self.assertEqual(c.poll(), AppEvent.NONE)

    def test_no_key_returns_none(self):
        c = MockControls(MINIMAL_CONFIG, key_reader=FakeKeyReader([]))
        self.assertEqual(c.poll(), AppEvent.NONE)

    def test_negative_values_treated_as_no_key(self):
        for sentinel in (-1, -2, -100):
            c = MockControls(MINIMAL_CONFIG, key_reader=FakeKeyReader([sentinel]))
            self.assertEqual(c.poll(), AppEvent.NONE, f"sentinel {sentinel}")

    def test_space_alias_resolves(self):
        c = MockControls(MINIMAL_CONFIG, key_reader=FakeKeyReader([ord(" ")]))
        self.assertEqual(c.poll(), AppEvent.FREEZE_TOGGLE)

    def test_short_and_long_press_are_separate_keys(self):
        s = MockControls(MINIMAL_CONFIG, key_reader=FakeKeyReader([ord("p")]))
        l = MockControls(MINIMAL_CONFIG, key_reader=FakeKeyReader([ord("P")]))
        self.assertEqual(s.poll(), AppEvent.POWER_SHORT_PRESS)
        self.assertEqual(l.poll(), AppEvent.POWER_LONG_PRESS)

    def test_high_byte_masked(self):
        c = MockControls(MINIMAL_CONFIG, key_reader=FakeKeyReader([0x100 | ord("c")]))
        self.assertEqual(c.poll(), AppEvent.CAPTURE_IMAGE)

    # ----- config validation ------------------------------------
    def _capture_warnings(self, fn):
        records = []
        logger = logging.getLogger("digital_magnifier.hal.mock_controls")
        h = logging.Handler()
        h.emit = lambda r: records.append(r)
        h.setLevel(logging.WARNING)
        logger.addHandler(h)
        try:
            result = fn()
        finally:
            logger.removeHandler(h)
        return result, records

    def test_empty_config_warns_and_returns_none(self):
        def build():
            return MockControls({}, key_reader=FakeKeyReader([ord("c")]))
        c, records = self._capture_warnings(build)
        self.assertTrue(any("no 'mock_keyboard_map'" in r.getMessage() for r in records))
        self.assertEqual(c.poll(), AppEvent.NONE)

    def test_unknown_event_name_is_skipped(self):
        def build():
            return MockControls(
                {"mock_keyboard_map": {"x": "NOT_A_REAL_EVENT"}},
                key_reader=FakeKeyReader([ord("x")]),
            )
        c, records = self._capture_warnings(build)
        self.assertEqual(c.poll(), AppEvent.NONE)
        self.assertTrue(any("unknown AppEvent" in r.getMessage() for r in records))

    def test_multi_char_key_is_skipped(self):
        def build():
            return MockControls(
                {"mock_keyboard_map": {"abc": "ZOOM_IN"}},
                key_reader=FakeKeyReader([]),
            )
        _, records = self._capture_warnings(build)
        self.assertTrue(any("single character or known alias" in r.getMessage() for r in records))

    def test_non_string_key_is_skipped(self):
        def build():
            return MockControls(
                {"mock_keyboard_map": {42: "ZOOM_IN"}},
                key_reader=FakeKeyReader([]),
            )
        _, records = self._capture_warnings(build)
        self.assertTrue(any("key is not a string" in r.getMessage() for r in records))

    def test_non_string_event_is_skipped(self):
        def build():
            return MockControls(
                {"mock_keyboard_map": {"x": ["not", "string"]}},
                key_reader=FakeKeyReader([]),
            )
        _, records = self._capture_warnings(build)
        self.assertTrue(any("event must be a string" in r.getMessage() for r in records))

    def test_duplicate_key_code_warns(self):
        def build():
            return MockControls(
                {"mock_keyboard_map": {"space": "FREEZE_TOGGLE", " ": "CAPTURE_IMAGE"}},
                key_reader=FakeKeyReader([ord(" ")]),
            )
        c, records = self._capture_warnings(build)
        self.assertTrue(any("re-mapped" in r.getMessage() for r in records))
        self.assertEqual(c.poll(), AppEvent.CAPTURE_IMAGE)

    # ----- lifecycle --------------------------------------------
    def test_start_and_stop_are_noops_by_default(self):
        c = MockControls(MINIMAL_CONFIG)
        c.start()
        c.stop()

    def test_context_manager_starts_and_stops(self):
        with MockControls(MINIMAL_CONFIG) as c:
            self.assertIsInstance(c, ControlsHAL)
            self.assertEqual(c.poll(), AppEvent.NONE)

    def test_context_manager_calls_stop_on_exception(self):
        calls = []

        class Tracking(MockControls):
            def stop(self):
                calls.append("stop")

        with self.assertRaises(RuntimeError):
            with Tracking(MINIMAL_CONFIG):
                raise RuntimeError("boom")
        self.assertEqual(calls, ["stop"])

    # ----- sequential polling -----------------------------------
    def test_event_sequence(self):
        keys = [ord("c"), -1, ord("p"), ord(" "), -1, ord("q")]
        c = MockControls(MINIMAL_CONFIG, key_reader=FakeKeyReader(keys))
        results = [c.poll() for _ in keys]
        self.assertEqual(results, [
            AppEvent.CAPTURE_IMAGE,
            AppEvent.NONE,
            AppEvent.POWER_SHORT_PRESS,
            AppEvent.FREEZE_TOGGLE,
            AppEvent.NONE,
            AppEvent.QUIT,
        ])

    # ----- real config integration ------------------------------
    def test_every_real_config_binding(self):
        here = pathlib.Path(__file__).resolve().parent
        config_path = here.parents[1] / "config" / "hardware_pins.yaml"
        with config_path.open() as f:
            real_config = yaml.safe_load(f)

        bindings = [
            (" ", AppEvent.FREEZE_TOGGLE),
            ("c", AppEvent.CAPTURE_IMAGE),
            ("f", AppEvent.FILTER_NEXT),
            ("g", AppEvent.GALLERY_OPEN),
            ("r", AppEvent.RESET_VIEW),
            ("p", AppEvent.POWER_SHORT_PRESS),
            ("P", AppEvent.POWER_LONG_PRESS),
            ("w", AppEvent.PAN_UP),
            ("s", AppEvent.PAN_DOWN),
            ("a", AppEvent.PAN_LEFT),
            ("d", AppEvent.PAN_RIGHT),
            ("x", AppEvent.RESET_VIEW),
            ("+", AppEvent.ZOOM_IN),
            ("=", AppEvent.ZOOM_IN),
            ("-", AppEvent.ZOOM_OUT),
            ("_", AppEvent.ZOOM_OUT),
            ("[", AppEvent.BACK),
            ("q", AppEvent.QUIT),
        ]
        for key_char, expected in bindings:
            c = MockControls(real_config, key_reader=FakeKeyReader([ord(key_char)]))
            self.assertEqual(c.poll(), expected, f"{key_char!r} -> {expected.name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)