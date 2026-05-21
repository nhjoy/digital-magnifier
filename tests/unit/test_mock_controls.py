"""Unit tests for the controls HAL and MockControls.

These tests do not require OpenCV: ``MockControls`` accepts an
injected ``key_reader`` callable, and the tests feed it canned
input. ``cv2`` is lazy-imported inside the default reader, so it
need not be installed for the test process.
"""

from __future__ import annotations

import pytest

from digital_magnifier.core.events import AppEvent
from digital_magnifier.hal.controls_base import ControlsHAL
from digital_magnifier.hal.mock_controls import MockControls


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
class FakeKeyReader:
    """Returns each value in ``queue`` once, then -1 forever.

    Lets a test simulate a sequence of frames: real key codes for
    frames where a key was pressed, -1 for idle frames.
    """

    def __init__(self, queue) -> None:
        self._queue = list(queue)

    def __call__(self) -> int:
        if not self._queue:
            return -1
        return self._queue.pop(0)


@pytest.fixture
def minimal_config():
    """A small but complete mock_keyboard_map for most tests."""
    return {
        "mock_keyboard_map": {
            "space": "FREEZE_TOGGLE",
            "c": "CAPTURE_IMAGE",
            "f": "FILTER_NEXT",
            "p": "POWER_SHORT_PRESS",
            "P": "POWER_LONG_PRESS",
            "q": "QUIT",
        }
    }


# --------------------------------------------------------------------------- #
# Key -> event mapping
# --------------------------------------------------------------------------- #
class TestKeyMapping:
    def test_known_key_returns_correct_event(self, minimal_config):
        controls = MockControls(minimal_config, key_reader=FakeKeyReader([ord("c")]))
        assert controls.poll() == AppEvent.CAPTURE_IMAGE

    def test_unknown_key_returns_none(self, minimal_config):
        controls = MockControls(minimal_config, key_reader=FakeKeyReader([ord("z")]))
        assert controls.poll() == AppEvent.NONE

    def test_no_key_returns_none(self, minimal_config):
        # Empty queue -> reader returns -1 -> no key
        controls = MockControls(minimal_config, key_reader=FakeKeyReader([]))
        assert controls.poll() == AppEvent.NONE

    def test_negative_values_treated_as_no_key(self, minimal_config):
        for sentinel in (-1, -2, -100):
            controls = MockControls(
                minimal_config, key_reader=FakeKeyReader([sentinel])
            )
            assert controls.poll() == AppEvent.NONE

    def test_space_alias_resolves(self, minimal_config):
        controls = MockControls(minimal_config, key_reader=FakeKeyReader([ord(" ")]))
        assert controls.poll() == AppEvent.FREEZE_TOGGLE

    def test_short_and_long_press_are_separate_keys(self, minimal_config):
        """'p' and 'P' must produce different events (case sensitive)."""
        short = MockControls(minimal_config, key_reader=FakeKeyReader([ord("p")]))
        long_ = MockControls(minimal_config, key_reader=FakeKeyReader([ord("P")]))
        assert short.poll() == AppEvent.POWER_SHORT_PRESS
        assert long_.poll() == AppEvent.POWER_LONG_PRESS

    def test_high_byte_masked(self, minimal_config):
        """cv2 may return key codes with high bytes set; we mask to ASCII."""
        # 0x100 | ord('c') = 256 + 99 = 355; & 0xFF -> 99 -> 'c'
        controls = MockControls(
            minimal_config, key_reader=FakeKeyReader([0x100 | ord("c")])
        )
        assert controls.poll() == AppEvent.CAPTURE_IMAGE


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
class TestConfigValidation:
    def test_empty_config_warns_and_returns_none(self, caplog):
        with caplog.at_level("WARNING"):
            controls = MockControls({}, key_reader=FakeKeyReader([ord("c")]))
        assert any(
            "no 'mock_keyboard_map'" in r.message for r in caplog.records
        )
        assert controls.poll() == AppEvent.NONE

    def test_unknown_event_name_is_skipped(self, caplog):
        config = {"mock_keyboard_map": {"x": "NOT_A_REAL_EVENT"}}
        with caplog.at_level("WARNING"):
            controls = MockControls(
                config, key_reader=FakeKeyReader([ord("x")])
            )
        assert controls.poll() == AppEvent.NONE
        assert any("unknown AppEvent" in r.message for r in caplog.records)

    def test_multi_char_key_is_skipped(self, caplog):
        config = {"mock_keyboard_map": {"abc": "ZOOM_IN"}}
        with caplog.at_level("WARNING"):
            controls = MockControls(config, key_reader=FakeKeyReader([]))
        assert any(
            "single character or known alias" in r.message
            for r in caplog.records
        )

    def test_non_string_key_is_skipped(self, caplog):
        config = {"mock_keyboard_map": {42: "ZOOM_IN"}}  # int key
        with caplog.at_level("WARNING"):
            controls = MockControls(config, key_reader=FakeKeyReader([]))
        assert any(
            "key is not a string" in r.message for r in caplog.records
        )

    def test_non_string_event_is_skipped(self, caplog):
        config = {"mock_keyboard_map": {"x": ["not", "a", "string"]}}
        with caplog.at_level("WARNING"):
            controls = MockControls(config, key_reader=FakeKeyReader([]))
        assert any(
            "event must be a string" in r.message for r in caplog.records
        )

    def test_duplicate_key_code_warns(self, caplog):
        # Both 'space' alias and ' ' resolve to ord(' '), so the
        # second entry overrides the first and we expect a warning.
        config = {
            "mock_keyboard_map": {
                "space": "FREEZE_TOGGLE",
                " ": "CAPTURE_IMAGE",
            }
        }
        with caplog.at_level("WARNING"):
            controls = MockControls(config, key_reader=FakeKeyReader([ord(" ")]))
        assert any("re-mapped" in r.message for r in caplog.records)
        # The later entry wins.
        assert controls.poll() == AppEvent.CAPTURE_IMAGE


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
class TestLifecycle:
    def test_start_and_stop_are_noops_by_default(self, minimal_config):
        controls = MockControls(minimal_config)
        controls.start()  # must not raise
        controls.stop()   # must not raise

    def test_context_manager_starts_and_stops(self, minimal_config):
        with MockControls(minimal_config) as controls:
            assert isinstance(controls, ControlsHAL)
            assert controls.poll() == AppEvent.NONE  # no reader queue

    def test_context_manager_calls_stop_on_exception(self, minimal_config):
        calls: list[str] = []

        class TrackingMock(MockControls):
            def stop(self) -> None:
                calls.append("stop")

        with pytest.raises(RuntimeError, match="boom"):
            with TrackingMock(minimal_config):
                raise RuntimeError("boom")

        assert calls == ["stop"]


# --------------------------------------------------------------------------- #
# Sequential polling (simulates the main loop)
# --------------------------------------------------------------------------- #
class TestSequentialPolling:
    def test_event_sequence(self, minimal_config):
        """Simulate six frames worth of input, including idle frames."""
        keys = [ord("c"), -1, ord("p"), ord(" "), -1, ord("q")]
        controls = MockControls(
            minimal_config, key_reader=FakeKeyReader(keys)
        )
        results = [controls.poll() for _ in keys]
        assert results == [
            AppEvent.CAPTURE_IMAGE,
            AppEvent.NONE,
            AppEvent.POWER_SHORT_PRESS,
            AppEvent.FREEZE_TOGGLE,
            AppEvent.NONE,
            AppEvent.QUIT,
        ]


# --------------------------------------------------------------------------- #
# Real config file integration (just enough to confirm it parses
# and produces the events we want from the six-button layout)
# --------------------------------------------------------------------------- #
class TestRealConfig:
    """Read the shipped hardware_pins.yaml and exercise every binding."""

    @pytest.fixture
    def real_config(self):
        import pathlib
        import yaml

        # Locate config relative to this test file: tests/unit -> ../../
        here = pathlib.Path(__file__).resolve()
        config_path = here.parents[2] / "config" / "hardware_pins.yaml"
        with config_path.open() as f:
            return yaml.safe_load(f)

    @pytest.mark.parametrize(
        "key_char, expected_event",
        [
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
        ],
    )
    def test_every_documented_binding_works(
        self, real_config, key_char, expected_event
    ):
        controls = MockControls(
            real_config, key_reader=FakeKeyReader([ord(key_char)])
        )
        assert controls.poll() == expected_event, (
            f"Key {key_char!r} should produce {expected_event.name}"
        )