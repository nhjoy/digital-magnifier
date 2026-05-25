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


# =========================================================================
# GPIOControls tests
# =========================================================================
#
# GPIOControls reads buttons / nav switch via a TCA6416A I/O expander and
# (optionally) a zoom pot via an MCP3221 ADC. These tests use an in-memory
# FakeBus and a FakeClock to drive the chip drivers deterministically.

from collections import deque
from typing import Optional

from digital_magnifier.hal.i2c_devices import (
    I2CError,
    MCP3221,
    TCA6416A,
)
from digital_magnifier.hal.mock_controls import GPIOControls


class _FakeBus:
    """Minimal in-memory I2C bus matching the I2CBusLike protocol."""

    def __init__(self) -> None:
        self.registers: dict[tuple[int, int], int] = {}
        self.raw_read_queue: list[list[int]] = []
        self.write_log: list[tuple[int, int, int]] = []
        self.fail_on_address: Optional[int] = None
        self.fail_on_raw_read: bool = False

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        if address == self.fail_on_address:
            raise I2CError("injected write failure")
        self.registers[(address, register)] = value
        self.write_log.append((address, register, value))

    def read_byte_data(self, address: int, register: int) -> int:
        if address == self.fail_on_address:
            raise I2CError("injected read failure")
        return self.registers.get((address, register), 0xFF)

    def read_raw_bytes(self, address: int, length: int) -> list[int]:
        if self.fail_on_raw_read or address == self.fail_on_address:
            raise I2CError("injected raw-read failure")
        if self.raw_read_queue:
            return self.raw_read_queue.pop(0)[:length]
        return [0] * length

    def close(self) -> None:
        pass


class _FakeClock:
    """Manually-advanced monotonic clock."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _gpio_config() -> dict:
    """A complete hardware_pins.yaml-shaped config for GPIOControls."""
    return {
        "i2c": {
            "bus": 1,
            "devices": {"io_expander": {"poll_interval_ms": 20}},
        },
        "tca6416a_pins": {
            "config_port0": 0xFF,
            "config_port1": 0xC7,
            "inputs": {
                "nav_left":   "P00",
                "nav_center": "P01",
                "nav_right":  "P02",
                "nav_up":     "P03",
                "nav_down":   "P04",
                "snapshot":   "P05",
                "album":      "P06",
                "clr_btn":    "P07",
                "add1":       "P10",
                "add2":       "P11",
                "add3":       "P12",
            },
            "outputs": {
                "led_blue":  "P13",
                "led_green": "P14",
                "led_red":   "P15",
            },
        },
        "button_events": {
            "album":    "GALLERY_OPEN",
            "clr_btn":  "FILTER_NEXT",
            "add1":     "ZOOM_IN",
            "add2":     "ZOOM_OUT",
        },
        "snapshot_button": {
            "input": "snapshot",
            "short_event": "FREEZE_TOGGLE",
            "long_event": "CAPTURE_IMAGE",
            "long_press_threshold_ms": 3000,
        },
        "nav_events": {
            "nav_up":     "PAN_UP",
            "nav_down":   "PAN_DOWN",
            "nav_left":   "PAN_LEFT",
            "nav_right":  "PAN_RIGHT",
            "nav_center": "RESET_VIEW",
        },
        "nav_repeat": {
            "enabled": True,
            "initial_delay_ms": 400,
            "interval_ms": 100,
        },
        "power_button": {
            "input": "add3",
            "long_press_threshold_ms": 1000,
        },
        "zoom_pot": {"buckets": 16, "invert": False, "debounce_reads": 1},
    }


def _build_gpio_controls(
    *,
    scripted_inputs: Optional[list[int]] = None,
    raw_adc: Optional[list[tuple[int, int]]] = None,
    cfg: Optional[dict] = None,
):
    """Construct a GPIOControls + clock + bus for testing.

    ``scripted_inputs`` is a list of 16-bit input words returned by
    successive ``read_inputs()`` calls (the chip's input register
    snapshots over time). ``raw_adc`` is a list of (high_byte,
    low_byte) pairs returned by successive MCP3221 reads. Probe
    consumption is handled here so callers can think in terms of
    "poll 1 reads X, poll 2 reads Y".
    """
    bus = _FakeBus()
    clock = _FakeClock()
    chip = TCA6416A(
        bus, address=0x20, config_port0=0xFF, config_port1=0xC7,
    )

    adc = None
    if raw_adc is not None:
        adc = MCP3221(bus, address=0x4D)
        # probe() consumes one queue entry at start(); push a dummy.
        bus.raw_read_queue.append([0x00, 0x00])
        for pair in raw_adc:
            bus.raw_read_queue.append(list(pair))

    if scripted_inputs is not None:
        seq = iter(scripted_inputs)

        def reader():
            try:
                return next(seq)
            except StopIteration:
                return 0xFFFF
        chip.read_inputs = reader  # type: ignore[method-assign]

    controls = GPIOControls(cfg or _gpio_config(), chip, adc=adc, clock=clock)
    controls.start()
    return controls, clock, bus


def _drain_pending(controls, max_events: int = 20) -> list[AppEvent]:
    """Pull events one at a time without advancing the clock.

    Subsequent polls hit the rate-limit and don't trigger fresh I2C reads,
    so we drain only what was queued by the previous read.
    """
    out = []
    for _ in range(max_events):
        ev = controls.poll()
        if ev is AppEvent.NONE:
            break
        out.append(ev)
    return out


class TestGPIOControlsButtons:
    def test_first_read_produces_no_spurious_events(self):
        controls, clock, _ = _build_gpio_controls(scripted_inputs=[0xFFFF])
        clock.advance(0.030)
        assert controls.poll() == AppEvent.NONE

    def test_snapshot_tap_fires_freeze_toggle(self):
        # P05 LOW = pressed snapshot; release quickly = FREEZE_TOGGLE
        pressed = 0xFFFF & ~(1 << 5)   # P05 LOW
        controls, clock, _ = _build_gpio_controls(
            scripted_inputs=[0xFFFF, pressed, 0xFFFF],
        )
        clock.advance(0.030)
        controls.poll()                          # anchor
        clock.advance(0.030)
        assert controls.poll() == AppEvent.NONE  # press edge — no event yet
        clock.advance(0.200)                     # release within 3s
        assert controls.poll() == AppEvent.FREEZE_TOGGLE

    def test_snapshot_hold_fires_capture_image(self):
        # Hold snapshot for 3+ seconds = CAPTURE_IMAGE
        pressed = 0xFFFF & ~(1 << 5)
        controls, clock, _ = _build_gpio_controls(
            scripted_inputs=[0xFFFF] + [pressed] * 5,
        )
        clock.advance(0.030)
        controls.poll()                          # anchor
        clock.advance(0.030)
        controls.poll()                          # press edge
        clock.advance(1.5)
        assert controls.poll() == AppEvent.NONE  # still under 3s
        clock.advance(2.0)                       # now 3.5s total
        assert controls.poll() == AppEvent.CAPTURE_IMAGE

    def test_snapshot_hold_then_release_does_not_double_fire(self):
        pressed = 0xFFFF & ~(1 << 5)
        controls, clock, _ = _build_gpio_controls(
            scripted_inputs=[0xFFFF, pressed, pressed, 0xFFFF],
        )
        clock.advance(0.030); controls.poll()    # anchor
        clock.advance(0.030); controls.poll()    # press
        clock.advance(3.5)
        assert controls.poll() == AppEvent.CAPTURE_IMAGE  # long fires
        clock.advance(0.030)
        assert controls.poll() == AppEvent.NONE  # release — no second event

    def test_filter_next_bound_to_clr_btn(self):
        # P07 LOW = pressed clr_btn (which now maps to FILTER_NEXT)
        controls, clock, _ = _build_gpio_controls(
            scripted_inputs=[0xFFFF, 0xFF7F],
        )
        clock.advance(0.030)
        controls.poll()
        clock.advance(0.030)
        assert controls.poll() == AppEvent.FILTER_NEXT

    def test_multiple_simultaneous_presses_are_queued(self):
        # album (P06) AND clr_btn (P07) pressed at the same time
        controls, clock, _ = _build_gpio_controls(
            scripted_inputs=[0xFFFF, 0xFF3F],
        )
        clock.advance(0.030)
        controls.poll()
        clock.advance(0.030)
        events = _drain_pending(controls)
        assert AppEvent.GALLERY_OPEN in events
        assert AppEvent.FILTER_NEXT in events


class TestGPIOControlsNav:
    def test_nav_up_emits_pan_up(self):
        controls, clock, _ = _build_gpio_controls(
            scripted_inputs=[0xFFFF, 0xFFF7],
        )
        clock.advance(0.030)
        controls.poll()
        clock.advance(0.030)
        assert controls.poll() == AppEvent.PAN_UP

    def test_nav_center_emits_reset_view(self):
        # P01 LOW = nav centre
        controls, clock, _ = _build_gpio_controls(
            scripted_inputs=[0xFFFF, 0xFFFD],
        )
        clock.advance(0.030)
        controls.poll()
        clock.advance(0.030)
        assert controls.poll() == AppEvent.RESET_VIEW

    def test_nav_repeat_fires_after_initial_delay_and_interval(self):
        readings = [0xFFFF] + [0xFFF7] * 30   # nav_up held
        controls, clock, _ = _build_gpio_controls(scripted_inputs=readings)

        events: list[AppEvent] = []
        clock.advance(0.030); events.append(controls.poll())   # anchor
        clock.advance(0.030); events.append(controls.poll())   # press edge
        clock.advance(0.300); events.append(controls.poll())   # below threshold
        clock.advance(0.110); events.append(controls.poll())   # first repeat
        clock.advance(0.110); events.append(controls.poll())   # second repeat

        assert events.count(AppEvent.PAN_UP) == 3   # press + 2 repeats


class TestGPIOControlsPowerButton:
    def test_short_press_fires_on_release(self):
        pressed = 0xFFFF & ~(1 << 10)            # P12 LOW = add3 pressed
        controls, clock, _ = _build_gpio_controls(
            scripted_inputs=[0xFFFF, pressed, 0xFFFF],
        )
        clock.advance(0.030); controls.poll()    # anchor
        clock.advance(0.030)
        assert controls.poll() == AppEvent.NONE  # press edge, no event yet
        clock.advance(0.200)
        assert controls.poll() == AppEvent.POWER_SHORT_PRESS

    def test_long_press_fires_at_threshold(self):
        pressed = 0xFFFF & ~(1 << 10)
        controls, clock, _ = _build_gpio_controls(
            scripted_inputs=[0xFFFF] + [pressed] * 4,
        )
        clock.advance(0.030); controls.poll()
        clock.advance(0.030); controls.poll()    # press edge
        clock.advance(0.500)
        assert controls.poll() == AppEvent.NONE  # still under 1000 ms
        clock.advance(0.600)
        assert controls.poll() == AppEvent.POWER_LONG_PRESS

    def test_long_press_then_release_does_not_emit_short(self):
        pressed = 0xFFFF & ~(1 << 10)
        controls, clock, _ = _build_gpio_controls(
            scripted_inputs=[0xFFFF, pressed, pressed, 0xFFFF],
        )
        clock.advance(0.030); controls.poll()
        clock.advance(0.030); controls.poll()
        clock.advance(1.100)
        assert controls.poll() == AppEvent.POWER_LONG_PRESS
        clock.advance(0.030)
        assert controls.poll() == AppEvent.NONE


class TestGPIOControlsZoomPot:
    def test_zoom_in_when_bucket_increases(self):
        # 16 buckets across 0..4095. raw 1024 → bucket 4.
        controls, clock, _ = _build_gpio_controls(
            scripted_inputs=[0xFFFF],
            raw_adc=[(0x00, 0x00), (0x04, 0x00)],
        )
        clock.advance(0.030); assert controls.poll() == AppEvent.NONE
        clock.advance(0.030)
        events = _drain_pending(controls)
        assert events.count(AppEvent.ZOOM_IN) == 4
        assert events.count(AppEvent.ZOOM_OUT) == 0

    def test_zoom_out_when_bucket_decreases(self):
        controls, clock, _ = _build_gpio_controls(
            scripted_inputs=[0xFFFF],
            raw_adc=[(0x08, 0x00), (0x04, 0x00)],   # bucket 8 → 4
        )
        clock.advance(0.030); assert controls.poll() == AppEvent.NONE
        clock.advance(0.030)
        events = _drain_pending(controls)
        assert events.count(AppEvent.ZOOM_OUT) == 4
        assert events.count(AppEvent.ZOOM_IN) == 0

    def test_invert_swaps_direction(self):
        cfg = _gpio_config()
        cfg["zoom_pot"]["invert"] = True
        controls, clock, _ = _build_gpio_controls(
            scripted_inputs=[0xFFFF],
            raw_adc=[(0x00, 0x00), (0x04, 0x00)],
            cfg=cfg,
        )
        clock.advance(0.030); assert controls.poll() == AppEvent.NONE
        clock.advance(0.030)
        events = _drain_pending(controls)
        # Inverted: raw bucket UP = inverted bucket DOWN = ZOOM_OUT
        assert events.count(AppEvent.ZOOM_OUT) == 4
        assert events.count(AppEvent.ZOOM_IN) == 0


class TestGPIOControlsResilience:
    def test_i2c_read_failure_returns_none_and_recovers(self):
        bus = _FakeBus()
        clock = _FakeClock()
        chip = TCA6416A(bus, address=0x20)
        chip.init()

        controls = GPIOControls(_gpio_config(), chip, adc=None, clock=clock)

        # First poll fails -> NONE, warning logged
        bus.fail_on_address = 0x20
        clock.advance(0.030)
        assert controls.poll() == AppEvent.NONE

        # Recovery: clear injection, the next poll anchors successfully.
        bus.fail_on_address = None
        clock.advance(0.030)
        controls.poll()                      # anchor (no events)

        # Now press album — should be detected normally.
        bus.registers[(0x20, 0x00)] = 0xBF   # P06 LOW = album
        bus.registers[(0x20, 0x01)] = 0xFF
        clock.advance(0.030)
        assert controls.poll() == AppEvent.GALLERY_OPEN