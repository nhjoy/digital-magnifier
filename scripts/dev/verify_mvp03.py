"""Sandbox verifier: exercise the MVP 0.3 I2C drivers + GPIOControls.

Uses unittest (stdlib) instead of pytest so it runs in environments
without pytest installed. The pytest-flavoured tests in tests/unit/
cover the same ground for the CM5 venv.
"""

from __future__ import annotations

import sys
import unittest

sys.path.insert(0, str((__import__("pathlib").Path(__file__).resolve().parents[2] / "src")))

from digital_magnifier.core.events import AppEvent
from digital_magnifier.hal.i2c_devices import (
    I2CError,
    MCP3221,
    TCA6416A,
    parse_address,
    pin_name_to_index,
)
from digital_magnifier.hal.mock_controls import GPIOControls


# -------------------------------------------------------------------
# Fake bus
# -------------------------------------------------------------------


class FakeBus:
    def __init__(self):
        self.registers = {}
        self.raw_read_queue = []
        self.write_log = []
        self.read_log = []
        self.raw_read_log = []
        self.fail_on_address = None
        self.fail_on_raw_read = False

    def write_byte_data(self, address, register, value):
        if address == self.fail_on_address:
            raise I2CError(f"injected at 0x{address:02x}")
        self.registers[(address, register)] = value
        self.write_log.append((address, register, value))

    def read_byte_data(self, address, register):
        if address == self.fail_on_address:
            raise I2CError(f"injected at 0x{address:02x}")
        value = self.registers.get((address, register), 0xFF)
        self.read_log.append((address, register, value))
        return value

    def read_raw_bytes(self, address, length):
        if self.fail_on_raw_read or address == self.fail_on_address:
            raise I2CError("injected raw-read failure")
        self.raw_read_log.append((address, length))
        if self.raw_read_queue:
            return self.raw_read_queue.pop(0)[:length]
        return [0] * length

    def close(self):
        pass


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def make_full_config():
    """A minimal complete hardware_pins-style config for tests."""
    return {
        "i2c": {
            "bus": 1,
            "devices": {
                "io_expander": {"poll_interval_ms": 20},
            },
        },
        "tca6416a_pins": {
            "config_port0": 0xFF,
            "config_port1": 0xC7,
            "inputs": {
                "nav_left": "P00",
                "nav_center": "P01",
                "nav_right": "P02",
                "nav_up": "P03",
                "nav_down": "P04",
                "snapshot": "P05",
                "album": "P06",
                "clr_btn": "P07",
                "add1": "P10",
                "add2": "P11",
                "add3": "P12",
            },
            "outputs": {
                "led_blue": "P13",
                "led_green": "P14",
                "led_red": "P15",
            },
        },
        "button_events": {
            "snapshot": "CAPTURE_IMAGE",
            "album": "GALLERY_OPEN",
            "clr_btn": "RESET_VIEW",
            "add1": "FREEZE_TOGGLE",
            "add2": "FILTER_NEXT",
        },
        "nav_events": {
            "nav_up": "PAN_UP",
            "nav_down": "PAN_DOWN",
            "nav_left": "PAN_LEFT",
            "nav_right": "PAN_RIGHT",
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
        "zoom_pot": {
            "buckets": 16,
            "invert": False,
            "debounce_reads": 1,
        },
    }


class FakeClock:
    """Manually-advanced monotonic clock for deterministic timing tests."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


class TestI2CHelpers(unittest.TestCase):
    def test_pin_name_to_index_port0(self):
        for i in range(8):
            self.assertEqual(pin_name_to_index(f"P0{i}"), i)

    def test_pin_name_to_index_port1(self):
        for i in range(8):
            self.assertEqual(pin_name_to_index(f"P1{i}"), 8 + i)

    def test_pin_name_to_index_invalid(self):
        for name in ("P08", "p00", "GPIO5", "", "5"):
            with self.assertRaises(ValueError):
                pin_name_to_index(name)

    def test_parse_address(self):
        self.assertEqual(parse_address(0x20), 0x20)
        self.assertEqual(parse_address("0x4D"), 0x4D)
        self.assertEqual(parse_address("32"), 32)
        with self.assertRaises(ValueError):
            parse_address(0x80)
        with self.assertRaises(ValueError):
            parse_address("bad")


class TestTCA6416A(unittest.TestCase):
    def test_init_writes_in_order(self):
        bus = FakeBus()
        chip = TCA6416A(
            bus, address=0x20,
            config_port0=0xFF, config_port1=0xC7,
            initial_outputs_port0=0xFF, initial_outputs_port1=0xFF,
        )
        chip.init()
        regs = [w[1] for w in bus.write_log]
        self.assertEqual(regs, [0x04, 0x05, 0x02, 0x03, 0x06, 0x07])
        # Last two are direction config
        self.assertEqual(bus.write_log[4], (0x20, 0x06, 0xFF))
        self.assertEqual(bus.write_log[5], (0x20, 0x07, 0xC7))

    def test_read_inputs_combines_ports(self):
        bus = FakeBus()
        chip = TCA6416A(bus, address=0x20)
        bus.registers[(0x20, 0x00)] = 0xAB
        bus.registers[(0x20, 0x01)] = 0xCD
        self.assertEqual(chip.read_inputs(), 0xCDAB)

    def test_write_output_caches(self):
        bus = FakeBus()
        chip = TCA6416A(
            bus, address=0x20,
            initial_outputs_port0=0xFF, initial_outputs_port1=0xFF,
        )
        chip.init()
        bus.write_log.clear()

        # Writing same value as cache → no-op
        chip.write_output_pin(11, True)
        self.assertEqual(bus.write_log, [])

        # Clear P13 (LED on) → one write
        chip.write_output_pin(11, False)
        self.assertEqual(bus.write_log, [(0x20, 0x03, 0xF7)])

        # Clear P14 (subsequent write uses cached 0xF7)
        chip.write_output_pin(12, False)
        self.assertEqual(bus.write_log[-1], (0x20, 0x03, 0xE7))

    def test_probe_failure(self):
        bus = FakeBus()
        bus.fail_on_address = 0x20
        chip = TCA6416A(bus, address=0x20)
        with self.assertRaises(I2CError):
            chip.probe()


class TestMCP3221(unittest.TestCase):
    def test_read_raw_decodes(self):
        bus = FakeBus()
        adc = MCP3221(bus, address=0x4D)
        bus.raw_read_queue.append([0x01, 0x23])
        self.assertEqual(adc.read_raw(), 0x123)

    def test_read_raw_full_scale(self):
        bus = FakeBus()
        adc = MCP3221(bus, address=0x4D)
        bus.raw_read_queue.append([0x0F, 0xFF])
        self.assertEqual(adc.read_raw(), 4095)

    def test_masks_top_nibble(self):
        bus = FakeBus()
        adc = MCP3221(bus, address=0x4D)
        bus.raw_read_queue.append([0xF1, 0x23])
        self.assertEqual(adc.read_raw(), 0x123)

    def test_read_voltage(self):
        bus = FakeBus()
        adc = MCP3221(bus, address=0x4D, vdd_volts=3.3)
        bus.raw_read_queue.append([0x0F, 0xFF])
        self.assertAlmostEqual(adc.read_voltage(), 3.3, places=3)

    def test_probe_failure(self):
        bus = FakeBus()
        bus.fail_on_raw_read = True
        adc = MCP3221(bus, address=0x4D)
        with self.assertRaises(I2CError):
            adc.probe()


class TestGPIOControls(unittest.TestCase):

    def _build(self, fake_inputs=None, raw_adc=None):
        """Make a GPIOControls wired to a FakeBus.

        ``fake_inputs`` is a callable returning the next 16-bit input
        word, or a list of words (one per poll). ``raw_adc`` is a list
        of (high, low) byte pairs to feed the MCP3221.

        Note: GPIOControls.start() calls adc.probe(), which consumes
        one queue entry. We push a dummy entry at the front of the
        queue so the first user-supplied reading is what the first
        actual poll sees.
        """
        bus = FakeBus()
        clock = FakeClock()

        chip = TCA6416A(
            bus, address=0x20,
            config_port0=0xFF, config_port1=0xC7,
        )
        adc = None
        if raw_adc is not None:
            adc = MCP3221(bus, address=0x4D)
            # probe() consumes one queue entry during start()
            bus.raw_read_queue.append([0x00, 0x00])
            for pair in raw_adc:
                bus.raw_read_queue.append(list(pair))

        # Patch the chip's read_inputs to return our scripted values
        if fake_inputs is not None:
            if callable(fake_inputs):
                chip.read_inputs = fake_inputs   # type: ignore[method-assign]
            else:
                seq = iter(fake_inputs)

                def reader():
                    try:
                        return next(seq)
                    except StopIteration:
                        return 0xFFFF
                chip.read_inputs = reader  # type: ignore[method-assign]

        controls = GPIOControls(
            make_full_config(),
            chip,
            adc=adc,
            clock=clock,
        )
        controls.start()
        return controls, clock, bus

    def _drain(self, controls, clock, max_events=10):
        """Drain all queued events without waiting between polls."""
        events = []
        for _ in range(max_events):
            ev = controls.poll()
            if ev is AppEvent.NONE:
                break
            events.append(ev)
        return events

    def test_first_read_no_spurious_events(self):
        controls, clock, _ = self._build(fake_inputs=[0xFFFF])
        clock.advance(0.030)
        ev = controls.poll()
        self.assertEqual(ev, AppEvent.NONE)

    def test_snapshot_button_press(self):
        controls, clock, _ = self._build(
            fake_inputs=[0xFFFF, 0xFFDF]   # P05 released, then pressed
        )
        clock.advance(0.030)
        controls.poll()                    # first read anchors state
        clock.advance(0.030)
        ev = controls.poll()
        self.assertEqual(ev, AppEvent.CAPTURE_IMAGE)

    def test_multiple_simultaneous_presses_queue(self):
        # snapshot (P05) AND album (P06) press at same time
        controls, clock, _ = self._build(
            fake_inputs=[0xFFFF, 0xFF9F]   # P05 and P06 both LOW
        )
        clock.advance(0.030)
        controls.poll()
        clock.advance(0.030)
        events = self._drain(controls, clock)
        self.assertIn(AppEvent.CAPTURE_IMAGE, events)
        self.assertIn(AppEvent.GALLERY_OPEN, events)

    def test_nav_up_emits_pan_up(self):
        controls, clock, _ = self._build(
            fake_inputs=[0xFFFF, 0xFFF7]   # P03 LOW = nav up
        )
        clock.advance(0.030)
        controls.poll()
        clock.advance(0.030)
        ev = controls.poll()
        self.assertEqual(ev, AppEvent.PAN_UP)

    def test_nav_repeat_fires_after_initial_delay(self):
        """Hold nav up: emit once on press, again after the initial delay,
        then once per interval."""
        # Frame the inputs so nav_up stays pressed across multiple polls.
        readings = [0xFFFF] + [0xFFF7] * 30

        controls, clock, _ = self._build(fake_inputs=readings)

        all_events = []

        # First poll: anchor state
        clock.advance(0.030)
        all_events.append(controls.poll())

        # Second poll: press edge — emit PAN_UP, schedule first repeat at +400ms
        clock.advance(0.030)
        all_events.append(controls.poll())

        # Just before the 400ms threshold: no new event
        clock.advance(0.300)
        all_events.append(controls.poll())

        # Cross the threshold: first repeat fires
        clock.advance(0.110)
        all_events.append(controls.poll())

        # After interval (100ms): second repeat
        clock.advance(0.110)
        all_events.append(controls.poll())

        pan_ups = [e for e in all_events if e == AppEvent.PAN_UP]
        self.assertEqual(len(pan_ups), 3)  # press + 2 repeats

    def test_power_short_press(self):
        # add3 = P12 = bit 10 of 16-bit input word.
        # P12 LOW = pressed
        pressed = 0xFFFF & ~(1 << 10)  # = 0xFBFF
        controls, clock, _ = self._build(
            fake_inputs=[0xFFFF, pressed, 0xFFFF]
        )
        clock.advance(0.030)
        controls.poll()                    # anchor
        clock.advance(0.030)
        ev = controls.poll()               # press edge — no event yet
        self.assertEqual(ev, AppEvent.NONE)
        clock.advance(0.200)               # hold for 200ms (< 1000ms)
        ev = controls.poll()               # release edge
        self.assertEqual(ev, AppEvent.POWER_SHORT_PRESS)

    def test_power_long_press_fires_at_threshold(self):
        pressed = 0xFFFF & ~(1 << 10)
        # Press and keep holding
        readings = [0xFFFF, pressed, pressed, pressed, pressed]
        controls, clock, _ = self._build(fake_inputs=readings)

        clock.advance(0.030)
        controls.poll()                    # anchor
        clock.advance(0.030)
        controls.poll()                    # press edge
        clock.advance(0.500)
        ev = controls.poll()               # still under threshold
        self.assertEqual(ev, AppEvent.NONE)
        clock.advance(0.600)               # now over 1000ms
        ev = controls.poll()
        self.assertEqual(ev, AppEvent.POWER_LONG_PRESS)

    def test_power_long_then_release_no_short(self):
        pressed = 0xFFFF & ~(1 << 10)
        readings = [0xFFFF, pressed, pressed, 0xFFFF]
        controls, clock, _ = self._build(fake_inputs=readings)

        clock.advance(0.030)
        controls.poll()                    # anchor
        clock.advance(0.030)
        controls.poll()                    # press
        clock.advance(1.100)
        e1 = controls.poll()               # long fires
        self.assertEqual(e1, AppEvent.POWER_LONG_PRESS)
        clock.advance(0.030)
        e2 = controls.poll()               # release — must NOT fire short
        self.assertEqual(e2, AppEvent.NONE)

    def test_io_failure_returns_none_and_recovers(self):
        bus = FakeBus()
        clock = FakeClock()
        chip = TCA6416A(bus, address=0x20)

        # First read fails, then we clear the injection
        bus.fail_on_address = 0x20
        controls = GPIOControls(make_full_config(), chip, adc=None, clock=clock)
        # Skip init (chip.init would fail); fake bus state instead
        bus.fail_on_address = None
        chip.init()

        bus.fail_on_address = 0x20
        clock.advance(0.030)
        self.assertEqual(controls.poll(), AppEvent.NONE)
        bus.fail_on_address = None
        clock.advance(0.030)
        # Should not raise and should recover
        controls.poll()  # anchor
        clock.advance(0.030)
        # Press snapshot
        bus.registers[(0x20, 0x00)] = 0xDF   # P05 = 0
        bus.registers[(0x20, 0x01)] = 0xFF
        ev = controls.poll()
        self.assertEqual(ev, AppEvent.CAPTURE_IMAGE)

    def test_zoom_pot_emits_zoom_in_when_increasing(self):
        # Bucket 0 → bucket 4 should queue 4 ZOOM_IN events.
        controls, clock, _ = self._build(
            fake_inputs=[0xFFFF],  # closure repeats 0xFFFF when exhausted
            raw_adc=[(0x00, 0x00), (0x04, 0x00)],  # bucket 0, bucket 4
        )
        # Poll 1: anchor inputs (last_inputs was -1) and ADC at bucket 0.
        clock.advance(0.030)
        self.assertEqual(controls.poll(), AppEvent.NONE)
        # Poll 2: ADC reads bucket 4, transitions from 0 → 4, queues 4 ZOOM_IN.
        clock.advance(0.030)
        events = self._drain(controls, clock, max_events=20)
        zoom_ins = [e for e in events if e == AppEvent.ZOOM_IN]
        self.assertEqual(len(zoom_ins), 4)
        zoom_outs = [e for e in events if e == AppEvent.ZOOM_OUT]
        self.assertEqual(len(zoom_outs), 0)

    def test_zoom_pot_emits_zoom_out_when_decreasing(self):
        # Bucket 8 → bucket 4 should queue 4 ZOOM_OUT events.
        controls, clock, _ = self._build(
            fake_inputs=[0xFFFF],
            raw_adc=[(0x08, 0x00), (0x04, 0x00)],
        )
        clock.advance(0.030)
        self.assertEqual(controls.poll(), AppEvent.NONE)   # anchor at 8
        clock.advance(0.030)
        events = self._drain(controls, clock, max_events=20)
        zoom_outs = [e for e in events if e == AppEvent.ZOOM_OUT]
        self.assertEqual(len(zoom_outs), 4)
        zoom_ins = [e for e in events if e == AppEvent.ZOOM_IN]
        self.assertEqual(len(zoom_ins), 0)

    def test_zoom_pot_invert(self):
        cfg = make_full_config()
        cfg["zoom_pot"]["invert"] = True

        bus = FakeBus()
        clock = FakeClock()
        chip = TCA6416A(bus, address=0x20)
        chip.init()
        # Hold all inputs HIGH so no button events interfere.
        chip.read_inputs = lambda: 0xFFFF  # type: ignore[method-assign]

        adc = MCP3221(bus, address=0x4D)
        # start() probes the ADC, consuming one queue entry.
        bus.raw_read_queue.append([0x00, 0x00])  # consumed by probe
        bus.raw_read_queue.append([0x00, 0x00])  # poll 1 anchor: raw bucket 0
        bus.raw_read_queue.append([0x04, 0x00])  # poll 2: raw bucket 4

        controls = GPIOControls(cfg, chip, adc=adc, clock=clock)
        controls.start()

        clock.advance(0.030)
        self.assertEqual(controls.poll(), AppEvent.NONE)
        clock.advance(0.030)
        events = self._drain(controls, clock, max_events=20)
        # With invert=True, raw bucket going UP means inverted bucket
        # going DOWN, so ZOOM_OUT fires.
        zoom_outs = [e for e in events if e == AppEvent.ZOOM_OUT]
        self.assertEqual(len(zoom_outs), 4)
        zoom_ins = [e for e in events if e == AppEvent.ZOOM_IN]
        self.assertEqual(len(zoom_ins), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)