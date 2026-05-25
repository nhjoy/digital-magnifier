#!/usr/bin/env python3
"""Live TCA6416A probe.

Reads the I/O expander on /dev/i2c-1, decodes the nav switch and
buttons against the pin map in ``config/hardware_pins.yaml``, and
streams press / release events to the terminal.

Why run this script
-------------------
It exercises the *production* I²C driver (``TCA6416A`` in
``src/digital_magnifier/hal/i2c_devices.py``) against real hardware,
so if it works the app's GPIO controls path also works. Use it when
bringing up a new breadboard or after a wiring change, before
firing up the whole magnifier.

Usage
-----
::

    # Just listen for button events (Ctrl-C to exit)
    python3 tests/hardware/probe_tca6416a.py

    # Cycle the RGB LED first, then listen
    python3 tests/hardware/probe_tca6416a.py --led-test

    # Bus / address overrides if you've moved things
    python3 tests/hardware/probe_tca6416a.py --bus 1 --address 0x20
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

# Make the production drivers importable from this script's location.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import yaml  # noqa: E402  (after sys.path tweak)

from digital_magnifier.hal.i2c_devices import (  # noqa: E402
    I2CError,
    TCA6416A,
    open_smbus_bus,
    parse_address,
    pin_name_to_index,
)


# ===================================================================
# Helpers
# ===================================================================


def load_pin_config(config_path: Path) -> dict:
    """Load and lightly validate the hardware_pins YAML."""
    if not config_path.exists():
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with config_path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        print("error: hardware_pins.yaml root must be a mapping", file=sys.stderr)
        sys.exit(1)
    return data


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def format_pin_name(name: str, width: int = 12) -> str:
    return f"{name:<{width}}"


# ===================================================================
# Optional LED self-test
# ===================================================================


def led_self_test(chip: TCA6416A, outputs_cfg: dict[str, str]) -> None:
    """Cycle through each named output once, then turn all off.

    Drives each output LOW for ~400 ms (common-anode LEDs light when
    pin is LOW), then HIGH again, then moves to the next colour.
    """
    if not outputs_cfg:
        print("(no outputs configured; skipping LED test)")
        return

    print("\nLED self-test:")
    for name, pin_name in outputs_cfg.items():
        try:
            idx = pin_name_to_index(pin_name)
        except ValueError as exc:
            print(f"  {name}: bad pin name {pin_name!r} ({exc})")
            continue
        try:
            print(f"  {name} on  ({pin_name})")
            chip.write_output_pin(idx, False)   # LOW = light up (common-anode)
            time.sleep(0.4)
            chip.write_output_pin(idx, True)    # HIGH = off
            time.sleep(0.1)
        except I2CError as exc:
            print(f"  {name}: I²C error: {exc}")
            return
    print("  all off\nLED test done.\n")


# ===================================================================
# Event loop
# ===================================================================


def event_loop(
    chip: TCA6416A,
    inputs_cfg: dict[str, str],
    poll_interval_s: float,
) -> None:
    """Continuously read the input register and report edge transitions."""
    # name -> pin index, name -> bit mask
    pins: dict[str, int] = {}
    masks: dict[str, int] = {}
    for name, pin_name in inputs_cfg.items():
        try:
            idx = pin_name_to_index(pin_name)
        except ValueError as exc:
            print(f"warning: ignoring input {name!r} with bad pin {pin_name!r}: {exc}")
            continue
        pins[name] = idx
        masks[name] = 1 << idx

    if not pins:
        print("error: no valid inputs configured; nothing to monitor", file=sys.stderr)
        sys.exit(1)

    print(f"\nConfigured inputs ({len(pins)}):")
    for name, idx in pins.items():
        pin_label = ("P0" if idx < 8 else "P1") + str(idx % 8)
        print(f"  {pin_label}  {name}")

    print("\nPress buttons to see events. Ctrl-C to exit.\n")

    # Anchor the initial state without producing events.
    try:
        last = chip.read_inputs()
    except I2CError as exc:
        print(f"error: initial read failed: {exc}", file=sys.stderr)
        sys.exit(1)

    press_times: dict[str, float] = {}
    consecutive_failures = 0

    while True:
        time.sleep(poll_interval_s)

        try:
            curr = chip.read_inputs()
        except I2CError as exc:
            consecutive_failures += 1
            if consecutive_failures <= 3 or consecutive_failures % 50 == 0:
                print(f"[warn] I²C read failed ({exc}); retrying...")
            if consecutive_failures > 200:
                print("error: too many consecutive failures, aborting", file=sys.stderr)
                sys.exit(1)
            continue
        if consecutive_failures:
            print(f"[info] I²C recovered after {consecutive_failures} failed reads")
            consecutive_failures = 0

        if curr == last:
            continue

        now = time.monotonic()
        stamp = time.strftime("%H:%M:%S")
        for name, mask in masks.items():
            was_pressed = (last & mask) == 0
            is_pressed = (curr & mask) == 0
            if not was_pressed and is_pressed:
                press_times[name] = now
                print(f"[{stamp}] {format_pin_name(name)}PRESSED")
            elif was_pressed and not is_pressed:
                held = now - press_times.pop(name, now)
                print(
                    f"[{stamp}] {format_pin_name(name)}released  "
                    f"(held {held:.2f}s)"
                )

        last = curr


# ===================================================================
# Main
# ===================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "config" / "hardware_pins.yaml",
        help="Path to hardware_pins.yaml (default: %(default)s)",
    )
    ap.add_argument(
        "--bus", type=int, default=None,
        help="I²C bus number (default: read from config, fallback 1)",
    )
    ap.add_argument(
        "--address", type=lambda s: int(s, 0), default=None,
        help="TCA6416A I²C address (default: read from config, fallback 0x20)",
    )
    ap.add_argument(
        "--led-test", action="store_true",
        help="Cycle the RGB LED once at startup as an output test",
    )
    ap.add_argument(
        "--poll-ms", type=int, default=20,
        help="Polling interval in milliseconds (default: %(default)s)",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable driver debug logging",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    cfg = load_pin_config(args.config)

    # Resolve bus and address with command-line overrides
    i2c_cfg = cfg.get("i2c", {})
    devices_cfg = i2c_cfg.get("devices", {})
    io_cfg = devices_cfg.get("io_expander", {})
    bus_number = args.bus if args.bus is not None else int(i2c_cfg.get("bus", 1))
    address = (
        args.address if args.address is not None
        else parse_address(io_cfg.get("address", 0x20))
    )
    poll_interval = max(args.poll_ms, 1) / 1000.0

    pins_cfg = cfg.get("tca6416a_pins", {})
    config_port0 = int(pins_cfg.get("config_port0", 0xFF))
    config_port1 = int(pins_cfg.get("config_port1", 0xFF))
    inputs_cfg = pins_cfg.get("inputs", {}) or {}
    outputs_cfg = pins_cfg.get("outputs", {}) or {}

    print(f"TCA6416A live probe")
    print(f"  config:   {args.config}")
    print(f"  bus:      /dev/i2c-{bus_number}")
    print(f"  address:  0x{address:02x}")
    print(f"  poll:     {args.poll_ms} ms")

    # Open the bus
    try:
        bus = open_smbus_bus(bus_number)
    except I2CError as exc:
        print(f"\nerror: failed to open I2C bus {bus_number}: {exc}", file=sys.stderr)
        print(f"hint: enable I²C with `sudo raspi-config nonint do_i2c 0`",
              file=sys.stderr)
        sys.exit(2)

    # Construct the driver
    try:
        chip = TCA6416A(
            bus, address=address,
            config_port0=config_port0,
            config_port1=config_port1,
        )
    except ValueError as exc:
        print(f"\nerror: invalid config: {exc}", file=sys.stderr)
        bus.close()
        sys.exit(2)

    # Probe — does the chip actually respond?
    try:
        chip.probe()
    except I2CError as exc:
        print(f"\nerror: TCA6416A at 0x{address:02x} not responding: {exc}",
              file=sys.stderr)
        print(f"hint: `i2cdetect -y {bus_number}` and look for the address",
              file=sys.stderr)
        bus.close()
        sys.exit(3)

    # Bring the chip up (writes polarity, outputs default HIGH, direction config)
    try:
        chip.init()
    except I2CError as exc:
        print(f"\nerror: TCA6416A init failed: {exc}", file=sys.stderr)
        bus.close()
        sys.exit(3)

    print(f"\nTCA6416A at 0x{address:02x}: detected and initialised.")

    # Ensure LEDs end up OFF on Ctrl-C
    def cleanup(*_args) -> None:
        try:
            for pin_name in outputs_cfg.values():
                try:
                    idx = pin_name_to_index(pin_name)
                    chip.write_output_pin(idx, True)  # HIGH = off
                except (ValueError, I2CError):
                    pass
        finally:
            bus.close()
            print("\nClosed I²C bus.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    if args.led_test:
        led_self_test(chip, outputs_cfg)

    try:
        event_loop(chip, inputs_cfg, poll_interval)
    except KeyboardInterrupt:
        cleanup()
    except Exception:
        import traceback
        traceback.print_exc()
        cleanup()
        sys.exit(4)


if __name__ == "__main__":
    main()
