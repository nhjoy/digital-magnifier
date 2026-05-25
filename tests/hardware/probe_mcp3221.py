#!/usr/bin/env python3
"""Live MCP3221 zoom-pot probe.

Continuously reads the 12-bit ADC and prints raw value, voltage,
quantised bucket, and an ASCII bar showing pot position. Use this
when the MCP3221 is wired up to verify the pot sweeps cleanly
across its full range before trusting it in the app.

Why run this script
-------------------
It exercises the *production* MCP3221 driver, the same one the app's
GPIOControls path uses. If the pot looks smooth here, the app's
zoom-pot logic will see a smooth signal too. If it jumps or sticks,
the problem is wiring / power / pot quality, not software.

Usage
-----
::

    # Default: 5 Hz read-out, exits on Ctrl-C
    python3 tests/hardware/probe_mcp3221.py

    # Faster sampling
    python3 tests/hardware/probe_mcp3221.py --rate 20

    # Override bus/address
    python3 tests/hardware/probe_mcp3221.py --bus 1 --address 0x4D
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

# Make the production drivers importable from this script's location.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import yaml  # noqa: E402

from digital_magnifier.hal.i2c_devices import (  # noqa: E402
    I2CError,
    MCP3221,
    open_smbus_bus,
    parse_address,
)


# ===================================================================
# Helpers
# ===================================================================


BAR_WIDTH = 30


def load_pin_config(config_path: Path) -> dict:
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


def render_bar(fraction: float, width: int = BAR_WIDTH) -> str:
    """Return an ASCII progress bar of length ``width`` for 0.0 .. 1.0."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


# ===================================================================
# Read loop
# ===================================================================


def read_loop(
    adc: MCP3221,
    sample_period_s: float,
    buckets: int,
    invert: bool,
    vdd: float,
) -> None:
    """Sample the ADC indefinitely and pretty-print each reading."""
    print(
        f"\nSampling at {1.0 / sample_period_s:.1f} Hz "
        f"({buckets} buckets, invert={invert}). Ctrl-C to exit.\n"
    )
    print(f"  {'raw':>5}  {'V':>5}  {'bucket':>9}  {'bar':<{BAR_WIDTH + 2}}")
    print(f"  {'-' * 5}  {'-' * 5}  {'-' * 9}  {'-' * (BAR_WIDTH + 2)}")

    last_bucket: int | None = None
    consecutive_failures = 0

    while True:
        start = time.monotonic()
        try:
            raw = adc.read_raw()
        except I2CError as exc:
            consecutive_failures += 1
            if consecutive_failures <= 3 or consecutive_failures % 50 == 0:
                print(f"  [warn] read failed ({exc}); retrying")
            if consecutive_failures > 200:
                print("\nerror: too many consecutive failures, aborting",
                      file=sys.stderr)
                sys.exit(1)
        else:
            if consecutive_failures:
                print(f"  [info] recovered after {consecutive_failures} failures")
                consecutive_failures = 0

            fraction = raw / MCP3221.MAX_RAW          # 0.0 .. 1.0
            voltage = fraction * vdd

            bucket = int(fraction * buckets)
            if bucket >= buckets:
                bucket = buckets - 1
            if invert:
                bucket = (buckets - 1) - bucket

            # Marker '*' when the bucket changes — handy for spotting
            # whether the boundary settings match what the app will see.
            marker = ""
            if last_bucket is not None and bucket != last_bucket:
                direction = "↑" if bucket > last_bucket else "↓"
                marker = f"  bucket {direction} ({last_bucket} → {bucket})"
            last_bucket = bucket

            print(
                f"  {raw:>5}  {voltage:>5.2f}  {bucket:>3}/{buckets:<3}  "
                f"{render_bar(fraction)}{marker}"
            )

        # Sleep just long enough to keep the requested sample rate
        elapsed = time.monotonic() - start
        if elapsed < sample_period_s:
            time.sleep(sample_period_s - elapsed)


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
        help="MCP3221 I²C address (default: read from config, fallback 0x4D)",
    )
    ap.add_argument(
        "--rate", type=float, default=5.0,
        help="Sample rate in Hz (default: %(default)s)",
    )
    ap.add_argument(
        "--vdd", type=float, default=None,
        help="Supply voltage for voltage display (default: from config, fallback 3.3)",
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
    i2c_cfg = cfg.get("i2c", {})
    devices_cfg = i2c_cfg.get("devices", {})
    adc_cfg = devices_cfg.get("zoom_adc", {})

    bus_number = args.bus if args.bus is not None else int(i2c_cfg.get("bus", 1))
    address = (
        args.address if args.address is not None
        else parse_address(adc_cfg.get("address", 0x4D))
    )
    vdd = (
        args.vdd if args.vdd is not None
        else float(adc_cfg.get("vdd_volts", 3.3))
    )

    zoom_cfg = cfg.get("zoom_pot", {})
    buckets = max(2, int(zoom_cfg.get("buckets", 16)))
    invert = bool(zoom_cfg.get("invert", False))

    sample_period_s = 1.0 / max(args.rate, 0.5)

    print(f"MCP3221 live probe")
    print(f"  config:   {args.config}")
    print(f"  bus:      /dev/i2c-{bus_number}")
    print(f"  address:  0x{address:02x}")
    print(f"  vdd:      {vdd:.2f} V")
    print(f"  rate:     {args.rate:.1f} Hz")

    try:
        bus = open_smbus_bus(bus_number)
    except I2CError as exc:
        print(f"\nerror: failed to open I2C bus {bus_number}: {exc}", file=sys.stderr)
        print(f"hint: enable I²C with `sudo raspi-config nonint do_i2c 0`",
              file=sys.stderr)
        sys.exit(2)

    try:
        adc = MCP3221(bus, address=address, vdd_volts=vdd)
    except ValueError as exc:
        print(f"\nerror: invalid config: {exc}", file=sys.stderr)
        bus.close()
        sys.exit(2)

    try:
        adc.probe()
    except I2CError as exc:
        print(f"\nerror: MCP3221 at 0x{address:02x} not responding: {exc}",
              file=sys.stderr)
        print(f"hint: `i2cdetect -y {bus_number}` and look for the address.",
              file=sys.stderr)
        print(f"      MCP3221A5T-I/OT defaults to 0x4D.", file=sys.stderr)
        bus.close()
        sys.exit(3)

    print(f"\nMCP3221 at 0x{address:02x}: detected.")

    def cleanup(*_args) -> None:
        bus.close()
        print("\nClosed I²C bus.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        read_loop(adc, sample_period_s, buckets, invert, vdd)
    except KeyboardInterrupt:
        cleanup()
    except Exception:
        import traceback
        traceback.print_exc()
        cleanup()
        sys.exit(4)


if __name__ == "__main__":
    main()
