#!/usr/bin/env python3
"""Live UPS HAT (E) probe.

Polls the Waveshare UPS HAT over I²C and prints battery state to the
terminal: pack voltage, charge percent, current direction, runtime
estimate, charging state, and per-cell voltages.

Why run this script
-------------------
Use it to confirm the UPS HAT is alive and talking before wiring its
state into the app's UI or shutdown logic. Useful for:
  * confirming a fresh wiring job (I²C address detected, ID matches),
  * watching the pack discharge or charge during bring-up testing,
  * sanity-checking per-cell balance on a 2S/3S/4S pack.

Usage
-----
::

    # Default: poll once per second, print a one-line summary
    python3 tests/hardware/probe_ups_hat.py

    # Faster polling
    python3 tests/hardware/probe_ups_hat.py --rate 5

    # Verbose: show per-cell voltages too
    python3 tests/hardware/probe_ups_hat.py --cells

    # Just one read, then exit (useful for scripts / cron)
    python3 tests/hardware/probe_ups_hat.py --once
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from digital_magnifier.hal.i2c_devices import (  # noqa: E402
    I2CError,
    UPSHatE,
    UPSStatus,
    open_smbus_bus,
)


DEFAULT_BUS = 1
DEFAULT_ADDRESS = 0x2D


# ===================================================================
# Formatting
# ===================================================================


def format_one_line(status: UPSStatus) -> str:
    direction = (
        "+" if status.battery_current_ma > 0
        else "-" if status.battery_current_ma < 0
        else "0"
    )

    # Runtime estimate, in whichever direction is meaningful
    if status.charging:
        runtime = (
            f"{status.remaining_charge_min} min to full"
            if status.remaining_charge_min and status.remaining_charge_min < 0xFFFF
            else "—"
        )
    else:
        runtime = (
            f"{status.remaining_discharge_min} min left"
            if status.remaining_discharge_min and status.remaining_discharge_min < 0xFFFF
            else "—"
        )

    state_marker = "⚡" if status.charging else "🔋" if not status.vbus_powered else "—"

    return (
        f"{state_marker} "
        f"{status.battery_percent:3d}% "
        f"({status.battery_voltage_mv / 1000.0:.2f} V) "
        f"{direction}{abs(status.battery_current_ma):>5d} mA  "
        f"VBUS={status.vbus_voltage_mv / 1000.0:.2f}V "
        f"{status.vbus_current_ma:>4d}mA  "
        f"state={status.charge_state:<17s} "
        f"{runtime}"
    )


def format_cells(status: UPSStatus) -> str:
    parts = []
    for i, mv in enumerate(status.cell_voltages_mv, start=1):
        if mv > 0:
            parts.append(f"C{i}={mv / 1000.0:.3f}V")
        else:
            parts.append(f"C{i}=—")
    return "  cells: " + "  ".join(parts)


# ===================================================================
# Main
# ===================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--bus", type=int, default=DEFAULT_BUS,
        help=f"I²C bus number (default: %(default)s)",
    )
    ap.add_argument(
        "--address", type=lambda s: int(s, 0), default=DEFAULT_ADDRESS,
        help=f"UPS HAT I²C address (default: 0x{DEFAULT_ADDRESS:02x})",
    )
    ap.add_argument(
        "--rate", type=float, default=1.0,
        help="Polling rate in Hz (default: %(default)s)",
    )
    ap.add_argument(
        "--cells", action="store_true",
        help="Also print per-cell voltages each tick",
    )
    ap.add_argument(
        "--once", action="store_true",
        help="Take one reading and exit; useful for cron / scripts",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true",
        help="Driver debug logging",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    print("UPS HAT (E) live probe")
    print(f"  bus:      /dev/i2c-{args.bus}")
    print(f"  address:  0x{args.address:02x}")
    print(f"  rate:     {args.rate:.1f} Hz" if not args.once else "  rate:     once")

    # ---- Open bus ----
    try:
        bus = open_smbus_bus(args.bus)
    except I2CError as exc:
        print(f"\nerror: failed to open I2C bus {args.bus}: {exc}", file=sys.stderr)
        print(f"hint: `sudo raspi-config nonint do_i2c 0`", file=sys.stderr)
        sys.exit(2)

    # ---- Build + probe driver ----
    try:
        ups = UPSHatE(bus, address=args.address)
    except ValueError as exc:
        print(f"\nerror: invalid config: {exc}", file=sys.stderr)
        bus.close()
        sys.exit(2)

    try:
        ups.probe()
    except I2CError as exc:
        print(f"\nerror: UPS HAT not detected: {exc}", file=sys.stderr)
        print(
            f"hint: `i2cdetect -y {args.bus}` — expect 0x{args.address:02x} "
            f"in the grid",
            file=sys.stderr,
        )
        bus.close()
        sys.exit(3)

    print(f"\nUPS HAT at 0x{args.address:02x}: detected. ID register reads 0x0A as expected.\n")

    # ---- Cleanup ----
    def cleanup(*_args) -> None:
        bus.close()
        print("\nClosed I²C bus.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # ---- Read loop ----
    period_s = 1.0 / max(args.rate, 0.1)
    consecutive_failures = 0

    try:
        while True:
            start = time.monotonic()
            try:
                status = ups.read_summary()
            except I2CError as exc:
                consecutive_failures += 1
                if consecutive_failures <= 3 or consecutive_failures % 30 == 0:
                    print(f"  [warn] read failed ({exc}); retrying")
                if consecutive_failures > 100:
                    print("\nerror: too many consecutive failures, aborting",
                          file=sys.stderr)
                    sys.exit(1)
            else:
                if consecutive_failures:
                    print(f"  [info] recovered after {consecutive_failures} failures")
                    consecutive_failures = 0
                stamp = time.strftime("%H:%M:%S")
                print(f"[{stamp}] {format_one_line(status)}")
                if args.cells:
                    print(format_cells(status))

            if args.once:
                break

            elapsed = time.monotonic() - start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)

    except KeyboardInterrupt:
        cleanup()
    except Exception:
        import traceback
        traceback.print_exc()
        cleanup()
        sys.exit(4)

    cleanup()


if __name__ == "__main__":
    main()
