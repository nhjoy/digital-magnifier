#!/usr/bin/env python3
"""Live buzzer probe (PS1740P02E passive piezo transducer on GPIO 24).

The PS1740P02E is a *passive* transducer — it has no internal
oscillator. A steady DC voltage just produces a faint click. To
make audible sound, you must feed it an alternating square wave at
its resonant frequency (4.0 kHz). This probe uses gpiozero's
``PWMOutputDevice`` to generate that square wave via the lgpio
backend's kernel-level ``tx_pwm()``.

Usage
-----
::

    # Default: cycle through all the named patterns at 4 kHz
    python3 tests/hardware/probe_buzzer.py

    # Just one pattern
    python3 tests/hardware/probe_buzzer.py --pattern startup

    # Tune the drive frequency (±200 Hz from resonance still works)
    python3 tests/hardware/probe_buzzer.py --freq 4000

    # Manual mode — press Enter to beep, type pattern names
    python3 tests/hardware/probe_buzzer.py --manual

    # Use a different pin
    python3 tests/hardware/probe_buzzer.py --pin 24
"""

from __future__ import annotations

import argparse
import signal
import sys
import time


DEFAULT_PIN = 24       # BCM pin (physical pin 18 on the 40-pin header)
DEFAULT_FREQ = 4000    # PS1740P02E resonant frequency = 4.0 kHz
DUTY_CYCLE = 0.5       # 50% duty cycle square wave


# ===================================================================
# Pattern definitions
# ===================================================================
#
# Each pattern is a list of (on_seconds, off_seconds) pairs.
# "on" means the PWM square wave is active; "off" means the pin
# is silent. The different timing makes them distinguishable by
# ear even though they all play the same pitch.

PATTERNS: dict[str, list[tuple[float, float]]] = {
    # Startup: two quick ascending blips. ~300 ms total.
    "startup":     [(0.08, 0.08), (0.15, 0.0)],

    # Shutdown: three descending pulses, getting longer. ~800 ms.
    "shutdown":    [(0.20, 0.10), (0.20, 0.10), (0.40, 0.0)],

    # Low battery: three short urgent beeps. ~600 ms. Repeatable.
    "low_battery": [(0.10, 0.10), (0.10, 0.10), (0.10, 0.0)],

    # Invalid press: one very short blip. ~80 ms.
    "invalid":     [(0.08, 0.0)],

    # Long test tone for confirming buzzer works at all.
    "test":        [(1.0, 0.0)],
}


# ===================================================================
# GPIO access — PWM via gpiozero (lgpio backend on Pi 5 / CM5)
# ===================================================================


class BuzzerError(Exception):
    """Anything that goes wrong opening or driving the GPIO."""


def open_buzzer(pin: int, frequency: int):
    """Open ``pin`` as a PWM output at ``frequency`` Hz.

    Returns an object with ``.value`` (0.0 = silent, 0.5 = 50%
    duty-cycle tone) and ``.close()``.
    """
    try:
        from gpiozero import PWMOutputDevice  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BuzzerError(
            "gpiozero not installed; on Pi OS: sudo apt install python3-gpiozero"
        ) from exc

    try:
        return PWMOutputDevice(
            pin,
            active_high=True,
            initial_value=0,       # start silent
            frequency=frequency,   # square-wave frequency in Hz
        )
    except Exception as exc:
        raise BuzzerError(
            f"could not claim GPIO {pin} for PWM: {exc}. "
            f"Is another process holding it? Try `sudo lsof /dev/gpiochip*`."
        ) from exc


# ===================================================================
# Playback
# ===================================================================


def beep_on(buzzer) -> None:
    """Start the tone (50% duty-cycle square wave)."""
    buzzer.value = DUTY_CYCLE


def beep_off(buzzer) -> None:
    """Silence the buzzer (0% duty = pin held low)."""
    buzzer.value = 0


def play_pattern(buzzer, pattern: list[tuple[float, float]]) -> None:
    """Play one pattern. Caller does the labelling / spacing."""
    for on_s, off_s in pattern:
        if on_s > 0:
            beep_on(buzzer)
            time.sleep(on_s)
            beep_off(buzzer)
        if off_s > 0:
            time.sleep(off_s)


def cycle_all(buzzer, gap_s: float = 0.8) -> None:
    """Play every named pattern in order, with a gap between them."""
    names = list(PATTERNS.keys())
    for i, name in enumerate(names):
        print(f"  [{i + 1}/{len(names)}] {name}")
        play_pattern(buzzer, PATTERNS[name])
        if i < len(names) - 1:
            time.sleep(gap_s)


def manual_mode(buzzer) -> None:
    """Beep on Enter. Type a pattern name to play it."""
    print("\nManual mode: press Enter to beep, type a pattern name, or 'q' to quit.")
    print(f"  Available patterns: {', '.join(PATTERNS)}")
    while True:
        try:
            line = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            play_pattern(buzzer, [(0.10, 0.0)])
            continue
        if line in PATTERNS:
            print(f"  playing: {line}")
            play_pattern(buzzer, PATTERNS[line])
            continue
        if line in {"q", "quit", "exit"}:
            return
        print(
            f"  unknown. Press Enter for a beep, or type a pattern name: "
            f"{', '.join(PATTERNS)}"
        )


# ===================================================================
# Main
# ===================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--pin", type=int, default=DEFAULT_PIN,
        help=f"BCM pin number (default: %(default)s = physical pin 18)",
    )
    ap.add_argument(
        "--freq", type=int, default=DEFAULT_FREQ,
        help=(
            f"Square-wave frequency in Hz (default: %(default)s). "
            f"The PS1740P02E resonates at 4000 Hz; driving off-resonance "
            f"causes a steep volume drop."
        ),
    )
    ap.add_argument(
        "--pattern", choices=list(PATTERNS), default=None,
        help="Play this single pattern then exit",
    )
    ap.add_argument(
        "--manual", action="store_true",
        help="Interactive mode — Enter to beep, named patterns by typing",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Buzzer probe (passive piezo, PWM-driven)")
    print(f"  pin:       BCM {args.pin}")
    print(f"  frequency: {args.freq} Hz (resonant = {DEFAULT_FREQ} Hz)")
    if args.pattern:
        print(f"  mode:      single pattern ({args.pattern})")
    elif args.manual:
        print(f"  mode:      manual / interactive")
    else:
        print(f"  mode:      cycle through all {len(PATTERNS)} patterns")

    try:
        buzzer = open_buzzer(args.pin, args.freq)
    except BuzzerError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        sys.exit(2)

    def cleanup(*_args) -> None:
        try:
            beep_off(buzzer)
        finally:
            buzzer.close()
        print("\nBuzzer released.")
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print()
    try:
        if args.pattern:
            print(f"Playing: {args.pattern}")
            play_pattern(buzzer, PATTERNS[args.pattern])
        elif args.manual:
            manual_mode(buzzer)
        else:
            cycle_all(buzzer)
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