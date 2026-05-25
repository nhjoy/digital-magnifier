"""Buzzer HAL for the PS1740P02E passive piezoelectric transducer.

The transducer needs a 4 kHz square wave to produce sound. This module
wraps gpiozero's ``PWMOutputDevice`` and exposes named patterns that
the app controller can trigger for startup, shutdown, low-battery,
and invalid-press feedback.

Patterns are blocking (the main loop pauses for the duration), which
is acceptable because the longest pattern is ~800 ms. For the 1 Hz
low-battery warning, the caller fires a single short beep per tick
rather than playing a long pattern.

Graceful degradation: if gpiozero is not installed or the pin cannot
be claimed, the module logs a warning and ``BuzzerController.play``
silently does nothing.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ===================================================================
# Named patterns  (on_seconds, off_seconds)
# ===================================================================

PATTERNS: dict[str, list[tuple[float, float]]] = {
    "startup":     [(0.08, 0.08), (0.15, 0.0)],
    "shutdown":    [(0.20, 0.10), (0.20, 0.10), (0.40, 0.0)],
    "low_battery": [(0.10, 0.10), (0.10, 0.10), (0.10, 0.0)],
    "invalid":     [(0.08, 0.0)],
    "battery_50":  [(0.15, 0.0)],
}


class BuzzerController:
    """High-level buzzer interface for the magnifier.

    Parameters
    ----------
    pin
        BCM GPIO pin connected to the transducer.
    frequency
        Square-wave drive frequency in Hz. The PS1740P02E resonates
        at 4 000 Hz; driving off-resonance causes a steep volume
        drop.
    """

    DEFAULT_PIN = 24
    DEFAULT_FREQ = 4000

    def __init__(
        self,
        pin: int = DEFAULT_PIN,
        frequency: int = DEFAULT_FREQ,
    ) -> None:
        self._device = None      # type: ignore[assignment]
        self._frequency = frequency
        try:
            from gpiozero import PWMOutputDevice  # type: ignore[import-not-found]
            self._device = PWMOutputDevice(
                pin,
                active_high=True,
                initial_value=0,
                frequency=frequency,
            )
            logger.info(
                "Buzzer ready on GPIO %d at %d Hz",
                pin, frequency,
            )
        except ImportError:
            logger.warning(
                "gpiozero not installed; buzzer disabled "
                "(sudo apt install python3-gpiozero)"
            )
        except Exception as exc:
            logger.warning(
                "Could not claim GPIO %d for buzzer: %s; buzzer disabled",
                pin, exc,
            )

    @property
    def available(self) -> bool:
        """True if the buzzer hardware was successfully opened."""
        return self._device is not None

    # ----- playback -----------------------------------------------

    def play(self, pattern_name: str) -> None:
        """Play a named pattern. No-op if buzzer isn't available."""
        pattern = PATTERNS.get(pattern_name)
        if pattern is None:
            logger.warning("Unknown buzzer pattern: %r", pattern_name)
            return
        self._play_raw(pattern)

    def beep(self, duration_s: float = 0.10) -> None:
        """Single beep of *duration_s* seconds."""
        self._play_raw([(duration_s, 0.0)])

    def _play_raw(self, pattern: list[tuple[float, float]]) -> None:
        if self._device is None:
            return
        for on_s, off_s in pattern:
            if on_s > 0:
                self._device.value = 0.5    # 50% duty-cycle = tone
                time.sleep(on_s)
                self._device.value = 0      # silent
            if off_s > 0:
                time.sleep(off_s)

    # ----- lifecycle -----------------------------------------------

    def close(self) -> None:
        if self._device is not None:
            self._device.value = 0
            self._device.close()
            self._device = None
            logger.info("Buzzer released")