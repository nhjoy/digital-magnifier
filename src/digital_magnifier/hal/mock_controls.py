"""Mock controls HAL: keyboard input → AppEvent.

Reads ``cv2.waitKey()`` and maps the result to an ``AppEvent`` using
the ``mock_keyboard_map`` section of ``config/hardware_pins.yaml``.
Used during development on laptops and in WSL. The future
``GPIOControls`` (MVP 0.3) will produce the exact same events from
real hardware so that everything above the HAL is unaffected by the
swap.

Design notes
------------
- ``cv2.waitKey()`` only returns key events when an OpenCV window
  is open and focused. The app controller satisfies this by calling
  ``cv2.imshow()`` every frame; ``MockControls`` itself does not
  manage windows.

- Power short-press and long-press are bound to **separate keys**
  (``'p'`` and ``'P'``) rather than emulated through hold-time
  detection. Rationale: ``cv2.waitKey()`` does not expose key-release
  events, only key-arrival events, so any hold-time heuristic on a
  keyboard fights against the OS keyboard-repeat delay (typically
  300–500 ms before repeat begins, which masks short-versus-long
  intent). Real GPIO has clean rising and falling edges and can do
  accurate hold-time detection; that logic will live in
  ``GPIOControls``, not here.

- The key reader is injected as a callable so unit tests can feed
  canned input without opening an OpenCV window. By default it
  calls ``cv2.waitKey(1)`` lazily so importing this module does not
  require OpenCV to be installed in test environments.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from digital_magnifier.core.events import AppEvent
from digital_magnifier.hal.controls_base import ControlsHAL


logger = logging.getLogger(__name__)


# Translate human-readable key names in YAML to the actual
# single-character form ``cv2.waitKey()`` returns. Add more
# aliases here if needed (arrow keys, function keys, etc.) once
# their cv2 codes are confirmed on the target platform.
_KEY_ALIASES: dict[str, str] = {
    "space": " ",
    "enter": "\r",
    "return": "\r",
    "tab": "\t",
    "esc": chr(27),
    "escape": chr(27),
}


def _default_key_reader() -> int:
    """Default backend: poll OpenCV for the next key event.

    Lazy-imports ``cv2`` so that importing :mod:`mock_controls` in
    a test environment does not require OpenCV. Tests inject a
    fake reader and never reach this function.
    """
    import cv2  # Lazy import on purpose.

    return cv2.waitKey(1)


class MockControls(ControlsHAL):
    """Keyboard-driven controls HAL for development.

    Parameters
    ----------
    hardware_config : dict
        Parsed contents of ``config/hardware_pins.yaml``. Only the
        ``mock_keyboard_map`` section is read.
    key_reader : Callable[[], int], optional
        Returns the next key code, or ``-1`` if no key is pending.
        Defaults to ``cv2.waitKey(1)``. Tests inject a fake.
    """

    def __init__(
        self,
        hardware_config: dict[str, Any],
        key_reader: Callable[[], int] | None = None,
    ) -> None:
        self._hardware_config = hardware_config
        self._key_reader: Callable[[], int] = key_reader or _default_key_reader
        self._key_code_to_event: dict[int, AppEvent] = {}

        self._build_key_map()

    # ----- ControlsHAL interface -------------------------------------

    def poll(self) -> AppEvent:
        raw = self._key_reader()

        # ``cv2.waitKey`` returns -1 (or other negative) when no key
        # is pending. Treat any negative value as "no input".
        if raw < 0:
            return AppEvent.NONE

        # Mask to a single byte; cv2 may return higher values for
        # special keys, but the map is keyed the same way.
        key_code = raw & 0xFF
        return self._key_code_to_event.get(key_code, AppEvent.NONE)

    # ----- internals -------------------------------------------------

    def _build_key_map(self) -> None:
        raw_map = self._hardware_config.get("mock_keyboard_map", {})

        if not raw_map:
            logger.warning(
                "hardware_pins.yaml has no 'mock_keyboard_map' section; "
                "MockControls will only ever return AppEvent.NONE"
            )
            return

        for key_str, event_name in raw_map.items():
            self._add_key_mapping(key_str, event_name)

        logger.info(
            "MockControls loaded %d key mapping(s)", len(self._key_code_to_event)
        )

    def _add_key_mapping(self, key_str: Any, event_name: Any) -> None:
        # --- validate the key side --------------------------------
        if not isinstance(key_str, str):
            logger.warning(
                "Skipping mock_keyboard_map entry: key is not a string (%r)", key_str
            )
            return

        normalized = _KEY_ALIASES.get(key_str.lower(), key_str)
        if len(normalized) != 1:
            logger.warning(
                "Skipping mock_keyboard_map entry %r: expected a single "
                "character or known alias, got length %d",
                key_str, len(normalized),
            )
            return

        # --- validate the event side ------------------------------
        if not isinstance(event_name, str):
            logger.warning(
                "Skipping mock_keyboard_map entry for key %r: "
                "event must be a string, got %r",
                key_str, event_name,
            )
            return

        try:
            event = AppEvent[event_name]
        except KeyError:
            logger.warning(
                "Skipping mock_keyboard_map entry for key %r: "
                "unknown AppEvent %r",
                key_str, event_name,
            )
            return

        # --- store, warning on duplicates -------------------------
        key_code = ord(normalized) & 0xFF
        if key_code in self._key_code_to_event:
            existing = self._key_code_to_event[key_code]
            logger.warning(
                "Key %r (code %d) re-mapped from %s to %s; "
                "later YAML entries win",
                key_str, key_code, existing.name, event.name,
            )
        self._key_code_to_event[key_code] = event