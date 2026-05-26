"""Controls HAL implementations.

Two classes live in this module:

- :class:`MockControls` — keyboard input → AppEvent, used during
  development on laptops and in WSL.

- :class:`GPIOControls` (MVP 0.3) — real hardware input via a
  TCA6416A I/O expander (buttons + nav switch) and an optional
  MCP3221 ADC (zoom potentiometer). Generates identical AppEvent
  values, so the app controller, state machine, and everything
  above the HAL are unaware of which one is plugged in.

The filename "mock_controls.py" predates MVP 0.3 and is now slightly
misleading. It is kept to avoid a rename per the project's
no-rename rule; both implementations live here side by side.

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


# ===================================================================
# GPIOControls — real hardware (MVP 0.3)
# ===================================================================

import time
from collections import deque
from typing import Optional

from digital_magnifier.hal.i2c_devices import (
    I2CError,
    MCP3221,
    TCA6416A,
    pin_name_to_index,
)


class _TimedButtonState:
    """Tracks a button that emits different events for short vs long press.

    Unlike regular ``button_events`` (fire on press edge), timed buttons
    fire on *release* if held less than ``threshold_s`` (short event) or
    as soon as the hold threshold is reached (long event). The long event
    fires once; subsequent release is a no-op to avoid double-firing.

    Either ``short_event`` or ``long_event`` may be None — for example,
    the status button has only a long press (no useful short action).
    """
    __slots__ = ("short_event", "long_event", "threshold_s",
                 "pressed_at", "long_fired")

    def __init__(
        self,
        short_event: Optional[AppEvent],
        long_event: Optional[AppEvent],
        threshold_s: float,
    ) -> None:
        self.short_event = short_event
        self.long_event = long_event
        self.threshold_s = threshold_s
        self.pressed_at: Optional[float] = None
        self.long_fired: bool = False


class GPIOControls(ControlsHAL):
    """Controls HAL backed by a TCA6416A I/O expander and optional MCP3221 ADC.

    Reads:
      * 5-way nav switch (P00..P04) → PAN_* / RESET_VIEW
      * 6 action buttons (P05..P07, P10..P12) → app events per config
      * Zoom pot via MCP3221 (optional) → ZOOM_IN / ZOOM_OUT

    Press detection is edge-based: events fire on the HIGH→LOW
    transition of an active-low input (button press), not while the
    button is held. Nav directions optionally repeat while held for
    "press and hold to pan further" UX.

    The power button is special: short press → POWER_SHORT_PRESS on
    release, long press → POWER_LONG_PRESS the moment the threshold
    is crossed (so the user gets immediate feedback when they hold
    for shutdown).

    Internal event queue
    --------------------
    The :class:`ControlsHAL` contract is one event per :meth:`poll`
    call. With 11 buttons and one ADC channel, a single poll can
    legitimately produce multiple events (e.g. user starts pressing
    two nav directions on the same I2C read). We buffer them in a
    deque and drain one per poll, performing I2C reads only when the
    deque is empty and enough time has elapsed since the last read.

    Resilience
    ----------
    I2C failures during steady-state operation are logged at WARNING
    and the read is skipped — the device does not crash if a wire
    falls off. The first failure logs; subsequent identical failures
    are rate-limited via ``_io_failure_logged`` to avoid log spam.
    """

    # Hard cap so a buggy YAML or stuck button can't fill memory.
    _MAX_QUEUED_EVENTS = 64

    def __init__(
        self,
        hardware_config: dict[str, Any],
        io_expander: TCA6416A,
        adc: Optional[MCP3221] = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._hardware_config = hardware_config
        self._io_expander = io_expander
        self._adc = adc
        self._clock = clock

        # ---- Resolved pin maps and event bindings ---------------
        # input_name → linear pin index (0..15)
        self._input_pins: dict[str, int] = {}
        # Lookup masks keyed by input_name for quick state checks.
        self._input_mask: dict[str, int] = {}
        # Action buttons: input_name → AppEvent (fires on press edge)
        self._button_events: dict[str, AppEvent] = {}
        # Nav directions: input_name → AppEvent (fires on press, optionally repeats)
        self._nav_events: dict[str, AppEvent] = {}

        # ---- Nav repeat config ----------------------------------
        self._nav_repeat_enabled: bool = True
        self._nav_repeat_initial_delay_s: float = 0.4
        self._nav_repeat_interval_s: float = 0.1

        # ---- Poll rate limit ------------------------------------
        # Re-read I2C at most this often. The main loop calls poll()
        # at the camera frame rate (~24 Hz), but several queued
        # events may be drained per frame without re-reading.
        self._poll_interval_s: float = 0.020   # 50 Hz default
        self._last_poll_time: float = -1.0

        # ---- Zoom pot config ------------------------------------
        self._zoom_buckets: int = 16
        self._zoom_invert: bool = False
        self._zoom_debounce_reads: int = 2

        # ---- Timed buttons (short vs long press) -----------------
        # Maps input name → tracking state. Loaded from
        # ``snapshot_button`` + ``power_button`` config sections.
        # Unlike regular ``button_events`` (which fire on press
        # edge), timed buttons fire on *release* (short press) or
        # when the hold threshold is reached (long press).
        self._timed_buttons: dict[str, _TimedButtonState] = {}

        # ---- Runtime state --------------------------------------
        # Last observed input word from the expander, used for
        # edge detection. -1 = "never read".
        self._last_inputs: int = -1
        # Per-input press start time (None = not pressed).
        self._press_started: dict[str, Optional[float]] = {}
        # Per-nav next repeat time (None = not held / not started).
        self._nav_next_repeat: dict[str, Optional[float]] = {}

        # ---- Zoom pot tracking ----------------------------------
        # Last bucket we emitted at. None = no read yet.
        self._zoom_last_bucket: Optional[int] = None
        # Candidate bucket (waiting to clear debounce) and how many
        # consecutive reads we've seen it.
        self._zoom_candidate_bucket: Optional[int] = None
        self._zoom_candidate_reads: int = 0

        # ---- Outbound event queue --------------------------------
        self._pending: deque[AppEvent] = deque()

        # ---- Error throttling -----------------------------------
        self._io_failure_logged: bool = False

        self._load_config()

    # ----- ControlsHAL interface -------------------------------------

    def start(self) -> None:
        """Initialise the I/O expander (and probe the ADC if present)."""
        try:
            self._io_expander.init()
        except I2CError:
            logger.exception("TCA6416A init failed; GPIO controls unavailable")
            raise

        if self._adc is not None:
            try:
                self._adc.probe()
                logger.info("MCP3221 ADC present; zoom pot enabled")
            except I2CError as exc:
                logger.warning(
                    "MCP3221 probe failed (%s); zoom pot disabled until wired", exc
                )
                self._adc = None

    def stop(self) -> None:
        """Best-effort: turn the LEDs off."""
        outputs = self._hardware_config.get("tca6416a_pins", {}).get("outputs", {})
        try:
            for pin_name in outputs.values():
                try:
                    index = pin_name_to_index(pin_name)
                except ValueError:
                    continue
                # HIGH = LED off for common-anode wiring.
                self._io_expander.write_output_pin(index, True)
        except I2CError:
            logger.debug("could not clear LED state on stop (probably OK)")

    def poll(self) -> AppEvent:
        # Pump the cv2 event queue so cv2.imshow() in the app controller
        # actually refreshes the display. cv2.imshow needs periodic
        # waitKey/pollKey calls to update the window; the main loop
        # (see app_controller._tick) expects controls.poll() to do this.
        # MockControls gets it for free via its keyboard reader;
        # GPIOControls has to do it explicitly. Without this, the
        # window stays black even though imshow is called every frame.
        self._pump_display_events()

        # If we already have queued events, drain one without an I2C read.
        if self._pending:
            return self._pending.popleft()

        now = self._clock()
        if (
            self._last_poll_time >= 0
            and (now - self._last_poll_time) < self._poll_interval_s
        ):
            return AppEvent.NONE
        self._last_poll_time = now

        # --- Read the I/O expander ----------------------------------
        try:
            inputs = self._io_expander.read_inputs()
        except I2CError as exc:
            if not self._io_failure_logged:
                logger.warning("TCA6416A read failed (%s); skipping poll", exc)
                self._io_failure_logged = True
            return AppEvent.NONE
        else:
            if self._io_failure_logged:
                logger.info("TCA6416A read recovered")
                self._io_failure_logged = False

        # --- Detect transitions vs previous read --------------------
        if self._last_inputs == -1:
            # First read: don't generate spurious events for any
            # buttons that happen to be held while we initialised.
            self._last_inputs = inputs
        else:
            self._process_transitions(self._last_inputs, inputs, now)
            self._last_inputs = inputs

        # --- Time-based events: long-press, nav repeat --------------
        self._process_long_press(inputs, now)
        self._process_nav_repeat(inputs, now)

        # --- Read the ADC ------------------------------------------
        if self._adc is not None:
            self._process_zoom_pot()

        # Cap queue length so a wedged button can't grow it forever.
        while len(self._pending) > self._MAX_QUEUED_EVENTS:
            self._pending.popleft()

        if self._pending:
            return self._pending.popleft()
        return AppEvent.NONE

    # ----- transition processing -------------------------------------

    def _process_transitions(self, prev: int, curr: int, now: float) -> None:
        # Walk every input pin we care about, compare its bit before
        # and after, dispatch on the edge.
        for name, pin_index in self._input_pins.items():
            mask = self._input_mask[name]
            was_pressed = (prev & mask) == 0
            is_pressed = (curr & mask) == 0

            if not was_pressed and is_pressed:
                # Press edge (HIGH → LOW). Remember when it started.
                self._press_started[name] = now
                self._on_press(name, now)
            elif was_pressed and not is_pressed:
                # Release edge (LOW → HIGH).
                self._on_release(name, now)
                self._press_started[name] = None

    def _on_press(self, name: str, now: float) -> None:
        # Timed buttons don't fire on press — they fire on release
        # (short) or after the hold threshold (long).
        if name in self._timed_buttons:
            self._timed_buttons[name].long_fired = False
            return

        # Action buttons fire their event immediately on press.
        if name in self._button_events:
            self._queue(self._button_events[name])

        # Nav directions: emit once on press; schedule the next
        # repeat if held.
        if name in self._nav_events:
            self._queue(self._nav_events[name])
            if self._nav_repeat_enabled:
                self._nav_next_repeat[name] = now + self._nav_repeat_initial_delay_s
            else:
                self._nav_next_repeat[name] = None

    def _on_release(self, name: str, now: float) -> None:
        # Timed buttons: short event fires on release if the hold
        # threshold wasn't reached; if it was, long already fired.
        tb = self._timed_buttons.get(name)
        if tb is not None:
            if not tb.long_fired and tb.short_event is not None:
                press_start = self._press_started.get(name)
                if press_start is not None:
                    if (now - press_start) < tb.threshold_s:
                        self._queue(tb.short_event)
            tb.long_fired = False
            return

        # Nav release: cancel any pending repeat.
        if name in self._nav_events:
            self._nav_next_repeat[name] = None

    def _process_long_press(self, inputs: int, now: float) -> None:
        for name, tb in self._timed_buttons.items():
            if tb.long_fired or tb.long_event is None:
                continue
            press_start = self._press_started.get(name)
            if press_start is None:
                continue
            mask = self._input_mask.get(name)
            if mask is None or (inputs & mask) != 0:
                continue   # button not currently pressed
            if (now - press_start) >= tb.threshold_s:
                self._queue(tb.long_event)
                tb.long_fired = True

    def _process_nav_repeat(self, inputs: int, now: float) -> None:
        if not self._nav_repeat_enabled:
            return
        for name, event in self._nav_events.items():
            next_at = self._nav_next_repeat.get(name)
            if next_at is None:
                continue
            mask = self._input_mask[name]
            if (inputs & mask) != 0:
                # Not pressed any more; release handler clears the timer.
                continue
            if now >= next_at:
                self._queue(event)
                self._nav_next_repeat[name] = now + self._nav_repeat_interval_s

    # ----- zoom pot --------------------------------------------------

    def _process_zoom_pot(self) -> None:
        try:
            raw = self._adc.read_raw()  # type: ignore[union-attr]
        except I2CError as exc:
            logger.warning("MCP3221 read failed (%s); disabling zoom pot", exc)
            self._adc = None
            return

        # Quantise to [0, buckets-1]. Use floor division so the
        # bucket boundaries are evenly spaced across 0..4095.
        bucket = (raw * self._zoom_buckets) // (MCP3221.MAX_RAW + 1)
        if bucket >= self._zoom_buckets:
            bucket = self._zoom_buckets - 1
        if self._zoom_invert:
            bucket = (self._zoom_buckets - 1) - bucket

        # Debounce: only accept the new bucket once we've seen it
        # for ``debounce_reads`` consecutive polls.
        if bucket == self._zoom_candidate_bucket:
            self._zoom_candidate_reads += 1
        else:
            self._zoom_candidate_bucket = bucket
            self._zoom_candidate_reads = 1

        if self._zoom_candidate_reads < self._zoom_debounce_reads:
            return

        # First-ever stable read just anchors the previous bucket.
        if self._zoom_last_bucket is None:
            self._zoom_last_bucket = bucket
            return

        if bucket > self._zoom_last_bucket:
            for _ in range(bucket - self._zoom_last_bucket):
                self._queue(AppEvent.ZOOM_IN)
        elif bucket < self._zoom_last_bucket:
            for _ in range(self._zoom_last_bucket - bucket):
                self._queue(AppEvent.ZOOM_OUT)
        self._zoom_last_bucket = bucket

    # ----- helpers ---------------------------------------------------

    def _queue(self, event: AppEvent) -> None:
        if event is not AppEvent.NONE:
            self._pending.append(event)

    def _pump_display_events(self) -> None:
        """Pump cv2's GUI event queue so the most recent imshow paints.

        Lazy-imported so this module remains importable in environments
        without cv2. Any failure (no window yet, headless CI, missing
        backend) is swallowed silently — it just means the display
        wasn't refreshed this tick, which is harmless.
        """
        try:
            import cv2
            cv2.waitKey(1)
        except Exception:  # noqa: BLE001 — intentionally broad
            pass

    # ----- config loading --------------------------------------------

    def _load_config(self) -> None:
        cfg = self._hardware_config

        # --- TCA6416A pin map -----------------------------------
        pin_map = cfg.get("tca6416a_pins", {}).get("inputs", {})
        for name, pin_name in pin_map.items():
            try:
                index = pin_name_to_index(pin_name)
            except ValueError as exc:
                logger.warning(
                    "Skipping input %r: %s", name, exc
                )
                continue
            self._input_pins[name] = index
            self._input_mask[name] = 1 << index
            self._press_started[name] = None

        # --- Action buttons ------------------------------------
        for name, event_name in cfg.get("button_events", {}).items():
            if name not in self._input_pins:
                logger.warning(
                    "button_events references unknown input %r; ignoring", name
                )
                continue
            event = self._parse_event(event_name, context=f"button {name!r}")
            if event is not None:
                self._button_events[name] = event

        # --- Nav switch ----------------------------------------
        for name, event_name in cfg.get("nav_events", {}).items():
            if name not in self._input_pins:
                logger.warning(
                    "nav_events references unknown input %r; ignoring", name
                )
                continue
            event = self._parse_event(event_name, context=f"nav {name!r}")
            if event is not None:
                self._nav_events[name] = event
                self._nav_next_repeat[name] = None

        # --- Nav repeat ---------------------------------------
        repeat_cfg = cfg.get("nav_repeat", {})
        self._nav_repeat_enabled = bool(repeat_cfg.get("enabled", True))
        try:
            self._nav_repeat_initial_delay_s = (
                float(repeat_cfg.get("initial_delay_ms", 400)) / 1000.0
            )
            self._nav_repeat_interval_s = (
                float(repeat_cfg.get("interval_ms", 100)) / 1000.0
            )
        except (TypeError, ValueError):
            logger.warning(
                "nav_repeat has non-numeric timing; using defaults 400ms/100ms"
            )
            self._nav_repeat_initial_delay_s = 0.4
            self._nav_repeat_interval_s = 0.1

        # --- Timed buttons (short / long press) -------------------
        # Power button: hardcoded events (backward compat).
        power_cfg = cfg.get("power_button", {})
        power_input = power_cfg.get("input")
        if power_input:
            if power_input in self._input_pins:
                try:
                    threshold_ms = int(
                        power_cfg.get("long_press_threshold_ms", 1000)
                    )
                except (TypeError, ValueError):
                    logger.warning(
                        "power_button.long_press_threshold_ms invalid; "
                        "using 1000"
                    )
                    threshold_ms = 1000
                self._timed_buttons[power_input] = _TimedButtonState(
                    short_event=AppEvent.POWER_SHORT_PRESS,
                    long_event=AppEvent.POWER_LONG_PRESS,
                    threshold_s=threshold_ms / 1000.0,
                )
                self._press_started.setdefault(power_input, None)
            else:
                logger.warning(
                    "power_button.input %r is not a defined input; "
                    "power button disabled",
                    power_input,
                )

        # Snapshot button: tap=freeze, hold=capture (configurable).
        snap_cfg = cfg.get("snapshot_button", {})
        snap_input = snap_cfg.get("input")
        if snap_input:
            if snap_input in self._input_pins:
                short_ev = self._parse_event(
                    snap_cfg.get("short_event", "FREEZE_TOGGLE"),
                    "snapshot_button.short_event",
                )
                long_ev = self._parse_event(
                    snap_cfg.get("long_event", "CAPTURE_IMAGE"),
                    "snapshot_button.long_event",
                )
                if short_ev and long_ev:
                    try:
                        threshold_ms = int(
                            snap_cfg.get("long_press_threshold_ms", 3000)
                        )
                    except (TypeError, ValueError):
                        logger.warning(
                            "snapshot_button.long_press_threshold_ms "
                            "invalid; using 3000"
                        )
                        threshold_ms = 3000
                    self._timed_buttons[snap_input] = _TimedButtonState(
                        short_event=short_ev,
                        long_event=long_ev,
                        threshold_s=threshold_ms / 1000.0,
                    )
                    self._press_started.setdefault(snap_input, None)
                    # Warn if it's also in button_events (double-fire risk).
                    if snap_input in self._button_events:
                        logger.warning(
                            "snapshot_button.input %r is also in "
                            "button_events; remove it from button_events "
                            "to avoid double-firing",
                            snap_input,
                        )
            else:
                logger.warning(
                    "snapshot_button.input %r is not a defined input; "
                    "snapshot button disabled",
                    snap_input,
                )

        # Status button: long press shows fullscreen status overlay.
        # No short_event (a tap does nothing).
        status_cfg = cfg.get("status_button", {})
        status_input = status_cfg.get("input")
        if status_input:
            if status_input in self._input_pins:
                short_raw = status_cfg.get("short_event")
                long_raw = status_cfg.get("long_event")
                short_ev = (
                    self._parse_event(short_raw, "status_button.short_event")
                    if short_raw else None
                )
                long_ev = (
                    self._parse_event(long_raw, "status_button.long_event")
                    if long_raw else None
                )
                if short_ev is not None or long_ev is not None:
                    try:
                        threshold_ms = int(
                            status_cfg.get("long_press_threshold_ms", 5000)
                        )
                    except (TypeError, ValueError):
                        logger.warning(
                            "status_button.long_press_threshold_ms "
                            "invalid; using 5000"
                        )
                        threshold_ms = 5000
                    self._timed_buttons[status_input] = _TimedButtonState(
                        short_event=short_ev,
                        long_event=long_ev,
                        threshold_s=threshold_ms / 1000.0,
                    )
                    self._press_started.setdefault(status_input, None)
            else:
                logger.warning(
                    "status_button.input %r is not a defined input; "
                    "status button disabled",
                    status_input,
                )

        # --- Poll interval ------------------------------------
        io_cfg = (
            cfg.get("i2c", {}).get("devices", {}).get("io_expander", {})
        )
        try:
            self._poll_interval_s = (
                float(io_cfg.get("poll_interval_ms", 20)) / 1000.0
            )
        except (TypeError, ValueError):
            logger.warning("io_expander.poll_interval_ms invalid; using 20ms")
            self._poll_interval_s = 0.020

        # --- Zoom pot calibration -----------------------------
        zoom_cfg = cfg.get("zoom_pot", {})
        try:
            self._zoom_buckets = max(2, int(zoom_cfg.get("buckets", 16)))
        except (TypeError, ValueError):
            self._zoom_buckets = 16
        self._zoom_invert = bool(zoom_cfg.get("invert", False))
        try:
            self._zoom_debounce_reads = max(
                1, int(zoom_cfg.get("debounce_reads", 2))
            )
        except (TypeError, ValueError):
            self._zoom_debounce_reads = 2

        timed_names = ", ".join(
            f"{n}(short={getattr(tb.short_event, 'name', 'none')}, "
            f"long={getattr(tb.long_event, 'name', 'none')})"
            for n, tb in self._timed_buttons.items()
        ) or "none"
        logger.info(
            "GPIOControls: %d inputs, %d action buttons, %d nav directions, "
            "timed=[%s], zoom_pot=%s",
            len(self._input_pins),
            len(self._button_events),
            len(self._nav_events),
            timed_names,
            "enabled" if self._adc is not None else "disabled",
        )

    def _parse_event(self, raw: Any, context: str) -> Optional[AppEvent]:
        if not isinstance(raw, str):
            logger.warning(
                "%s: event must be a string, got %r", context, raw
            )
            return None
        try:
            return AppEvent[raw]
        except KeyError:
            logger.warning("%s: unknown AppEvent %r", context, raw)
            return None