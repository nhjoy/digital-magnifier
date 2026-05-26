"""
State machine for the Digital Magnifier application.

This module owns the rules for which application states are valid and
which events cause transitions between them. It does NOT execute any
side effects (camera, image processing, file I/O, audio feedback);
those are the app controller's responsibility. Keeping the state
machine free of side effects makes it trivial to test in isolation
and keeps the architecture's HAL → Core → Processing → UI boundary
honest.

Three categories of events
--------------------------
From any given state, an incoming event falls into one of:

  1. Transition event: defined in TRANSITION_TABLE or GLOBAL_TRANSITIONS.
     ``handle()`` returns the new AppState. The caller runs entry
     actions for the new state (e.g., starting the capture-flash timer).

  2. In-state event: listed in IN_STATE_EVENTS for the current state.
     The event is legal here but does not change state (e.g., ZOOM_IN
     in LIVE_VIEW). ``handle()`` returns None. The caller (app
     controller) is expected to act on the event in place.

  3. Invalid event: anything else. ``handle()`` returns None and logs
     a warning, but does not raise. This is deliberate: a flaky button
     bounce or a stray key press should never crash the device for a
     child mid-reading.

Timer-driven transitions
------------------------
There are no timers inside the state machine. Timed transitions (e.g.,
CAPTURE_FLASH auto-returning to LIVE_VIEW after the configured flash
duration) are driven by the app controller calling
``force_transition``. This keeps the state machine purely functional
in input/output terms.

Safety rule
-----------
``POWER_LONG_PRESS`` is registered as a global transition to SHUTDOWN.
It will be honoured from every state, including CAPTURE_FLASH. This is
intentional: the child must always be able to power the device down,
even mid-animation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

from digital_magnifier.core.events import AppEvent


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# States
# --------------------------------------------------------------------------- #
class AppState(Enum):
    """All top-level states the application can be in."""

    STARTUP = auto()
    LIVE_VIEW = auto()
    FROZEN_VIEW = auto()
    CAPTURE_FLASH = auto()
    GALLERY_VIEW = auto()
    MENU_VIEW = auto()
    SHUTDOWN = auto()


@dataclass(frozen=True)
class Transition:
    """A record of a single state transition.

    Stored in the state machine's history for debugging and consulted
    by tests. ``event`` is ``AppEvent.NONE`` for forced transitions
    (e.g., a timer firing) since there was no user input.
    """

    from_state: AppState
    event: AppEvent
    to_state: AppState


# --------------------------------------------------------------------------- #
# Transition rules
# --------------------------------------------------------------------------- #
# Mapping of (current_state, event) -> new_state. Anything not in this
# table (and not in IN_STATE_EVENTS) is treated as invalid for the
# current state.
TRANSITION_TABLE: dict[tuple[AppState, AppEvent], AppState] = {
    # --- STARTUP --------------------------------------------------------
    (AppState.STARTUP, AppEvent.STARTUP_COMPLETE): AppState.LIVE_VIEW,

    # --- LIVE_VIEW ------------------------------------------------------
    (AppState.LIVE_VIEW, AppEvent.FREEZE_TOGGLE):     AppState.FROZEN_VIEW,
    (AppState.LIVE_VIEW, AppEvent.CAPTURE_IMAGE):     AppState.CAPTURE_FLASH,
    (AppState.LIVE_VIEW, AppEvent.GALLERY_OPEN):      AppState.GALLERY_VIEW,
    (AppState.LIVE_VIEW, AppEvent.POWER_SHORT_PRESS): AppState.MENU_VIEW,

    # --- FROZEN_VIEW ----------------------------------------------------
    (AppState.FROZEN_VIEW, AppEvent.FREEZE_TOGGLE):     AppState.LIVE_VIEW,
    (AppState.FROZEN_VIEW, AppEvent.CAPTURE_IMAGE):     AppState.CAPTURE_FLASH,
    (AppState.FROZEN_VIEW, AppEvent.GALLERY_OPEN):      AppState.GALLERY_VIEW,
    (AppState.FROZEN_VIEW, AppEvent.POWER_SHORT_PRESS): AppState.MENU_VIEW,

    # --- CAPTURE_FLASH --------------------------------------------------
    # No manual transitions; the app controller calls force_transition
    # to return to LIVE_VIEW once the configured flash duration elapses.

    # --- GALLERY_VIEW ---------------------------------------------------
    (AppState.GALLERY_VIEW, AppEvent.GALLERY_OPEN): AppState.LIVE_VIEW,  # toggle
    (AppState.GALLERY_VIEW, AppEvent.BACK):         AppState.LIVE_VIEW,

    # --- MENU_VIEW ------------------------------------------------------
    (AppState.MENU_VIEW, AppEvent.POWER_SHORT_PRESS): AppState.LIVE_VIEW,  # toggle
    (AppState.MENU_VIEW, AppEvent.BACK):              AppState.LIVE_VIEW,
}


# Events that are legal in a state but do not cause a transition. The
# app controller acts on them in place (zoom, pan, filter cycle,
# reset). Every AppState must have an entry here, even if empty;
# tests enforce that invariant.
IN_STATE_EVENTS: dict[AppState, frozenset[AppEvent]] = {
    AppState.STARTUP: frozenset(),

    AppState.LIVE_VIEW: frozenset({
        AppEvent.ZOOM_IN,
        AppEvent.ZOOM_OUT,
        AppEvent.PAN_UP,
        AppEvent.PAN_DOWN,
        AppEvent.PAN_LEFT,
        AppEvent.PAN_RIGHT,
        AppEvent.FILTER_NEXT,
        AppEvent.RESET_VIEW,
        AppEvent.STATUS_TOGGLE,
    }),

    AppState.FROZEN_VIEW: frozenset({
        AppEvent.ZOOM_IN,
        AppEvent.ZOOM_OUT,
        AppEvent.PAN_UP,
        AppEvent.PAN_DOWN,
        AppEvent.PAN_LEFT,
        AppEvent.PAN_RIGHT,
        AppEvent.FILTER_NEXT,
        AppEvent.RESET_VIEW,
        AppEvent.STATUS_TOGGLE,
    }),

    # During the brief capture-flash, ignore all user input except
    # global shutdown. The flash is short enough (configurable, ~500ms
    # default) that no useful interaction is missed.
    AppState.CAPTURE_FLASH: frozenset(),

    # Gallery interaction (MVP 0.4). Same six view-manipulation
    # events as live/frozen, plus CAPTURE_IMAGE which the gallery
    # repurposes as "delete with confirmation" (snapshot button has
    # no other useful meaning in gallery view). The app controller
    # delegates these to the Gallery instance, which owns its own
    # zoom / pan / filter state separate from the live view.
    AppState.GALLERY_VIEW: frozenset({
        AppEvent.ZOOM_IN,
        AppEvent.ZOOM_OUT,
        AppEvent.PAN_UP,
        AppEvent.PAN_DOWN,
        AppEvent.PAN_LEFT,
        AppEvent.PAN_RIGHT,
        AppEvent.FILTER_NEXT,
        AppEvent.RESET_VIEW,
        AppEvent.CAPTURE_IMAGE,
    }),

    # Same idea for the menu stub.
    AppState.MENU_VIEW: frozenset({
        AppEvent.PAN_UP,
        AppEvent.PAN_DOWN,
    }),

    AppState.SHUTDOWN: frozenset(),
}


# Transitions that fire from any state. Used for safety guarantees
# such as power-off and developer-convenience quit.
GLOBAL_TRANSITIONS: dict[AppEvent, AppState] = {
    AppEvent.POWER_LONG_PRESS: AppState.SHUTDOWN,
    AppEvent.QUIT:             AppState.SHUTDOWN,
}


# --------------------------------------------------------------------------- #
# StateMachine
# --------------------------------------------------------------------------- #
class StateMachine:
    """Event-driven state machine for the magnifier app.

    Typical use in the app controller:

        sm = StateMachine(initial_state=AppState.STARTUP)
        sm.handle(AppEvent.STARTUP_COMPLETE)
        ...
        new_state = sm.handle(event)
        if new_state is not None:
            self._on_enter_state(new_state)
        else:
            # The event was either in-state (zoom, pan, ...) or invalid;
            # the controller's dispatch dict handles the in-state case.
            self._dispatch_in_state(event)
    """

    def __init__(
        self,
        initial_state: AppState = AppState.STARTUP,
        log_transitions: bool = True,
    ) -> None:
        self._current_state: AppState = initial_state
        self._previous_state: AppState | None = None
        self._log_transitions: bool = log_transitions
        self._history: list[Transition] = []

    # ----- public API ------------------------------------------------------

    @property
    def current_state(self) -> AppState:
        return self._current_state

    @property
    def previous_state(self) -> AppState | None:
        return self._previous_state

    @property
    def history(self) -> list[Transition]:
        """A copy of the transitions taken since construction or last reset."""
        return list(self._history)

    def is_in(self, state: AppState) -> bool:
        """Whether the machine is currently in ``state``."""
        return self._current_state == state

    def can_handle(self, event: AppEvent) -> bool:
        """Whether ``event`` is legal in the current state.

        Returns True if the event would either cause a transition or be
        accepted as an in-state event. Useful for callers that want to
        decide whether to play a button-click sound or a "blocked" beep
        without actually firing the event.
        """
        if event == AppEvent.NONE:
            return True
        if event in GLOBAL_TRANSITIONS:
            return True
        if (self._current_state, event) in TRANSITION_TABLE:
            return True
        if event in IN_STATE_EVENTS.get(self._current_state, frozenset()):
            return True
        return False

    def handle(self, event: AppEvent) -> AppState | None:
        """Process an event.

        Returns the new state if a transition occurred, or None
        otherwise. A return value of None covers two cases:
          - The event is valid in the current state but does not cause
            a transition (an in-state event such as ZOOM_IN).
          - The event is invalid in the current state (logged as a
            warning so the controller does not have to).

        Callers that need to distinguish the two cases should consult
        ``can_handle`` before calling ``handle``.
        """
        if event == AppEvent.NONE:
            return None

        # Global transitions take precedence so safety events (shutdown,
        # developer quit) are honoured from every state.
        if event in GLOBAL_TRANSITIONS:
            target = GLOBAL_TRANSITIONS[event]
            return self._transition_to(target, event)

        # Per-state transitions.
        target = TRANSITION_TABLE.get((self._current_state, event))
        if target is not None:
            return self._transition_to(target, event)

        # In-state events are legal but do not change state.
        if event in IN_STATE_EVENTS.get(self._current_state, frozenset()):
            return None

        # Anything else is invalid in this state. Log and ignore.
        logger.warning(
            "Ignored event %s in state %s (no transition or in-state rule)",
            event.name,
            self._current_state.name,
        )
        return None

    def force_transition(
        self,
        new_state: AppState,
        reason: str = "forced",
    ) -> AppState:
        """Unconditionally move to ``new_state``.

        Intended for timer-driven transitions (e.g., the
        ``CAPTURE_FLASH`` flash duration elapsing) and for setting up
        test fixtures. ``reason`` is included in the log line for
        traceability.
        """
        previous = self._current_state
        self._previous_state = previous
        self._current_state = new_state
        # AppEvent.NONE in the history record flags the transition as
        # non-event-driven.
        self._history.append(Transition(previous, AppEvent.NONE, new_state))
        if self._log_transitions:
            logger.info(
                "Forced transition: %s -> %s (%s)",
                previous.name,
                new_state.name,
                reason,
            )
        return new_state

    def reset(self, to_state: AppState = AppState.STARTUP) -> None:
        """Reset to a known state and clear transition history.

        Intended for tests and emergency recovery. Does not log.
        """
        self._current_state = to_state
        self._previous_state = None
        self._history.clear()

    # ----- internals ------------------------------------------------------

    def _transition_to(
        self,
        new_state: AppState,
        triggering_event: AppEvent,
    ) -> AppState:
        previous = self._current_state
        self._previous_state = previous
        self._current_state = new_state
        self._history.append(Transition(previous, triggering_event, new_state))
        if self._log_transitions:
            logger.info(
                "Transition: %s --[%s]--> %s",
                previous.name,
                triggering_event.name,
                new_state.name,
            )
        return new_state