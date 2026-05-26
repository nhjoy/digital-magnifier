"""
Application events.

All inputs (mock keyboard, real GPIO buttons, the navigation switch, the
zoom potentiometer, timers, and system signals) are normalised into
AppEvent values before they reach the state machine or the app
controller. Hardware-level details stay inside the HAL; everything above
the HAL only sees AppEvent.

When adding a new event:
  1. Add it here.
  2. Decide whether it is a transition event, an in-state event, or a
     global event, and register it in state_machine.py accordingly.
  3. Map a mock keyboard key (or real GPIO pin) to it in
     config/hardware_pins.yaml.
"""

from enum import Enum, auto


class AppEvent(Enum):
    # No input detected on this poll. Returned by the controls HAL when
    # the user has not pressed anything; the state machine treats this
    # as a silent no-op.
    NONE = auto()

    # --- lifecycle ----------------------------------------------------
    STARTUP_COMPLETE = auto()

    # --- in-state controls (do not change state by themselves) -------
    ZOOM_IN = auto()
    ZOOM_OUT = auto()
    PAN_UP = auto()
    PAN_DOWN = auto()
    PAN_LEFT = auto()
    PAN_RIGHT = auto()
    FREEZE_TOGGLE = auto()
    FILTER_NEXT = auto()
    RESET_VIEW = auto()

    # --- one-shot actions --------------------------------------------
    CAPTURE_IMAGE = auto()

    # --- mode switches ------------------------------------------------
    GALLERY_OPEN = auto()
    BACK = auto()

    # --- power button -------------------------------------------------
    # Short press toggles the menu, long press triggers a clean shutdown.
    POWER_SHORT_PRESS = auto()
    POWER_LONG_PRESS = auto()

    # --- status overlay ----------------------------------------------
    # Long press of ADD3 (5s) toggles a fullscreen status screen
    # showing battery, state, zoom, filter, etc. in big text.
    STATUS_TOGGLE = auto()

    # --- developer convenience ---------------------------------------
    # The mock keyboard maps 'q' to this so the dev can exit without
    # exercising the power-long-press code path during testing.
    QUIT = auto()