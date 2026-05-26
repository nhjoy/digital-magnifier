"""Main application controller for the Digital Magnifier.

Orchestrates the camera HAL, controls HAL, image processing, UI
overlay, image saver, and state machine. Runs the main loop until
the state machine reaches ``AppState.SHUTDOWN``.

Architecture
------------
::

      ControlsHAL          CameraSensor
           │                    │
           │ poll()              │ get_frame()
           ▼                    ▼
       AppEvent             np.ndarray
           │                    │
           │                    │
           ▼                    ▼
        ┌─────────────────────────────┐
        │       MagnifierApp          │
        │   ┌─────────────────────┐   │
        │   │   StateMachine      │   │
        │   └─────────────────────┘   │
        │   ┌─────────────────────┐   │
        │   │  in-state dispatch  │   │
        │   └─────────────────────┘   │
        │   ┌─────────────────────┐   │
        │   │  on-enter dispatch  │   │
        │   └─────────────────────┘   │
        └─────────────────────────────┘
                     │
                     ▼
              cv2.imshow + ImageSaver

Design notes
------------
- All hardware-touching dependencies are injected through the
  constructor. The controller never imports a specific camera or
  controls implementation, only the ABCs. Swapping mock for real
  hardware in MVP 0.2 / 0.3 is one-line changes in ``main.py``.

- The state machine owns "what state we are in"; the controller
  owns "what to do while in that state". The dispatch tables
  ``_in_state_handlers`` and ``_on_enter_handlers`` keep this
  explicit and easy to extend.

- In ``FROZEN_VIEW`` the processed display frame is cached. The
  cache is invalidated by any change to zoom, pan, or filter, and
  cleared on exit. This satisfies the blueprint rule "Lower
  processing during frozen mode" — re-zooming a 720p frame at
  24 FPS allocates ~60 MB/s; the cache reduces that to allocations
  on input only.

- Pan offsets reset to zero when zoom returns to 1× (panning has
  no meaning at 1× and stale pan would jump the view back on the
  next zoom in).

- Capture flash is a timed transition driven from the main loop:
  on entry, the flash start time is recorded and the captured
  frame is saved; each tick checks the elapsed time and force-
  transitions back to LIVE_VIEW when the flash duration expires.

- Every tick is wrapped in try/except. A single bad frame, a
  transient camera error, or a misbehaving filter cannot crash the
  device — the consecutive-failure counter triggers a clean
  shutdown only after sustained trouble.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import cv2
import numpy as np

from digital_magnifier.core.events import AppEvent
from digital_magnifier.core.gallery import Gallery
from digital_magnifier.core.state_machine import AppState, StateMachine
from digital_magnifier.hal.camera_base import CameraSensor
from digital_magnifier.hal.controls_base import ControlsHAL
from digital_magnifier.processing.magnifier import apply_zoom
from digital_magnifier.processing.vision_filters import apply_filter
from digital_magnifier.storage.image_saver import ImageSaver


logger = logging.getLogger(__name__)


class MagnifierApp:
    """The application orchestrator."""

    WINDOW_NAME: str = "Digital Magnifier"

    # If this many consecutive ticks raise, give up and shut down
    # cleanly. At 24 FPS this is ~1.25 s of sustained failure,
    # which is the threshold beyond which a child would notice
    # the device is stuck.
    MAX_CONSECUTIVE_FAILURES: int = 30

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        camera: CameraSensor,
        controls: ControlsHAL,
        image_saver: ImageSaver,
        config: dict[str, Any],
        *,
        ups: Any = None,
        buzzer: Any = None,
    ) -> None:
        # --- injected dependencies --------------------------------
        self._camera = camera
        self._controls = controls
        self._image_saver = image_saver
        self._config = config

        # --- optional UPS HAT for battery display -----------------
        self._ups = ups
        self._battery_percent: int = -1      # -1 = unknown / no UPS
        self._battery_charging: bool = False
        self._battery_next_read: float = 0.0
        self._BATTERY_POLL_INTERVAL_S: float = 5.0

        # --- battery alert thresholds -----------------------------
        self._battery_warned_50: bool = False
        self._battery_warned_15: bool = False
        self._battery_shutdown_armed: bool = False
        self._low_battery_last_beep: float = 0.0

        # --- optional buzzer for audio feedback -------------------
        self._buzzer = buzzer

        # --- status overlay (toggled by ADD3 long-press) ---------
        self._show_status: bool = False

        # --- loading splash (startup_time set here; duration
        #     read from app config below once app_cfg exists) -----
        self._startup_time: float = time.monotonic()

        # --- read app config --------------------------------------
        app_cfg = config.get("app", {})
        self._target_fps: float = float(app_cfg.get("target_fps", 24))
        self._show_overlay: bool = bool(app_cfg.get("show_overlay", True))
        self._fullscreen: bool = bool(app_cfg.get("fullscreen", False))
        self._SPLASH_DURATION_S: float = float(
            app_cfg.get("splash_duration_s", 5.0)
        )

        mag_cfg = config.get("magnifier", {})
        self._zoom_min: float = float(mag_cfg.get("min_zoom", 1.0))
        self._zoom_max: float = float(mag_cfg.get("max_zoom", 8.0))
        self._zoom_step: float = float(mag_cfg.get("zoom_step", 0.25))
        self._zoom_default: float = float(mag_cfg.get("default_zoom", 1.0))
        self._pan_step: float = float(mag_cfg.get("pan_step", 0.1))

        filter_cfg = config.get("filters", {})
        self._filter_default: str = str(filter_cfg.get("default", "normal"))
        self._filters_available: list[str] = list(
            filter_cfg.get("available", ["normal"])
        )
        # Per-filter parameters — passed to apply_filter as kwargs.
        # Shape: {filter_name: {param: value, ...}}
        self._filter_config: dict[str, dict[str, Any]] = (
            filter_cfg.get("config", {}) or {}
        )

        capture_cfg = config.get("capture", {})
        self._capture_flash_ms: float = float(
            capture_cfg.get("flash_duration_ms", 500)
        )

        sm_cfg = config.get("state_machine", {})
        log_transitions: bool = bool(sm_cfg.get("log_transitions", True))

        cam_cfg = config.get("camera", {})
        self._cam_width: int = int(cam_cfg.get("width", 1280))
        self._cam_height: int = int(cam_cfg.get("height", 720))

        # --- runtime state ----------------------------------------
        self._state_machine = StateMachine(
            initial_state=AppState.STARTUP,
            log_transitions=log_transitions,
        )

        self._zoom: float = self._zoom_default
        self._pan_x: float = 0.0  # normalized -1..1; ratio of crop offset
        self._pan_y: float = 0.0
        self._filter_index: int = self._filter_index_for(self._filter_default)

        # frozen-frame caching
        self._frozen_raw_frame: np.ndarray | None = None
        self._cached_display_frame: np.ndarray | None = None
        self._cache_dirty: bool = True

        # capture-flash timing
        self._capture_flash_started_at: float = 0.0

        # error handling
        self._consecutive_failures: int = 0

        # display init flag (so we open the window only once)
        self._window_opened: bool = False

        # --- gallery (MVP 0.4) -------------------------------------
        # Owns its own zoom / pan / filter state, separate from the
        # live view, so opening the gallery doesn't carry over the
        # last zoom level. App controller delegates to it whenever
        # the state machine is in GALLERY_VIEW.
        self._gallery = Gallery(
            image_saver=self._image_saver,
            display_width=self._cam_width,
            display_height=self._cam_height,
            filters_available=self._filters_available,
            initial_filter=self._filter_default,
            filter_config=self._filter_config,
            zoom_min=self._zoom_min,
            zoom_max=self._zoom_max,
            zoom_step=self._zoom_step,
            pan_step=self._pan_step,
        )

        # --- dispatch tables --------------------------------------
        self._in_state_handlers: dict[AppEvent, Callable[[], None]] = {
            AppEvent.ZOOM_IN:      self._handle_zoom_in,
            AppEvent.ZOOM_OUT:     self._handle_zoom_out,
            AppEvent.PAN_UP:       self._handle_pan_up,
            AppEvent.PAN_DOWN:     self._handle_pan_down,
            AppEvent.PAN_LEFT:     self._handle_pan_left,
            AppEvent.PAN_RIGHT:    self._handle_pan_right,
            AppEvent.FILTER_NEXT:  self._handle_filter_next,
            AppEvent.RESET_VIEW:   self._handle_reset_view,
            AppEvent.STATUS_TOGGLE: self._handle_status_toggle,
            # In GALLERY_VIEW only: the snapshot button is repurposed
            # as a "delete current image with confirmation" trigger.
            # In LIVE/FROZEN this event is a transition, so this
            # handler doesn't fire from those states.
            AppEvent.CAPTURE_IMAGE: self._handle_capture_image_in_state,
        }

        self._on_enter_handlers: dict[AppState, Callable[[], None]] = {
            AppState.LIVE_VIEW:     self._on_enter_live,
            AppState.FROZEN_VIEW:   self._on_enter_frozen,
            AppState.CAPTURE_FLASH: self._on_enter_capture_flash,
            AppState.GALLERY_VIEW:  self._on_enter_gallery,
            AppState.MENU_VIEW:     self._on_enter_menu,
            AppState.SHUTDOWN:      self._on_enter_shutdown,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run until the state machine reaches SHUTDOWN.

        Opens both HALs as context managers so resources are
        released even on exception. Catches KeyboardInterrupt at
        the top level so Ctrl+C in dev mode exits cleanly.
        """
        logger.info("MagnifierApp starting")

        try:
            with self._camera, self._controls:
                # Leave STARTUP for LIVE_VIEW. We bypass the
                # standard dispatch because there is no event to
                # generate STARTUP_COMPLETE from outside.
                self._state_machine.handle(AppEvent.STARTUP_COMPLETE)
                self._run_handler_safe(self._on_enter_live, "on_enter_live")
                if self._buzzer is not None:
                    self._buzzer.play("startup")
                self._run_loop()
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_loop(self) -> None:
        frame_duration_s = 1.0 / self._target_fps

        while not self._state_machine.is_in(AppState.SHUTDOWN):
            tick_start = time.monotonic()

            try:
                self._tick()
                self._consecutive_failures = 0
            except KeyboardInterrupt:
                logger.info("keyboard interrupt; shutting down")
                self._state_machine.force_transition(
                    AppState.SHUTDOWN, reason="keyboard interrupt"
                )
                break
            except Exception:
                self._consecutive_failures += 1
                logger.exception(
                    "tick failed (consecutive failures: %d)",
                    self._consecutive_failures,
                )
                if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    logger.error(
                        "too many consecutive failures (%d); shutting down",
                        self._consecutive_failures,
                    )
                    self._state_machine.force_transition(
                        AppState.SHUTDOWN, reason="excessive failures"
                    )
                    break

            # FPS cap
            elapsed = time.monotonic() - tick_start
            sleep_s = frame_duration_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _tick(self) -> None:
        """One iteration of the main loop.

        Order is deliberate:
          1. Check timer-driven transitions (capture flash timeout).
          2. Acquire a frame appropriate to the current state.
          3. Render (process + overlay).
          4. Display.
          5. Poll input — this also pumps the cv2 event queue so
             ``imshow`` from step 4 actually appears on screen.
          6. Dispatch the event.
        """
        self._check_timed_transitions()
        self._poll_battery()
        self._check_low_battery_beep()

        frame = self._acquire_frame_for_current_state()
        display = self._render_for_current_state(frame)
        self._display(display)

        event = self._controls.poll()
        self._dispatch(event)

    # ------------------------------------------------------------------
    # Battery
    # ------------------------------------------------------------------

    def _poll_battery(self) -> None:
        """Read battery state from the UPS HAT, at most every few seconds."""
        if self._ups is None:
            return
        now = time.monotonic()
        if now < self._battery_next_read:
            return
        self._battery_next_read = now + self._BATTERY_POLL_INTERVAL_S
        try:
            status = self._ups.charging_status()
            self._battery_percent = self._ups.battery_percent()
            self._battery_charging = bool(status.get("charging", False))
        except Exception:
            logger.debug("UPS battery read failed; will retry", exc_info=True)
            return

        pct = self._battery_percent

        # Reset warnings when charging back above thresholds
        if self._battery_charging or pct > 55:
            self._battery_warned_50 = False
        if self._battery_charging or pct > 20:
            self._battery_warned_15 = False
            self._battery_shutdown_armed = False

        # One-time alert crossing below 50%
        if pct <= 50 and not self._battery_warned_50 and not self._battery_charging:
            self._battery_warned_50 = True
            if self._buzzer is not None:
                self._buzzer.play("battery_50")
            logger.info("Battery at %d%% — below 50%% threshold", pct)

        # Entering 15% warning zone
        if pct <= 15 and not self._battery_warned_15 and not self._battery_charging:
            self._battery_warned_15 = True
            if self._buzzer is not None:
                self._buzzer.play("low_battery")
            logger.warning("Battery at %d%% — charge soon", pct)

        # Critical: initiate shutdown at 10%
        if pct <= 10 and not self._battery_shutdown_armed and not self._battery_charging:
            self._battery_shutdown_armed = True
            logger.critical("Battery at %d%% — initiating safe shutdown", pct)
            self._initiate_low_battery_shutdown()

    def _check_low_battery_beep(self) -> None:
        """Called every tick (~24 Hz). Fires a 1 Hz warning beep when
        battery is between 10% and 15% and not charging."""
        if (
            self._buzzer is None
            or self._battery_percent < 0
            or self._battery_charging
        ):
            return
        if 10 < self._battery_percent <= 15:
            now = time.monotonic()
            if now - self._low_battery_last_beep >= 1.0:
                self._buzzer.beep(0.08)
                self._low_battery_last_beep = now

    def _initiate_low_battery_shutdown(self) -> None:
        """Play shutdown tone, tell UPS to arm power-off, then halt."""
        import os

        if self._buzzer is not None:
            self._buzzer.play("shutdown")

        # Tell UPS HAT to cut power once the Pi is down, and to
        # auto-restart when external power returns.
        if self._ups is not None:
            try:
                self._ups.request_power_off()
                logger.info("UPS HAT power-off armed")
            except Exception:
                logger.warning("Failed to arm UPS power-off", exc_info=True)

        # Trigger OS-level shutdown. This ends the process.
        logger.info("Calling 'sudo shutdown -h now'")
        os.system("sudo shutdown -h now")

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, event: AppEvent) -> None:
        """Route an event through the state machine and handlers."""
        if event == AppEvent.NONE:
            return

        new_state = self._state_machine.handle(event)

        if new_state is not None:
            # Transition happened; run the entry handler.
            handler = self._on_enter_handlers.get(new_state)
            if handler is not None:
                self._run_handler_safe(handler, f"on_enter_{new_state.name}")
            return

        # No transition. If the event is valid in this state and we
        # have an in-state handler for it, run it. (Invalid events
        # are already logged by the state machine.)
        if not self._state_machine.can_handle(event):
            return
        handler = self._in_state_handlers.get(event)
        if handler is not None:
            self._run_handler_safe(handler, f"in_state_{event.name}")

    def _run_handler_safe(self, fn: Callable[[], None], label: str) -> None:
        """Invoke a handler, logging but not propagating exceptions."""
        try:
            fn()
        except Exception:
            logger.exception("handler %s failed", label)

    # ------------------------------------------------------------------
    # On-enter handlers (state transitions)
    # ------------------------------------------------------------------

    def _on_enter_live(self) -> None:
        logger.debug("entering LIVE_VIEW")
        self._frozen_raw_frame = None
        self._cached_display_frame = None
        self._cache_dirty = True

    def _on_enter_frozen(self) -> None:
        logger.debug("entering FROZEN_VIEW")
        try:
            self._frozen_raw_frame = self._camera.get_frame().copy()
        except Exception:
            logger.exception("failed to capture frozen frame; staying live")
            self._state_machine.force_transition(
                AppState.LIVE_VIEW, reason="freeze acquisition failed"
            )
            return
        self._cache_dirty = True

    def _on_enter_capture_flash(self) -> None:
        logger.info("CAPTURE")
        self._capture_flash_started_at = time.monotonic()

        # Acquire a frame to save. If we came from LIVE_VIEW we
        # need a fresh frame; if we came from FROZEN_VIEW the
        # stored frame is what the user was looking at.
        prev = self._state_machine.previous_state
        if prev == AppState.LIVE_VIEW:
            try:
                self._frozen_raw_frame = self._camera.get_frame().copy()
            except Exception:
                logger.exception("camera read failed during capture")
                self._state_machine.force_transition(
                    AppState.LIVE_VIEW, reason="capture acquisition failed"
                )
                return

        if self._frozen_raw_frame is None:
            logger.error("no frame available for capture")
            return

        # Save the processed frame (zoom + filter applied, no UI
        # chrome). That's what the user was reading.
        try:
            processed = self._process_frame(self._frozen_raw_frame)
            self._image_saver.save(processed)
        except Exception:
            logger.exception("failed to save capture")

    def _on_enter_gallery(self) -> None:
        logger.info("entering GALLERY_VIEW")
        self._gallery.open()

    def _on_enter_menu(self) -> None:
        logger.info("entering MENU_VIEW (stub)")

    def _on_enter_shutdown(self) -> None:
        logger.info("entering SHUTDOWN")

    # ------------------------------------------------------------------
    # In-state handlers (no transition)
    # ------------------------------------------------------------------
    #
    # Each handler routes by current state:
    #   * In gallery view, delegate to :class:`Gallery` (which owns
    #     its own zoom / pan / filter state).
    #   * In live or frozen view, mutate the live view state as before.
    #   * Otherwise (startup, menu, capture-flash, shutdown), no-op
    #     — the event won't have reached this handler if the state
    #     machine wasn't expecting it, but defensive guards stop
    #     incidental no-ops from corrupting state.

    def _handle_zoom_in(self) -> None:
        if self._is_in_gallery():
            self._gallery.zoom_in()
            return
        if not self._is_view_state():
            return
        self._zoom = min(self._zoom + self._zoom_step, self._zoom_max)
        self._cache_dirty = True
        logger.debug("zoom -> %.2f", self._zoom)

    def _handle_zoom_out(self) -> None:
        if self._is_in_gallery():
            self._gallery.zoom_out()
            return
        if not self._is_view_state():
            return
        new_zoom = max(self._zoom - self._zoom_step, self._zoom_min)
        # Returning to 1x: pan has no meaning, reset it so a later
        # zoom-in doesn't jump back to the old offset.
        if new_zoom <= 1.0 and self._zoom > 1.0:
            self._pan_x = 0.0
            self._pan_y = 0.0
            logger.debug("zoom returned to 1x; pan reset")
        self._zoom = new_zoom
        self._cache_dirty = True
        logger.debug("zoom -> %.2f", self._zoom)

    def _handle_pan_up(self) -> None:
        if self._is_in_gallery():
            self._gallery.pan_up()
            return
        if not self._is_view_state():
            return
        self._pan_y = max(self._pan_y - self._pan_step, -1.0)
        self._cache_dirty = True

    def _handle_pan_down(self) -> None:
        if self._is_in_gallery():
            self._gallery.pan_down()
            return
        if not self._is_view_state():
            return
        self._pan_y = min(self._pan_y + self._pan_step, 1.0)
        self._cache_dirty = True

    def _handle_pan_left(self) -> None:
        # In gallery, left = previous image (the nav switch is the
        # natural way to flip through captures).
        if self._is_in_gallery():
            self._gallery.prev()
            return
        if not self._is_view_state():
            return
        self._pan_x = max(self._pan_x - self._pan_step, -1.0)
        self._cache_dirty = True

    def _handle_pan_right(self) -> None:
        # In gallery, right = next image.
        if self._is_in_gallery():
            self._gallery.next()
            return
        if not self._is_view_state():
            return
        self._pan_x = min(self._pan_x + self._pan_step, 1.0)
        self._cache_dirty = True

    def _handle_filter_next(self) -> None:
        if self._is_in_gallery():
            self._gallery.filter_next()
            return
        if not self._is_view_state():
            return
        self._filter_index = (
            (self._filter_index + 1) % len(self._filters_available)
        )
        self._cache_dirty = True
        logger.debug("filter -> %s", self._current_filter())

    def _handle_reset_view(self) -> None:
        if self._is_in_gallery():
            self._gallery.reset_view()
            return
        if not self._is_view_state():
            return
        self._zoom = self._zoom_default
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._filter_index = self._filter_index_for(self._filter_default)
        self._cache_dirty = True

    def _handle_status_toggle(self) -> None:
        """Toggle the fullscreen status overlay (battery, state, zoom, ...).

        Triggered by a 5-second hold of ADD3. The overlay sits *on top*
        of the normal render — when active, ``_render_for_current_state``
        returns a status frame instead of the usual view, but the
        underlying state machine is unchanged so zoom / pan / filter
        actions still take effect behind it (visible again when the
        overlay is dismissed).
        """
        self._show_status = not self._show_status
        logger.info(
            "Status overlay %s",
            "shown" if self._show_status else "hidden",
        )

    def _handle_capture_image_in_state(self) -> None:
        """CAPTURE_IMAGE event arrived as an *in-state* event.

        That only happens in GALLERY_VIEW (in LIVE/FROZEN it's a
        transition to CAPTURE_FLASH and never reaches an in-state
        dispatch). In the gallery it means "delete the current image
        with confirmation"; see :meth:`Gallery.request_delete`.
        """
        if self._is_in_gallery():
            self._gallery.request_delete()
        logger.info("view reset")

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------

    def _acquire_frame_for_current_state(self) -> np.ndarray:
        state = self._state_machine.current_state

        if state == AppState.LIVE_VIEW:
            return self._camera.get_frame()

        if state in (AppState.FROZEN_VIEW, AppState.CAPTURE_FLASH):
            # If somehow we got here without a frozen frame
            # (shouldn't happen but be defensive), drop back to
            # live and re-enter from there next tick.
            if self._frozen_raw_frame is None:
                logger.warning(
                    "no frozen frame in %s; falling back to LIVE_VIEW",
                    state.name,
                )
                self._state_machine.force_transition(
                    AppState.LIVE_VIEW, reason="missing frozen frame"
                )
                return self._camera.get_frame()
            return self._frozen_raw_frame

        if state == AppState.GALLERY_VIEW:
            # Gallery owns its own rendering — image load, zoom, pan,
            # filter, and overlays (count, filename, delete prompt).
            return self._gallery.render()

        if state == AppState.MENU_VIEW:
            return self._placeholder_frame("MENU", "Press [P] or [ to exit")

        # STARTUP / SHUTDOWN — black frame is fine.
        return self._placeholder_frame("", "")

    def _placeholder_frame(self, title: str, subtitle: str) -> np.ndarray:
        """Render a centered title/subtitle on a black canvas.

        Used as the temporary display for stubbed states
        (gallery, menu) until those features are implemented in
        MVP 0.4.
        """
        frame = np.zeros((self._cam_height, self._cam_width, 3), dtype=np.uint8)
        if title:
            self._draw_centered_text(
                frame, title, y_offset=0, scale=2.5, thickness=5,
                color=(255, 255, 255),
            )
        if subtitle:
            self._draw_centered_text(
                frame, subtitle, y_offset=60, scale=1.0, thickness=2,
                color=(200, 200, 200),
            )
        return frame

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_for_current_state(self, frame: np.ndarray) -> np.ndarray:
        state = self._state_machine.current_state

        # Loading splash for the first N seconds of the session, so
        # the user sees something while picamera2 initialises (which
        # can take 2-3 seconds on the CM5).
        elapsed = time.monotonic() - self._startup_time
        if elapsed < self._SPLASH_DURATION_S:
            return self._render_splash_screen(frame, elapsed)

        # Status overlay: a long-press of ADD3 toggles this. It
        # replaces the normal render with a big, readable summary
        # of device state. Underlying state machine continues to
        # run, so the view restores cleanly on toggle-off.
        if self._show_status:
            return self._render_status_screen(frame)

        # Gallery short-circuit: the frame returned by
        # _acquire_frame_for_current_state in GALLERY_VIEW is
        # already fully rendered by Gallery.render() — zoom, filter,
        # and gallery-specific overlays already applied. Running it
        # through _process_frame again would zoom-zoom and filter-
        # filter using the live-view state, which is wrong.
        if state == AppState.GALLERY_VIEW:
            return frame

        # Cache fast path for frozen view
        if (
            state == AppState.FROZEN_VIEW
            and not self._cache_dirty
            and self._cached_display_frame is not None
        ):
            return self._cached_display_frame

        processed = self._process_frame(frame)
        display = (
            self._draw_overlay(processed) if self._show_overlay else processed
        )

        if state == AppState.FROZEN_VIEW:
            self._cached_display_frame = display
            self._cache_dirty = False

        if state == AppState.CAPTURE_FLASH:
            display = self._apply_capture_flash(display)

        return display

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply zoom, pan, and filter. Pure of state — easy to test."""
        zoomed = apply_zoom(frame, self._zoom, self._pan_x, self._pan_y)
        return apply_filter(zoomed, self._current_filter(), self._filter_config)

    def _render_splash_screen(
        self, frame: np.ndarray, elapsed: float,
    ) -> np.ndarray:
        """Fullscreen loading splash shown for the first few seconds.

        Solid dark background with the device name and a subtle dot
        progress indicator. We render programmatically (no image
        asset needed) so the first run on a fresh CM5 doesn't fail
        for a missing file.
        """
        h, w = frame.shape[:2]
        splash = np.full((h, w, 3), 0, dtype=np.uint8)
        splash[:, :] = (32, 24, 16)            # BGR: warm dark

        # Title
        self._draw_centered_text(
            splash, "MAGNIFIER", -30,
            scale=3.0, thickness=8, color=(255, 255, 255),
        )
        # Subtitle
        self._draw_centered_text(
            splash, "Loading...", 60,
            scale=1.2, thickness=2, color=(180, 180, 180),
        )

        # Progress dots — fill in as elapsed approaches SPLASH_DURATION_S.
        steps = int((elapsed / self._SPLASH_DURATION_S) * 4)
        steps = max(0, min(3, steps))
        cx, cy = w // 2, h // 2 + 130
        for i in range(3):
            x = cx - 40 + i * 40
            color = (0, 220, 0) if i < steps else (80, 80, 80)
            cv2.circle(splash, (x, cy), 8, color, -1)

        return splash

    def _render_status_screen(self, frame: np.ndarray) -> np.ndarray:
        """Fullscreen device-status overlay (ADD3 long-press).

        Big, high-contrast text for low-vision users. Shows the
        key things a parent or older child might want to glance at
        without entering the menu.
        """
        h, w = frame.shape[:2]
        out = np.full((h, w, 3), 0, dtype=np.uint8)
        out[:, :] = (40, 30, 20)               # BGR: dark navy/grey

        # Title bar
        cv2.rectangle(out, (0, 0), (w, 70), (60, 50, 40), -1)
        self._draw_centered_text(
            out, "DEVICE STATUS", -(h // 2 - 46),
            scale=1.3, thickness=3, color=(255, 255, 255),
        )

        # Battery — biggest line on the screen.
        if self._battery_percent >= 0:
            if self._battery_charging:
                batt_text = f"BATTERY: {self._battery_percent}%  CHARGING"
                batt_color = (200, 200, 0)     # cyan = charging
            else:
                pct = self._battery_percent
                batt_text = f"BATTERY: {pct}%"
                batt_color = (
                    (0, 220, 0) if pct > 50 else
                    (0, 200, 255) if pct > 20 else
                    (0, 0, 220)
                )
        else:
            batt_text = "BATTERY: not detected"
            batt_color = (180, 180, 180)
        self._draw_centered_text(
            out, batt_text, -80,
            scale=1.6, thickness=4, color=batt_color,
        )

        # Lower info lines
        state_name = self._state_machine.current_state.name.replace("_", " ")
        try:
            photos = len(self._image_saver.list_images())
        except Exception:
            photos = 0
        info_lines = [
            ("MODE",   state_name),
            ("ZOOM",   f"{self._zoom:.1f}x"),
            ("FILTER", self._current_filter()),
            ("PHOTOS", f"{photos} saved"),
        ]
        for i, (label, value) in enumerate(info_lines):
            line = f"{label}:  {value}"
            self._draw_centered_text(
                out, line, 10 + i * 60,
                scale=1.1, thickness=3, color=(230, 230, 230),
            )

        # Dismiss hint at the bottom
        self._draw_centered_text(
            out, "Hold ADD3 for 5 seconds to dismiss",
            h // 2 - 60,
            scale=0.7, thickness=2, color=(150, 150, 150),
        )
        return out

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """High-contrast overlay bar for low-vision use.

        Layout (1280×800 display, 50px top bar):

            ┌───────────────────────────────────────────────────┐
            │  LIVE VIEW      2.0x  high_contrast    ██  72%  │
            └───────────────────────────────────────────────────┘

        State label is large and color-coded. Battery bar appears
        only if a UPS HAT is connected. Zoom / filter labels are
        suppressed when at default values (1.0x / normal) so the
        overlay stays uncluttered for the simplest use case.
        """
        h, w = frame.shape[:2]
        out = frame.copy()

        # ----- Dimensions tuned for 1280×800 / 800×480 displays -----
        bar_h = 50
        font = cv2.FONT_HERSHEY_SIMPLEX
        thick = 2

        # Semi-transparent top bar background
        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, out, 0.3, 0, out)

        # ----- State label (left side, large, color-coded) -----------
        state = self._state_machine.current_state
        _STATE_COLORS = {
            AppState.LIVE_VIEW:     (0, 230, 0),     # green
            AppState.FROZEN_VIEW:   (255, 200, 0),   # cyan-ish (BGR)
            AppState.CAPTURE_FLASH: (255, 255, 255), # white
            AppState.GALLERY_VIEW:  (0, 220, 255),   # yellow-ish (BGR)
            AppState.MENU_VIEW:     (200, 200, 200), # grey
            AppState.SHUTDOWN:      (0, 0, 200),     # red
        }
        state_label = state.name.replace("_", " ")
        state_color = _STATE_COLORS.get(state, (255, 255, 255))
        cv2.putText(
            out, state_label, (12, 36),
            font, 0.9, state_color, thick,
        )

        # ----- Zoom + filter (centre, smaller, only when non-default) -
        info_parts = []
        if self._zoom > 1.05:
            info_parts.append(f"{self._zoom:.1f}x")
        filt = self._current_filter()
        if filt != "normal":
            info_parts.append(filt)
        if info_parts:
            info_text = "  ".join(info_parts)
            (tw, _), _ = cv2.getTextSize(info_text, font, 0.7, thick)
            cv2.putText(
                out, info_text, ((w - tw) // 2, 34),
                font, 0.7, (0, 255, 255), thick,
            )

        # ----- Battery bar (right side) ------------------------------
        if self._battery_percent >= 0:
            self._draw_battery_bar(out, w, bar_h)

        return out

    def _draw_battery_bar(
        self, frame: np.ndarray, frame_w: int, bar_h: int,
    ) -> None:
        """Draw a large, high-contrast battery indicator in the top-right."""
        pct = max(0, min(100, self._battery_percent))

        # Bar geometry
        bar_w = 100
        bar_inner_h = 22
        margin_right = 14
        margin_top = (bar_h - bar_inner_h) // 2

        x2 = frame_w - margin_right
        x1 = x2 - bar_w
        y1 = margin_top
        y2 = y1 + bar_inner_h

        # Color based on level
        if pct > 50:
            color = (0, 200, 0)      # green
        elif pct > 20:
            color = (0, 200, 255)    # yellow (BGR)
        else:
            color = (0, 0, 220)      # red

        # Charging indicator: use a brighter version
        if self._battery_charging:
            color = (200, 200, 0)    # cyan = charging (BGR)

        # Border
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2),
                       (200, 200, 200), 1)
        # Terminal nub
        cv2.rectangle(frame, (x2 + 2, y1 + 4), (x2 + 6, y2 - 4),
                       (200, 200, 200), -1)

        # Fill
        fill_w = int((pct / 100.0) * bar_w)
        if fill_w > 0:
            cv2.rectangle(frame, (x1, y1), (x1 + fill_w, y2), color, -1)

        # Percentage text to the left of the bar
        label = f"{pct}%"
        if self._battery_charging:
            label = f"CHG {pct}%"
        (tw, _), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2,
        )
        cv2.putText(
            frame, label, (x1 - tw - 8, y2 - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    def _apply_capture_flash(self, frame: np.ndarray) -> np.ndarray:
        """White-tint the frame to indicate a capture just happened."""
        white = np.full_like(frame, 255)
        return cv2.addWeighted(frame, 0.4, white, 0.6, 0)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display(self, frame: np.ndarray) -> None:
        if not self._window_opened:
            # WINDOW_GUI_NORMAL suppresses the Qt-enhanced toolbar and
            # pixel-inspector side panels that OpenCV's Qt backend
            # (used by python3-opencv on Pi OS) draws by default. Without
            # this flag, the toolbar and panels appear in the fullscreen
            # window and partially occlude the camera view.
            cv2.namedWindow(
                self.WINDOW_NAME,
                cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL,
            )
            if self._fullscreen:
                cv2.setWindowProperty(
                    self.WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN,
                )
            self._window_opened = True
        cv2.imshow(self.WINDOW_NAME, frame)

    # ------------------------------------------------------------------
    # Timer-driven transitions
    # ------------------------------------------------------------------

    def _check_timed_transitions(self) -> None:
        if self._state_machine.is_in(AppState.CAPTURE_FLASH):
            elapsed_ms = (
                time.monotonic() - self._capture_flash_started_at
            ) * 1000.0
            if elapsed_ms >= self._capture_flash_ms:
                self._state_machine.force_transition(
                    AppState.LIVE_VIEW, reason="capture flash timeout"
                )
                self._run_handler_safe(self._on_enter_live, "on_enter_live")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_view_state(self) -> bool:
        """True if the current state accepts zoom/pan/filter actions."""
        return self._state_machine.current_state in (
            AppState.LIVE_VIEW,
            AppState.FROZEN_VIEW,
        )

    def _is_in_gallery(self) -> bool:
        """True if the current state is GALLERY_VIEW.

        Used by the in-state handlers to route events to the
        :class:`Gallery` instance instead of mutating the live-view
        zoom / pan / filter state.
        """
        return self._state_machine.current_state == AppState.GALLERY_VIEW

    def _current_filter(self) -> str:
        return self._filters_available[self._filter_index]

    def _filter_index_for(self, name: str) -> int:
        try:
            return self._filters_available.index(name)
        except ValueError:
            logger.warning(
                "configured default filter %r not in available list; "
                "using first available (%r)",
                name, self._filters_available[0],
            )
            return 0

    def _draw_centered_text(
        self,
        frame: np.ndarray,
        text: str,
        y_offset: int,
        scale: float,
        thickness: int,
        color: tuple[int, int, int],
    ) -> None:
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness,
        )
        x = (w - tw) // 2
        y = h // 2 + y_offset + th // 2
        cv2.putText(
            frame, text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness,
        )

    def _shutdown(self) -> None:
        logger.info("MagnifierApp shutting down")
        if self._buzzer is not None:
            try:
                self._buzzer.play("shutdown")
            except Exception:
                pass
        try:
            if self._window_opened:
                cv2.destroyAllWindows()
        except Exception:
            logger.exception("error destroying windows")
        if self._buzzer is not None:
            try:
                self._buzzer.close()
            except Exception:
                pass