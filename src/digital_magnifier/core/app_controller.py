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
    ) -> None:
        # --- injected dependencies --------------------------------
        self._camera = camera
        self._controls = controls
        self._image_saver = image_saver
        self._config = config

        # --- read app config --------------------------------------
        app_cfg = config.get("app", {})
        self._target_fps: float = float(app_cfg.get("target_fps", 24))
        self._show_overlay: bool = bool(app_cfg.get("show_overlay", True))

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

        # --- dispatch tables --------------------------------------
        self._in_state_handlers: dict[AppEvent, Callable[[], None]] = {
            AppEvent.ZOOM_IN:     self._handle_zoom_in,
            AppEvent.ZOOM_OUT:    self._handle_zoom_out,
            AppEvent.PAN_UP:      self._handle_pan_up,
            AppEvent.PAN_DOWN:    self._handle_pan_down,
            AppEvent.PAN_LEFT:    self._handle_pan_left,
            AppEvent.PAN_RIGHT:   self._handle_pan_right,
            AppEvent.FILTER_NEXT: self._handle_filter_next,
            AppEvent.RESET_VIEW:  self._handle_reset_view,
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

        frame = self._acquire_frame_for_current_state()
        display = self._render_for_current_state(frame)
        self._display(display)

        event = self._controls.poll()
        self._dispatch(event)

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
        logger.info("entering GALLERY_VIEW (stub)")

    def _on_enter_menu(self) -> None:
        logger.info("entering MENU_VIEW (stub)")

    def _on_enter_shutdown(self) -> None:
        logger.info("entering SHUTDOWN")

    # ------------------------------------------------------------------
    # In-state handlers (no transition)
    # ------------------------------------------------------------------

    def _handle_zoom_in(self) -> None:
        if not self._is_view_state():
            return
        self._zoom = min(self._zoom + self._zoom_step, self._zoom_max)
        self._cache_dirty = True
        logger.debug("zoom -> %.2f", self._zoom)

    def _handle_zoom_out(self) -> None:
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
        if not self._is_view_state():
            return
        self._pan_y = max(self._pan_y - self._pan_step, -1.0)
        self._cache_dirty = True

    def _handle_pan_down(self) -> None:
        if not self._is_view_state():
            return
        self._pan_y = min(self._pan_y + self._pan_step, 1.0)
        self._cache_dirty = True

    def _handle_pan_left(self) -> None:
        if not self._is_view_state():
            return
        self._pan_x = max(self._pan_x - self._pan_step, -1.0)
        self._cache_dirty = True

    def _handle_pan_right(self) -> None:
        if not self._is_view_state():
            return
        self._pan_x = min(self._pan_x + self._pan_step, 1.0)
        self._cache_dirty = True

    def _handle_filter_next(self) -> None:
        if not self._is_view_state():
            return
        self._filter_index = (
            (self._filter_index + 1) % len(self._filters_available)
        )
        self._cache_dirty = True
        logger.debug("filter -> %s", self._current_filter())

    def _handle_reset_view(self) -> None:
        if not self._is_view_state():
            return
        self._zoom = self._zoom_default
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._filter_index = self._filter_index_for(self._filter_default)
        self._cache_dirty = True
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
            return self._placeholder_frame("GALLERY", "Press [G] or [ to exit")

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
        return apply_filter(zoomed, self._current_filter())

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Minimal inline overlay for MVP 0.1.

        Will be replaced by the accessibility UI components in
        MVP 0.4 (large icons, high-contrast indicators, optional
        audio cues). Kept inline here to avoid coupling MVP 0.1
        to a UI implementation that doesn't exist yet.
        """
        h, w = frame.shape[:2]
        out = frame.copy()

        # Top bar background
        cv2.rectangle(out, (0, 0), (w, 40), (0, 0, 0), -1)

        state_label = self._state_machine.current_state.name.replace("_", " ")
        cv2.putText(
            out, state_label, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

        zoom_label = f"{self._zoom:.1f}x"
        (tw, _), _ = cv2.getTextSize(
            zoom_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2,
        )
        cv2.putText(
            out, zoom_label, ((w - tw) // 2, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )

        filter_label = self._current_filter()
        (tw, _), _ = cv2.getTextSize(
            filter_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2,
        )
        cv2.putText(
            out, filter_label, (w - tw - 10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        return out

    def _apply_capture_flash(self, frame: np.ndarray) -> np.ndarray:
        """White-tint the frame to indicate a capture just happened."""
        white = np.full_like(frame, 255)
        return cv2.addWeighted(frame, 0.4, white, 0.6, 0)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display(self, frame: np.ndarray) -> None:
        if not self._window_opened:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
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
        try:
            if self._window_opened:
                cv2.destroyAllWindows()
        except Exception:
            logger.exception("error destroying windows")