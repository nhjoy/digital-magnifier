"""Gallery view for browsing previously-captured images.

Lives in its own module so the view's state (current index, zoom,
pan, filter, delete-confirmation timer) is contained and testable
in isolation. The app controller delegates to a single
:class:`Gallery` instance when the state machine is in
:class:`~digital_magnifier.core.state_machine.AppState.GALLERY_VIEW`.

Design notes
------------
**Separate zoom / pan / filter state from the live view.** When a
user opens the gallery, they probably want to see the whole
captured image first, not arrive at whatever zoom level the live
view was last at. Going back to live view restores the live state.

**Image cache of one.** Loading a 2304×1296 PNG isn't free
(~30 ms on the CM5), so we cache the decoded image and only reload
when the index changes. Re-applying zoom / filter / pan is cheap
in comparison and runs every render tick.

**Delete confirmation.** A single-button delete is dangerous — one
stray press wipes a capture. The confirmation flow is:

    snapshot button → "Press snapshot again to delete" overlay
                      shown for ``confirm_window_s`` seconds
    snapshot again  → file removed, index advances to next image
    any other input → silently cancels the arming

This needs zero new buttons (the snapshot button is otherwise
unused in gallery view) and gives the user an obvious bail-out.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from digital_magnifier.processing.magnifier import apply_zoom
from digital_magnifier.processing.vision_filters import apply_filter
from digital_magnifier.storage.image_saver import ImageSaver, ImageSaverError


logger = logging.getLogger(__name__)


class Gallery:
    """Browse / zoom / filter / delete over previously-captured images.

    Parameters
    ----------
    image_saver
        Source of truth for the captures directory. Gallery calls
        :meth:`ImageSaver.list_images` on open and on delete, and
        :meth:`ImageSaver.delete_image` for the destructive op.
    display_width, display_height
        Used to size the empty-state placeholder and the
        confirmation overlay. Should match the camera/display
        canvas the app controller uses.
    filters_available
        Ordered list of filter names that :func:`apply_filter`
        understands. Same list the live view uses, so the user's
        mental model carries across.
    initial_filter
        Filter to start on each time the gallery is opened.
    zoom_min, zoom_max, zoom_step, pan_step
        Same semantics as the live-view magnifier.
    confirm_window_s
        How long a "press snapshot again to delete" prompt stays
        valid. 2 seconds is short enough that you can't forget you
        armed it, long enough that you can comfortably re-press.
    clock
        Source of monotonic time. Injectable for tests.
    """

    # Maximum length of the filename shown in the overlay.
    _FILENAME_OVERLAY_CHARS = 32

    def __init__(
        self,
        image_saver: ImageSaver,
        *,
        display_width: int,
        display_height: int,
        filters_available: list[str],
        initial_filter: str = "normal",
        filter_config: Optional[dict[str, dict[str, Any]]] = None,
        zoom_min: float = 1.0,
        zoom_max: float = 8.0,
        zoom_step: float = 0.5,
        pan_step: float = 0.1,
        confirm_window_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._saver = image_saver
        self._w = int(display_width)
        self._h = int(display_height)
        self._filters_available = list(filters_available)
        if not self._filters_available:
            self._filters_available = ["normal"]
        self._initial_filter = initial_filter
        self._filter_config = filter_config or {}
        self._zoom_min = float(zoom_min)
        self._zoom_max = float(zoom_max)
        self._zoom_step = float(zoom_step)
        self._pan_step = float(pan_step)
        self._confirm_window_s = float(confirm_window_s)
        self._clock = clock

        # Mutable view state — populated on open()
        self._images: list[Path] = []
        self._index: int = 0
        self._loaded_image: Optional[np.ndarray] = None
        self._loaded_index: Optional[int] = None
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        self._filter_index: int = self._index_for_filter(self._initial_filter)

        # Delete confirmation timer (monotonic). ``None`` means not armed.
        self._delete_armed_until: Optional[float] = None

    # ----- lifecycle ------------------------------------------------

    def open(self) -> None:
        """Called by the app controller when GALLERY_VIEW is entered.

        Rescans the captures directory, resets the index to 0, and
        clears any leftover view state. Cheap enough to call every
        time even with hundreds of images.
        """
        self._images = self._saver.list_images(newest_first=True)
        self._index = 0
        self._loaded_image = None
        self._loaded_index = None
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._filter_index = self._index_for_filter(self._initial_filter)
        self._delete_armed_until = None
        logger.info("Gallery opened with %d image(s)", len(self._images))

    def close(self) -> None:
        """Drop any cached frame so memory stays low when not browsing."""
        self._loaded_image = None
        self._loaded_index = None
        self._delete_armed_until = None

    # ----- queries --------------------------------------------------

    @property
    def is_empty(self) -> bool:
        return not self._images

    @property
    def count(self) -> int:
        return len(self._images)

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def delete_armed(self) -> bool:
        """Whether a delete confirmation is currently active and unexpired."""
        if self._delete_armed_until is None:
            return False
        if self._clock() >= self._delete_armed_until:
            # Window expired — clear the flag now so the renderer sees
            # the right state. This is the only place the timer gets
            # cleared autonomously.
            self._delete_armed_until = None
            return False
        return True

    @property
    def current_path(self) -> Optional[Path]:
        if not self._images:
            return None
        return self._images[self._index]

    # ----- navigation ----------------------------------------------

    def next(self) -> None:
        if not self._images:
            return
        self._cancel_delete_arm()
        self._index = (self._index + 1) % len(self._images)
        self._on_index_changed()

    def prev(self) -> None:
        if not self._images:
            return
        self._cancel_delete_arm()
        self._index = (self._index - 1) % len(self._images)
        self._on_index_changed()

    def _on_index_changed(self) -> None:
        """Reset zoom and force a reload when the user moves to a new image.

        Filter is intentionally preserved: a user who's chosen
        "high_contrast" probably wants it on every image they look
        at next, not to have to re-cycle on each one.
        """
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        # _loaded_image will be reloaded on next render

    # ----- view manipulation ---------------------------------------

    def zoom_in(self) -> None:
        if not self._images:
            return
        self._cancel_delete_arm()
        self._zoom = min(self._zoom + self._zoom_step, self._zoom_max)

    def zoom_out(self) -> None:
        if not self._images:
            return
        self._cancel_delete_arm()
        new_zoom = max(self._zoom - self._zoom_step, self._zoom_min)
        if new_zoom <= 1.0 and self._zoom > 1.0:
            self._pan_x = 0.0
            self._pan_y = 0.0
        self._zoom = new_zoom

    def pan_up(self) -> None:
        if not self._images or self._zoom <= 1.0:
            self._cancel_delete_arm()
            return
        self._cancel_delete_arm()
        self._pan_y = max(self._pan_y - self._pan_step, -1.0)

    def pan_down(self) -> None:
        if not self._images or self._zoom <= 1.0:
            self._cancel_delete_arm()
            return
        self._cancel_delete_arm()
        self._pan_y = min(self._pan_y + self._pan_step, 1.0)

    def filter_next(self) -> None:
        if not self._images:
            return
        self._cancel_delete_arm()
        self._filter_index = (
            (self._filter_index + 1) % len(self._filters_available)
        )

    def reset_view(self) -> None:
        if not self._images:
            return
        self._cancel_delete_arm()
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._filter_index = self._index_for_filter(self._initial_filter)

    # ----- delete with confirmation --------------------------------

    def request_delete(self) -> Optional[Path]:
        """Press-once-to-arm, press-again-to-delete flow.

        First call: arms the delete timer. Returns ``None`` and
        leaves the image alone. The caller is expected to render
        the confirmation overlay (see :attr:`delete_armed`).

        Second call within :attr:`confirm_window_s`: actually
        deletes the current image, advances the index to a sensible
        place, returns the deleted path.

        Returns
        -------
        Optional[Path]
            The deleted file's path, or ``None`` if no deletion
            happened (either arming, or no images present).
        """
        if not self._images:
            return None

        if self.delete_armed:
            # Confirmed — perform the delete.
            path = self._images[self._index]
            try:
                self._saver.delete_image(path)
            except ImageSaverError:
                logger.exception("delete failed for %s", path)
                self._delete_armed_until = None
                return None

            del self._images[self._index]
            if self._images:
                # Keep the cursor near where it was, but clamp.
                self._index = min(self._index, len(self._images) - 1)
            else:
                self._index = 0
            self._loaded_image = None
            self._loaded_index = None
            self._zoom = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0
            self._delete_armed_until = None
            logger.info("Gallery deleted %s", path)
            return path

        # Arm the timer; await confirmation.
        self._delete_armed_until = self._clock() + self._confirm_window_s
        logger.debug(
            "delete armed for %.1fs on %s",
            self._confirm_window_s, self._images[self._index].name,
        )
        return None

    def _cancel_delete_arm(self) -> None:
        if self._delete_armed_until is not None:
            self._delete_armed_until = None

    # ----- rendering -----------------------------------------------

    def render(self) -> np.ndarray:
        """Produce the current gallery frame for display.

        Returns a fresh BGR uint8 array sized to the configured
        display dimensions. Includes any overlays (count, filename,
        delete confirmation prompt, empty-state placeholder).
        """
        if not self._images:
            return self._render_empty()

        image = self._get_or_load_current()
        if image is None:
            # File vanished or unreadable. Tell the user, leave the
            # image list as it was (next/prev still work, the user can
            # navigate away).
            return self._render_message(
                "(image unreadable)",
                f"{self._images[self._index].name}",
            )

        zoomed = apply_zoom(image, self._zoom, self._pan_x, self._pan_y)
        filtered = apply_filter(zoomed, self._current_filter(), self._filter_config)

        # Resize to match the display canvas so the overlay sits
        # at expected positions regardless of source resolution.
        if filtered.shape[1] != self._w or filtered.shape[0] != self._h:
            filtered = cv2.resize(
                filtered, (self._w, self._h),
                interpolation=cv2.INTER_AREA,
            )

        return self._draw_overlay(filtered)

    def _get_or_load_current(self) -> Optional[np.ndarray]:
        """Load the current image off disk, caching one frame at a time."""
        if (
            self._loaded_image is not None
            and self._loaded_index == self._index
        ):
            return self._loaded_image
        path = self._images[self._index]
        try:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        except Exception:
            logger.exception("failed reading %s", path)
            return None
        if img is None:
            logger.warning("cv2.imread returned None for %s", path)
            return None
        self._loaded_image = img
        self._loaded_index = self._index
        return img

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add gallery-specific overlays: index / filename / delete prompt."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Top bar: "GALLERY  N / M  •  filename"
        cv2.rectangle(out, (0, 0), (w, 40), (0, 0, 0), -1)
        path = self._images[self._index]
        filename = path.name
        if len(filename) > self._FILENAME_OVERLAY_CHARS:
            filename = filename[: self._FILENAME_OVERLAY_CHARS - 1] + "…"
        header = f"GALLERY  {self._index + 1} / {len(self._images)}  -  {filename}"
        cv2.putText(
            out, header, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        # Zoom + filter labels (top-right, like the live view overlay)
        info = f"{self._zoom:.1f}x  {self._current_filter()}"
        (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(
            out, info, (w - tw - 10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )

        # Delete confirmation banner (bottom, red, hard to miss).
        if self.delete_armed:
            banner_h = 60
            cv2.rectangle(
                out, (0, h - banner_h), (w, h), (0, 0, 180), -1,
            )
            msg = "Press SNAPSHOT again to DELETE"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
            cv2.putText(
                out, msg, ((w - tw) // 2, h - (banner_h - th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3,
            )

        return out

    def _render_empty(self) -> np.ndarray:
        return self._render_message(
            "No captures yet",
            "Press SNAPSHOT in live view to take a picture",
        )

    def _render_message(self, title: str, subtitle: str) -> np.ndarray:
        frame = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        # Title (centred, large).
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)
        cv2.putText(
            frame, title, ((self._w - tw) // 2, self._h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4,
        )
        # Subtitle (centred, below).
        (sw, _), _ = cv2.getTextSize(subtitle, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(
            frame, subtitle, ((self._w - sw) // 2, self._h // 2 + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2,
        )
        return frame

    # ----- helpers --------------------------------------------------

    def _current_filter(self) -> str:
        return self._filters_available[self._filter_index]

    def _index_for_filter(self, name: str) -> int:
        try:
            return self._filters_available.index(name)
        except ValueError:
            return 0