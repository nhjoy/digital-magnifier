"""Abstract base class for the controls HAL.

The controls HAL turns raw user input — mock keyboard during
development, real GPIO buttons + nav switch + ADC potentiometer on
the CM5 — into AppEvent values that the rest of the application
consumes. Everything above this layer should be unaware of whether
the underlying input is a keyboard, GPIO, or anything else.

Implementations must:
  - Inherit ``ControlsHAL``.
  - Override ``poll()`` and return ``AppEvent.NONE`` when no input
    is pending. ``poll()`` is called once per main-loop iteration,
    so it MUST be non-blocking (or only briefly blocking, e.g.
    ``cv2.waitKey(1)``).
  - Optionally override ``start()`` and ``stop()`` if the
    implementation needs to acquire and release resources (GPIO
    claims, ADC handles, background threads, etc.).

Implementations must NOT:
  - Perform image processing, display, or state-machine logic.
  - Block for arbitrary durations (the camera view would freeze).
  - Contain hardcoded key bindings or pin numbers; load them from
    config.

Lifecycle
---------
Manual::

    controls = MockControls(config)
    controls.start()
    try:
        while running:
            event = controls.poll()
            ...
    finally:
        controls.stop()

Context manager (preferred)::

    with MockControls(config) as controls:
        while running:
            event = controls.poll()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

from digital_magnifier.core.events import AppEvent


class ControlsHAL(ABC):
    """Abstract input device."""

    def start(self) -> None:
        """Acquire any resources the implementation needs.

        Default is a no-op so trivial subclasses (such as the
        keyboard mock) need not override it.
        """

    def stop(self) -> None:
        """Release any resources acquired in :meth:`start`.

        Default is a no-op. Override to clean up GPIO claims,
        background threads, file handles, etc.
        """

    @abstractmethod
    def poll(self) -> AppEvent:
        """Return the next pending event, or ``AppEvent.NONE``.

        Called once per main-loop iteration. Must be non-blocking
        or only briefly blocking. Implementations should never
        raise from this method in steady state; log and return
        ``AppEvent.NONE`` on transient errors.
        """
        raise NotImplementedError

    # ----- context-manager sugar -------------------------------------

    def __enter__(self) -> "ControlsHAL":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()