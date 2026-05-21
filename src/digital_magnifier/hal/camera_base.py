"""Abstract base class for the camera sensor HAL.

The camera HAL delivers BGR uint8 frames to the application. Two
implementations are planned:

  - ``MockCameraSensor`` (MVP 0.1): USB webcam via ``cv2.VideoCapture``
    with a fallback to a procedurally-generated test frame when no
    webcam is connected (useful in WSL).
  - ``PiCameraSensor`` (MVP 0.2): the Raspberry Pi Camera Module 3
    via the ``picamera2`` library.

The contract above the HAL is identical: code consuming a
``CameraSensor`` doesn't know or care which implementation it is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

import numpy as np


class CameraError(Exception):
    """Raised when the camera cannot deliver a frame.

    The app controller's main loop catches this, increments its
    consecutive-failure counter, and decides whether to retry or
    shut down. A single bad frame should never crash the device.
    """


class CameraSensor(ABC):
    """Abstract camera input.

    Lifecycle:
        camera = MockCameraSensor(config)
        camera.start()
        try:
            while running:
                frame = camera.get_frame()
                ...
        finally:
            camera.stop()

    Or as a context manager::

        with MockCameraSensor(config) as camera:
            frame = camera.get_frame()
    """

    def start(self) -> None:
        """Acquire the camera. Default no-op."""

    def stop(self) -> None:
        """Release the camera. Default no-op."""

    @abstractmethod
    def get_frame(self) -> np.ndarray:
        """Return the latest frame as a BGR uint8 array.

        Raises
        ------
        CameraError
            If no frame is available. The controller's error
            handling decides whether to retry, swap the sensor,
            or shut down.
        """
        raise NotImplementedError

    # ----- context-manager sugar -------------------------------------

    def __enter__(self) -> "CameraSensor":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()