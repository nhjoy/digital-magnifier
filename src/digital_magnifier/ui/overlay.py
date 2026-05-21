import cv2
import numpy as np


def draw_overlay(
    frame: np.ndarray,
    zoom: float,
    filter_name: str,
    frozen: bool,
    fps: float,
) -> np.ndarray:
    output = frame.copy()

    status = "FROZEN" if frozen else "LIVE"

    text = f"{status} | Zoom: {zoom:.1f}x | Filter: {filter_name} | FPS: {fps:.1f}"

    cv2.rectangle(output, (0, 0), (output.shape[1], 50), (0, 0, 0), -1)

    cv2.putText(
        output,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output