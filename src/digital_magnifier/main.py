"""Digital Magnifier — application entry point.

Constructs the HAL implementations, processing stack, image
storage, and the ``MagnifierApp`` orchestrator, and starts the
main loop.

Platform detection
------------------
This file picks the camera HAL based on the configured platform:

- ``"auto"`` (default): try to import ``picamera2``; if it works,
  use :class:`PiCameraSensor`, otherwise fall back to
  :class:`MockCameraSensor`. Same config works on WSL and CM5.
- ``"raspberrypi_cm5"``: force :class:`PiCameraSensor`. Fails fast
  if picamera2 is not installed.
- ``"wsl_mock"``: force :class:`MockCameraSensor`. Skips the
  picamera2 import attempt entirely.

The platform string is read from ``hardware_pins.yaml`` under
``hardware.platform``, and can be overridden by the ``--platform``
CLI flag.

Usage::

    # If you ran `pip install -e .` once:
    digital-magnifier
    digital-magnifier --log-level DEBUG --platform wsl_mock

    # Or, without installing:
    PYTHONPATH=src python -m digital_magnifier.main
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from digital_magnifier.core.app_controller import MagnifierApp
from digital_magnifier.hal.camera_base import CameraSensor
from digital_magnifier.hal.mock_controls import MockControls
from digital_magnifier.storage.image_saver import ImageSaver
from digital_magnifier.utils.config_loader import (
    ConfigError,
    load_all_configs,
)
from digital_magnifier.utils.logger import setup_logging


logger = logging.getLogger(__name__)


# Platform strings used in hardware_pins.yaml under `hardware.platform`.
PLATFORM_AUTO = "auto"
PLATFORM_WSL_MOCK = "wsl_mock"
PLATFORM_RPI_CM5 = "raspberrypi_cm5"
KNOWN_PLATFORMS = (PLATFORM_AUTO, PLATFORM_WSL_MOCK, PLATFORM_RPI_CM5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="digital_magnifier",
        description=(
            "Portable digital magnifier for low-vision children."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Override the log level from app_config.yaml.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help=(
            "Override the config directory. Defaults to "
            "<project_root>/config."
        ),
    )
    parser.add_argument(
        "--platform",
        choices=list(KNOWN_PLATFORMS),
        default=None,
        help=(
            "Override the platform setting from hardware_pins.yaml. "
            "'auto' detects picamera2 at runtime."
        ),
    )
    return parser.parse_args(argv)


def _resolve_platform(
    configs: dict[str, dict],
    force_platform: str | None = None,
) -> str:
    """Determine which platform we're running on.

    Priority: CLI flag > YAML setting > "auto".
    """
    if force_platform:
        return force_platform

    yaml_platform = (
        configs.get("hardware_pins", {})
        .get("hardware", {})
        .get("platform", PLATFORM_AUTO)
    )
    if yaml_platform not in KNOWN_PLATFORMS:
        logger.warning(
            "Unknown platform %r in hardware_pins.yaml; falling back to %s",
            yaml_platform, PLATFORM_AUTO,
        )
        return PLATFORM_AUTO
    return yaml_platform


def _detect_platform_auto() -> str:
    """Probe for picamera2; report resolved platform."""
    try:
        import picamera2  # noqa: F401
        import libcamera  # noqa: F401
    except ImportError as e:
        logger.info(
            "picamera2/libcamera not importable (%s); using %s",
            e, PLATFORM_WSL_MOCK,
        )
        return PLATFORM_WSL_MOCK
    logger.info("picamera2 available; using %s", PLATFORM_RPI_CM5)
    return PLATFORM_RPI_CM5


def _build_camera(
    platform: str,
    camera_config: dict[str, Any],
) -> CameraSensor:
    """Construct the appropriate CameraSensor for the platform.

    Camera classes are lazy-imported so that an environment missing
    one (e.g. WSL without picamera2) doesn't fail at app startup.
    """
    if platform == PLATFORM_AUTO:
        platform = _detect_platform_auto()

    if platform == PLATFORM_RPI_CM5:
        from digital_magnifier.hal.camera_sensor import PiCameraSensor
        logger.info("Camera: PiCameraSensor")
        return PiCameraSensor(camera_config)

    # Default for "wsl_mock" and anything unrecognised
    from digital_magnifier.hal.camera_sensor import MockCameraSensor
    logger.info("Camera: MockCameraSensor")
    return MockCameraSensor(camera_config)


def _build_app_config(
    app_cfg: dict[str, Any],
    camera_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Merge camera resolution into the app config.

    The app controller reads ``config["camera"]["width"]`` /
    ``["height"]`` for placeholder frames. The source of truth lives
    in ``camera_config.yaml``, so we inject it here.
    """
    merged = dict(app_cfg)
    res_cfg = camera_cfg.get("resolution", {})
    merged["camera"] = {
        "width": int(res_cfg.get("width", 1280)),
        "height": int(res_cfg.get("height", 720)),
    }
    return merged


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- load configs (fatal if any are missing or malformed) -----
    try:
        configs = load_all_configs(config_dir=args.config_dir)
    except ConfigError as e:
        logging.basicConfig(
            level=logging.ERROR,
            format="%(levelname)s: %(message)s",
        )
        logger.error("configuration error: %s", e)
        return 2

    # --- set up logging (CLI overrides config) --------------------
    log_cfg = dict(configs["app"].get("logging", {}))
    if args.log_level:
        log_cfg["level"] = args.log_level
    setup_logging(log_cfg)

    logger.info("Digital Magnifier starting (v0.2)")
    if args.config_dir:
        logger.info("using config dir: %s", args.config_dir)

    # --- wire dependencies ---------------------------------------
    try:
        platform = _resolve_platform(configs, force_platform=args.platform)
        logger.info("resolved platform: %s", platform)

        camera = _build_camera(platform, configs["camera"])
        controls = MockControls(configs["hardware_pins"])

        capture_cfg = configs["app"].get("capture", {})
        saver = ImageSaver(
            output_directory=capture_cfg.get(
                "output_directory", "captures"
            ),
            file_format=capture_cfg.get("file_format", "png"),
        )

        app_config = _build_app_config(
            configs["app"], configs["camera"]
        )
        app = MagnifierApp(
            camera=camera,
            controls=controls,
            image_saver=saver,
            config=app_config,
        )
    except Exception:
        logger.exception("failed to construct application")
        return 1

    # --- run ------------------------------------------------------
    try:
        app.run()
    except Exception:
        logger.exception("unhandled exception during run loop")
        return 1

    logger.info("Digital Magnifier exited cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())