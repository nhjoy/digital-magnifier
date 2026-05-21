"""Digital Magnifier — application entry point.

Constructs the HAL implementations, processing stack, image
storage, and the ``MagnifierApp`` orchestrator, and starts the
main loop. Swapping mock HALs for real CM5 hardware in
MVP 0.2 / 0.3 is a one-line change in this file: pick a
different ``CameraSensor`` or ``ControlsHAL`` subclass; everything
else is untouched.

Usage::

    # From the repo root, with venv active:
    python -m digital_magnifier.main
    python -m digital_magnifier.main --log-level DEBUG
    python -m digital_magnifier.main --config-dir /path/to/custom/config

The OpenCV window must be focused for the keyboard mock to
receive input — clicking the window once is enough.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from digital_magnifier.core.app_controller import MagnifierApp
from digital_magnifier.hal.camera_sensor import MockCameraSensor
from digital_magnifier.hal.mock_controls import MockControls
from digital_magnifier.storage.image_saver import ImageSaver
from digital_magnifier.utils.config_loader import (
    ConfigError,
    load_all_configs,
)
from digital_magnifier.utils.logger import setup_logging


logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="digital_magnifier",
        description=(
            "Portable digital magnifier for low-vision children "
            "(MVP 0.1, keyboard-mocked controls)."
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
    return parser.parse_args(argv)


def _build_app_config(
    app_cfg: dict,
    camera_cfg: dict,
) -> dict:
    """Merge camera resolution into the app config.

    The app controller reads ``config["camera"]["width"]`` and
    ``config["camera"]["height"]`` for the placeholder frames
    (gallery, menu). The source of truth for those dimensions is
    ``camera_config.yaml``, so we inject them here rather than
    duplicating them across two YAML files.
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
        # Bootstrap logger just to report the error before exit.
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

    logger.info("Digital Magnifier MVP 0.1 starting")
    if args.config_dir:
        logger.info("using config dir: %s", args.config_dir)

    # --- wire dependencies ---------------------------------------
    try:
        camera = MockCameraSensor(configs["camera"])
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