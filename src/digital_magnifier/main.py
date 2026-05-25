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


def _build_controls(
    platform: str,
    hardware_config: dict[str, Any],
):  # -> ControlsHAL  (annotation omitted to avoid eager import)
    """Construct the appropriate ControlsHAL for the platform.

    On the CM5, attempt to open the I2C bus and probe the TCA6416A.
    If that succeeds, use :class:`GPIOControls` (with the MCP3221
    ADC if it also probes successfully — the zoom pot is optional
    so a missing ADC degrades gracefully to "zoom pot does nothing").
    If anything in the GPIO path fails (smbus2 missing, I2C disabled,
    chip not wired), fall back to :class:`MockControls` so the
    keyboard can still drive the device for debugging.
    """
    if platform == PLATFORM_AUTO:
        platform = _detect_platform_auto()

    if platform != PLATFORM_RPI_CM5:
        logger.info("Controls: MockControls (keyboard)")
        return MockControls(hardware_config)

    # CM5 path: try to bring up GPIOControls.
    try:
        from digital_magnifier.hal.i2c_devices import (
            I2CError,
            MCP3221,
            TCA6416A,
            open_smbus_bus,
            parse_address,
        )
        from digital_magnifier.hal.mock_controls import GPIOControls
    except ImportError as exc:
        logger.warning(
            "GPIO dependencies missing (%s); falling back to MockControls", exc,
        )
        return MockControls(hardware_config)

    i2c_cfg = hardware_config.get("i2c", {})
    bus_number = int(i2c_cfg.get("bus", 1))
    devices_cfg = i2c_cfg.get("devices", {})
    io_cfg = devices_cfg.get("io_expander", {})
    adc_cfg = devices_cfg.get("zoom_adc", {})

    # Open the bus
    try:
        bus = open_smbus_bus(bus_number)
    except I2CError as exc:
        logger.warning(
            "Cannot open I2C bus %d (%s); falling back to MockControls",
            bus_number, exc,
        )
        return MockControls(hardware_config)

    # Build TCA6416A
    try:
        io_expander = TCA6416A(
            bus,
            address=parse_address(io_cfg.get("address", 0x20)),
            config_port0=int(
                hardware_config.get("tca6416a_pins", {}).get("config_port0", 0xFF)
            ),
            config_port1=int(
                hardware_config.get("tca6416a_pins", {}).get("config_port1", 0xFF)
            ),
        )
        io_expander.probe()
    except (I2CError, ValueError) as exc:
        logger.warning(
            "TCA6416A not detected (%s); falling back to MockControls", exc,
        )
        bus.close()
        return MockControls(hardware_config)

    # Build MCP3221 (optional)
    adc: MCP3221 | None = None
    if adc_cfg:
        try:
            adc = MCP3221(
                bus,
                address=parse_address(adc_cfg.get("address", 0x4D)),
                vdd_volts=float(adc_cfg.get("vdd_volts", 3.3)),
            )
            adc.probe()
        except (I2CError, ValueError) as exc:
            optional = bool(adc_cfg.get("optional", True))
            if optional:
                logger.info(
                    "MCP3221 not present (%s); zoom pot disabled "
                    "(this is fine until you wire it in)",
                    exc,
                )
                adc = None
            else:
                logger.error(
                    "MCP3221 not present and configured as required (%s); "
                    "falling back to MockControls",
                    exc,
                )
                bus.close()
                return MockControls(hardware_config)

    logger.info("Controls: GPIOControls (TCA6416A + %s)",
                "MCP3221" if adc else "no ADC")
    return GPIOControls(hardware_config, io_expander, adc)


def _probe_ups(hardware_config: dict) -> Any:
    """Try to connect to a Waveshare UPS HAT (E) on I²C.

    Returns a ``UPSHatE`` instance if the HAT is present, or
    ``None`` if it can't be reached. Does not raise — the UPS is
    an optional accessory for battery-percentage display.
    """
    try:
        from digital_magnifier.hal.i2c_devices import (
            I2CError,
            UPSHatE,
            open_smbus_bus,
            parse_address,
        )
    except ImportError:
        return None

    i2c_cfg = hardware_config.get("i2c", {})
    bus_number = int(i2c_cfg.get("bus", 1))
    devices_cfg = i2c_cfg.get("devices", {})
    ups_cfg = devices_cfg.get("ups_hat", {})
    address = parse_address(ups_cfg.get("address", 0x2D))

    try:
        bus = open_smbus_bus(bus_number)
    except I2CError:
        return None

    try:
        ups = UPSHatE(bus, address=address)
        ups.probe()
        pct = ups.battery_percent()
        logger.info(
            "UPS HAT (E) at 0x%02x: detected (battery %d%%)",
            address, pct,
        )
        return ups
    except (I2CError, ValueError) as exc:
        logger.info(
            "UPS HAT not detected at 0x%02x (%s); battery display disabled",
            address, exc,
        )
        bus.close()
        return None


def _probe_buzzer(hardware_config: dict) -> Any:
    """Try to open the buzzer on the configured GPIO pin.

    Returns a ``BuzzerController`` instance if the pin was claimed,
    or ``None`` if gpiozero is missing or the pin can't be opened.
    """
    buzzer_cfg = hardware_config.get("buzzer", {})
    if not buzzer_cfg.get("enabled", True):
        logger.info("Buzzer disabled in config")
        return None

    pin = int(buzzer_cfg.get("pin", 24))
    freq = int(buzzer_cfg.get("frequency_hz", 4000))

    try:
        from digital_magnifier.hal.buzzer import BuzzerController
        buz = BuzzerController(pin=pin, frequency=freq)
        if buz.available:
            return buz
        return None
    except Exception as exc:
        logger.info("Buzzer not available: %s", exc)
        return None


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

    logger.info("Digital Magnifier starting (v0.3)")
    if args.config_dir:
        logger.info("using config dir: %s", args.config_dir)

    # --- wire dependencies ---------------------------------------
    try:
        platform = _resolve_platform(configs, force_platform=args.platform)
        logger.info("resolved platform: %s", platform)

        camera = _build_camera(platform, configs["camera"])
        controls = _build_controls(platform, configs["hardware_pins"])

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

        # Probe UPS HAT (optional — battery display disabled if absent)
        ups = _probe_ups(configs["hardware_pins"])

        # Probe buzzer (optional — audio feedback disabled if absent)
        buzzer = _probe_buzzer(configs["hardware_pins"])

        app = MagnifierApp(
            camera=camera,
            controls=controls,
            image_saver=saver,
            config=app_config,
            ups=ups,
            buzzer=buzzer,
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