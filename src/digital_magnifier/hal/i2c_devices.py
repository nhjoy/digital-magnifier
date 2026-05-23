"""I2C device drivers used by the GPIO controls HAL.

Two chips live here:

- :class:`TCA6416A` — 16-bit I/O expander hosting the navigation
  switch, six buttons, and an RGB status LED. Used for digital
  input (active-low buttons) and digital output (LED cathodes).

- :class:`MCP3221` — single-channel 12-bit ADC for the zoom
  potentiometer. The MCP3221 has no command registers; a read is
  simply ``[start, address|R, ack, byte_hi, ack, byte_lo, nack, stop]``.
  Driver wraps that protocol behind a clean Python interface.

Bus injection
-------------
Both drivers take a bus-like object in ``__init__`` rather than
opening :mod:`smbus2` themselves. This makes them trivially testable
with an in-memory fake bus, and keeps :mod:`smbus2` an optional
runtime dependency (the rest of the app works on a dev laptop where
:mod:`smbus2` may not be installed).

The required bus interface:

- ``write_byte_data(address: int, register: int, value: int) -> None``
- ``read_byte_data(address: int, register: int) -> int``
- ``read_raw_bytes(address: int, length: int) -> list[int]``
- ``close() -> None`` (optional)

Use :func:`open_smbus_bus` to obtain a real bus on the CM5;
:class:`FakeBus` in the tests for unit tests.

Errors
------
The drivers raise :class:`I2CError` on any bus-level failure. The
calling code in :class:`GPIOControls` catches these, logs, and keeps
the main loop running (it does not crash the device when a wire
falls off).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol


logger = logging.getLogger(__name__)


# ===================================================================
# Errors
# ===================================================================


class I2CError(Exception):
    """A bus-level I2C failure (NACK, device missing, IO error)."""


# ===================================================================
# Bus abstraction
# ===================================================================


class I2CBusLike(Protocol):
    """Structural type describing the bus operations we use."""

    def write_byte_data(
        self, address: int, register: int, value: int
    ) -> None: ...

    def read_byte_data(self, address: int, register: int) -> int: ...

    def read_raw_bytes(self, address: int, length: int) -> list[int]: ...

    def close(self) -> None: ...


class SmbusBus:
    """Thin wrapper over :mod:`smbus2` that matches :class:`I2CBusLike`.

    Imports :mod:`smbus2` lazily so this module is importable in test
    environments that do not have the C-extension installed.
    """

    def __init__(self, bus_number: int) -> None:
        try:
            from smbus2 import SMBus, i2c_msg  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - covered by integration
            raise I2CError(
                "smbus2 not installed; install python3-smbus2 (apt) "
                "or smbus2 (pip) on the CM5"
            ) from exc

        try:
            self._bus = SMBus(bus_number)
        except OSError as exc:
            raise I2CError(
                f"failed to open I2C bus {bus_number}: {exc}. "
                f"Is I2C enabled in raspi-config?"
            ) from exc

        self._i2c_msg = i2c_msg
        self._bus_number = bus_number

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        try:
            self._bus.write_byte_data(address, register, value)
        except OSError as exc:
            raise I2CError(
                f"write_byte_data(addr=0x{address:02x}, reg=0x{register:02x}, "
                f"value=0x{value:02x}) failed: {exc}"
            ) from exc

    def read_byte_data(self, address: int, register: int) -> int:
        try:
            return self._bus.read_byte_data(address, register)
        except OSError as exc:
            raise I2CError(
                f"read_byte_data(addr=0x{address:02x}, "
                f"reg=0x{register:02x}) failed: {exc}"
            ) from exc

    def read_raw_bytes(self, address: int, length: int) -> list[int]:
        """Read raw bytes with no register pointer (for MCP3221).

        Uses i2c_rdwr so the bus transaction is just
        ``[start, address|R, ack, byte..., nack, stop]`` with no
        leading write of a register address. The MCP3221 expects
        exactly this.
        """
        try:
            msg = self._i2c_msg.read(address, length)
            self._bus.i2c_rdwr(msg)
            return list(msg)
        except OSError as exc:
            raise I2CError(
                f"read_raw_bytes(addr=0x{address:02x}, length={length}) "
                f"failed: {exc}"
            ) from exc

    def close(self) -> None:
        try:
            self._bus.close()
        except OSError:  # pragma: no cover - best-effort cleanup
            logger.exception("error closing I2C bus %d", self._bus_number)


def open_smbus_bus(bus_number: int) -> SmbusBus:
    """Factory: open the real :mod:`smbus2` bus, raise :class:`I2CError` on failure."""
    return SmbusBus(bus_number)


# ===================================================================
# TCA6416A — 16-bit I/O expander
# ===================================================================


class TCA6416A:
    """Driver for the Texas Instruments TCA6416A 16-bit I2C I/O expander.

    Register map (from the datasheet, table 8-4):

    =====  ==========================  =====
    Reg    Name                        R/W
    =====  ==========================  =====
    0x00   Input Port 0 (P00..P07)     R
    0x01   Input Port 1 (P10..P17)     R
    0x02   Output Port 0               R/W
    0x03   Output Port 1               R/W
    0x04   Polarity Inversion 0        R/W
    0x05   Polarity Inversion 1        R/W
    0x06   Configuration 0 (direction) R/W
    0x07   Configuration 1 (direction) R/W
    =====  ==========================  =====

    For the configuration registers, ``1`` = input, ``0`` = output.
    Polarity inversion is left at 0 (no inversion); active-low logic
    is handled in software by :class:`GPIOControls`.

    Parameters
    ----------
    bus
        Anything matching :class:`I2CBusLike`.
    address
        7-bit I2C address (0x20 with ADDR pin to GND, 0x21 with
        ADDR to VCCP).
    config_port0, config_port1
        Direction bytes written on :meth:`init`.
    initial_outputs_port0, initial_outputs_port1
        Initial state of the output port registers. Defaults to
        0xFF (all high), which is the chip's power-on default and
        also the correct "LEDs off" state for common-anode LEDs.
    """

    REG_INPUT_PORT_0 = 0x00
    REG_INPUT_PORT_1 = 0x01
    REG_OUTPUT_PORT_0 = 0x02
    REG_OUTPUT_PORT_1 = 0x03
    REG_POLARITY_0 = 0x04
    REG_POLARITY_1 = 0x05
    REG_CONFIG_0 = 0x06
    REG_CONFIG_1 = 0x07

    def __init__(
        self,
        bus: I2CBusLike,
        address: int,
        *,
        config_port0: int = 0xFF,
        config_port1: int = 0xFF,
        initial_outputs_port0: int = 0xFF,
        initial_outputs_port1: int = 0xFF,
    ) -> None:
        if not (0x00 <= address <= 0x7F):
            raise ValueError(f"I2C address out of range: 0x{address:02x}")
        for name, val in (
            ("config_port0", config_port0),
            ("config_port1", config_port1),
            ("initial_outputs_port0", initial_outputs_port0),
            ("initial_outputs_port1", initial_outputs_port1),
        ):
            if not (0x00 <= val <= 0xFF):
                raise ValueError(f"{name} must be 0..0xFF, got 0x{val:x}")

        self._bus = bus
        self._address = address
        self._config_port0 = config_port0
        self._config_port1 = config_port1
        self._initial_outputs_port0 = initial_outputs_port0
        self._initial_outputs_port1 = initial_outputs_port1

        # Cached output state so write_output_pin doesn't need to
        # read-modify-write on every call.
        self._cached_output_port0 = initial_outputs_port0
        self._cached_output_port1 = initial_outputs_port1
        self._initialized = False

    # ----- lifecycle --------------------------------------------------

    def init(self) -> None:
        """Configure the chip: polarity, outputs, then directions.

        Order matters: set output values BEFORE flipping pins to
        outputs, otherwise the pin briefly drives whatever was last
        in the output register (probably 0xFF from power-on, so
        usually harmless, but doing it in this order is safer).
        """
        # Polarity: no inversion. Active-low is handled in software.
        self._bus.write_byte_data(self._address, self.REG_POLARITY_0, 0x00)
        self._bus.write_byte_data(self._address, self.REG_POLARITY_1, 0x00)

        # Pre-load the output registers with the safe default.
        self._bus.write_byte_data(
            self._address, self.REG_OUTPUT_PORT_0, self._initial_outputs_port0
        )
        self._bus.write_byte_data(
            self._address, self.REG_OUTPUT_PORT_1, self._initial_outputs_port1
        )

        # Now apply the direction config — pins flip from input
        # (Hi-Z) to whichever direction we want.
        self._bus.write_byte_data(
            self._address, self.REG_CONFIG_0, self._config_port0
        )
        self._bus.write_byte_data(
            self._address, self.REG_CONFIG_1, self._config_port1
        )

        self._initialized = True
        logger.info(
            "TCA6416A at 0x%02x initialised: "
            "config=(0x%02x, 0x%02x) outputs=(0x%02x, 0x%02x)",
            self._address,
            self._config_port0, self._config_port1,
            self._initial_outputs_port0, self._initial_outputs_port1,
        )

    # ----- reads ------------------------------------------------------

    def read_inputs(self) -> int:
        """Return all 16 input pins as one int, P17..P00 packed MSB-first.

        Bit 0 of the result is P00, bit 7 is P07, bit 8 is P10, bit
        15 is P17. Reads BOTH port registers; bits configured as
        outputs reflect whatever is actually on the pin (the
        datasheet says input registers reflect the pin state
        regardless of direction).
        """
        port0 = self._bus.read_byte_data(self._address, self.REG_INPUT_PORT_0)
        port1 = self._bus.read_byte_data(self._address, self.REG_INPUT_PORT_1)
        return (port1 << 8) | port0

    # ----- writes -----------------------------------------------------

    def write_output_pin(self, pin_index: int, value: bool) -> None:
        """Set one output pin HIGH (True) or LOW (False).

        ``pin_index`` is the linear pin number 0..15, where 0..7 are
        P00..P07 and 8..15 are P10..P17. Caller is responsible for
        configuring the pin as an output via ``config_port*`` first;
        writes to input pins are harmless but have no visible effect.
        """
        if not (0 <= pin_index <= 15):
            raise ValueError(f"pin_index must be 0..15, got {pin_index}")

        if pin_index < 8:
            bit = 1 << pin_index
            new_value = (
                self._cached_output_port0 | bit if value
                else self._cached_output_port0 & ~bit & 0xFF
            )
            if new_value != self._cached_output_port0:
                self._bus.write_byte_data(
                    self._address, self.REG_OUTPUT_PORT_0, new_value
                )
                self._cached_output_port0 = new_value
        else:
            bit = 1 << (pin_index - 8)
            new_value = (
                self._cached_output_port1 | bit if value
                else self._cached_output_port1 & ~bit & 0xFF
            )
            if new_value != self._cached_output_port1:
                self._bus.write_byte_data(
                    self._address, self.REG_OUTPUT_PORT_1, new_value
                )
                self._cached_output_port1 = new_value

    # ----- probe ------------------------------------------------------

    def probe(self) -> None:
        """Read the configuration register to confirm the chip responds.

        Useful as a presence check during startup. Raises
        :class:`I2CError` if the chip is missing or unresponsive.
        """
        # Reading any register exercises a full I2C transaction.
        # Config 0 is convenient since its power-on default is
        # known (0xFF — all inputs), so we can also sanity-check
        # the read.
        _ = self._bus.read_byte_data(self._address, self.REG_CONFIG_0)


# ===================================================================
# MCP3221 — 12-bit single-channel ADC
# ===================================================================


class MCP3221:
    """Driver for the Microchip MCP3221 12-bit I2C ADC.

    The MCP3221 has no command interface. A read transaction is:

      [Start | addr<<1 | R=1] [ack] [byte_hi] [ack] [byte_lo] [nack] [Stop]

    ``byte_hi`` carries 4 leading zero bits then the top 4 ADC bits;
    ``byte_lo`` carries the bottom 8 ADC bits. Result range is
    0..4095 corresponding to 0V..VDD on AIN.

    Parameters
    ----------
    bus
        Anything matching :class:`I2CBusLike`. Must implement
        ``read_raw_bytes`` (the MCP3221 cannot be read via the
        register-style ``read_byte_data``).
    address
        7-bit I2C address. The A5 part variant defaults to 0x4D.
    vdd_volts
        Supply voltage on the chip's VDD pin. Used only for
        :meth:`read_voltage`; the raw and normalised reads are
        independent of supply.
    """

    MAX_RAW = 4095  # 12-bit full-scale value

    def __init__(
        self,
        bus: I2CBusLike,
        address: int,
        *,
        vdd_volts: float = 3.3,
    ) -> None:
        if not (0x00 <= address <= 0x7F):
            raise ValueError(f"I2C address out of range: 0x{address:02x}")
        if vdd_volts <= 0:
            raise ValueError(f"vdd_volts must be positive, got {vdd_volts}")

        self._bus = bus
        self._address = address
        self._vdd = vdd_volts

    # ----- reads ------------------------------------------------------

    def read_raw(self) -> int:
        """Read one conversion, return raw 12-bit value (0..4095).

        The conversion takes ~10 µs internally and overlaps with the
        I2C transaction time, so back-to-back reads at 400 kHz I2C
        give roughly 22 kSPS, far more than we need for a zoom knob.
        """
        data = self._bus.read_raw_bytes(self._address, 2)
        if len(data) != 2:
            raise I2CError(
                f"MCP3221 at 0x{self._address:02x} returned {len(data)} "
                f"bytes; expected 2"
            )
        # Top 4 bits live in the lower nibble of byte 0 (the upper
        # nibble is always zero per datasheet section 5.3.2).
        high = data[0] & 0x0F
        low = data[1] & 0xFF
        return (high << 8) | low

    def read_normalized(self) -> float:
        """Read one conversion, return value as 0.0..1.0 fraction of full scale."""
        return self.read_raw() / self.MAX_RAW

    def read_voltage(self) -> float:
        """Read one conversion, return value in volts (assuming VDD reference)."""
        return self.read_normalized() * self._vdd

    # ----- probe ------------------------------------------------------

    def probe(self) -> None:
        """Confirm the chip ACKs by attempting one read.

        Raises :class:`I2CError` if the chip is absent. Used at
        startup to decide whether to wire up the zoom control or
        skip it.
        """
        _ = self.read_raw()


# ===================================================================
# Helpers
# ===================================================================


_PIN_NAME_RE_PORT0 = ("P00", "P01", "P02", "P03", "P04", "P05", "P06", "P07")
_PIN_NAME_RE_PORT1 = ("P10", "P11", "P12", "P13", "P14", "P15", "P16", "P17")
_ALL_PIN_NAMES = _PIN_NAME_RE_PORT0 + _PIN_NAME_RE_PORT1


def pin_name_to_index(pin_name: str) -> int:
    """Convert a TCA6416A pin name like ``P05`` or ``P13`` to linear 0..15.

    ``P00..P07`` → ``0..7``
    ``P10..P17`` → ``8..15``

    Raises :class:`ValueError` on any other input.
    """
    if not isinstance(pin_name, str):
        raise ValueError(f"pin name must be a string, got {type(pin_name).__name__}")
    if pin_name in _PIN_NAME_RE_PORT0:
        return _PIN_NAME_RE_PORT0.index(pin_name)
    if pin_name in _PIN_NAME_RE_PORT1:
        return 8 + _PIN_NAME_RE_PORT1.index(pin_name)
    raise ValueError(
        f"unknown TCA6416A pin name {pin_name!r}; "
        f"expected one of {_ALL_PIN_NAMES}"
    )


def parse_address(raw: Any) -> int:
    """Parse a YAML-loaded I2C address.

    YAML may give us an int (0x20 written as ``0x20``), a string
    (``"0x20"``), or even a decimal int. Normalise to a 7-bit int.
    """
    if isinstance(raw, int):
        addr = raw
    elif isinstance(raw, str):
        addr = int(raw, 0)  # honours 0x / 0o / 0b / plain decimal
    else:
        raise ValueError(
            f"I2C address must be int or string, got {type(raw).__name__}"
        )

    if not (0x00 <= addr <= 0x7F):
        raise ValueError(f"I2C address 0x{addr:02x} is outside 7-bit range")
    return addr