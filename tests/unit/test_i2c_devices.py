"""Unit tests for the TCA6416A and MCP3221 I2C drivers.

Uses an in-memory ``FakeBus`` (no smbus2 required) so the tests run
identically on the dev laptop and the CM5.
"""

from __future__ import annotations

import pytest

from digital_magnifier.hal.i2c_devices import (
    I2CError,
    MCP3221,
    TCA6416A,
    parse_address,
    pin_name_to_index,
)


# ===================================================================
# Fake bus
# ===================================================================


class FakeBus:
    """In-memory I2C bus.

    Behaviour intentionally mirrors what smbus2 does at the API
    level: register-style reads/writes via ``write_byte_data`` /
    ``read_byte_data``, and raw reads via ``read_raw_bytes``.
    """

    def __init__(self) -> None:
        # {(address, register): value}
        self.registers: dict[tuple[int, int], int] = {}
        # Sequence of (address, length, bytes) to return for raw reads.
        # Each call to read_raw_bytes pops one entry from the head.
        self.raw_read_queue: list[list[int]] = []
        self.write_log: list[tuple[int, int, int]] = []
        self.read_log: list[tuple[int, int, int]] = []
        self.raw_read_log: list[tuple[int, int]] = []
        self.closed = False
        # Optional: per-address failure injection.
        self.fail_on_address: int | None = None
        self.fail_on_raw_read: bool = False

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        if address == self.fail_on_address:
            raise I2CError(f"injected write failure at 0x{address:02x}")
        self.registers[(address, register)] = value
        self.write_log.append((address, register, value))

    def read_byte_data(self, address: int, register: int) -> int:
        if address == self.fail_on_address:
            raise I2CError(f"injected read failure at 0x{address:02x}")
        value = self.registers.get((address, register), 0xFF)
        self.read_log.append((address, register, value))
        return value

    def read_raw_bytes(self, address: int, length: int) -> list[int]:
        if self.fail_on_raw_read or address == self.fail_on_address:
            raise I2CError(f"injected raw-read failure at 0x{address:02x}")
        self.raw_read_log.append((address, length))
        if self.raw_read_queue:
            data = self.raw_read_queue.pop(0)
            return data[:length] if len(data) >= length else data + [0] * (length - len(data))
        return [0] * length

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def bus() -> FakeBus:
    return FakeBus()


# ===================================================================
# pin_name_to_index helper
# ===================================================================


class TestPinNameToIndex:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("P00", 0), ("P01", 1), ("P02", 2), ("P03", 3),
            ("P04", 4), ("P05", 5), ("P06", 6), ("P07", 7),
            ("P10", 8), ("P11", 9), ("P12", 10), ("P13", 11),
            ("P14", 12), ("P15", 13), ("P16", 14), ("P17", 15),
        ],
    )
    def test_valid_names(self, name, expected):
        assert pin_name_to_index(name) == expected

    @pytest.mark.parametrize(
        "name",
        ["P08", "P09", "P18", "P19", "P20", "p00", "", "GPIO5", "5"],
    )
    def test_invalid_names(self, name):
        with pytest.raises(ValueError):
            pin_name_to_index(name)

    def test_non_string(self):
        with pytest.raises(ValueError):
            pin_name_to_index(5)  # type: ignore[arg-type]


# ===================================================================
# parse_address helper
# ===================================================================


class TestParseAddress:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (0x20, 0x20),
            (32, 0x20),
            (0x4D, 0x4D),
            ("0x20", 0x20),
            ("0x4d", 0x4D),
            ("32", 32),
        ],
    )
    def test_valid(self, raw, expected):
        assert parse_address(raw) == expected

    @pytest.mark.parametrize("raw", [-1, 0x80, 0xFF, "0x80"])
    def test_out_of_range(self, raw):
        with pytest.raises(ValueError):
            parse_address(raw)

    def test_wrong_type(self):
        with pytest.raises(ValueError):
            parse_address(None)  # type: ignore[arg-type]

    def test_garbage_string(self):
        with pytest.raises(ValueError):
            parse_address("not a number")


# ===================================================================
# TCA6416A
# ===================================================================


class TestTCA6416AConstruction:
    def test_default_init(self, bus):
        chip = TCA6416A(bus, address=0x20)
        assert chip is not None

    @pytest.mark.parametrize("address", [-1, 0x80, 0xFF])
    def test_bad_address_raises(self, bus, address):
        with pytest.raises(ValueError):
            TCA6416A(bus, address=address)

    @pytest.mark.parametrize(
        "kwarg",
        ["config_port0", "config_port1",
         "initial_outputs_port0", "initial_outputs_port1"],
    )
    def test_bad_byte_value_raises(self, bus, kwarg):
        with pytest.raises(ValueError):
            TCA6416A(bus, address=0x20, **{kwarg: 0x100})


class TestTCA6416AInit:
    def test_writes_polarity_outputs_then_config(self, bus):
        chip = TCA6416A(
            bus, address=0x20,
            config_port0=0xFF, config_port1=0xC7,
            initial_outputs_port0=0xFF, initial_outputs_port1=0xFF,
        )
        chip.init()

        # All six writes should have happened
        assert len(bus.write_log) == 6

        # In the expected order: polarity, outputs, config
        assert bus.write_log[0] == (0x20, 0x04, 0x00)  # polarity 0
        assert bus.write_log[1] == (0x20, 0x05, 0x00)  # polarity 1
        assert bus.write_log[2] == (0x20, 0x02, 0xFF)  # output 0
        assert bus.write_log[3] == (0x20, 0x03, 0xFF)  # output 1
        assert bus.write_log[4] == (0x20, 0x06, 0xFF)  # config 0
        assert bus.write_log[5] == (0x20, 0x07, 0xC7)  # config 1

    def test_io_failure_propagates(self, bus):
        bus.fail_on_address = 0x20
        chip = TCA6416A(bus, address=0x20)
        with pytest.raises(I2CError):
            chip.init()


class TestTCA6416AReadInputs:
    def test_combines_both_ports(self, bus):
        chip = TCA6416A(bus, address=0x20)
        bus.registers[(0x20, 0x00)] = 0xAB  # port 0
        bus.registers[(0x20, 0x01)] = 0xCD  # port 1
        assert chip.read_inputs() == 0xCDAB

    def test_all_zero(self, bus):
        chip = TCA6416A(bus, address=0x20)
        bus.registers[(0x20, 0x00)] = 0x00
        bus.registers[(0x20, 0x01)] = 0x00
        assert chip.read_inputs() == 0x0000

    def test_all_high(self, bus):
        chip = TCA6416A(bus, address=0x20)
        bus.registers[(0x20, 0x00)] = 0xFF
        bus.registers[(0x20, 0x01)] = 0xFF
        assert chip.read_inputs() == 0xFFFF

    def test_specific_pin_low(self, bus):
        """Pressing nav switch A (P00) should show bit 0 cleared."""
        chip = TCA6416A(bus, address=0x20)
        # All HIGH except P00 (button on P00 pressed = active-low)
        bus.registers[(0x20, 0x00)] = 0b11111110
        bus.registers[(0x20, 0x01)] = 0xFF
        result = chip.read_inputs()
        # Bit 0 cleared, all others set
        assert result == 0xFFFE
        assert (result & 0x0001) == 0  # P00 = pressed

    def test_io_failure_propagates(self, bus):
        chip = TCA6416A(bus, address=0x20)
        bus.fail_on_address = 0x20
        with pytest.raises(I2CError):
            chip.read_inputs()


class TestTCA6416AWriteOutput:
    def _init_with_known_cache(self, bus) -> TCA6416A:
        chip = TCA6416A(
            bus, address=0x20,
            initial_outputs_port0=0xFF,
            initial_outputs_port1=0xFF,
        )
        chip.init()
        bus.write_log.clear()  # discard init writes
        return chip

    def test_set_pin_high_when_already_high_is_noop(self, bus):
        chip = self._init_with_known_cache(bus)
        chip.write_output_pin(13, True)  # P15, currently HIGH
        # No write because the cached state already matches
        assert bus.write_log == []

    def test_clear_pin_writes_new_value(self, bus):
        chip = self._init_with_known_cache(bus)
        # P13 (index 11, bit 3 of port 1) — turn LED on (drive LOW)
        chip.write_output_pin(11, False)
        # Should clear bit 3 of port 1's cached 0xFF
        assert bus.write_log == [(0x20, 0x03, 0xF7)]

    def test_consecutive_writes_use_cache(self, bus):
        chip = self._init_with_known_cache(bus)
        chip.write_output_pin(11, False)  # P13 LOW: 0xF7
        chip.write_output_pin(12, False)  # P14 LOW: 0xF7 & ~0x10 = 0xE7
        chip.write_output_pin(13, False)  # P15 LOW: 0xE7 & ~0x20 = 0xC7
        assert bus.write_log == [
            (0x20, 0x03, 0xF7),
            (0x20, 0x03, 0xE7),
            (0x20, 0x03, 0xC7),
        ]

    def test_port0_pin(self, bus):
        chip = self._init_with_known_cache(bus)
        chip.write_output_pin(0, False)  # P00 LOW
        assert bus.write_log == [(0x20, 0x02, 0xFE)]

    @pytest.mark.parametrize("index", [-1, 16, 100])
    def test_bad_index_raises(self, bus, index):
        chip = self._init_with_known_cache(bus)
        with pytest.raises(ValueError):
            chip.write_output_pin(index, True)


class TestTCA6416AProbe:
    def test_probe_reads_config_register(self, bus):
        chip = TCA6416A(bus, address=0x20)
        chip.probe()
        assert (0x20, 0x06, 0xFF) in bus.read_log  # default 0xFF for missing key

    def test_probe_raises_on_missing_chip(self, bus):
        chip = TCA6416A(bus, address=0x20)
        bus.fail_on_address = 0x20
        with pytest.raises(I2CError):
            chip.probe()


# ===================================================================
# MCP3221
# ===================================================================


class TestMCP3221Construction:
    def test_default_init(self, bus):
        adc = MCP3221(bus, address=0x4D)
        assert adc is not None

    @pytest.mark.parametrize("address", [-1, 0x80])
    def test_bad_address_raises(self, bus, address):
        with pytest.raises(ValueError):
            MCP3221(bus, address=address)

    @pytest.mark.parametrize("vdd", [0, -3.3])
    def test_non_positive_vdd_raises(self, bus, vdd):
        with pytest.raises(ValueError):
            MCP3221(bus, address=0x4D, vdd_volts=vdd)


class TestMCP3221ReadRaw:
    def test_decodes_12_bit_value(self, bus):
        adc = MCP3221(bus, address=0x4D)
        # Upper byte has 4 zero bits + top 4 ADC bits; lower has 8 ADC bits.
        # For raw = 0x123: upper = 0x01, lower = 0x23
        bus.raw_read_queue.append([0x01, 0x23])
        assert adc.read_raw() == 0x123

    def test_decodes_full_scale(self, bus):
        adc = MCP3221(bus, address=0x4D)
        bus.raw_read_queue.append([0x0F, 0xFF])
        assert adc.read_raw() == 4095

    def test_decodes_zero(self, bus):
        adc = MCP3221(bus, address=0x4D)
        bus.raw_read_queue.append([0x00, 0x00])
        assert adc.read_raw() == 0

    def test_decodes_mid_scale(self, bus):
        adc = MCP3221(bus, address=0x4D)
        bus.raw_read_queue.append([0x08, 0x00])
        assert adc.read_raw() == 2048

    def test_ignores_upper_nibble_of_high_byte(self, bus):
        """Datasheet says top 4 bits of byte 0 are always zero; we mask anyway."""
        adc = MCP3221(bus, address=0x4D)
        bus.raw_read_queue.append([0xF1, 0x23])  # garbage top nibble
        assert adc.read_raw() == 0x123

    def test_short_response_raises(self, bus):
        adc = MCP3221(bus, address=0x4D)
        # Push only one byte then back-fill — read_raw_bytes pads with zero.
        # Override the fake so we hit the length-check path: monkey-patch a
        # one-shot wrapper.
        original = bus.read_raw_bytes

        def short_read(address, length):  # noqa: ARG001
            original(address, length)
            return [0x01]

        bus.read_raw_bytes = short_read  # type: ignore[assignment]
        with pytest.raises(I2CError, match="returned 1 bytes"):
            adc.read_raw()

    def test_io_failure_propagates(self, bus):
        adc = MCP3221(bus, address=0x4D)
        bus.fail_on_raw_read = True
        with pytest.raises(I2CError):
            adc.read_raw()


class TestMCP3221Conversions:
    def test_read_normalized(self, bus):
        adc = MCP3221(bus, address=0x4D)
        bus.raw_read_queue.append([0x08, 0x00])  # 2048
        result = adc.read_normalized()
        assert abs(result - 2048 / 4095) < 1e-6

    def test_read_voltage_default_vdd(self, bus):
        adc = MCP3221(bus, address=0x4D, vdd_volts=3.3)
        bus.raw_read_queue.append([0x0F, 0xFF])  # 4095 = full scale
        result = adc.read_voltage()
        assert abs(result - 3.3) < 1e-6

    def test_read_voltage_5v(self, bus):
        adc = MCP3221(bus, address=0x4D, vdd_volts=5.0)
        bus.raw_read_queue.append([0x08, 0x00])  # 2048 ≈ half
        result = adc.read_voltage()
        assert abs(result - 2.5) < 5e-3   # ~half of 5V

    def test_read_voltage_zero(self, bus):
        adc = MCP3221(bus, address=0x4D, vdd_volts=3.3)
        bus.raw_read_queue.append([0x00, 0x00])
        assert adc.read_voltage() == 0.0


class TestMCP3221Probe:
    def test_probe_does_one_read(self, bus):
        adc = MCP3221(bus, address=0x4D)
        bus.raw_read_queue.append([0x05, 0x55])
        adc.probe()
        assert bus.raw_read_log == [(0x4D, 2)]

    def test_probe_raises_when_missing(self, bus):
        adc = MCP3221(bus, address=0x4D)
        bus.fail_on_raw_read = True
        with pytest.raises(I2CError):
            adc.probe()