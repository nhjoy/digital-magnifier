# tests/hardware — live-hardware probes

These scripts talk to real I²C devices on a running CM5. They are
**not** run as part of `pytest tests/` (the unit suite mocks all I/O
deliberately) and they require:

- a wired-up breadboard with the relevant chip,
- I²C enabled in `raspi-config`,
- the `python3-smbus2` apt package or `pip install smbus2`.

Run them by hand when bringing up new hardware, after a wiring change,
or when chasing an intermittent fault. Each probe exercises the same
driver code that the app uses, so if a probe works, the app's
controls path will work.

## Scripts

| Script | What it does |
|---|---|
| `probe_tca6416a.py` | Streams nav-switch / button press-release events to the terminal. Optional `--led-test` cycles the RGB LED to confirm the output side. |
| `probe_mcp3221.py` | Continuous read-out of the zoom potentiometer. Shows raw value, voltage, quantised bucket, and an ASCII bar. Use this to confirm the pot sweeps cleanly across its range before trusting it in the app. |

## How to run

From the project root:

```bash
# Buttons & nav switch (Ctrl-C to exit)
python3 tests/hardware/probe_tca6416a.py

# Plus an LED self-test up front
python3 tests/hardware/probe_tca6416a.py --led-test

# Zoom pot (Ctrl-C to exit). Requires MCP3221 wired.
python3 tests/hardware/probe_mcp3221.py
```

Both scripts read `config/hardware_pins.yaml`, so renaming pins or
changing the I²C addresses in YAML is automatically reflected here too.

## What "good" looks like

**probe_tca6416a.py** — every button press logs a `PRESSED` line, every
release logs a `released` line with the hold-time. Held buttons stay
quiet between press and release (no spurious events). Unwired pins
stay HIGH (logical "not pressed") and produce nothing.

**probe_mcp3221.py** — at one end of the pot's travel you should see
`raw≈0` and `V≈0.0`. At the other end, `raw≈4095` and `V≈3.3`. The
bucket counter should march cleanly from 0 to 15 (or 15 to 0) as you
turn the knob, with at most ±1 jitter at rest. Wild jumps or a stuck
bucket mean either a bad wiper, a missing decoupling cap, or a noisy
3V3 rail.

## When a probe fails

Most failure modes log a single descriptive line and exit:

- `failed to open I2C bus 1` — I²C not enabled (`sudo raspi-config nonint do_i2c 0`).
- `TCA6416A at 0x20: not responding` — chip not detected. Check
  `i2cdetect -y 1`; you should see `20` in the grid.
- `MCP3221 at 0x4D: not responding` — same idea (`4d` in the grid).
- `smbus2 not installed` — `sudo apt install python3-smbus2`.

Anything else is an unexpected bug — capture the traceback and file
an issue (or just rerun under `python3 -X dev` and paste the output).
