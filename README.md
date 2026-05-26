# Digital Magnifier

Portable digital magnifier for low-vision children, built on a Raspberry Pi
Compute Module 5 with the Pi Camera Module 3.

**Current version: MVP 0.4** — gallery, buzzer, battery monitoring, status overlay.

See [USER_GUIDE.md](USER_GUIDE.md) for how to use the device (written for
parents, caregivers, and older children).

## What works

- Live magnified view from the Pi Camera Module 3 (or a USB webcam / synthetic
  frame on a dev machine)
- Digital zoom 1× to 8× with digital pan
- Seven vision filters (configurable in YAML): normal, grayscale,
  high-contrast (CLAHE), inverted, binary, yellow-on-black, white-on-black
- Filter parameters adjustable in `config/app_config.yaml` (block size,
  threshold offset, CLAHE clip limit) — no code changes needed
- Freeze / unfreeze with frozen-frame cache (snapshot button tap)
- Capture-to-disk with timestamped PNGs (snapshot button 3s hold)
- Gallery UI: browse, zoom, filter, and delete captured images
- Reset view (one-press return to defaults via nav centre)
- **Buzzer feedback** (PS1740P02E passive piezo at 4 kHz): startup, shutdown,
  low-battery alerts, and audio cues
- **Battery monitoring** via Waveshare UPS HAT (E) at I²C 0x2D: live battery
  bar in the overlay, one-time 50% warning, continuous 15% alert (1 Hz beep),
  automatic safe shutdown at 10%
- **Status overlay**: hold ADD3 for 5 seconds to see a fullscreen device
  status screen (battery, mode, zoom, filter, photo count)
- **Loading splash**: 5-second startup splash with progress dots while the
  camera initialises
- Enhanced state-aware overlay with colour-coded state labels and battery bar
- Pi Cam 3 continuous autofocus, configurable AWB / AE / HDR / rotation
- Fullscreen display on the DSI panel
- **GPIO controls**: navigation switch + 6 buttons via TCA6416A I²C I/O
  expander (0x20), keyboard fallback when GPIO isn't available
- **Hardware power button** on GPIO 26 via `dtoverlay=gpio-shutdown` — clean
  OS-level shutdown, separate from app logic
- Resilient main loop — bad frames, save failures, and handler exceptions
  are logged but never crash the device

## Stubbed (deferred to later MVPs)

- Zoom potentiometer via MCP3221 ADC — driver written, awaiting hardware.
  ADD1 / ADD2 buttons stand in for ZOOM_IN / ZOOM_OUT until the ADC arrives.
- RGB status LED — pins are configured as outputs but not yet wired into
  app logic (MVP 0.5)
- Menu UI (MVP 0.5)
- systemd auto-start, watchdog (MVP 0.5)

## Project layout

```
digital-magnifier/
├── pyproject.toml             # package definition + dependencies
├── README.md                  # developer documentation (this file)
├── USER_GUIDE.md              # how-to for parents and caregivers
├── .gitignore
├── config/
│   ├── app_config.yaml        # app behaviour, filters, capture, logging
│   ├── camera_config.yaml     # camera source, resolution, Pi cam settings
│   └── hardware_pins.yaml     # I²C devices + button/nav event bindings
├── src/
│   └── digital_magnifier/
│       ├── main.py
│       ├── core/
│       │   ├── events.py          # AppEvent enum
│       │   ├── state_machine.py   # AppState + transitions
│       │   ├── app_controller.py  # main orchestrator
│       │   └── gallery.py         # gallery UI
│       ├── hal/
│       │   ├── camera_base.py     # CameraSensor ABC
│       │   ├── camera_sensor.py   # webcam / synthetic backends
│       │   ├── pi_camera_sensor.py# Pi Camera Module 3 backend
│       │   ├── controls_base.py   # ControlsHAL ABC
│       │   ├── mock_controls.py   # keyboard + GPIO controls
│       │   ├── i2c_devices.py     # TCA6416A, MCP3221, UPS HAT (E)
│       │   └── buzzer.py          # PS1740P02E passive piezo driver
│       ├── processing/
│       │   ├── magnifier.py       # apply_zoom
│       │   └── vision_filters.py  # apply_filter + 7 filters
│       ├── storage/
│       │   └── image_saver.py     # timestamped PNG saving + listing
│       └── utils/
│           ├── config_loader.py
│           └── logger.py
├── tests/
│   ├── unit/                  # 370+ pytest tests
│   └── hardware/              # on-device probes (buzzer, UPS, I2C)
├── scripts/
│   └── dev/                   # alternative test runners + MVP milestone proofs
└── captures/                  # output of capture button (gitignored)
```

## Setup on the Raspberry Pi CM5

This is the primary development and runtime environment.

```bash
# 1) System dependencies (one-time, via apt — gets hardware-accelerated builds)
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y \
    python3-picamera2 \
    python3-opencv \
    python3-numpy \
    python3-yaml \
    python3-smbus2 \
    python3-pytest \
    i2c-tools \
    git

# 2) Enable I²C if you haven't already
sudo raspi-config nonint do_i2c 0

# 3) Clone (one-time)
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/<you>/digital-magnifier.git
cd digital-magnifier

# 4) Create a venv that can see the apt-installed picamera2 + smbus2
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -e ".[dev]"

# 5) Sanity check
pytest tests/ -v
rpicam-hello --timeout 5000     # confirm the camera hardware is alive
i2cdetect -y 1                  # confirm 0x20 (TCA6416A) shows up

# 6) Run
digital-magnifier --log-level INFO
```

The `--system-site-packages` flag is the important detail — it lets the venv
see the apt-installed `python3-picamera2`, `python3-libcamera`, and
`python3-smbus2`. Without it the app falls back to mock camera and mock
controls.

If the TCA6416A doesn't show up on `i2cdetect`, the app degrades gracefully
to keyboard mock — useful while you're still wiring the breadboard.

After step 4, you only need `source venv/bin/activate` at the start of each
session. Add it to `~/.bashrc` if you want it automatic on SSH login.

## Setup on a development machine (WSL / Linux / macOS, optional)

If you want to develop without the CM5 attached — useful for refactoring
core logic — the same code runs with the mock camera:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
digital-magnifier --log-level INFO
```

Without picamera2 the app auto-detects this and uses `MockCameraSensor`
(webcam if available, otherwise a procedurally-generated test frame).

## Hardware controls (MVP 0.3)

The TCA6416A I/O expander at 0x20 hosts the nav switch and six buttons.
Pin mappings and event bindings live in `config/hardware_pins.yaml`.

| Button     | Action                          | Notes                              |
|------------|---------------------------------|------------------------------------|
| nav up     | Pan up                          | SKRHABE010 directional (P03)       |
| nav down   | Pan down                        | P04                                |
| nav left   | Pan left                        | P00                                |
| nav right  | Pan right                       | P02                                |
| nav centre | Reset view                      | P01                                |
| snapshot   | Capture image                   | P05 — saves timestamped PNG        |
| album      | Open gallery (stub)             | P06                                |
| CLR_BTN    | Reset view                      | P07                                |
| ADD1       | **Zoom in** *(temporary)*       | P10 — moves to FREEZE post-ADC     |
| ADD2       | **Zoom out** *(temporary)*      | P11 — moves to FILTER_NEXT post-ADC|
| ADD3       | Power short / long press        | P12 — held 1s = shutdown           |

The MCP3221 ADC at 0x4D will host the zoom potentiometer once wired. Until
then, ADD1 and ADD2 stand in for ZOOM_IN / ZOOM_OUT. Flip the mapping back
by editing the `button_events:` block in `config/hardware_pins.yaml`.

## Keyboard fallback (development)

When the TCA6416A isn't present (dev laptop, or breadboard not yet wired),
the app falls back to keyboard input. Same events, different transport:

| Key       | Action                          |
|-----------|---------------------------------|
| `space`   | Freeze / Unfreeze               |
| `c`       | Capture image                   |
| `f`       | Cycle filter                    |
| `g`       | Open / close gallery (stub)     |
| `r`       | Reset view                      |
| `p` / `P` | Power short / long press        |
| `w/a/s/d` | Pan up/left/down/right          |
| `x`       | Reset view                      |
| `+` / `=` | Zoom in                         |
| `-` / `_` | Zoom out                        |
| `[`       | Back (exit gallery / menu)      |
| `q`       | Quit (dev convenience)          |

## Configuration

Three YAML files in `config/` drive behaviour. All keys are documented
inline; the most useful tweaks:

- `camera_config.yaml` — `pi_camera.af_mode` (`continuous` / `manual`),
  `pi_camera.lens_position` (manual focus distance in diopters),
  `resolution.width/height` (default 2304×1296 uses the Pi Cam 3's
  sweet-spot binned mode).
- `app_config.yaml` — `app.fullscreen` (true on the CM5, false during
  desktop dev), `capture.flash_duration_ms`, filter order in
  `filters.available`.
- `hardware_pins.yaml` — `hardware.platform: auto` is the default and
  picks the right HAL automatically.

## Running tests

```bash
pytest                       # everything
pytest tests/unit/test_camera_sensor.py -v   # just the camera tests
pytest tests/unit/test_camera_sensor.py::TestPiStart -v   # one class
```

## Development workflow

Edit on the CM5 (via VS Code Remote-SSH or directly), test locally on the
CM5 against the real camera, commit small and often:

```bash
git add . && git commit -m "..." && git push
```

The CM5's SD/eMMC is the single point of failure — push frequently. The
`captures/` directory is gitignored; back it up separately with `rsync`
if you need to preserve test images.

## License

MIT.