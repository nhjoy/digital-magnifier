# Digital Magnifier

Portable digital magnifier for low-vision children, built on a Raspberry Pi
Compute Module 5 with the Pi Camera Module 3.

**Current version: MVP 0.2** — real Pi camera input on the CM5.

## What works

- Live magnified view from the Pi Camera Module 3 (or a USB webcam / synthetic
  frame on a dev machine)
- Digital zoom 1× to 8× with digital pan
- Five vision filters: normal, grayscale, high-contrast (CLAHE),
  inverted, binary (adaptive threshold)
- Freeze / unfreeze with frozen-frame cache
- Capture-to-disk with timestamped PNGs
- Reset view (one-press return to defaults)
- Resilient main loop — bad frames, save failures, and handler exceptions
  are logged but never crash the device
- Pi Cam 3 continuous autofocus, configurable AWB / AE / HDR / rotation
- Fullscreen display on the DSI panel

## Stubbed (deferred to later MVPs)

- Gallery UI showing captured images (MVP 0.4)
- Menu UI (MVP 0.4)
- Real GPIO buttons + nav switch + ADC pot (MVP 0.3 — currently keyboard mock)
- Accessibility polish, audio feedback (MVP 0.4)
- systemd auto-start, watchdog, safe shutdown (MVP 0.5)

## Project layout

```
digital-magnifier/
├── pyproject.toml           # package definition + dependencies
├── README.md
├── requirements.txt         # convenience (pyproject is source of truth)
├── .gitignore
├── config/
│   ├── app_config.yaml      # app behaviour, filters, capture, logging
│   ├── camera_config.yaml   # camera source, resolution, Pi cam settings
│   └── hardware_pins.yaml   # mock keyboard + future GPIO map
├── src/
│   └── digital_magnifier/
│       ├── main.py
│       ├── core/
│       │   ├── app_controller.py    # main orchestrator
│       │   ├── events.py            # AppEvent enum
│       │   └── state_machine.py
│       ├── hal/
│       │   ├── camera_base.py       # CameraSensor ABC
│       │   ├── camera_sensor.py     # MockCameraSensor + PiCameraSensor
│       │   ├── controls_base.py     # ControlsHAL ABC
│       │   └── mock_controls.py     # keyboard → AppEvent
│       ├── processing/
│       │   ├── magnifier.py         # apply_zoom (digital zoom + pan)
│       │   └── vision_filters.py    # filters incl. adaptive-threshold binary
│       ├── storage/
│       │   └── image_saver.py
│       └── utils/
│           ├── config_loader.py
│           └── logger.py
├── tests/
│   └── unit/                # 100+ pytest tests
└── captures/                # output of capture button (gitignored)
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
    python3-pytest \
    git

# 2) Clone (one-time)
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/<you>/digital-magnifier.git
cd digital-magnifier

# 3) Create a venv that can see the apt-installed picamera2
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install -e ".[dev]"

# 4) Sanity check
pytest tests/ -v
rpicam-hello --timeout 5000   # confirm the camera hardware is alive

# 5) Run
digital-magnifier --log-level INFO
```

The `--system-site-packages` flag is the important detail — it lets the venv
see the apt-installed `python3-picamera2` and `python3-libcamera`. Without it
the app falls back to the mock camera.

After step 3, you only need `source venv/bin/activate` at the start of each
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

## Keyboard controls

| Key       | Action                          | Notes                  |
|-----------|---------------------------------|------------------------|
| `space`   | Freeze / Unfreeze               | Button 1               |
| `c`       | Capture image                   | Button 2 — saves PNG   |
| `f`       | Cycle filter                    | Button 3               |
| `g`       | Open / close gallery (stub)     | Button 4               |
| `r`       | Reset view                      | Button 5               |
| `p` / `P` | Power short / long press        | Button 6               |
| `w/a/s/d` | Pan up/left/down/right          | Nav switch directional |
| `x`       | Reset view                      | Nav switch centre      |
| `+` / `=` | Zoom in                         | Pot increase           |
| `-` / `_` | Zoom out                        | Pot decrease           |
| `[`       | Back                            | Exit gallery / menu    |
| `q`       | Quit                            | Dev convenience        |

In MVP 0.3 these map to real GPIO inputs; the events are identical.

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