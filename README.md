# Digital Magnifier — MVP 0.1

Portable digital magnifier for low-vision children, built on a
Raspberry Pi Compute Module 5. MVP 0.1 is the **software
foundation**: the full application architecture runs on a laptop
with keyboard-mocked controls, ready for real hardware to drop
in during MVP 0.2 (Pi Camera) and MVP 0.3 (GPIO buttons + nav
switch + pot).

## What works in MVP 0.1

- Real-time digital zoom (1× to 8×) with digital pan
- Five vision filters: normal, grayscale, high-contrast, inverted,
  binary (adaptive threshold)
- Freeze / unfreeze with frozen-frame cache (no re-processing
  while paused)
- Capture-to-disk with timestamped PNG filenames and visual flash
- Reset view (one-press return to defaults)
- State machine with seven states and full transition table
- Resilient main loop — bad frames, save failures, and handler
  exceptions are logged but never crash the device
- Mock camera with auto-fallback (webcam → fallback image →
  procedurally-generated text-on-page)
- Stub screens for gallery and menu (UI fills in MVP 0.4)

## Project layout

```
digital-magnifier/
├── config/
│   ├── app_config.yaml          # app behaviour, filters, capture, logging
│   ├── camera_config.yaml       # camera source & resolution
│   └── hardware_pins.yaml       # mock keyboard + future GPIO map
├── src/
│   └── digital_magnifier/
│       ├── main.py              # entry point
│       ├── core/
│       │   ├── app_controller.py # orchestrator
│       │   ├── events.py         # AppEvent enum
│       │   └── state_machine.py
│       ├── hal/
│       │   ├── camera_base.py    # ABC
│       │   ├── camera_sensor.py  # MockCameraSensor
│       │   ├── controls_base.py  # ABC
│       │   └── mock_controls.py  # keyboard → AppEvent
│       ├── processing/
│       │   ├── magnifier.py      # apply_zoom (digital zoom + pan)
│       │   └── vision_filters.py # filters incl. adaptive-threshold binary
│       ├── storage/
│       │   └── image_saver.py
│       └── utils/
│           ├── config_loader.py
│           └── logger.py
├── tests/
│   └── unit/                    # ~100 pytest tests
├── captures/                    # output of the capture button (gitignored)
├── requirements.txt
└── .gitignore
```

## Quick start

```bash
# 1) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# or: .\venv\Scripts\activate    # Windows

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
PYTHONPATH=src python -m digital_magnifier.main

# Or: increase log verbosity
PYTHONPATH=src python -m digital_magnifier.main --log-level DEBUG
```

The application opens an OpenCV window titled "Digital Magnifier".
**Click the window once** to give it keyboard focus, then use the
keys below.

If a webcam is available it will be used; otherwise a synthetic
text-on-page frame is generated automatically (look for the red
"SYNTHETIC TEST FRAME" watermark in the bottom-left).

## Keyboard controls (mock)

These are the bindings in `config/hardware_pins.yaml`. They
correspond to the six physical buttons + 4-way nav + pot that the
real device will use in MVP 0.3.

| Key       | Action                          | Notes                  |
|-----------|---------------------------------|------------------------|
| `space`   | Freeze / Unfreeze               | Button 1               |
| `c`       | Capture image                   | Button 2 — saves PNG   |
| `f`       | Cycle filter                    | Button 3               |
| `g`       | Open / close gallery (stub)     | Button 4               |
| `r`       | Reset view                      | Button 5               |
| `p`       | Power short press (toggle menu) | Button 6 short         |
| `P`       | Power long press (shutdown)     | Button 6 long (shift+p)|
| `w/a/s/d` | Pan up/left/down/right          | Nav switch directional |
| `x`       | Reset view                      | Nav switch centre      |
| `+` / `=` | Zoom in                         | Pot increase           |
| `-` / `_` | Zoom out                        | Pot decrease           |
| `[`       | Back                            | Exit gallery / menu    |
| `q`       | Quit (dev convenience)          |                        |

## Running the tests

```bash
PYTHONPATH=src pytest tests/ -v
```

## Cleanup pass (one-time, recommended after dropping in MVP 0.1)

```bash
# Remove the legacy scratch directory shipped in the early zip
rm -rf src/.temp

# Remove all __pycache__ directories
find . -type d -name "__pycache__" -exec rm -rf {} +

# Convert any CRLF line endings to LF (requires dos2unix)
find src/ tests/ config/ -type f \( -name "*.py" -o -name "*.yaml" \) -exec dos2unix {} +
```

## What's *not* in MVP 0.1

Deferred to later MVPs:

- MVP 0.2 — Real Pi Camera Module 3 via `picamera2`
- MVP 0.3 — Real GPIO buttons + nav switch + ADC pot, including
  edge-based long-press timing for the power button
- MVP 0.4 — Real gallery UI, real menu UI, accessibility polish
  (large indicators, audio feedback)
- MVP 0.5 — systemd auto-start, watchdog, safe shutdown, power
  monitoring, optional OCR/TTS

The architecture is set up so each of these is mostly a HAL
swap or a new module: nothing above the HAL needs to change.# digital-magnifer
