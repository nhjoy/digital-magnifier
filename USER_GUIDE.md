# Magnifier — User Guide

A portable digital magnifier to help children with low vision read
books, worksheets, and other printed material.

## Getting started

1. **Plug in or charge** the device using a USB-C cable connected to
   the UPS battery pack. The battery bar in the top-right corner of the
   screen shows the current charge level. "CHG" appears when charging.

2. **Turn on** by pressing the power button (on the side of the case).
   You will see a "MAGNIFIER — Loading..." splash screen for a few
   seconds, then hear two short beeps and the camera view appears.

3. **Place reading material** under the camera. The device focuses
   automatically — no manual adjustment needed.

## Buttons

| Button     | Tap (quick press)     | Hold (3+ seconds)       |
|------------|-----------------------|-------------------------|
| Snapshot   | **Freeze** the view   | **Take a picture**      |
| Album      | Open the gallery      | —                       |
| CLR        | Next filter           | —                       |
| ADD1       | Zoom in               | —                       |
| ADD2       | Zoom out              | —                       |
| ADD3       | — (nothing)           | Show device status (5s) |
| Nav stick  | Pan up/down/left/right| Continuous panning      |
| Nav centre | Reset zoom + pan      | —                       |
| Power      | Shut down safely      | —                       |

### Freezing and capturing

**Tap** the snapshot button to freeze the view. The screen shows
"FROZEN VIEW" in yellow. The text stays still so you can study it
without the page moving. Tap again to return to the live camera.

**Hold** the snapshot button for 3 seconds to save a picture. You will
see a brief white flash on screen. The picture is saved to the SD card.

### Viewing saved pictures

Press the **Album** button to open the gallery. Use the nav stick
left/right to flip through pictures. The overlay shows which picture
you are on ("1 of 5") and the filename.

To **delete** a picture: tap the snapshot button once (the bar turns
red, "DELETE? tap again"), then tap again within 2 seconds to confirm.
If you wait longer than 2 seconds, the delete is cancelled.

Press the **back** button or Album again to return to the live view.

## Filters

Press **CLR** to cycle through the available filters. Each filter
changes how the image looks to make text easier to read:

- **normal** — no change (full colour)
- **high_contrast** — boosts contrast so faint text pops
- **inverted** — white text on a dark background (easier on the eyes)
- **binary** — pure black and white (maximum contrast)
- **yellow_on_black** — yellow text on a pure black background

Your teacher or parent can enable additional filters (grayscale,
white_on_black) or adjust filter settings by editing the configuration
file.

## The status screen

Hold **ADD3** for 5 seconds to see a fullscreen status display showing
battery percentage, current mode, zoom level, active filter, and how
many pictures are saved. Hold ADD3 for 5 seconds again to dismiss it.

## What the sounds mean

| Sound                        | Meaning                          |
|------------------------------|----------------------------------|
| Two quick beeps              | Device is ready (startup)        |
| Three descending tones       | Device is shutting down           |
| One short beep               | Battery is at 50% — plug in soon |
| Three urgent beeps + repeat  | Battery is at 15% — charge now   |
| Continuous beeping (1/sec)   | Battery is critically low         |
| (automatic shutdown)         | Battery reached 10% — safe stop  |

## Battery and charging

The battery bar in the top-right corner shows the charge level:

- **Green bar** — above 50%, plenty of charge
- **Yellow bar** — between 20% and 50%
- **Red bar** — below 20%, charge soon
- **Cyan bar + "CHG"** — currently charging

When the battery drops to 10%, the device plays the shutdown tone and
turns itself off automatically to protect the SD card from corruption.

## Troubleshooting

**Screen is black on startup** — wait 5 seconds for the camera to
initialise. If it stays black, check the camera cable connection.

**Image is blurry** — the camera focuses automatically but needs a
moment. Hold the material still for 1–2 seconds. Make sure the text
is within 5–30 cm of the camera lens.

**No sound from the buzzer** — check that the buzzer is enabled in the
config file (`buzzer: enabled: true`) and that it's connected to
GPIO 24.

**Buttons don't respond** — the I/O expander communicates over I²C.
Check the cable between the button board and the Pi. Run
`i2cdetect -y 1` and confirm you see address `0x20`.

**Battery bar not showing** — the UPS HAT may not be detected. Check
the I²C connection and run `python3 tests/hardware/probe_ups_hat.py`.
