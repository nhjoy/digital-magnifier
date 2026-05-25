"""Stdlib verification for the MVP 0.1 integration delivery.

Covers: config_loader, MockCameraSensor, magnifier, vision_filters,
and an end-to-end smoke test that loads real configs and runs
the full MagnifierApp with the headless display stubbed.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str((__import__("pathlib").Path(__file__).resolve().parents[2] / "src")))

import cv2  # for image I/O in the fallback test

from digital_magnifier.core.app_controller import MagnifierApp
from digital_magnifier.core.events import AppEvent
from digital_magnifier.core.state_machine import AppState
from digital_magnifier.hal.camera_base import CameraError
from digital_magnifier.hal.camera_sensor import (
    MockCameraSensor,
    _MODE_FALLBACK_IMAGE,
    _MODE_STOPPED,
    _MODE_SYNTHETIC,
)
from digital_magnifier.hal.mock_controls import MockControls
from digital_magnifier.processing.magnifier import apply_zoom
from digital_magnifier.processing.vision_filters import (
    AVAILABLE_FILTERS,
    FILTER_BINARY,
    FILTER_GRAYSCALE,
    FILTER_HIGH_CONTRAST,
    FILTER_INVERTED,
    FILTER_NORMAL,
    apply_filter,
)
from digital_magnifier.storage.image_saver import ImageSaver
from digital_magnifier.utils.config_loader import (
    ConfigError,
    DEFAULT_CONFIG_FILES,
    find_project_root,
    load_all_configs,
    load_config,
)
from digital_magnifier.utils.logger import setup_logging


# ============================================================
# config_loader
# ============================================================
class ConfigLoaderTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_yaml(self):
        f = self.tmp_path / "good.yaml"
        f.write_text("a: 1\nb:\n  c: 2\n")
        result = load_config("good.yaml", config_dir=self.tmp_path)
        self.assertEqual(result, {"a": 1, "b": {"c": 2}})

    def test_missing_file_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            load_config("missing.yaml", config_dir=self.tmp_path)
        self.assertIn("not found", str(ctx.exception))

    def test_malformed_yaml_raises(self):
        f = self.tmp_path / "bad.yaml"
        f.write_text("a: [unclosed\n")
        with self.assertRaises(ConfigError):
            load_config("bad.yaml", config_dir=self.tmp_path)

    def test_empty_file_returns_empty_dict(self):
        f = self.tmp_path / "empty.yaml"
        f.write_text("")
        self.assertEqual(load_config("empty.yaml", config_dir=self.tmp_path), {})

    def test_root_must_be_mapping(self):
        f = self.tmp_path / "list.yaml"
        f.write_text("- 1\n- 2\n")
        with self.assertRaises(ConfigError):
            load_config("list.yaml", config_dir=self.tmp_path)

    def test_load_all_configs(self):
        (self.tmp_path / "app_config.yaml").write_text("app: {target_fps: 24}\n")
        (self.tmp_path / "camera_config.yaml").write_text(
            "resolution:\n  width: 800\n  height: 600\n"
        )
        (self.tmp_path / "hardware_pins.yaml").write_text(
            "mock_keyboard_map:\n  q: QUIT\n"
        )
        configs = load_all_configs(config_dir=self.tmp_path)
        self.assertEqual(set(configs.keys()), {"app", "camera", "hardware_pins"})
        self.assertEqual(configs["app"]["app"]["target_fps"], 24)
        self.assertEqual(configs["camera"]["resolution"]["width"], 800)

    def test_find_project_root_finds_pyproject(self):
        (self.tmp_path / "pyproject.toml").write_text("")
        sub = self.tmp_path / "sub" / "deeper"
        sub.mkdir(parents=True)
        self.assertEqual(find_project_root(start=sub), self.tmp_path)

    def test_find_project_root_finds_config_dir(self):
        (self.tmp_path / "config").mkdir()
        sub = self.tmp_path / "sub"
        sub.mkdir()
        self.assertEqual(find_project_root(start=sub), self.tmp_path)


# ============================================================
# camera_sensor
# ============================================================
class CameraSensorTests(unittest.TestCase):

    def test_synthetic_frame_path(self):
        cfg = {
            "device": {"source": "test_image"},
            "resolution": {"width": 320, "height": 240},
        }
        cam = MockCameraSensor(cfg)
        cam.start()
        self.assertEqual(cam._mode, _MODE_SYNTHETIC)
        frame = cam.get_frame()
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)
        cam.stop()
        self.assertEqual(cam._mode, _MODE_STOPPED)

    def test_synthetic_frames_are_deterministic(self):
        cfg = {
            "device": {"source": "test_image"},
            "resolution": {"width": 320, "height": 240},
        }
        c1 = MockCameraSensor(cfg)
        c2 = MockCameraSensor(cfg)
        c1.start()
        c2.start()
        self.assertTrue(np.array_equal(c1.get_frame(), c2.get_frame()))

    def test_fallback_image_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "fallback.png"
            original = np.full((100, 200, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(img_path), original)

            cfg = {
                "device": {
                    "source": "test_image",
                    "fallback_image": str(img_path),
                },
                "resolution": {"width": 320, "height": 240},
            }
            cam = MockCameraSensor(cfg)
            cam.start()
            self.assertEqual(cam._mode, _MODE_FALLBACK_IMAGE)
            self.assertEqual(cam.get_frame().shape, (240, 320, 3))

    def test_missing_fallback_falls_through_to_synthetic(self):
        cfg = {
            "device": {
                "source": "test_image",
                "fallback_image": "/nonexistent/path.png",
            },
            "resolution": {"width": 320, "height": 240},
        }
        cam = MockCameraSensor(cfg)
        cam.start()
        self.assertEqual(cam._mode, _MODE_SYNTHETIC)

    def test_context_manager(self):
        cfg = {"device": {"source": "test_image"}, "resolution": {"width": 320, "height": 240}}
        with MockCameraSensor(cfg) as cam:
            frame = cam.get_frame()
            self.assertEqual(frame.shape, (240, 320, 3))
        self.assertEqual(cam._mode, _MODE_STOPPED)


# ============================================================
# magnifier
# ============================================================
class MagnifierTests(unittest.TestCase):

    def _make_frame(self):
        f = np.zeros((100, 200, 3), dtype=np.uint8)
        f[40:60, 90:110] = [0, 0, 255]
        f[0:5, :] = [0, 128, 0]
        f[-5:, :] = [0, 128, 0]
        f[:, 0:5] = [0, 128, 0]
        f[:, -5:] = [0, 128, 0]
        return f

    def test_zoom_one_returns_input(self):
        f = self._make_frame()
        self.assertTrue(np.array_equal(apply_zoom(f, 1.0), f))

    def test_zoom_two_centred_finds_red(self):
        f = self._make_frame()
        out = apply_zoom(f, 2.0)
        cy, cx = out.shape[0] // 2, out.shape[1] // 2
        self.assertGreater(out[cy, cx, 2], 200, "expected red centre")

    def test_zoom_preserves_shape(self):
        f = self._make_frame()
        for z in [1.5, 2.0, 3.0, 4.0]:
            out = apply_zoom(f, z)
            self.assertEqual(out.shape, f.shape, f"zoom={z}")

    def test_pan_clamped(self):
        f = self._make_frame()
        # Out-of-range pan must not raise
        out = apply_zoom(f, 2.0, pan_x=5.0, pan_y=-3.0)
        self.assertEqual(out.shape, f.shape)


# ============================================================
# vision_filters
# ============================================================
class VisionFiltersTests(unittest.TestCase):

    def _make_frame(self):
        h, w = 100, 200
        f = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            f[y, :, 2] = int(255 * y / h)
        for x in range(w):
            f[:, x, 0] = int(255 * x / w)
        f[:, :, 1] = 128
        return f

    def test_all_filters_preserve_shape(self):
        f = self._make_frame()
        for name in AVAILABLE_FILTERS:
            out = apply_filter(f, name)
            self.assertEqual(out.shape, f.shape, f"{name}")
            self.assertEqual(out.dtype, f.dtype, f"{name}")

    def test_normal_is_passthrough(self):
        f = self._make_frame()
        self.assertTrue(np.array_equal(apply_filter(f, FILTER_NORMAL), f))

    def test_grayscale_channels_equal(self):
        f = self._make_frame()
        out = apply_filter(f, FILTER_GRAYSCALE)
        self.assertTrue(np.array_equal(out[..., 0], out[..., 1]))
        self.assertTrue(np.array_equal(out[..., 1], out[..., 2]))

    def test_inverted(self):
        f = self._make_frame()
        out = apply_filter(f, FILTER_INVERTED)
        self.assertTrue(np.array_equal(out, 255 - f))

    def test_binary_only_extreme_values(self):
        f = self._make_frame()
        out = apply_filter(f, FILTER_BINARY)
        unique = np.unique(out)
        self.assertTrue(set(unique.tolist()).issubset({0, 255}))

    def test_unknown_filter_passes_through(self):
        f = self._make_frame()
        with self.assertLogs(
            "digital_magnifier.processing.vision_filters", level="WARNING"
        ) as cm:
            out = apply_filter(f, "nope")
        self.assertTrue(np.array_equal(out, f))
        self.assertTrue(any("unknown filter" in m for m in cm.output))


# ============================================================
# End-to-end smoke: load real configs, run a scripted session
# ============================================================
class EndToEndTests(unittest.TestCase):
    """Wire up the entire app from real config files and run a session."""

    def setUp(self):
        # Stub cv2 display so the test is headless.
        self._orig_imshow = cv2.imshow
        self._orig_named = cv2.namedWindow
        self._orig_destroy = cv2.destroyAllWindows
        self._orig_set_prop = cv2.setWindowProperty
        cv2.imshow = lambda *a, **kw: None
        cv2.namedWindow = lambda *a, **kw: None
        cv2.destroyAllWindows = lambda *a, **kw: None
        cv2.setWindowProperty = lambda *a, **kw: None

    def tearDown(self):
        cv2.imshow = self._orig_imshow
        cv2.namedWindow = self._orig_named
        cv2.destroyAllWindows = self._orig_destroy
        cv2.setWindowProperty = self._orig_set_prop

    def test_full_session_with_real_configs(self):
        # Load the real config files from /home/claude/work/config/
        configs = load_all_configs(config_dir=Path("config"))

        # Build app using a scripted controls layer
        scripted = [
            AppEvent.ZOOM_IN,
            AppEvent.ZOOM_IN,
            AppEvent.PAN_RIGHT,
            AppEvent.FREEZE_TOGGLE,
            AppEvent.FILTER_NEXT,    # cache must invalidate
            AppEvent.CAPTURE_IMAGE,
            AppEvent.QUIT,
        ]

        from digital_magnifier.hal.controls_base import ControlsHAL

        class Script(ControlsHAL):
            def __init__(self, evs):
                self.evs = list(evs)
                self.start_calls = 0
                self.stop_calls = 0
            def start(self): self.start_calls += 1
            def stop(self): self.stop_calls += 1
            def poll(self):
                return self.evs.pop(0) if self.evs else AppEvent.NONE

        with tempfile.TemporaryDirectory() as out_dir:
            camera = MockCameraSensor(configs["camera"])
            controls = Script(scripted)
            saver = ImageSaver(output_directory=out_dir)

            # Merge camera res into app config (what main.py does)
            app_cfg = dict(configs["app"])
            res = configs["camera"].get("resolution", {})
            app_cfg["camera"] = {
                "width": int(res.get("width", 1280)),
                "height": int(res.get("height", 720)),
            }
            # Speed up the test
            app_cfg["app"] = {**app_cfg.get("app", {}), "target_fps": 1000}
            app_cfg["capture"] = {
                **app_cfg.get("capture", {}),
                "flash_duration_ms": 1,  # tiny so we exit quickly
            }

            app = MagnifierApp(camera, controls, saver, app_cfg)
            app.run()

            # End state
            self.assertTrue(app._state_machine.is_in(AppState.SHUTDOWN))
            self.assertEqual(controls.start_calls, 1)
            self.assertEqual(controls.stop_calls, 1)

            # Capture was saved
            saved = list(Path(out_dir).glob("*.png"))
            self.assertEqual(len(saved), 1, f"expected 1 capture, found {saved}")

            # Final magnifier state — zoom_step in real app_config.yaml is 0.5,
            # so two ZOOM_IN events = 1.0 + 0.5 + 0.5 = 2.0
            self.assertEqual(app._zoom, 2.0)
            # Pan was applied
            self.assertGreater(app._pan_x, 0)
            # Filter advanced
            self.assertEqual(app._current_filter(), "grayscale")


if __name__ == "__main__":
    # Quieten the loggers a bit so test output is readable
    logging.getLogger().setLevel(logging.ERROR)
    unittest.main(verbosity=2)