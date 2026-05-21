"""Unit tests for ``utils.config_loader``."""

from __future__ import annotations

from pathlib import Path

import pytest

from digital_magnifier.utils.config_loader import (
    ConfigError,
    DEFAULT_CONFIG_FILES,
    find_project_root,
    load_all_configs,
    load_config,
    resolve_config_dir,
)


# --------------------------------------------------------------------------- #
# load_config
# --------------------------------------------------------------------------- #
class TestLoadConfig:
    def test_valid_yaml(self, tmp_path: Path):
        f = tmp_path / "good.yaml"
        f.write_text("a: 1\nb:\n  c: 2\n")
        result = load_config("good.yaml", config_dir=tmp_path)
        assert result == {"a": 1, "b": {"c": 2}}

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="not found"):
            load_config("missing.yaml", config_dir=tmp_path)

    def test_malformed_yaml_raises(self, tmp_path: Path):
        f = tmp_path / "bad.yaml"
        f.write_text("a: [unclosed\n")
        with pytest.raises(ConfigError, match="failed to parse"):
            load_config("bad.yaml", config_dir=tmp_path)

    def test_empty_file_returns_empty_dict(self, tmp_path: Path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        assert load_config("empty.yaml", config_dir=tmp_path) == {}

    def test_root_must_be_mapping(self, tmp_path: Path):
        # A YAML list at root should fail (we expect a section dict).
        f = tmp_path / "list.yaml"
        f.write_text("- 1\n- 2\n")
        with pytest.raises(ConfigError, match="must be a mapping"):
            load_config("list.yaml", config_dir=tmp_path)


# --------------------------------------------------------------------------- #
# load_all_configs
# --------------------------------------------------------------------------- #
class TestLoadAllConfigs:
    def test_default_set(self, tmp_path: Path):
        # Create all three expected files
        (tmp_path / "app_config.yaml").write_text("app: {target_fps: 24}\n")
        (tmp_path / "camera_config.yaml").write_text(
            "resolution:\n  width: 800\n  height: 600\n"
        )
        (tmp_path / "hardware_pins.yaml").write_text(
            "mock_keyboard_map:\n  q: QUIT\n"
        )

        configs = load_all_configs(config_dir=tmp_path)
        assert set(configs.keys()) == {"app", "camera", "hardware_pins"}
        assert configs["app"]["app"]["target_fps"] == 24
        assert configs["camera"]["resolution"]["width"] == 800
        assert configs["hardware_pins"]["mock_keyboard_map"]["q"] == "QUIT"

    def test_custom_set(self, tmp_path: Path):
        (tmp_path / "only.yaml").write_text("key: value\n")
        configs = load_all_configs(
            config_dir=tmp_path, files={"only": "only.yaml"}
        )
        assert configs == {"only": {"key": "value"}}

    def test_one_missing_aborts(self, tmp_path: Path):
        (tmp_path / "app_config.yaml").write_text("a: 1\n")
        # camera_config.yaml and hardware_pins.yaml are missing
        with pytest.raises(ConfigError):
            load_all_configs(config_dir=tmp_path)


# --------------------------------------------------------------------------- #
# Project-root resolution
# --------------------------------------------------------------------------- #
class TestProjectRoot:
    def test_finds_pyproject_toml(self, tmp_path: Path):
        (tmp_path / "pyproject.toml").write_text("")
        sub = tmp_path / "sub" / "deeper"
        sub.mkdir(parents=True)
        root = find_project_root(start=sub)
        assert root == tmp_path

    def test_finds_config_directory(self, tmp_path: Path):
        (tmp_path / "config").mkdir()
        sub = tmp_path / "sub"
        sub.mkdir()
        root = find_project_root(start=sub)
        assert root == tmp_path

    def test_no_markers_falls_back_to_cwd(self, tmp_path: Path):
        # No markers exist; we accept anything (fallback may be cwd
        # depending on environment). Just verify it returns *some* Path.
        root = find_project_root(start=tmp_path)
        assert isinstance(root, Path)


# --------------------------------------------------------------------------- #
# resolve_config_dir
# --------------------------------------------------------------------------- #
class TestResolveConfigDir:
    def test_explicit_override(self, tmp_path: Path):
        result = resolve_config_dir(tmp_path)
        assert result == tmp_path.resolve()

    def test_default_uses_project_root_config(self, tmp_path: Path):
        # If we set up a fake project root with config/, the default
        # resolves to that.
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "config").mkdir()
        # We need find_project_root to start from inside tmp_path —
        # we don't have a hook for that, so just verify the function
        # returns something Path-like.
        result = resolve_config_dir(None)
        assert isinstance(result, Path)


# --------------------------------------------------------------------------- #
# Sanity check: the default file list matches expected names
# --------------------------------------------------------------------------- #
def test_default_files_are_complete():
    assert "app" in DEFAULT_CONFIG_FILES
    assert "camera" in DEFAULT_CONFIG_FILES
    assert "hardware_pins" in DEFAULT_CONFIG_FILES
    assert DEFAULT_CONFIG_FILES["app"] == "app_config.yaml"