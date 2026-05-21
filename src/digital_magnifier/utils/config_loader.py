"""YAML configuration loader.

Resolves config files relative to the project root, so the same code
works whether ``main.py`` is invoked from the repo root, from ``src/``,
or via an installed entry point. ``load_all_configs()`` returns a
dict mapping config name to parsed contents, ready for ``main.py``
to wire into the constructed components.

Project root resolution looks for any of these markers, walking
upward from this module:

- ``pyproject.toml``
- ``setup.py``
- a ``config/`` directory

If none is found, falls back to the current working directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml


logger = logging.getLogger(__name__)


# The set of config files the application expects, mapped to a
# short name used as the dict key by ``load_all_configs``. Adding
# a new config means adding one entry here.
DEFAULT_CONFIG_FILES: dict[str, str] = {
    "app": "app_config.yaml",
    "camera": "camera_config.yaml",
    "hardware_pins": "hardware_pins.yaml",
}


class ConfigError(Exception):
    """Raised when a config file is missing, unreadable, or malformed."""


def find_project_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` looking for a project-root marker.

    Markers (any one is sufficient): ``pyproject.toml``,
    ``setup.py``, or a ``config/`` directory.

    Falls back to the current working directory if no marker is
    found anywhere on the path.
    """
    here = (start or Path(__file__)).resolve()
    candidates = [here] + list(here.parents)
    markers = ("pyproject.toml", "setup.py", "config")
    for parent in candidates:
        if any((parent / m).exists() for m in markers):
            return parent
    return Path.cwd()


def resolve_config_dir(config_dir: Path | None = None) -> Path:
    """Return the directory to load configs from."""
    if config_dir is not None:
        return Path(config_dir).expanduser().resolve()
    return find_project_root() / "config"


def load_config(
    filename: str,
    config_dir: Path | None = None,
) -> dict[str, Any]:
    """Load and parse a single YAML config file.

    Parameters
    ----------
    filename : str
        File name relative to ``config_dir``, e.g. ``"app_config.yaml"``.
    config_dir : Path, optional
        Override the config directory. Defaults to
        ``<project_root>/config``.

    Raises
    ------
    ConfigError
        If the file is missing, unreadable, malformed, or its
        top-level value is not a mapping.
    """
    cfg_dir = resolve_config_dir(config_dir)
    path = (cfg_dir / filename).resolve()

    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"failed to parse {path}: {e}") from e
    except OSError as e:
        raise ConfigError(f"failed to read {path}: {e}") from e

    if data is None:
        # Empty file is technically valid YAML; treat as empty dict.
        data = {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"config root must be a mapping in {path}; "
            f"got {type(data).__name__}"
        )

    logger.debug("loaded config %s", path)
    return data


def load_all_configs(
    config_dir: Path | None = None,
    files: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load every known config file.

    Returns a dict mapping each entry in ``files`` to the parsed
    contents of its YAML file. Defaults to :data:`DEFAULT_CONFIG_FILES`.

    Raises
    ------
    ConfigError
        Propagated from :func:`load_config` if any file is bad.
        Loading is sequential, so the first failure stops further
        loads — failing fast at startup is preferable to a partly-
        configured run.
    """
    files = files if files is not None else DEFAULT_CONFIG_FILES
    return {
        name: load_config(filename, config_dir=config_dir)
        for name, filename in files.items()
    }