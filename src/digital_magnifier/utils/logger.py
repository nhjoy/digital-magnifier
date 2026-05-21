"""Logging setup helpers.

The application uses the stdlib ``logging`` module, configured from
the ``logging:`` section of ``app_config.yaml``. One root
configuration is enough for MVP 0.1; per-module log levels and file
handlers can be added later alongside the systemd integration in
MVP 0.5.

Usage:

    from digital_magnifier.utils.logger import setup_logging

    setup_logging({"level": "DEBUG"})

After this call, every module's ``logging.getLogger(__name__)`` will
use the configured formatter and level.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


DEFAULT_FORMAT: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DEFAULT_DATE_FORMAT: str = "%H:%M:%S"
DEFAULT_LEVEL: str = "INFO"


def setup_logging(config: dict[str, Any] | None = None) -> None:
    """Configure the root logger.

    Parameters
    ----------
    config : dict, optional
        The ``logging`` section of ``app_config.yaml``. Recognised
        keys:

        - ``level`` (str): ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
          ``"ERROR"``. Default ``"INFO"``.
        - ``format`` (str): log line format string. Default is a
          human-readable timestamp/level/name/message layout.
        - ``date_format`` (str): strftime format for ``%(asctime)s``.
        - ``use_stderr`` (bool): log to stderr instead of stdout.
          Default false (stdout) so logs interleave naturally with
          the console output of ``main.py``. Set true if you want
          to pipe stdout cleanly somewhere.

    Calling this replaces any existing root-logger handlers, so
    calling more than once is safe — useful when a CLI flag
    overrides the configured level.
    """
    cfg = config or {}

    level_name = str(cfg.get("level", DEFAULT_LEVEL)).upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        # Defensive: a typo in YAML shouldn't crash startup.
        level = logging.INFO

    fmt = str(cfg.get("format", DEFAULT_FORMAT))
    date_fmt = str(cfg.get("date_format", DEFAULT_DATE_FORMAT))
    stream = sys.stderr if cfg.get("use_stderr", False) else sys.stdout

    root = logging.getLogger()
    # Replace existing handlers (e.g. from bootstrap basicConfig).
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt, datefmt=date_fmt))
    root.addHandler(handler)
    root.setLevel(level)

    # Silence noisy third-party loggers (cv2 doesn't use Python
    # logging, but PIL/matplotlib sometimes leak through if added).
    for noisy in ("PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger.

    Thin convenience wrapper so individual modules can avoid
    importing ``logging`` if they only need ``get_logger``.
    """
    return logging.getLogger(name)