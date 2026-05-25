"""Unit tests for ImageSaver, focusing on the MVP 0.4 gallery support.

The save() path is exercised in test_app_controller; here we cover
list_images() and delete_image() in isolation.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from digital_magnifier.storage.image_saver import ImageSaver, ImageSaverError


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _touch(path: Path, mtime: float) -> Path:
    """Create an empty file at ``path`` and set its modification time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    import os
    os.utime(path, (mtime, mtime))
    return path


def _make_image(path: Path, width: int = 4, height: int = 3) -> Path:
    """Write a real PNG to disk so the file is recognisably an image.

    Used by tests that don't care about content, only existence.
    """
    import cv2
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), frame)
    assert ok, f"setup failure: could not write {path}"
    return path


# --------------------------------------------------------------------------- #
# list_images
# --------------------------------------------------------------------------- #
class TestListImages:
    def test_empty_directory_returns_empty(self, tmp_path: Path):
        saver = ImageSaver(tmp_path)
        assert saver.list_images() == []

    def test_missing_directory_returns_empty(self, tmp_path: Path):
        target = tmp_path / "does_not_exist"
        # ImageSaver creates the dir on init; delete it after to
        # simulate a runtime disappearance.
        saver = ImageSaver(target)
        target.rmdir()
        assert saver.list_images() == []

    def test_finds_png_jpg_jpeg_only(self, tmp_path: Path):
        saver = ImageSaver(tmp_path)
        _touch(tmp_path / "a.png", 1000.0)
        _touch(tmp_path / "b.jpg", 1001.0)
        _touch(tmp_path / "c.jpeg", 1002.0)
        _touch(tmp_path / "d.txt", 1003.0)       # not an image
        _touch(tmp_path / "e.gif", 1004.0)       # not in allow-list
        _touch(tmp_path / ".hidden.png", 1005.0)  # still a .png, should be included

        results = saver.list_images()
        names = {p.name for p in results}
        assert names == {"a.png", "b.jpg", "c.jpeg", ".hidden.png"}

    def test_case_insensitive_extension(self, tmp_path: Path):
        saver = ImageSaver(tmp_path)
        _touch(tmp_path / "upper.PNG", 1000.0)
        _touch(tmp_path / "mixed.JpG", 1001.0)
        results = saver.list_images()
        assert len(results) == 2

    def test_newest_first_by_mtime(self, tmp_path: Path):
        saver = ImageSaver(tmp_path)
        _touch(tmp_path / "old.png", 1000.0)
        _touch(tmp_path / "mid.png", 2000.0)
        _touch(tmp_path / "new.png", 3000.0)
        names = [p.name for p in saver.list_images(newest_first=True)]
        assert names == ["new.png", "mid.png", "old.png"]

    def test_oldest_first_optional(self, tmp_path: Path):
        saver = ImageSaver(tmp_path)
        _touch(tmp_path / "old.png", 1000.0)
        _touch(tmp_path / "new.png", 2000.0)
        names = [p.name for p in saver.list_images(newest_first=False)]
        assert names == ["old.png", "new.png"]

    def test_ignores_subdirectories(self, tmp_path: Path):
        saver = ImageSaver(tmp_path)
        (tmp_path / "nested").mkdir()
        _touch(tmp_path / "nested" / "deep.png", 1000.0)
        _touch(tmp_path / "shallow.png", 1001.0)
        results = saver.list_images()
        assert [p.name for p in results] == ["shallow.png"]


# --------------------------------------------------------------------------- #
# delete_image
# --------------------------------------------------------------------------- #
class TestDeleteImage:
    def test_deletes_file(self, tmp_path: Path):
        saver = ImageSaver(tmp_path)
        target = _make_image(tmp_path / "victim.png")
        assert target.exists()
        saver.delete_image(target)
        assert not target.exists()

    def test_refuses_path_outside_output_dir(self, tmp_path: Path):
        saver = ImageSaver(tmp_path / "captures")
        outside = tmp_path / "elsewhere.png"
        _make_image(outside)
        with pytest.raises(ImageSaverError, match="not inside"):
            saver.delete_image(outside)
        assert outside.exists(), "file outside dir must not be deleted"

    def test_refuses_relative_traversal(self, tmp_path: Path):
        captures = tmp_path / "captures"
        outside = tmp_path / "secret.png"
        saver = ImageSaver(captures)
        _make_image(outside)
        traversal = captures / ".." / "secret.png"
        with pytest.raises(ImageSaverError):
            saver.delete_image(traversal)
        assert outside.exists()

    def test_missing_file_does_not_raise(self, tmp_path: Path):
        saver = ImageSaver(tmp_path)
        ghost = tmp_path / "already_gone.png"
        # Not created; delete should be a no-op (warning logged).
        saver.delete_image(ghost)   # must not raise

    def test_delete_then_list_excludes_file(self, tmp_path: Path):
        saver = ImageSaver(tmp_path)
        a = _make_image(tmp_path / "a.png")
        b = _make_image(tmp_path / "b.png")
        saver.delete_image(a)
        names = {p.name for p in saver.list_images()}
        assert names == {"b.png"}
