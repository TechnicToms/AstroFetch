"""Unit tests for the disposable window cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from astrofetch.data.cache import WindowCache, default_cache_dir
from astrofetch.data.grid import TargetGrid


def _grid(width: int = 8) -> TargetGrid:
    return TargetGrid(bbox=(0.0, 0.0, 1.0, 1.0), width=width, height=8)


def test_roundtrips_image_and_mask(tmp_path: Path) -> None:
    cache = WindowCache(tmp_path)
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    mask = image > 30

    assert cache.get("kaguya_tc_dtm", _grid()) is None
    cache.put("kaguya_tc_dtm", _grid(), image, mask)
    cached = cache.get("kaguya_tc_dtm", _grid())

    assert cached is not None
    got_image, got_mask = cached
    np.testing.assert_array_equal(got_image, image)
    np.testing.assert_array_equal(got_mask, mask)


def test_distinct_grids_do_not_collide(tmp_path: Path) -> None:
    cache = WindowCache(tmp_path)
    image = np.ones((8, 8), dtype=np.float32)
    cache.put("layer", _grid(width=8), image, image.astype(bool))

    # Same layer, different grid width -> different key -> miss.
    assert cache.get("layer", _grid(width=16)) is None


def test_clear_removes_entries(tmp_path: Path) -> None:
    cache = WindowCache(tmp_path)
    image = np.zeros((8, 8), dtype=np.float32)
    cache.put("layer", _grid(), image, image.astype(bool))

    cache.clear()

    assert cache.get("layer", _grid()) is None
    assert not list(tmp_path.glob("*.npz"))


def test_put_leaves_no_temp_files(tmp_path: Path) -> None:
    cache = WindowCache(tmp_path)
    image = np.zeros((8, 8), dtype=np.float32)
    cache.put("layer", _grid(), image, image.astype(bool))

    assert not list(tmp_path.glob("*.tmp.npz"))


def test_default_dir_honors_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ASTROFETCH_CACHE", str(tmp_path / "custom"))
    assert default_cache_dir() == tmp_path / "custom"
