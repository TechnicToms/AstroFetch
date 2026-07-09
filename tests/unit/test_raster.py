"""Unit tests for windowed COG reads — local GeoTIFFs only, never the network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from astrofetch.data.grid import GEOGRAPHIC_CRS, TargetGrid
from astrofetch.data.raster import read_window
from astrofetch.errors import EndpointError


def _write_geotiff(
    path: Path,
    data: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    crs: str = GEOGRAPHIC_CRS,
    nodata: float | None = None,
    scale: float = 1.0,
    offset: float = 0.0,
) -> str:
    height, width = data.shape
    west, south, east, north = bbox
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=from_bounds(west, south, east, north, width, height),
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
        dst.scales = (scale,)
        dst.offsets = (offset,)
    return str(path)


def test_reads_grid_shape_and_dtype(tmp_path: Path) -> None:
    src = np.arange(64, dtype=np.float32).reshape(8, 8)
    href = _write_geotiff(tmp_path / "src.tif", src, (0.0, 0.0, 8.0, 8.0))
    grid = TargetGrid(bbox=(0.0, 0.0, 8.0, 8.0), width=8, height=8)

    image, mask = read_window(href, grid)

    assert image.shape == (8, 8)
    assert image.dtype == np.float32
    assert mask.dtype == np.bool_
    assert mask.all()


def test_applies_scale_and_offset(tmp_path: Path) -> None:
    src = np.full((8, 8), 100, dtype=np.int16)
    href = _write_geotiff(
        tmp_path / "radiance.tif", src, (0.0, 0.0, 8.0, 8.0), scale=0.013, offset=2.0
    )
    grid = TargetGrid(bbox=(0.0, 0.0, 8.0, 8.0), width=8, height=8)

    image, _ = read_window(href, grid, resampling=rasterio.enums.Resampling.nearest)

    assert image == pytest.approx(100 * 0.013 + 2.0)


def test_nodata_becomes_invalid_mask(tmp_path: Path) -> None:
    src = np.ones((8, 8), dtype=np.float32)
    src[:, :4] = -9999.0
    href = _write_geotiff(tmp_path / "gappy.tif", src, (0.0, 0.0, 8.0, 8.0), nodata=-9999.0)
    grid = TargetGrid(bbox=(0.0, 0.0, 8.0, 8.0), width=8, height=8)

    image, mask = read_window(href, grid, resampling=rasterio.enums.Resampling.nearest)

    assert not mask[:, :4].any()
    assert mask[:, 4:].all()
    # Invalid pixels are zero-filled, never left as the nodata sentinel.
    assert (image[~mask] == 0.0).all()


def test_area_outside_source_is_invalid(tmp_path: Path) -> None:
    src = np.ones((8, 8), dtype=np.float32)
    href = _write_geotiff(tmp_path / "small.tif", src, (0.0, 0.0, 4.0, 4.0), nodata=0.0)
    # Grid extends east of the source coverage.
    grid = TargetGrid(bbox=(0.0, 0.0, 8.0, 4.0), width=16, height=8)

    _, mask = read_window(href, grid, resampling=rasterio.enums.Resampling.nearest)

    assert mask[:, :8].all()
    assert not mask[:, 8:].any()


def test_missing_file_raises_endpoint_error(tmp_path: Path) -> None:
    grid = TargetGrid(bbox=(0.0, 0.0, 1.0, 1.0), width=8, height=8)
    with pytest.raises(EndpointError):
        read_window(str(tmp_path / "does_not_exist.tif"), grid)
