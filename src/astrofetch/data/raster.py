"""Windowed COG reads reprojected onto a :class:`TargetGrid` (rasterio).

Reads only the region of a Cloud Optimized GeoTIFF that covers the grid — via
HTTP range requests against the right overview level — reprojects it onto the
grid's CRS and resolution, applies the band's scale/offset to recover physical
values, and returns the data with a boolean validity mask (``False`` where the
source had nodata or simply does not cover the grid).

All map-projection and resampling math is delegated to rasterio/GDAL (AGENTS
rule 1): this module wires it up, it does not reimplement it.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from astrofetch.data.grid import TargetGrid
from astrofetch.errors import EndpointError

_FILL = np.float32(0.0)
"""Value written into invalid pixels; always paired with a False mask entry."""


def read_window(
    href: str,
    grid: TargetGrid,
    band: int = 1,
    resampling: Resampling = Resampling.bilinear,
    nodata_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one COG band, reprojected onto ``grid``, in physical units.

    Args:
        href: COG URL or local path.
        grid: output grid; defines CRS, size, and extent.
        band: 1-based band index to read.
        resampling: resampling used when reprojecting to the grid.
        nodata_override: nodata value to use in place of the source's own
            (which may be unset). Needed for a handful of PDS products that
            omit nodata from their label even though the raster does not
            cover its full extent (unwarped pixels would otherwise silently
            read back as valid zeros); prefer the source's own declared
            nodata whenever a product provides one.

    Returns:
        ``(image, mask)`` where ``image`` is a ``(grid.height, grid.width)``
        float32 array of physical values (scale/offset applied) with invalid
        pixels set to 0, and ``mask`` is a same-shaped bool array, ``True``
        where the pixel is valid.

    Raises:
        EndpointError: the COG could not be opened or read.
    """
    try:
        with rasterio.open(href) as src:
            nodata = src.nodata if nodata_override is None else nodata_override
            scale = float(src.scales[band - 1])
            offset = float(src.offsets[band - 1])
            with WarpedVRT(
                src,
                crs=grid.crs,
                transform=grid.transform,
                width=grid.width,
                height=grid.height,
                resampling=resampling,
                src_nodata=nodata,
                nodata=nodata,
            ) as vrt:
                raw = vrt.read(band)
    except RasterioIOError as exc:
        raise EndpointError(href, f"could not read COG: {exc}") from exc

    if nodata is not None:
        mask = raw != nodata
    else:
        mask = np.ones(raw.shape, dtype=bool)

    image = raw.astype(np.float32) * np.float32(scale) + np.float32(offset)
    image[~mask] = _FILL
    return image, mask


def read_full(
    href: str,
    window: Window | None = None,
    bands: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a raster's own pixels with no reprojection, in physical units.

    Unlike :func:`read_window`, this does not warp onto a :class:`TargetGrid`
    -- it reads the source in its own native geometry, for rasters that have
    no map projection to warp to (e.g. raw camera-frame swaths). Used by
    :mod:`astrofetch.moon.granules`.

    Args:
        href: raster URL or local path.
        window: pixel window ``(col_off, row_off, width, height)`` to read;
            ``None`` reads the full raster.
        bands: 1-based band indices to read; ``None`` reads every band.

    Returns:
        ``(image, mask)`` where ``image`` is a ``(bands, height, width)``
        float32 array of physical values (per-band scale/offset applied)
        with invalid pixels set to 0, and ``mask`` is a same-shaped bool
        array, ``True`` where the pixel is valid.

    Raises:
        EndpointError: the raster could not be opened or read.
    """
    try:
        with rasterio.open(href) as src:
            band_list = bands if bands is not None else list(range(1, src.count + 1))
            raw = src.read(band_list, window=window)
            nodata = src.nodata
            scales = np.array([src.scales[b - 1] for b in band_list], dtype=np.float32)
            offsets = np.array([src.offsets[b - 1] for b in band_list], dtype=np.float32)
    except RasterioIOError as exc:
        raise EndpointError(href, f"could not read raster: {exc}") from exc

    if nodata is not None:
        mask = raw != nodata
    else:
        mask = np.ones(raw.shape, dtype=bool)

    image = raw.astype(np.float32) * scales.reshape(-1, 1, 1) + offsets.reshape(-1, 1, 1)
    image[~mask] = _FILL
    return image, mask
