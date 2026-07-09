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

from astrofetch.data.grid import TargetGrid
from astrofetch.errors import EndpointError

_FILL = np.float32(0.0)
"""Value written into invalid pixels; always paired with a False mask entry."""


def read_window(
    href: str,
    grid: TargetGrid,
    band: int = 1,
    resampling: Resampling = Resampling.bilinear,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one COG band, reprojected onto ``grid``, in physical units.

    Args:
        href: COG URL or local path.
        grid: output grid; defines CRS, size, and extent.
        band: 1-based band index to read.
        resampling: resampling used when reprojecting to the grid.

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
            nodata = src.nodata
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
