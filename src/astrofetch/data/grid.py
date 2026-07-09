"""Target grid definition and unit helpers (body-agnostic).

A :class:`TargetGrid` is the common raster every layer is reprojected onto so
that channels coregister exactly. Phase 1 uses a plate-carree (equirectangular)
grid in the IAU 2015 Moon geographic CRS; ``data/raster.py`` reprojects each
source COG — whatever its native projection, equirectangular or polar
stereographic — onto it. Keeping the grid geographic means the ``bbox`` a caller
passes and the ``crs`` a sample advertises are the same coordinate system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from affine import Affine
from rasterio.transform import from_bounds

BBox = tuple[float, float, float, float]
"""(west, south, east, north) in degrees."""

GEOGRAPHIC_CRS = "IAU_2015:30100"
"""Ocentric IAU 2015 Moon geographic CRS (degrees); the target grid CRS."""

MOON_RADIUS_M = 1737400.0
"""IAU 2015 Moon sphere radius in metres (IAU code 30100; see any ARD COG WKT)."""


@dataclass(frozen=True)
class TargetGrid:
    """A fixed output raster: a bbox rasterized to ``width`` x ``height`` pixels.

    Args:
        bbox: (west, south, east, north) in degrees, west < east, south < north.
        width: output width in pixels.
        height: output height in pixels.
        crs: grid CRS; defaults to the IAU 2015 Moon geographic CRS.
    """

    bbox: BBox
    width: int
    height: int
    crs: str = GEOGRAPHIC_CRS

    def __post_init__(self) -> None:
        west, south, east, north = self.bbox
        if not (west < east and south < north):
            raise ValueError(f"invalid bbox (west, south, east, north): {self.bbox}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"grid size must be positive, got {self.width}x{self.height}")

    @property
    def transform(self) -> Affine:
        """Affine mapping pixel (col, row) to grid CRS coordinates."""
        west, south, east, north = self.bbox
        return from_bounds(west, south, east, north, self.width, self.height)


def meters_to_degrees(meters: float, latitude: float) -> tuple[float, float]:
    """Approximate the (dlon, dlat) degree span of a ground distance in metres.

    Spherical arc length on the IAU 2015 Moon sphere, used only to size a
    sampling window from ``patch_size * resolution``; the exact pixel geometry
    is fixed later by :class:`TargetGrid`, so this need only be close. Longitude
    degrees shrink with ``cos(latitude)``.

    Args:
        meters: ground distance in metres.
        latitude: latitude in degrees where the span is measured.

    Returns:
        (dlon, dlat) span in degrees.
    """
    dlat = math.degrees(meters / MOON_RADIUS_M)
    coslat = math.cos(math.radians(latitude))
    # Guard the poles, where a bounded ground distance spans unbounded longitude.
    dlon = dlat / coslat if coslat > 1e-6 else 360.0
    return dlon, dlat
