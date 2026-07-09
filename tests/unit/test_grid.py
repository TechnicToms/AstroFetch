import math

import pytest

from astrofetch.data.grid import (
    GEOGRAPHIC_CRS,
    MOON_RADIUS_M,
    TargetGrid,
    meters_to_degrees,
)


def test_transform_spans_bbox() -> None:
    grid = TargetGrid(bbox=(-10.0, -5.0, 10.0, 5.0), width=200, height=100)
    transform = grid.transform
    # Top-left pixel corner maps to (west, north); bottom-right to (east, south).
    assert transform * (0, 0) == pytest.approx((-10.0, 5.0))
    assert transform * (200, 100) == pytest.approx((10.0, -5.0))


def test_default_crs_is_geographic() -> None:
    assert TargetGrid(bbox=(0.0, 0.0, 1.0, 1.0), width=8, height=8).crs == GEOGRAPHIC_CRS


def test_rejects_inverted_bbox() -> None:
    with pytest.raises(ValueError):
        TargetGrid(bbox=(10.0, 0.0, -10.0, 5.0), width=8, height=8)


def test_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError):
        TargetGrid(bbox=(0.0, 0.0, 1.0, 1.0), width=0, height=8)


def test_meters_to_degrees_at_equator_is_isotropic() -> None:
    dlon, dlat = meters_to_degrees(MOON_RADIUS_M * math.radians(1.0), latitude=0.0)
    assert dlon == pytest.approx(1.0)
    assert dlat == pytest.approx(1.0)


def test_longitude_span_grows_toward_the_pole() -> None:
    dlon, dlat = meters_to_degrees(1000.0, latitude=60.0)
    # cos(60) = 0.5, so a fixed ground distance spans twice the longitude.
    assert dlon == pytest.approx(2.0 * dlat)


def test_pole_longitude_span_is_clamped() -> None:
    dlon, _ = meters_to_degrees(1000.0, latitude=90.0)
    assert dlon == 360.0
