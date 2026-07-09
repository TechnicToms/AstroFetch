"""Live endpoint tests — hit real government servers, manual trigger only.

These never run in CI or in the default ``pytest`` invocation; they are
deselected by the ``-m 'not live'`` default in ``pyproject.toml``. Run them
explicitly, one at a time, when verifying a real endpoint::

    uv run pytest tests/live -m live

They exercise the Phase 1 STAC/COG path against the USGS ARD catalog over a
small, known-covered lunar region, so they stay fast and single-shot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import astrofetch as af
from astrofetch.data import stac
from astrofetch.data.cache import WindowCache

# A small box with known Kaguya TC USGS DTM coverage (near -50 deg latitude).
_COVERED_BBOX = (-26.3, -50.6, -25.5, -49.7)


@pytest.mark.live
def test_stac_catalog_returns_covering_cogs() -> None:
    """The USGS ARD catalog is reachable and returns DTM COGs for a known box."""
    hrefs = stac.find_asset_hrefs(af.KaguyaTC.collection, "dtm", _COVERED_BBOX, max_items=3)
    assert hrefs
    assert all(href.endswith(".tif") for href in hrefs)


@pytest.mark.live
def test_fetches_a_real_two_layer_patch(tmp_path: Path) -> None:
    """End to end: a coregistered (2, H, W) DTM+ortho patch of real physical data."""
    moondata = af.KaguyaTC(
        products=["dtm", "ortho"],
        bbox=_COVERED_BBOX,
        resolution=50.0,
        patch_size=64,
        length=1,
        seed=3,
        cache=WindowCache(tmp_path),
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (2, 64, 64)
    assert sample["layers"] == ["kaguya_tc_dtm", "kaguya_tc_ortho"]
    # At least some pixels are valid over a covered region.
    assert bool(sample["mask"].any())
