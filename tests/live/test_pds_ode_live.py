"""Live endpoint tests for the PDS-ODE-backed and fixed-mosaic datasets —
hit real government/archive servers, manual trigger only.

These never run in CI or in the default ``pytest`` invocation; they are
deselected by the ``-m 'not live'`` default in ``pyproject.toml``. Run them
explicitly, one at a time, when verifying a real endpoint::

    uv run pytest tests/live -m live

Each test also doubles as the live verification step for that dataset's
filename-pattern regexes (AGENTS testing rules): if an archive changes its
naming convention, one of these fails loudly instead of a user silently
getting an all-invalid mask.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import astrofetch as af
from astrofetch.data.cache import WindowCache

# A small box around the Apollo 15 NAC stereo DTM site (known ODE SDNDTM
# coverage), safely within SLDEM2015's 60S-60N extent too.
_APOLLO15_AREA = (3.0, 25.0, 4.5, 26.5)


@pytest.mark.live
def test_nac_dtm_fetches_a_real_patch(tmp_path: Path) -> None:
    """End to end: PDS ODE search, footprint-constrained sampling, PDS4
    label-over-HTTPS read, and the elevation-product nodata override."""
    moondata = af.LROCNACDTM(
        products=["dtm", "ortho"],
        bbox=_APOLLO15_AREA,
        resolution=5.0,
        patch_size=64,
        length=1,
        seed=1,
        cache=WindowCache(tmp_path),
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (2, 64, 64)
    assert sample["layers"] == ["lroc_nac_dtm", "lroc_nac_ortho"]
    assert bool(sample["mask"].any())


@pytest.mark.live
def test_wac_mosaic_fetches_a_real_patch(tmp_path: Path) -> None:
    """The fixed-URL WAC global mosaic opens and reads at its native 100 m."""
    moondata = af.LROCWACMosaic(
        bbox=_APOLLO15_AREA,
        resolution=100.0,
        patch_size=64,
        length=1,
        seed=1,
        cache=WindowCache(tmp_path),
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (1, 64, 64)
    assert bool(sample["mask"].any())


@pytest.mark.live
def test_lola_dem_fetches_a_real_patch(tmp_path: Path) -> None:
    """The fixed-URL global LOLA DEM opens (detached PDS3 label) and reads."""
    moondata = af.LOLA(
        bbox=_APOLLO15_AREA,
        resolution=200.0,
        patch_size=32,
        length=1,
        seed=1,
        cache=WindowCache(tmp_path),
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (1, 32, 32)
    assert bool(sample["mask"].any())


@pytest.mark.live
def test_sldem2015_fetches_a_real_patch(tmp_path: Path) -> None:
    """SLDEM2015 within its 60S-60N coverage band reads valid elevation."""
    moondata = af.SLDEM2015(
        bbox=_APOLLO15_AREA,
        resolution=200.0,
        patch_size=32,
        length=1,
        seed=1,
        cache=WindowCache(tmp_path),
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (1, 32, 32)
    assert bool(sample["mask"].any())


@pytest.mark.live
def test_minirf_fetches_a_real_patch(tmp_path: Path) -> None:
    """Mini-RF global CPR mosaic: detached PDS3 label read via PDS ODE search."""
    moondata = af.MiniRF(
        products=["cpr"],
        bbox=(23.0, 18.0, 25.0, 20.0),
        resolution=500.0,
        patch_size=32,
        length=1,
        seed=1,
        cache=WindowCache(tmp_path),
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (1, 32, 32)
    assert bool(sample["mask"].any())
    valid = sample["image"][sample["mask"]]
    assert bool((valid >= 0.0).all()) and bool((valid <= 3.0).all())


@pytest.mark.live
def test_diviner_fetches_a_real_patch(tmp_path: Path) -> None:
    """Diviner rock abundance: the latest cumulative global mosaic, detached
    PDS3 label with correctly declared nodata. Rock abundance coverage is
    real but incomplete even in the latest cumulative mosaic (insufficient
    nighttime passes in some regions -- verified live 2026-07-21 that
    (23, 18, 25, 20) is a genuine gap), so this uses a bbox confirmed to
    have data rather than asserting coverage anywhere."""
    moondata = af.DivinerGDR(
        products=["rock_abundance"],
        bbox=(60.0, 55.0, 62.0, 57.0),
        resolution=1000.0,
        patch_size=16,
        length=1,
        seed=1,
        cache=WindowCache(tmp_path),
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (1, 16, 16)
    assert bool(sample["mask"].any())
    valid = sample["image"][sample["mask"]]
    assert bool((valid >= 0.0).all()) and bool((valid <= 1.0).all())


@pytest.mark.live
def test_wac_gld100_fetches_a_real_patch(tmp_path: Path) -> None:
    """WAC GLD100: 100 m tiled DTM, searched and opened via its .IMG data
    file (never the PDS4 .xml label -- rule 1)."""
    moondata = af.WACGLD100(
        bbox=(23.0, 18.0, 25.0, 20.0),
        resolution=100.0,
        patch_size=32,
        length=1,
        seed=1,
        cache=WindowCache(tmp_path),
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (1, 32, 32)
    assert bool(sample["mask"].any())
    valid = sample["image"][sample["mask"]]
    assert bool((valid >= -9500.0).all()) and bool((valid <= 10800.0).all())


@pytest.mark.live
def test_wac_tio2_fetches_a_real_patch(tmp_path: Path) -> None:
    """WAC TiO2 abundance map over a mare region (TiO2-rich basalt)."""
    moondata = af.WACTiO2(
        bbox=(23.0, 18.0, 25.0, 20.0),
        resolution=500.0,
        patch_size=16,
        length=1,
        seed=1,
        cache=WindowCache(tmp_path),
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (1, 16, 16)
    assert bool(sample["mask"].any())


@pytest.mark.live
def test_wac_global_tiled_fetches_a_real_patch(tmp_path: Path) -> None:
    """LROC WAC global mosaic, tiled/searched sibling of LROCWACMosaic: one
    E-family quadrant tile, opened via its .IMG data file."""
    moondata = af.LROCWACGlobal(
        bbox=(23.0, 18.0, 25.0, 20.0),
        resolution=100.0,
        patch_size=32,
        length=1,
        seed=1,
        cache=WindowCache(tmp_path),
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (1, 32, 32)
    assert bool(sample["mask"].any())


@pytest.mark.live
def test_wac_color_fetches_a_real_patch(tmp_path: Path) -> None:
    """LROC WAC 7-color reflectance, 643 nm band."""
    moondata = af.LROCWACColor(
        products=["refl_643nm"],
        bbox=(23.0, 18.0, 25.0, 20.0),
        resolution=500.0,
        patch_size=16,
        length=1,
        seed=1,
        cache=WindowCache(tmp_path),
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (1, 16, 16)
    assert bool(sample["mask"].any())
    valid = sample["image"][sample["mask"]]
    assert bool((valid >= 0.0).all()) and bool((valid <= 1.0).all())


@pytest.mark.live
def test_nac_roi_fetches_a_real_patch(tmp_path: Path) -> None:
    """NAC ROI: footprint-constrained sampling over named sites, 5 m mosaic."""
    moondata = af.LROCNACROI(
        products=["mosaic_5m"],
        resolution=5.0,
        patch_size=32,
        length=1,
        seed=1,
        cache=WindowCache(tmp_path),
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (1, 32, 32)
    assert bool(sample["mask"].any())


@pytest.mark.live
def test_shadowcam_fetches_a_real_patch(tmp_path: Path) -> None:
    """ShadowCam: footprint-constrained sampling over a PSR site mosaic."""
    moondata = af.ShadowCam(
        products=["mosaic"],
        resolution=5.0,
        patch_size=32,
        length=1,
        seed=1,
        cache=WindowCache(tmp_path),
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (1, 32, 32)
    assert bool(sample["mask"].any())


@pytest.mark.live
def test_nac_raw_granule_reads_a_row_slice() -> None:
    """EXPERIMENTAL granule path: PDS4 raw NAC strip, partial row read."""
    dataset = af.LROCNACRaw(bbox=_APOLLO15_AREA, max_products=1, rows=slice(0, 64))
    assert len(dataset) >= 1
    sample = dataset[0]
    assert sample["image"].shape[-2] == 64
    assert sample["pdsid"]


@pytest.mark.live
def test_m3_granule_reads_radiance_and_geolocation() -> None:
    """EXPERIMENTAL granule path: ENVI-format M3 radiance cube (opened via
    its .IMG data file directly -- unlike NAC/WAC, M3's driver rejects the
    .HDR header file) plus its lon/lat/elevation backplane."""
    dataset = af.M3(bbox=(3.0, 20.0, 8.0, 30.0), max_products=1, rows=slice(0, 32))
    assert len(dataset) >= 1
    sample = dataset[0]
    assert sample["image"].shape[0] == 85  # M3 L1B band count
    assert sample["image"].shape[-2] == 32
    assert sample["loc"].shape[0] == 3  # longitude, latitude, elevation
    assert sample["loc"].shape[-2] == 32
