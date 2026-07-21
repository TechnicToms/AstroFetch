"""Unit tests for the instrument datasets.

The read path is real (STAC/ODE search + raster reproject + cache), so these
tests stub :mod:`astrofetch.data.stac`, :mod:`astrofetch.data.ode`, and
:mod:`astrofetch.data.raster` — no network, no real rasters — and redirect
the cache to a temp dir. A deterministic fake read lets us assert shapes,
layer wiring, reproducibility, and composition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import astrofetch as af
from astrofetch.data import ode
from astrofetch.moon import LAYERS, MOON
from astrofetch.moon import datasets as ds

SAMPLE_KEYS = {"image", "mask", "layers", "bbox", "crs", "resolution"}

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "ode"


def _phase_c_files(pt_key: str) -> tuple[ode.ODEFile, ...]:
    raw = json.loads((_FIXTURES / "phase_c_listings.json").read_text())[pt_key]
    return tuple(ode.ODEFile(f["FileName"], f["Type"], f["URL"]) for f in raw)


def _assert_pattern_selects(spec: ds.ODEAsset, pt_key: str, expected_suffix: str) -> None:
    files = _phase_c_files(pt_key)
    urls = ode.match_files(files, spec.pattern, spec.file_type)
    assert len(urls) == 1, f"{spec.layer}: expected exactly one match, got {urls}"
    assert urls[0].endswith(expected_suffix)


def _fake_find_asset_hrefs(
    collection: str, asset: str, bbox: tuple, max_items: int = 20, root: str | None = None
) -> list[str]:
    return [f"{collection}|{asset}"]


def _fake_read_window(href, grid, band=1, resampling=None, nodata_override=None):
    # Deterministic in (href, window): identical requests read identical data,
    # which is what makes seeded sampling reproducible under mocking.
    value = float(hash((href, grid.bbox)) % 997)
    image = np.full((grid.height, grid.width), value, dtype=np.float32)
    return image, np.ones((grid.height, grid.width), dtype=bool)


def _fake_find_file_urls(
    ihid: str,
    iid: str,
    pt: str,
    pattern: str,
    bbox: tuple,
    max_products: int = 20,
    file_type: str | None = "Product",
    root: str | None = None,
) -> list[str]:
    return [f"{ihid}|{iid}|{pt}"]


def _fake_query_products(
    ihid: str, iid: str, pt: str, bbox: tuple, max_products: int = 20, root: str | None = None
) -> list:
    # No footprints by default: exercises the uniform-sampling fallback unless
    # a test overrides this to supply real footprints.
    return []


@pytest.fixture(autouse=True)
def _mock_reads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ASTROFETCH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(ds.stac, "find_asset_hrefs", _fake_find_asset_hrefs)
    monkeypatch.setattr(ds.raster, "read_window", _fake_read_window)
    monkeypatch.setattr(ds.ode, "find_file_urls", _fake_find_file_urls)
    monkeypatch.setattr(ds.ode, "query_products", _fake_query_products)


def test_instrument_yields_sample_dicts() -> None:
    moondata = af.KaguyaTC(
        products=["dtm", "ortho"],
        bbox=(-60.0, 5.0, -55.0, 10.0),
        patch_size=32,
        length=3,
        seed=0,
    )
    samples = list(moondata)
    assert len(samples) == 3
    for sample in samples:
        assert set(sample) == SAMPLE_KEYS
        assert sample["image"].shape == (2, 32, 32)
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].shape == (2, 32, 32)
        assert sample["mask"].dtype == torch.bool
        assert sample["layers"] == ["kaguya_tc_dtm", "kaguya_tc_ortho"]
        assert sample["crs"] == "IAU_2015:30100"


def test_products_default_to_all() -> None:
    moondata = af.KaguyaTC(patch_size=16, length=1)
    assert moondata.layers == ["kaguya_tc_dtm", "kaguya_tc_ortho"]


def test_read_queries_stac_by_collection_and_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def _spy(collection, asset, bbox, max_items=20, root=None):
        calls.append((collection, asset))
        return [f"{collection}|{asset}"]

    monkeypatch.setattr(ds.stac, "find_asset_hrefs", _spy)
    next(iter(af.KaguyaTC(products=["ortho"], patch_size=8, length=1, seed=0)))
    assert calls == [("kaguya_terrain_camera_usgs_dtms", "orthoimage")]


def test_supports_indexing() -> None:
    moondata = af.KaguyaTC(products=["dtm"], patch_size=8, length=5, seed=0)
    sample = moondata[0]
    assert set(sample) == SAMPLE_KEYS
    assert sample["image"].shape == (1, 8, 8)
    # Indexing is deterministic and negative indices work.
    assert torch.equal(moondata[0]["image"], moondata[0]["image"])
    assert torch.equal(moondata[-1]["image"], moondata[4]["image"])


def test_index_out_of_range_raises() -> None:
    moondata = af.KaguyaTC(products=["dtm"], patch_size=8, length=3, seed=0)
    with pytest.raises(IndexError):
        moondata[3]


def test_patch_size_sets_image_dimensions() -> None:
    moondata = af.KaguyaTCImagery(patch_size=48, length=1, seed=0)
    assert moondata[0]["image"].shape == (1, 48, 48)


def test_shuffled_dataloader_covers_every_index() -> None:
    moondata = af.KaguyaTC(products=["dtm"], patch_size=8, length=6, seed=0)
    loader = DataLoader(moondata, batch_size=2, shuffle=True)
    total = sum(batch["image"].shape[0] for batch in loader)
    assert total == 6


def test_seed_is_reproducible() -> None:
    def sampler() -> af.KaguyaTCImagery:
        return af.KaguyaTCImagery(products=["image"], patch_size=16, length=2, seed=42)

    first = [s["image"] for s in sampler()]
    second = [s["image"] for s in sampler()]
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_intersection_stacks_channels() -> None:
    moondata = af.KaguyaTC(
        products=["dtm"], bbox=(-30.0, -10.0, -20.0, 0.0), patch_size=16, length=4, seed=1
    ) & af.KaguyaTCImagery(bbox=(-25.0, -15.0, -15.0, 5.0), patch_size=16, length=6, seed=1)
    assert moondata.bbox == (-25.0, -10.0, -20.0, 0.0)
    assert len(moondata) == 4
    for sample in moondata:
        assert set(sample) == SAMPLE_KEYS
        assert sample["image"].shape == (2, 16, 16)
        assert sample["mask"].shape == (2, 16, 16)
        assert sample["layers"] == ["kaguya_tc_dtm", "kaguya_tc_image"]


def test_intersection_nests() -> None:
    moondata = (
        af.KaguyaTC(products=["dtm"], patch_size=16, length=2, seed=0)
        & af.KaguyaTC(products=["ortho"], patch_size=16, length=2, seed=0)
        & af.KaguyaTCImagery(patch_size=16, length=2, seed=0)
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (3, 16, 16)
    assert sample["layers"] == ["kaguya_tc_dtm", "kaguya_tc_ortho", "kaguya_tc_image"]


def test_works_with_default_dataloader_collation() -> None:
    moondata = af.KaguyaTCImagery(patch_size=16, length=4, seed=0)
    loader = DataLoader(moondata, batch_size=2)
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["image"].shape == (2, 1, 16, 16)
    assert batches[0]["mask"].dtype == torch.bool


def test_rejects_empty_products() -> None:
    with pytest.raises(ValueError):
        af.KaguyaTC(products=[])


def test_rejects_unknown_product() -> None:
    with pytest.raises(ValueError):
        af.KaguyaTC(products=["thermal"])


def test_rejects_invalid_bbox() -> None:
    with pytest.raises(ValueError):
        af.KaguyaTCImagery(bbox=(10.0, 0.0, -10.0, 5.0))


def test_rejects_disjoint_intersection() -> None:
    with pytest.raises(ValueError):
        af.KaguyaTC(bbox=(-60.0, 5.0, -55.0, 10.0)) & af.KaguyaTCImagery(
            bbox=(30.0, -10.0, 40.0, 0.0)
        )


def test_rejects_mismatched_grids() -> None:
    with pytest.raises(ValueError):
        af.KaguyaTC(patch_size=16) & af.KaguyaTCImagery(patch_size=32)


def test_catalog_points_at_dataset_classes() -> None:
    assert MOON.probes["kaguya"].instruments["tc_dtm"].dataset is af.KaguyaTC
    assert MOON.probes["kaguya"].instruments["tc_imagery"].dataset is af.KaguyaTCImagery


def test_catalog_and_registry_agree() -> None:
    spec = MOON.probes["kaguya"].instruments["tc_dtm"].products["dtm"]
    assert spec is LAYERS["kaguya_tc_dtm"]
    assert spec.collection == af.KaguyaTC.collection
    assert spec.asset == "dtm"


def test_mosaic_prefers_earlier_items_and_fills_gaps(monkeypatch: pytest.MonkeyPatch) -> None:
    from astrofetch.data.grid import TargetGrid

    def _coverage(href, grid, band=1, resampling=None, nodata_override=None):
        image = np.full((grid.height, grid.width), float(href[-1]), dtype=np.float32)
        mask = np.zeros((grid.height, grid.width), dtype=bool)
        if href.endswith("1"):  # first item covers only the left half
            mask[:, : grid.width // 2] = True
        else:  # second item covers everything
            mask[:] = True
        image[~mask] = 0.0
        return image, mask

    monkeypatch.setattr(ds.raster, "read_window", _coverage)
    grid = TargetGrid(bbox=(0.0, 0.0, 1.0, 1.0), width=8, height=8)

    image, mask = ds._mosaic(["item1", "item2"], grid)

    assert mask.all()
    assert (image[:, :4] == 1.0).all()  # earlier item wins where it has data
    assert (image[:, 4:] == 2.0).all()  # later item fills the gap


def test_mosaic_with_no_items_is_all_invalid() -> None:
    from astrofetch.data.grid import TargetGrid

    grid = TargetGrid(bbox=(0.0, 0.0, 1.0, 1.0), width=8, height=8)
    image, mask = ds._mosaic([], grid)
    assert not mask.any()
    assert (image == 0.0).all()


def test_mosaic_passes_band_and_nodata_override() -> None:
    from astrofetch.data.grid import TargetGrid

    seen: list[tuple[int, float | None]] = []

    def _spy(href, grid, band=1, resampling=None, nodata_override=None):
        seen.append((band, nodata_override))
        shape = (grid.height, grid.width)
        return np.zeros(shape, dtype=np.float32), np.ones(shape, dtype=bool)

    grid = TargetGrid(bbox=(0.0, 0.0, 1.0, 1.0), width=4, height=4)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ds.raster, "read_window", _spy)
        ds._mosaic(["item"], grid, band=2, nodata_override=-999.0)
    assert seen == [(2, -999.0)]


# --- ODEInstrumentDataset -----------------------------------------------


def test_ode_instrument_yields_sample_dicts() -> None:
    moondata = af.LROCNACDTM(
        products=["dtm", "ortho"],
        bbox=(-60.0, 5.0, -55.0, 10.0),
        patch_size=32,
        length=3,
        seed=0,
        footprint_sampling=False,
    )
    samples = list(moondata)
    assert len(samples) == 3
    for sample in samples:
        assert set(sample) == SAMPLE_KEYS
        assert sample["image"].shape == (2, 32, 32)
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].shape == (2, 32, 32)
        assert sample["layers"] == ["lroc_nac_dtm", "lroc_nac_ortho"]


def test_ode_read_queries_ode_by_ihid_iid_pt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, str]] = []

    def _spy(ihid, iid, pt, pattern, bbox, max_products=20, file_type="Product", root=None):
        calls.append((ihid, iid, pt))
        return [f"{ihid}|{iid}|{pt}"]

    monkeypatch.setattr(ds.ode, "find_file_urls", _spy)
    next(
        iter(
            af.LROCNACDTM(
                products=["dtm"], patch_size=8, length=1, seed=0, footprint_sampling=False
            )
        )
    )
    assert calls == [("LRO", "LROC", "SDNDTM")]


def test_ode_asset_file_type_defaults_to_product() -> None:
    assert ds.ODEAsset("layer", "PT", r".+").file_type == "Product"


def test_ode_read_passes_asset_file_type(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def _spy(ihid, iid, pt, pattern, bbox, max_products=20, file_type="Product", root=None):
        seen.append(file_type)
        return [f"{ihid}|{iid}|{pt}"]

    class _Referenced(ds.ODEInstrumentDataset):
        probe = "Test Probe"
        instrument = "Test Instrument"
        ihid = "X"
        iid = "Y"
        all_products = {
            "data": ds.ODEAsset("test_layer", "PT", r".+", file_type="Referenced"),
        }

    monkeypatch.setattr(ds.ode, "find_file_urls", _spy)
    next(iter(_Referenced(products=["data"], patch_size=8, length=1, seed=0)))
    assert seen == ["Referenced"]


def test_ode_dataset_default_products_is_quantitative_only() -> None:
    # Slope/shade are rendered visualizations (AGENTS rule 3); only
    # elevation, orthoimage, and confidence are offered.
    assert set(af.LROCNACDTM.all_products) == {"dtm", "ortho", "confidence"}


def test_ode_rejects_unknown_product() -> None:
    with pytest.raises(ValueError):
        af.LROCNACDTM(products=["slope"])


# --- MosaicDataset -------------------------------------------------------


def test_mosaic_dataset_reads_fixed_href_without_search(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_hrefs: list[str] = []

    def _spy(href, grid, band=1, resampling=None, nodata_override=None):
        seen_hrefs.append(href)
        return _fake_read_window(href, grid, band, resampling, nodata_override)

    monkeypatch.setattr(ds.raster, "read_window", _spy)
    monkeypatch.setattr(
        ds.ode,
        "query_products",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("MosaicDataset must not search ODE")),
    )
    moondata = af.LROCWACMosaic(patch_size=8, length=1, seed=0)
    sample = moondata[0]
    assert set(sample) == SAMPLE_KEYS
    assert sample["image"].shape == (1, 8, 8)
    assert seen_hrefs == [ds.endpoints.LROC_WAC_MOSAIC_100M_URL]


def test_lola_and_sldem_read_their_fixed_hrefs(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_hrefs: list[str] = []
    monkeypatch.setattr(
        ds.raster,
        "read_window",
        lambda href, grid, band=1, resampling=None, nodata_override=None: (
            seen_hrefs.append(href),
            _fake_read_window(href, grid),
        )[1],
    )
    af.LOLA(patch_size=8, length=1, seed=0)[0]
    af.SLDEM2015(patch_size=8, length=1, seed=0)[0]
    assert seen_hrefs == [ds.endpoints.LOLA_DEM_128_URL, ds.endpoints.SLDEM2015_URL]


# --- footprint-constrained sampling ---------------------------------------

_FOOTPRINTS = [(-60.0, 5.0, -59.0, 6.0), (10.0, -20.0, 11.0, -19.0)]


def _fake_footprint_products(*_args, **_kwargs) -> list:
    return [
        ds.ode.ODEProduct(pdsid=f"p{i}", files=(), bbox=fp, metadata={})
        for i, fp in enumerate(_FOOTPRINTS)
    ]


def test_footprint_sampling_draws_windows_inside_a_footprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ds.ode, "query_products", _fake_footprint_products)
    moondata = af.LROCNACDTM(products=["dtm"], patch_size=8, length=20, seed=0)
    assert moondata.footprint_sampling is True
    for sample in moondata:
        west, south, east, north = sample["bbox"]
        assert any(
            fw <= west and east <= fe and fs <= south and north <= fn
            for fw, fs, fe, fn in _FOOTPRINTS
        )


def test_footprint_sampling_is_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ds.ode, "query_products", _fake_footprint_products)

    def sampler() -> af.LROCNACDTM:
        return af.LROCNACDTM(products=["dtm"], patch_size=8, length=3, seed=7)

    first = [s["bbox"] for s in sampler()]
    second = [s["bbox"] for s in sampler()]
    assert first == second


def test_footprint_sampling_falls_back_to_uniform_with_no_footprints() -> None:
    # Default autouse fixture's fake query_products returns [].
    moondata = af.LROCNACDTM(
        products=["dtm"], bbox=(-60.0, 5.0, -55.0, 10.0), patch_size=8, length=1, seed=0
    )
    west, south, east, north = moondata[0]["bbox"]
    assert -60.0 <= west and east <= -55.0
    assert 5.0 <= south and north <= 10.0


def test_footprint_sampling_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(
        ds.ode, "query_products", lambda *a, **k: calls.append(1) or _fake_footprint_products()
    )
    moondata = af.LROCNACDTM(
        products=["dtm"],
        bbox=(-60.0, 5.0, -55.0, 10.0),
        patch_size=8,
        length=1,
        seed=0,
        footprint_sampling=False,
    )
    _ = moondata[0]
    assert calls == []


# --- catalog / registry for the new datasets ------------------------------


def test_catalog_includes_lro_probe() -> None:
    lro = MOON.probes["lro"]
    assert lro.instruments["nac_dtm"].dataset is af.LROCNACDTM
    assert lro.instruments["wac_mosaic"].dataset is af.LROCWACMosaic
    assert lro.instruments["lola"].dataset is af.LOLA
    assert lro.instruments["sldem2015"].dataset is af.SLDEM2015
    assert lro.granules["nac_raw"] is af.LROCNACRaw
    assert lro.granules["wac_raw"] is af.LROCWACRaw


def test_catalog_includes_chandrayaan1_granules() -> None:
    assert MOON.probes["chandrayaan1"].granules["m3"] is af.M3
    assert MOON.probes["chandrayaan1"].instruments == {}


def test_registry_agrees_for_ode_layer() -> None:
    spec = MOON.probes["lro"].instruments["nac_dtm"].products["dtm"]
    assert spec is LAYERS["lroc_nac_dtm"]
    assert spec.source == "ode"
    assert spec.ihid == "LRO"
    assert spec.iid == "LROC"
    assert spec.pt == "SDNDTM"


def test_registry_agrees_for_mosaic_layer() -> None:
    spec = MOON.probes["lro"].instruments["wac_mosaic"].products["morphology"]
    assert spec is LAYERS["lroc_wac_mosaic"]
    assert spec.source == "mosaic"
    assert spec.href == ds.endpoints.LROC_WAC_MOSAIC_100M_URL


def test_registry_marks_stac_layers_with_source() -> None:
    assert LAYERS["kaguya_tc_dtm"].source == "stac"


def test_registry_carries_ode_asset_file_type() -> None:
    assert LAYERS["lroc_nac_dtm"].file_type == "Product"


# --- Phase C: wider PDS ODE roster ----------------------------------------


def test_minirf_patterns_select_the_right_global_mosaic() -> None:
    _assert_pattern_selects(
        af.MiniRF.all_products["cpr"], "LRO/MRFLRO/MOSDDR", "128ppd_simp_0c.lbl"
    )
    _assert_pattern_selects(
        af.MiniRF.all_products["sc"], "LRO/MRFLRO/MOSDDR", "global_sc_128ppd_simp_0c.lbl"
    )
    _assert_pattern_selects(
        af.MiniRF.all_products["oc"], "LRO/MRFLRO/MOSDDR", "global_oc_128ppd_simp_0c.lbl"
    )


def test_minirf_yields_sample_dicts() -> None:
    moondata = af.MiniRF(products=["cpr", "sc"], patch_size=16, length=1, seed=0)
    sample = moondata[0]
    assert set(sample) == SAMPLE_KEYS
    assert sample["layers"] == ["lro_minirf_cpr", "lro_minirf_sc"]


def test_catalog_includes_minirf() -> None:
    assert MOON.probes["lro"].instruments["minirf"].dataset is af.MiniRF
    spec = MOON.probes["lro"].instruments["minirf"].products["cpr"]
    assert spec is LAYERS["lro_minirf_cpr"]
    assert spec.source == "ode"
    assert spec.ihid == "LRO"
    assert spec.iid == "MRFLRO"
    assert spec.pt == "MOSDDR"
