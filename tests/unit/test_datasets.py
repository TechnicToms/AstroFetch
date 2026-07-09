"""Unit tests for the instrument datasets.

The read path is real (STAC search + COG reproject + cache), so these tests
stub :mod:`astrofetch.data.stac` and :mod:`astrofetch.data.raster` — no network,
no real COGs — and redirect the cache to a temp dir. A deterministic fake read
lets us assert shapes, layer wiring, reproducibility, and composition.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import astrofetch as af
from astrofetch.moon import LAYERS, MOON
from astrofetch.moon import datasets as ds

SAMPLE_KEYS = {"image", "mask", "layers", "bbox", "crs", "resolution"}


def _fake_find_asset_hrefs(
    collection: str, asset: str, bbox: tuple, max_items: int = 20, root: str | None = None
) -> list[str]:
    return [f"{collection}|{asset}"]


def _fake_read_window(href, grid, band=1, resampling=None):
    # Deterministic in (href, window): identical requests read identical data,
    # which is what makes seeded sampling reproducible under mocking.
    value = float(hash((href, grid.bbox)) % 997)
    image = np.full((grid.height, grid.width), value, dtype=np.float32)
    return image, np.ones((grid.height, grid.width), dtype=bool)


@pytest.fixture(autouse=True)
def _mock_reads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ASTROFETCH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(ds.stac, "find_asset_hrefs", _fake_find_asset_hrefs)
    monkeypatch.setattr(ds.raster, "read_window", _fake_read_window)


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

    def _coverage(href, grid, band=1, resampling=None):
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
