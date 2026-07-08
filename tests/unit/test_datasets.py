import pytest
import torch
from torch.utils.data import DataLoader

import astrofetch as af
from astrofetch.moon import LAYERS, MOON

SAMPLE_KEYS = {"image", "mask", "layers", "bbox", "crs", "resolution"}


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


def test_seed_is_reproducible() -> None:
    kwargs = dict(products=["mosaic"], patch_size=16, length=2, seed=42)
    first = [s["image"] for s in af.LROCWAC(**kwargs)]
    second = [s["image"] for s in af.LROCWAC(**kwargs)]
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_intersection_stacks_channels() -> None:
    moondata = af.KaguyaTC(
        products=["dtm"], bbox=(-30.0, -10.0, -20.0, 0.0), patch_size=16, length=4, seed=1
    ) & af.LROCWAC(bbox=(-25.0, -15.0, -15.0, 5.0), patch_size=16, length=6, seed=1)
    assert moondata.bbox == (-25.0, -10.0, -20.0, 0.0)
    assert len(moondata) == 4
    for sample in moondata:
        assert set(sample) == SAMPLE_KEYS
        assert sample["image"].shape == (2, 16, 16)
        assert sample["mask"].shape == (2, 16, 16)
        assert sample["layers"] == ["kaguya_tc_dtm", "lroc_wac"]


def test_intersection_nests() -> None:
    moondata = (
        af.KaguyaTC(products=["dtm"], patch_size=16, length=2, seed=0)
        & af.KaguyaTC(products=["ortho"], patch_size=16, length=2, seed=0)
        & af.LROCWAC(patch_size=16, length=2, seed=0)
    )
    sample = next(iter(moondata))
    assert sample["image"].shape == (3, 16, 16)
    assert sample["layers"] == ["kaguya_tc_dtm", "kaguya_tc_ortho", "lroc_wac"]


def test_works_with_default_dataloader_collation() -> None:
    moondata = af.LROCWAC(patch_size=16, length=4, seed=0)
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
        af.LROCWAC(bbox=(10.0, 0.0, -10.0, 5.0))


def test_rejects_disjoint_intersection() -> None:
    with pytest.raises(ValueError):
        af.KaguyaTC(bbox=(-60.0, 5.0, -55.0, 10.0)) & af.LROCWAC(bbox=(30.0, -10.0, 40.0, 0.0))


def test_rejects_mismatched_grids() -> None:
    with pytest.raises(ValueError):
        af.KaguyaTC(patch_size=16) & af.LROCWAC(patch_size=32)


def test_catalog_points_at_dataset_classes() -> None:
    assert MOON.probes["lro"].instruments["lroc_wac"].dataset is af.LROCWAC
    assert MOON.probes["kaguya"].instruments["tc"].dataset is af.KaguyaTC


def test_catalog_and_registry_agree() -> None:
    spec = MOON.probes["lro"].instruments["lroc_wac"].products["mosaic"]
    assert spec is LAYERS["lroc_wac"]
    assert spec.probe == af.LROCWAC.probe
    assert spec.instrument == af.LROCWAC.instrument
