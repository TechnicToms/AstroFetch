import pytest
import torch
from torch.utils.data import DataLoader

import astrofetch as af
from astrofetch.moon.datasets import Patch


def test_iterates_patches() -> None:
    moondata = af.LunarMoon(
        layers=["kaguya_tc_dtm", "lroc_wac"],
        bbox=(-60.0, 5.0, -55.0, 10.0),
        patch_size=32,
        length=3,
        seed=0,
    )
    patches = list(moondata)
    assert len(patches) == 3
    for patch in patches:
        assert isinstance(patch, Patch)
        assert patch.tensor.shape == (2, 32, 32)
        assert patch.meta["layers"] == ["kaguya_tc_dtm", "lroc_wac"]


def test_seed_is_reproducible() -> None:
    kwargs = dict(layers=["lroc_wac"], patch_size=16, length=2, seed=42)
    first = [p.tensor for p in af.LunarMoon(**kwargs)]
    second = [p.tensor for p in af.LunarMoon(**kwargs)]
    for a, b in zip(first, second):
        assert torch.equal(a, b)


def test_works_with_dataloader() -> None:
    moondata = af.LunarMoon(layers=["lroc_wac"], patch_size=16, length=4, seed=0)
    loader = DataLoader(moondata, batch_size=2, collate_fn=lambda batch: batch)
    batches = list(loader)
    assert len(batches) == 2
    assert all(len(batch) == 2 for batch in batches)


def test_rejects_empty_layers() -> None:
    with pytest.raises(ValueError):
        af.LunarMoon(layers=[])


def test_rejects_invalid_bbox() -> None:
    with pytest.raises(ValueError):
        af.LunarMoon(layers=["lroc_wac"], bbox=(10.0, 0.0, -10.0, 5.0))
