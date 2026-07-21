"""Unit tests for the experimental raw-granule datasets — local GeoTIFFs
and a stubbed :mod:`astrofetch.data.ode`, never the network.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_bounds

from astrofetch.data import ode
from astrofetch.errors import EndpointError
from astrofetch.moon import granules


def _write_geotiff(path: Path, data: np.ndarray) -> str:
    height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=from_bounds(0.0, 0.0, width, height, width, height),
    ) as dst:
        dst.write(data, 1)
    return str(path)


def _fake_product(pdsid: str, main_url: str, extra_url: str | None = None) -> ode.ODEProduct:
    files = [ode.ODEFile("MAIN.TIF", "Product", main_url)]
    if extra_url is not None:
        files.append(ode.ODEFile("EXTRA.TIF", "Product", extra_url))
    return ode.ODEProduct(
        pdsid=pdsid, files=tuple(files), bbox=(1.0, 2.0, 3.0, 4.0), metadata={"k": "v"}
    )


class _FakeGranules(granules.GranuleDataset):
    probe = "Test Probe"
    instrument = "Test Instrument"
    ihid = "TEST"
    iid = "TESTI"
    pt = "TESTPT"
    file_pattern = r"MAIN\.TIF"
    extra_patterns = {"extra": r"EXTRA\.TIF"}


def test_len_matches_ode_query(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    main = _write_geotiff(tmp_path / "main.tif", np.zeros((4, 4), dtype=np.float32))
    products = [_fake_product("p1", main), _fake_product("p2", main)]
    monkeypatch.setattr(granules.ode, "query_products", lambda *a, **k: products)
    dataset = _FakeGranules(bbox=(-1.0, -1.0, 1.0, 1.0), max_products=10)
    assert len(dataset) == 2


def test_getitem_reads_main_file_and_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    main = _write_geotiff(tmp_path / "main.tif", data)
    monkeypatch.setattr(granules.ode, "query_products", lambda *a, **k: [_fake_product("p1", main)])
    sample = _FakeGranules()[0]
    assert {"image", "mask", "pdsid", "bbox", "meta"} <= set(sample)
    assert sample["pdsid"] == "p1"
    assert sample["bbox"] == (1.0, 2.0, 3.0, 4.0)
    assert sample["meta"] == {"k": "v"}
    assert sample["image"].shape == (1, 8, 8)
    assert torch.equal(sample["image"][0], torch.from_numpy(data))
    assert sample["mask"].all()


def test_getitem_reads_extra_patterns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    main = _write_geotiff(tmp_path / "main.tif", np.zeros((4, 4), dtype=np.float32))
    extra = _write_geotiff(tmp_path / "extra.tif", np.ones((4, 4), dtype=np.float32))
    product = _fake_product("p1", main, extra_url=extra)
    monkeypatch.setattr(granules.ode, "query_products", lambda *a, **k: [product])
    sample = _FakeGranules()[0]
    assert "extra" in sample
    assert "extra_mask" in sample
    assert sample["extra"].shape == (1, 4, 4)
    assert sample["extra"][0, 0, 0] == 1.0


def test_negative_index_and_out_of_range(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    main = _write_geotiff(tmp_path / "main.tif", np.zeros((4, 4), dtype=np.float32))
    products = [_fake_product("p1", main), _fake_product("p2", main)]
    monkeypatch.setattr(granules.ode, "query_products", lambda *a, **k: products)
    dataset = _FakeGranules()
    assert dataset[-1]["pdsid"] == "p2"
    with pytest.raises(IndexError):
        dataset[2]


def test_no_matching_file_raises_endpoint_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    product = ode.ODEProduct(pdsid="p1", files=(), bbox=None, metadata={})
    monkeypatch.setattr(granules.ode, "query_products", lambda *a, **k: [product])
    with pytest.raises(EndpointError):
        _FakeGranules()[0]


def test_rows_reads_only_the_requested_row_range(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    main = _write_geotiff(tmp_path / "main.tif", data)
    monkeypatch.setattr(granules.ode, "query_products", lambda *a, **k: [_fake_product("p1", main)])
    sample = _FakeGranules(rows=slice(2, 5))[0]
    assert sample["image"].shape == (1, 3, 8)
    assert torch.equal(sample["image"][0], torch.from_numpy(data[2:5]))


def test_max_pixels_guard_raises_without_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    main = _write_geotiff(tmp_path / "main.tif", np.zeros((8, 8), dtype=np.float32))
    monkeypatch.setattr(granules.ode, "query_products", lambda *a, **k: [_fake_product("p1", main)])
    dataset = _FakeGranules(max_pixels=10)  # 8*8*1 = 64 > 10
    with pytest.raises(ValueError, match="max_pixels"):
        dataset[0]


def test_invalid_bbox_raises() -> None:
    with pytest.raises(ValueError):
        _FakeGranules(bbox=(10.0, 0.0, -10.0, 5.0))


def test_lroc_nac_raw_classvars() -> None:
    assert granules.LROCNACRaw.ihid == "LRO"
    assert granules.LROCNACRaw.iid == "LROC"
    assert granules.LROCNACRaw.pt == "CDRNAC4"


def test_lroc_wac_raw_classvars() -> None:
    assert granules.LROCWACRaw.ihid == "LRO"
    assert granules.LROCWACRaw.pt == "CDRWAM4"


def test_m3_classvars_and_extra_patterns() -> None:
    assert granules.M3.ihid == "CH1-ORB"
    assert granules.M3.iid == "M3"
    assert granules.M3.pt == "CALIMG"
    assert "loc" in granules.M3.extra_patterns
