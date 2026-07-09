"""Unit tests for the STAC wrapper — the client is stubbed, never the network."""

from __future__ import annotations

import pytest
from pystac_client.exceptions import APIError

from astrofetch.data import stac
from astrofetch.errors import EndpointError


class _Asset:
    def __init__(self, href: str) -> None:
        self.href = href


class _Item:
    def __init__(self, item_id: str, assets: dict[str, _Asset]) -> None:
        self.id = item_id
        self.assets = assets


class _Search:
    def __init__(self, items: list[_Item]) -> None:
        self._items = items

    def items(self) -> list[_Item]:
        return self._items


class _Client:
    def __init__(self, items: list[_Item]) -> None:
        self._items = items
        self.calls: list[dict] = []

    def search(self, **kwargs: object) -> _Search:
        self.calls.append(kwargs)
        return _Search(self._items)


def _patch_catalog(monkeypatch: pytest.MonkeyPatch, client: _Client) -> None:
    monkeypatch.setattr(stac, "catalog", lambda root=stac.STAC_API_ROOT: client)


def test_returns_hrefs_in_item_order(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client([
        _Item("a", {"dtm": _Asset("http://x/a.tif")}),
        _Item("b", {"dtm": _Asset("http://x/b.tif")}),
    ])
    _patch_catalog(monkeypatch, client)

    hrefs = stac.find_asset_hrefs("coll", "dtm", (-1.0, -1.0, 1.0, 1.0), max_items=5)

    assert hrefs == ["http://x/a.tif", "http://x/b.tif"]
    assert client.calls == [
        {"collections": ["coll"], "bbox": [-1.0, -1.0, 1.0, 1.0], "max_items": 5}
    ]


def test_no_overlap_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_catalog(monkeypatch, _Client([]))
    assert stac.find_asset_hrefs("coll", "dtm", (0.0, 0.0, 1.0, 1.0)) == []


def test_missing_asset_raises_endpoint_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_catalog(monkeypatch, _Client([_Item("a", {"thumbnail": _Asset("http://x/a.jpg")})]))
    with pytest.raises(EndpointError, match="no asset 'dtm'"):
        stac.find_asset_hrefs("coll", "dtm", (0.0, 0.0, 1.0, 1.0))


def test_api_error_becomes_endpoint_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom(_Client):
        def search(self, **kwargs: object) -> _Search:
            raise APIError("boom")

    _patch_catalog(monkeypatch, _Boom([]))
    with pytest.raises(EndpointError, match="STAC search failed"):
        stac.find_asset_hrefs("coll", "dtm", (0.0, 0.0, 1.0, 1.0))
