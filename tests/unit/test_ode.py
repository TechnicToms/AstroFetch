"""Unit tests for the PDS ODE client — the HTTP session is stubbed, never
the network. Fixture bodies in ``tests/fixtures/ode/`` mirror ODE's real
response shapes (verified against the live endpoint): a lone match comes
back as a dict instead of a one-element list, an empty result is the string
``"No Products Found"``, and errors are HTTP 200 responses carrying
``Status: "ERROR"`` in the body.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from astrofetch.data import ode
from astrofetch.errors import EndpointError

_FIXTURES = Path(__file__).parent.parent / "fixtures" / "ode"


def _fixture(name: str) -> dict:
    return json.loads((_FIXTURES / f"{name}.json").read_text())


_SINGLE_PRODUCT_BODY = _fixture("single_product")
_MULTI_PRODUCT_BODY = _fixture("multi_product")
_EMPTY_BODY = _fixture("no_products")
_ERROR_BODY = _fixture("error")


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _Session:
    """Serves ``pages`` in order, one per call; extra calls get an empty result."""

    def __init__(self, pages: list[dict]) -> None:
        self._pages = list(pages)
        self.calls: list[dict] = []

    def get(self, url: str, params: dict | None = None, timeout: float | None = None) -> _Response:
        self.calls.append(dict(params or {}))
        payload = self._pages.pop(0) if self._pages else _EMPTY_BODY
        return _Response(payload)


def _page(pdsids: list[str]) -> dict:
    return {
        "ODEResults": {
            "Status": "Success",
            "Products": {
                "Product": [{"pdsid": pid, "Product_files": {"Product_file": []}} for pid in pdsids]
            },
        }
    }


class _PagingSession:
    """Realistically slices a backing id list by the request's offset/limit."""

    def __init__(self, pdsids: list[str]) -> None:
        self._pdsids = pdsids
        self.calls: list[dict] = []

    def get(self, url: str, params: dict | None = None, timeout: float | None = None) -> _Response:
        params = dict(params or {})
        self.calls.append(params)
        offset, limit = int(params["offset"]), int(params["limit"])
        page_ids = self._pdsids[offset : offset + limit]
        return _Response(_page(page_ids) if page_ids else _EMPTY_BODY)


def _patch_session(monkeypatch: pytest.MonkeyPatch, session: object) -> None:
    monkeypatch.setattr(ode, "_session", lambda: session)


# --- query_products: JSON quirk normalization -----------------------------


def test_single_product_dict_is_normalized_to_a_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session(monkeypatch, _Session([_SINGLE_PRODUCT_BODY]))
    products = ode.query_products("LRO", "LROC", "SDNDTM", (3.0, 25.5, 4.5, 26.5))
    assert len(products) == 1
    assert products[0].pdsid == "sdp.nac_dtm.apollo15"
    assert products[0].bbox == pytest.approx((3.5, 25.8, 4.0, 26.2))
    assert products[0].files == (
        ode.ODEFile("NAC_DTM_APOLLO15.XML", "Product", "https://pds.example/NAC_DTM_APOLLO15.xml"),
    )


def test_multi_product_list_preserves_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session(monkeypatch, _Session([_MULTI_PRODUCT_BODY]))
    products = ode.query_products("LRO", "LROC", "SDNDTM", (0.0, 0.0, 1.0, 1.0))
    assert [p.pdsid for p in products] == ["a", "b"]
    assert len(products[0].files) == 2
    assert len(products[1].files) == 1


def test_no_products_found_returns_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session(monkeypatch, _Session([_EMPTY_BODY]))
    assert ode.query_products("LRO", "LROC", "SDNDTM", (0.0, 0.0, 1.0, 1.0)) == []


def test_error_status_raises_endpoint_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session(monkeypatch, _Session([_ERROR_BODY]))
    with pytest.raises(EndpointError, match="Invalid IIPT"):
        ode.query_products("LRO", "LROC", "BOGUS", (0.0, 0.0, 1.0, 1.0))


def test_request_exception_becomes_endpoint_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def get(self, *args: object, **kwargs: object) -> _Response:
            raise requests.ConnectionError("boom")

    _patch_session(monkeypatch, _Boom())
    with pytest.raises(EndpointError, match="ODE query failed"):
        ode.query_products("LRO", "LROC", "SDNDTM", (0.0, 0.0, 1.0, 1.0))


# --- longitude conversion ---------------------------------------------------


def test_longitude_converted_to_0_360(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _Session([_EMPTY_BODY])
    _patch_session(monkeypatch, session)
    ode.query_products("LRO", "LROC", "SDNDTM", (-26.3, -50.6, -25.5, -49.7))
    assert session.calls[0]["westernlon"] == pytest.approx(333.7)
    assert session.calls[0]["easternlon"] == pytest.approx(334.5)


def test_full_moon_bbox_does_not_collapse_to_zero_width(monkeypatch: pytest.MonkeyPatch) -> None:
    # A naive `lon % 360` on each bound independently sends (-180, 180) to
    # (180, 180): a zero-width query that would silently return nothing.
    session = _Session([_EMPTY_BODY])
    _patch_session(monkeypatch, session)
    ode.query_products("LRO", "LROC", "SDNDTM", (-180.0, -90.0, 180.0, 90.0))
    assert session.calls[0]["westernlon"] == 0.0
    assert session.calls[0]["easternlon"] == 360.0


# --- pagination --------------------------------------------------------------


def test_pagination_collects_multiple_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _PagingSession(["a", "b", "c"])
    _patch_session(monkeypatch, session)
    monkeypatch.setattr(ode, "_PAGE_SIZE", 2)
    products = ode.query_products("LRO", "LROC", "SDNDTM", (0.0, 0.0, 1.0, 1.0), max_products=10)
    assert [p.pdsid for p in products] == ["a", "b", "c"]
    assert [c["offset"] for c in session.calls] == [0, 2]


def test_pagination_stops_at_max_products(monkeypatch: pytest.MonkeyPatch) -> None:
    session = _PagingSession(["a", "b", "c", "d", "e", "f"])
    _patch_session(monkeypatch, session)
    monkeypatch.setattr(ode, "_PAGE_SIZE", 2)
    products = ode.query_products("LRO", "LROC", "SDNDTM", (0.0, 0.0, 1.0, 1.0), max_products=3)
    assert [p.pdsid for p in products] == ["a", "b", "c"]
    assert len(session.calls) == 2  # never asked for a third page


# --- footprint bbox parsing --------------------------------------------------


def test_product_bbox_parses_valid_footprint() -> None:
    meta = {
        "Westernmost_longitude": 3.5,
        "Easternmost_longitude": 4.0,
        "Minimum_latitude": 25.8,
        "Maximum_latitude": 26.2,
    }
    assert ode._product_bbox(meta) == pytest.approx((3.5, 25.8, 4.0, 26.2))


def test_product_bbox_returns_none_for_global_footprint() -> None:
    meta = {
        "Westernmost_longitude": 0,
        "Easternmost_longitude": 360,
        "Minimum_latitude": -90,
        "Maximum_latitude": 90,
    }
    assert ode._product_bbox(meta) is None


def test_product_bbox_returns_none_for_missing_fields() -> None:
    assert ode._product_bbox({}) is None


# --- match_files / find_file_urls --------------------------------------------


def test_match_files_filters_by_pattern_and_type() -> None:
    # URLs end in their real filename, as in actual ODE responses -- the sort
    # key is the URL's basename, so this also exercises sort-by-filename.
    files = (
        ode.ODEFile("A_DTM.TIF", "Product", "https://x/a_dtm.tif"),
        ode.ODEFile("A_BROWSE.JPG", "Browse", "https://x/a_browse.jpg"),
        ode.ODEFile("A_SHADE.TIF", "Product", "https://x/a_shade.tif"),
    )
    assert ode.match_files(files, r"A_DTM\.TIF") == ["https://x/a_dtm.tif"]
    assert ode.match_files(files, r"A_\w+\.TIF") == ["https://x/a_dtm.tif", "https://x/a_shade.tif"]
    assert ode.match_files(files, r".*", file_type=None) == [
        "https://x/a_browse.jpg",
        "https://x/a_dtm.tif",
        "https://x/a_shade.tif",
    ]


def test_match_files_is_case_insensitive() -> None:
    files = (ode.ODEFile("a_dtm.tif", "Product", "https://x/a"),)
    assert ode.match_files(files, r"A_DTM\.TIF") == ["https://x/a"]


def test_find_file_urls_flattens_across_products(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session(monkeypatch, _Session([_MULTI_PRODUCT_BODY]))
    urls = ode.find_file_urls("LRO", "LROC", "SDNDTM", r"\w+_DTM\.TIF", (0.0, 0.0, 1.0, 1.0))
    assert urls == ["https://x/a_dtm.tif", "https://x/b_dtm.tif"]


def test_find_file_urls_no_match_is_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_session(monkeypatch, _Session([_MULTI_PRODUCT_BODY]))
    urls = ode.find_file_urls("LRO", "LROC", "SDNDTM", r"NOTHING_MATCHES", (0.0, 0.0, 1.0, 1.0))
    assert urls == []
