"""Product search against the NASA PDS Orbital Data Explorer (ODE), politely.

Mirrors :mod:`astrofetch.data.stac`: a single retrying, backed-off HTTP session
(AGENTS rule 5) and failures normalized into :class:`EndpointError`, so callers
never have to know ODE's JSON quirks — a lone result comes back as a dict
instead of a list, an empty result is the string ``"No Products Found"``
instead of an empty list, and errors are HTTP 200 responses with an error
message in the body.
"""

from __future__ import annotations

import re
from functools import cache
from typing import Any, NamedTuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from astrofetch.data.endpoints import ODE_API_ROOT
from astrofetch.data.grid import BBox
from astrofetch.errors import EndpointError

_TIMEOUT_S = 30
"""Per-request timeout; a stuck archive should fail, not hang a dataloader."""

_RETRY = Retry(
    total=4,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET"}),
)
"""Exponential backoff on rate-limit and transient server errors."""

_PAGE_SIZE = 100
"""ODE ``limit`` per request; keeps any single request small and polite."""

_MAX_PAGES = 20
"""Hard cap on pages fetched regardless of ``max_products``, so a caller
requesting an unreasonably large ``max_products`` cannot loop indefinitely."""


class ODEFile(NamedTuple):
    """One file attached to an ODE product."""

    filename: str
    type: str
    """ODE file role, e.g. ``"Product"``, ``"Browse"``, ``"Derived"``."""
    url: str


class ODEProduct(NamedTuple):
    """One ODE product: its id, files, footprint, and raw metadata."""

    pdsid: str
    files: tuple[ODEFile, ...]
    bbox: BBox | None
    """(west, south, east, north) in degrees, -180 to 180; ``None`` when the
    footprint is not representable as a simple bbox (crosses the antimeridian,
    or spans exactly 360 degrees of longitude)."""
    metadata: dict[str, Any]
    """Raw per-product ODE metadata, unmodified."""


@cache
def _session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _as_list(value: Any) -> list[Any]:
    # ODE returns a bare dict instead of a one-element list when exactly one
    # item matches; normalize both shapes to a list.
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _normalize_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def _to_ode_lon_range(west: float, east: float) -> tuple[float, float]:
    # ODE wants 0-360 westernlon/easternlon with westernlon < easternlon.
    # Taking `% 360` of each bound independently collapses a full-Moon bbox
    # like (-180, 180) to (180, 180) -- a zero-width query that silently
    # returns nothing. Shifting `west` into 0-360 and adding back the
    # original span avoids that, for any span up to a full 360 degrees.
    span = east - west
    if span >= 360.0:
        return 0.0, 360.0
    w = west % 360.0
    return w, w + span


def _product_bbox(meta: dict[str, Any]) -> BBox | None:
    try:
        west = float(meta["Westernmost_longitude"])
        east = float(meta["Easternmost_longitude"])
        south = float(meta["Minimum_latitude"])
        north = float(meta["Maximum_latitude"])
    except (KeyError, TypeError, ValueError):
        return None
    west, east = _normalize_lon(west), _normalize_lon(east)
    if not (west < east and south < north):
        return None
    return (west, south, east, north)


def _parse_product(raw: dict[str, Any]) -> ODEProduct:
    file_entries = _as_list(raw.get("Product_files", {}).get("Product_file"))
    files = tuple(
        ODEFile(f.get("FileName", ""), f.get("Type", ""), f.get("URL", "")) for f in file_entries
    )
    return ODEProduct(
        pdsid=raw.get("pdsid", ""), files=files, bbox=_product_bbox(raw), metadata=raw
    )


def query_products(
    ihid: str,
    iid: str,
    pt: str,
    bbox: BBox,
    max_products: int = 20,
    root: str = ODE_API_ROOT,
) -> list[ODEProduct]:
    """Search ODE for products of one instrument and product type in ``bbox``.

    Args:
        ihid: ODE instrument host id, e.g. ``"LRO"``.
        iid: ODE instrument id, e.g. ``"LROC"``.
        pt: ODE product type, e.g. ``"SDNDTM"``.
        bbox: (west, south, east, north) in degrees, -180 to 180.
        max_products: cap on products returned; bounds request volume and
            paging (fetched in pages of up to 100).
        root: ODE API root; defaults to the configured endpoint.

    Returns:
        Matching products, possibly empty if none overlap ``bbox``.

    Raises:
        EndpointError: the query failed or ODE reported an error.

    Example:
        >>> from astrofetch.data.ode import query_products
        >>> query_products("LRO", "LROC", "SDNDTM", bbox=(3.0, 25.5, 4.5, 26.5))  # doctest: +SKIP
        [ODEProduct(pdsid='sdp.nac_dtm.apollo15...', ...), ...]
    """
    west, south, east, north = bbox
    lon_west, lon_east = _to_ode_lon_range(west, east)
    endpoint = f"{ihid}/{iid}/{pt}"
    products: list[ODEProduct] = []
    offset = 0
    for _ in range(_MAX_PAGES):
        remaining = max_products - len(products)
        if remaining <= 0:
            break
        limit = min(_PAGE_SIZE, remaining)
        params = {
            "query": "product",
            "results": "fmp",
            "output": "JSON",
            "target": "moon",
            "ihid": ihid,
            "iid": iid,
            "pt": pt,
            "westernlon": lon_west,
            "easternlon": lon_east,
            "minlat": south,
            "maxlat": north,
            "limit": limit,
            "offset": offset,
        }
        try:
            response = _session().get(root, params=params, timeout=_TIMEOUT_S)
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise EndpointError(endpoint, f"ODE query failed: {exc}") from exc

        results = body.get("ODEResults", {})
        if results.get("Status") == "ERROR":
            raise EndpointError(endpoint, f"ODE error: {results.get('Error')}")

        raw_products = results.get("Products", [])
        if isinstance(raw_products, str):  # "No Products Found"
            break
        page = [_parse_product(p) for p in _as_list(raw_products.get("Product"))]
        products.extend(page)
        if len(page) < limit:
            break
        offset += limit
    return products[:max_products]


def match_files(
    files: tuple[ODEFile, ...], pattern: str, file_type: str | None = "Product"
) -> list[str]:
    """Return URLs of ``files`` whose name matches ``pattern`` and ``file_type``.

    Args:
        files: files to filter, typically ``product.files``.
        pattern: regex, matched against the filename with ``fullmatch`` and
            case-insensitively (ODE filenames are inconsistently cased).
        file_type: required ODE file role, e.g. ``"Product"``; ``None`` skips
            this filter.

    Returns:
        Matching URLs, sorted by filename for deterministic ordering.
    """
    regex = re.compile(pattern, re.IGNORECASE)
    return sorted(
        (
            f.url
            for f in files
            if (file_type is None or f.type == file_type) and regex.fullmatch(f.filename)
        ),
        key=lambda url: url.rsplit("/", 1)[-1],
    )


def find_file_urls(
    ihid: str,
    iid: str,
    pt: str,
    pattern: str,
    bbox: BBox,
    max_products: int = 20,
    file_type: str | None = "Product",
    root: str = ODE_API_ROOT,
) -> list[str]:
    """Return file URLs matching ``pattern`` across products in ``bbox``.

    The ODE analogue of :func:`astrofetch.data.stac.find_asset_hrefs`: a
    product bundle can contain many files (data, browse, derived), so this
    searches products then filters their files by name and role. A product
    with no matching file is not an error — masks report coverage truthfully
    instead.

    Args:
        ihid: ODE instrument host id, e.g. ``"LRO"``.
        iid: ODE instrument id, e.g. ``"LROC"``.
        pt: ODE product type, e.g. ``"SDNDTM"``.
        pattern: regex matched against filenames, case-insensitive.
        bbox: (west, south, east, north) in degrees, -180 to 180.
        max_products: cap on products searched.
        file_type: required ODE file role; ``None`` skips this filter.
        root: ODE API root; defaults to the configured endpoint.

    Returns:
        Matching URLs, product order then filename order.

    Raises:
        EndpointError: the search failed or ODE reported an error.
    """
    products = query_products(ihid, iid, pt, bbox, max_products, root)
    hrefs: list[str] = []
    for product in products:
        hrefs.extend(match_files(product.files, pattern, file_type))
    return hrefs
