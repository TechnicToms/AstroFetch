"""STAC search against the USGS ARD catalog (pystac-client), politely.

Wraps pystac-client with a retrying, backed-off HTTP session — AGENTS rule 5:
never hammer archive servers — and normalizes failures into :class:`EndpointError`
so callers never have to know pystac-client's exception surface. A single client
is reused across searches (one polite connection pool, not one per request).
"""

from __future__ import annotations

from functools import cache

from pystac_client import Client
from pystac_client.exceptions import APIError
from pystac_client.stac_api_io import StacApiIO
from urllib3.util.retry import Retry

from astrofetch.data.endpoints import STAC_API_ROOT
from astrofetch.data.grid import BBox
from astrofetch.errors import EndpointError

_TIMEOUT_S = 30
"""Per-request timeout; a stuck archive should fail, not hang a dataloader."""

_RETRY = Retry(
    total=4,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset({"GET", "POST"}),
)
"""Exponential backoff on rate-limit and transient server errors."""


@cache
def catalog(root: str = STAC_API_ROOT) -> Client:
    """Open (and memoize) the STAC API client for ``root``.

    Raises:
        EndpointError: the catalog root could not be opened.
    """
    try:
        return Client.open(root, stac_io=StacApiIO(max_retries=_RETRY, timeout=_TIMEOUT_S))
    except APIError as exc:
        raise EndpointError(root, f"could not open STAC catalog: {exc}") from exc


def find_asset_hrefs(
    collection: str,
    asset_key: str,
    bbox: BBox,
    max_items: int = 20,
    root: str = STAC_API_ROOT,
) -> list[str]:
    """Return hrefs of one asset across items of ``collection`` intersecting ``bbox``.

    Args:
        collection: STAC collection id to search.
        asset_key: asset key to pull from each matching item, e.g. ``"dtm"``.
        bbox: (west, south, east, north) in degrees.
        max_items: cap on items returned; bounds the request volume per read.
        root: STAC API root; defaults to the configured USGS ARD catalog.

    Returns:
        Asset hrefs, one per intersecting item (possibly empty if none overlap).

    Raises:
        EndpointError: the search failed or a matched item lacks ``asset_key``.
    """
    client = catalog(root)
    try:
        search = client.search(collections=[collection], bbox=list(bbox), max_items=max_items)
        items = list(search.items())
    except APIError as exc:
        raise EndpointError(collection, f"STAC search failed: {exc}") from exc

    hrefs = []
    for item in items:
        asset = item.assets.get(asset_key)
        if asset is None:
            raise EndpointError(
                collection,
                f"item {item.id} has no asset {asset_key!r}; available: {sorted(item.assets)}",
            )
        hrefs.append(asset.href)
    return hrefs
