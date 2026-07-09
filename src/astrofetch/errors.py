"""AstroFetch exception hierarchy.

Public errors live here so callers can ``except astrofetch.errors.EndpointError``
without importing internal modules.
"""

from __future__ import annotations


class AstroFetchError(Exception):
    """Base class for every error AstroFetch raises on purpose."""


class EndpointError(AstroFetchError):
    """A remote archive endpoint failed or returned something unusable.

    Carries the offending endpoint (a collection id, URL, or asset href) and
    always points back at ``astrofetch/data/endpoints.py`` — the single place
    external URLs are configured — so a moved or flaky service is easy to trace.
    """

    def __init__(self, endpoint: str, message: str) -> None:
        self.endpoint = endpoint
        super().__init__(
            f"{endpoint}: {message} (endpoint URLs are configured in astrofetch/data/endpoints.py)"
        )
