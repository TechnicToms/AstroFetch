"""External archive endpoints — the ONLY place URLs live (AGENTS rule 2).

Every COG asset href is discovered through the STAC API at request time, so the
API root below is the single external URL AstroFetch hard-codes. When a service
moves (as QuickMap's domain once did), this is the only file to change.
"""

from __future__ import annotations

STAC_API_ROOT = "https://stac.astrogeology.usgs.gov/api/"
"""USGS Astrogeology Analysis Ready Data STAC API root (pystac-client entry)."""
