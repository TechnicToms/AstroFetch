"""External archive endpoints — the ONLY place URLs live (AGENTS rule 2).

Every COG asset href is discovered through the STAC API at request time, so the
API root below is the single external URL AstroFetch hard-codes. When a service
moves (as QuickMap's domain once did), this is the only file to change.
"""

from __future__ import annotations

STAC_API_ROOT = "https://stac.astrogeology.usgs.gov/api/"
"""USGS Astrogeology Analysis Ready Data STAC API root (pystac-client entry)."""

ODE_API_ROOT = "https://oderest.rsl.wustl.edu/live2/"
"""NASA PDS Orbital Data Explorer (ODE) REST API root, Washington Univ. St.
Louis. Used to search PDS3/PDS4 products (LROC, LOLA, M3, ...) by instrument
and bounding box; the USGS ARD STAC catalog does not carry these instruments.
Last verified 2026-07-20."""

LROC_WAC_MOSAIC_100M_URL = (
    "https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/"
    "Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
)
"""LRO LROC WAC global morphology mosaic, 100 m/px, equirectangular.

Not a Cloud Optimized GeoTIFF (striped, no overviews): windowed reads at
native resolution (100 m) are efficient; heavily downsampled reads are not.
Last verified 2026-07-20.
"""

LOLA_DEM_128_URL = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/"
    "data/lola_gdr/cylindrical/float_img/ldem_128_float.lbl"
)
"""LOLA global DEM, 128 px/degree (~237 m/px at the equator), float32 metres
above the IAU 2015 Moon reference sphere. Detached PDS3 label; GDAL's PDS
driver resolves the sibling ``.img`` over HTTPS. Last verified 2026-07-20."""

SLDEM2015_URL = (
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/"
    "data/sldem2015/global/float_img/sldem2015_128_60s_60n_000_360_float.lbl"
)
"""SLDEM2015: LOLA + Kaguya Terrain Camera co-registered DEM, 128 px/degree,
float32 metres. Source coverage is 60S-60N only (not a bug); windows outside
that band read back with ``mask`` all ``False``. Last verified 2026-07-20."""
