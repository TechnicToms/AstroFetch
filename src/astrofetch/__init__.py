"""AstroFetch: PyTorch-friendly, ML-ready access to planetary science data.

One dataset class per instrument; combine them with ``&`` to receive
coregistered multichannel samples. AstroFetch is a thin layer over existing
archive tooling (STAC, COGs, PDS ODE), never a mirror of any archive.
"""

from astrofetch import moon
from astrofetch.moon import (
    LOLA,
    LROCNACDTM,
    M3,
    MOON,
    SLDEM2015,
    WACGLD100,
    DivinerGDR,
    IntersectionDataset,
    KaguyaTC,
    KaguyaTCImagery,
    LROCNACRaw,
    LROCWACMosaic,
    LROCWACRaw,
    MiniRF,
    WACTiO2,
)

__version__ = "0.1.0"

__all__ = [
    "LOLA",
    "MOON",
    "M3",
    "DivinerGDR",
    "IntersectionDataset",
    "KaguyaTC",
    "KaguyaTCImagery",
    "LROCNACDTM",
    "LROCNACRaw",
    "LROCWACMosaic",
    "LROCWACRaw",
    "MiniRF",
    "SLDEM2015",
    "WACGLD100",
    "WACTiO2",
    "moon",
    "__version__",
]
