"""AstroFetch: PyTorch-friendly, ML-ready access to planetary science data.

One dataset class per instrument; combine them with ``&`` to receive
coregistered multichannel samples. AstroFetch is a thin layer over existing
archive tooling (STAC, COGs), never a mirror.
"""

from astrofetch import moon
from astrofetch.moon import MOON, IntersectionDataset, KaguyaTC, LROCWAC

__version__ = "0.1.0"

__all__ = ["MOON", "IntersectionDataset", "KaguyaTC", "LROCWAC", "moon", "__version__"]
