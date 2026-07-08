"""AstroFetch: PyTorch-friendly, ML-ready access to planetary science data.

Request a bounding box, receive a coregistered multichannel tensor. AstroFetch
is a thin layer over existing archive tooling (STAC, COGs), never a mirror.
"""

from astrofetch import moon
from astrofetch.moon.datasets import LunarMoon

__version__ = "0.1.0"

__all__ = ["LunarMoon", "moon", "__version__"]
