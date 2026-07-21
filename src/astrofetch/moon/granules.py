"""EXPERIMENTAL: map-style datasets over raw PDS granules, camera geometry.

Unlike every dataset in :mod:`astrofetch.moon.datasets`, these are **not**
reprojected onto a common grid: each item is one raw PDS product (an NAC/WAC
calibrated strip, an M3 radiance cube, ...), read in its own native
camera/instrument geometry with no reprojection, resampling, ISIS, or SPICE
processing. This is a deliberate, documented carve-out from AGENTS.md's
normal coregistered-tensor contract:

- No bbox windowing: ``__getitem__`` returns a whole granule (or a row range,
  via ``rows=``), not a patch cropped to a requested extent.
- Ragged shapes across items -- the default ``DataLoader`` collation will not
  work; use ``batch_size=None`` or a custom ``collate_fn``.
- No ``&`` composition -- there is no shared grid to stack channels onto.

``len()`` is the number of PDS ODE products matching a bbox, fetched once and
eagerly in ``__init__`` so ``len()`` never needs a network call.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import rasterio
import torch
from rasterio.windows import Window
from torch.utils.data import Dataset

from astrofetch.data import ode, raster
from astrofetch.data.grid import BBox
from astrofetch.errors import EndpointError

logger = logging.getLogger(__name__)

_DEFAULT_MAX_PIXELS = 512 * 1024 * 1024 // 4
"""Default guard: about 512 MiB as float32 (bands * width * height * 4 bytes)."""


class GranuleDataset(Dataset[dict]):
    """EXPERIMENTAL map-style dataset over one instrument's raw PDS granules.

    Args:
        bbox: region to search as (west, south, east, north) degrees.
        max_products: cap on granules in the dataset (one ODE search, eager).
        rows: pixel row range read from every granule, e.g. ``slice(0,
            512)``; ``None`` reads the full granule, subject to
            ``max_pixels``.
        max_pixels: raise instead of reading a granule (bands * width *
            height) larger than this; NAC/WAC strips can be gigapixel.

    Raises:
        EndpointError: the ODE search failed.
    """

    probe: ClassVar[str]
    """Name of the probe (spacecraft) carrying this instrument."""

    instrument: ClassVar[str]
    """Human-readable instrument name."""

    ihid: ClassVar[str]
    """ODE instrument host id, e.g. ``"LRO"``."""

    iid: ClassVar[str]
    """ODE instrument id, e.g. ``"LROC"``."""

    pt: ClassVar[str]
    """ODE product type, e.g. ``"CDRNAC4"``."""

    file_pattern: ClassVar[str]
    """Regex (fullmatch, case-insensitive) selecting the main data file."""

    extra_patterns: ClassVar[dict[str, str]] = {}
    """Extra sample-dict keys read the same way, e.g. geolocation backplanes."""

    def __init__(
        self,
        bbox: BBox = (-180.0, -90.0, 180.0, 90.0),
        max_products: int = 100,
        rows: slice | None = None,
        max_pixels: int = _DEFAULT_MAX_PIXELS,
    ) -> None:
        west, south, east, north = bbox
        if not (west < east and south < north):
            raise ValueError(f"invalid bbox (west, south, east, north): {bbox}")
        self.bbox = bbox
        self.rows = rows
        self.max_pixels = max_pixels
        self.products = ode.query_products(self.ihid, self.iid, self.pt, bbox, max_products)

    def __len__(self) -> int:
        return len(self.products)

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self.products)
        if not 0 <= index < len(self.products):
            raise IndexError(f"index out of range for length {len(self.products)}")
        product = self.products[index]

        main_urls = ode.match_files(product.files, self.file_pattern)
        if not main_urls:
            raise EndpointError(
                product.pdsid, f"no file matched {self.file_pattern!r} in this granule"
            )
        image, mask = self._read_granule(main_urls[0])
        sample: dict = {
            "image": image,
            "mask": mask,
            "pdsid": product.pdsid,
            "bbox": product.bbox,
            "meta": product.metadata,
        }
        for key, pattern in self.extra_patterns.items():
            urls = ode.match_files(product.files, pattern)
            if not urls:
                logger.warning("%s: no file matched %r for %r", product.pdsid, pattern, key)
                continue
            extra_image, extra_mask = self._read_granule(urls[0])
            sample[key] = extra_image
            sample[f"{key}_mask"] = extra_mask
        return sample

    def _read_granule(self, url: str) -> tuple[torch.Tensor, torch.Tensor]:
        image, mask = raster.read_full(url, window=self._window_for(url))
        return torch.from_numpy(image), torch.from_numpy(mask)

    def _window_for(self, url: str) -> Window | None:
        if self.rows is not None:
            with rasterio.open(url) as src:
                width = src.width
            return Window.from_slices(self.rows, slice(0, width))
        with rasterio.open(url) as src:
            pixel_count = src.width * src.height * src.count
        if pixel_count > self.max_pixels:
            raise ValueError(
                f"{url}: granule has {pixel_count:,} pixels, over "
                f"max_pixels={self.max_pixels:,}; pass rows=slice(...) to "
                "GranuleDataset(...) to read a row range instead of the whole granule"
            )
        return None


class LROCNACRaw(GranuleDataset):
    """EXPERIMENTAL: LRO LROC NAC calibrated strips, raw camera geometry.

    Radiometrically calibrated (I/F) but not map-projected: no bbox
    windowing, reprojection, ISIS, or SPICE. See :class:`GranuleDataset`.
    """

    probe = "Lunar Reconnaissance Orbiter"
    instrument = "LROC NAC (raw calibrated strips)"
    ihid = "LRO"
    iid = "LROC"
    pt = "CDRNAC4"
    file_pattern = r"M\d+[LR]C\.XML"


class LROCWACRaw(GranuleDataset):
    """EXPERIMENTAL: LRO LROC WAC monochrome calibrated strips, raw camera
    geometry. See :class:`GranuleDataset`.
    """

    probe = "Lunar Reconnaissance Orbiter"
    instrument = "LROC WAC (raw calibrated strips, mono)"
    ihid = "LRO"
    iid = "LROC"
    pt = "CDRWAM4"
    file_pattern = r"M\d+MC\.XML"


class M3(GranuleDataset):
    """EXPERIMENTAL: Chandrayaan-1 Moon Mineralogy Mapper (M3) L1B radiance
    cubes (85 bands), raw camera geometry, with per-pixel geolocation.

    Not map-projected: no bbox windowing, no reprojection. The ``loc`` key
    holds the 3-band (longitude, latitude, elevation) geolocation backplane
    at the same pixel grid as ``image``, for georeferencing samples
    yourself. See :class:`GranuleDataset`.
    """

    probe = "Chandrayaan-1"
    instrument = "Moon Mineralogy Mapper (M3), L1B radiance"
    ihid = "CH1-ORB"
    iid = "M3"
    pt = "CALIMG"
    file_pattern = r"M3G\w+_V\d+_RDN\.IMG"
    extra_patterns = {"loc": r"M3G\w+_V\d+_LOC\.IMG"}
