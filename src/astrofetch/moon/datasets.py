"""Torch dataset classes over lunar instrument data.

Each instrument dataset samples random windows inside its ``bbox`` and, for each
window, reads its product COGs from the USGS ARD STAC catalog, reprojects them
onto a common target grid, and stacks them into a coregistered ``(C, H, W)``
tensor. Combine instruments with ``&`` to stack their channels over the region
they share.

The Phase 1 read path is real: :mod:`astrofetch.data.stac` finds the COG assets
covering a window, :mod:`astrofetch.data.raster` reprojects and reads them, and
:class:`astrofetch.data.cache.WindowCache` memoizes the result on disk.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar, NamedTuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from astrofetch.data import raster, stac
from astrofetch.data.cache import WindowCache
from astrofetch.data.grid import GEOGRAPHIC_CRS, TargetGrid, meters_to_degrees

BBox = tuple[float, float, float, float]
"""(west, south, east, north) in degrees, IAU 2015 Moon."""

CRS = GEOGRAPHIC_CRS
"""Ocentric IAU 2015 Moon geographic CRS carried by every sample."""


class Product(NamedTuple):
    """One user-facing product: its sample-dict layer id and its STAC asset key."""

    layer: str
    asset: str


class _WindowedDataset(IterableDataset):
    """Shared sampling loop: draw random windows, delegate to ``read``.

    Subclasses set ``bbox``, ``resolution``, ``patch_size``, ``length``,
    ``seed``, and ``layers``, and implement ``read``.
    """

    bbox: BBox
    resolution: float
    patch_size: int
    length: int
    seed: int | None
    layers: list[str]

    def read(self, bbox: BBox) -> dict:
        """Read one window as a sample dict.

        Returns:
            dict with keys ``image`` (C, H, W) float32, ``mask`` (C, H, W)
            bool validity, ``layers``, ``bbox``, ``crs``, and ``resolution``.
        """
        raise NotImplementedError

    def __iter__(self) -> Iterator[dict]:
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        for _ in range(self.length):
            yield self.read(self._sample_bbox(generator))

    def __len__(self) -> int:
        return self.length

    def __and__(self, other: _WindowedDataset) -> IntersectionDataset:
        return IntersectionDataset(self, other)

    def _sample_bbox(self, generator: torch.Generator) -> BBox:
        west, south, east, north = self.bbox
        span_lon, span_lat = east - west, north - south
        # Window ground size from patch_size * resolution, converted to degrees
        # at the region's centre latitude; clamp to the bbox so it stays inside.
        center_lat = (south + north) / 2.0
        win_lon, win_lat = meters_to_degrees(self.patch_size * self.resolution, center_lat)
        win_lon, win_lat = min(win_lon, span_lon), min(win_lat, span_lat)
        u, v = torch.rand(2, generator=generator).tolist()
        west0 = west + u * (span_lon - win_lon)
        south0 = south + v * (span_lat - win_lat)
        return (west0, south0, west0 + win_lon, south0 + win_lat)


class InstrumentDataset(_WindowedDataset):
    """Iterable dataset of patches from a single instrument.

    Samples random bounding boxes within ``bbox`` and yields sample dicts:
    ``image`` is a (C, H, W) float tensor with one channel per requested
    product (physical values), ``mask`` a same-shaped bool validity tensor
    (orbital swaths do not cover everything), plus ``layers``/``bbox``/``crs``/
    ``resolution`` provenance. Combine instruments with ``&`` to stack their
    channels over the overlapping region.

    Args:
        products: product names to stack, e.g. ``["dtm"]``; defaults to all
            products the instrument offers.
        bbox: region to sample from as (west, south, east, north) degrees.
        resolution: target resolution in metres per pixel.
        patch_size: output height and width in pixels.
        length: number of patches per epoch.
        seed: RNG seed for reproducible sampling.
        max_items: cap on STAC items mosaicked per layer per window.
        cache: window cache; defaults to the shared on-disk cache.

    Example:
        >>> moondata = KaguyaTC(products=["dtm"], bbox=(-26.4, -50.7, -25.4, -49.6))
        >>> for sample in moondata:  # doctest: +SKIP
        ...     sample["image"]  # torch.Tensor (C, H, W)
    """

    probe: ClassVar[str]
    """Name of the probe (spacecraft) carrying this instrument."""

    instrument: ClassVar[str]
    """Human-readable instrument name."""

    collection: ClassVar[str]
    """USGS ARD STAC collection id backing this instrument's products."""

    all_products: ClassVar[dict[str, Product]]
    """Product name -> (layer id, STAC asset key)."""

    def __init__(
        self,
        products: list[str] | None = None,
        bbox: BBox = (-180.0, -90.0, 180.0, 90.0),
        resolution: float = 100.0,
        patch_size: int = 256,
        length: int = 1000,
        seed: int | None = None,
        max_items: int = 20,
        cache: WindowCache | None = None,
    ) -> None:
        if products is None:
            products = list(self.all_products)
        if not products:
            raise ValueError("at least one product is required")
        unknown = [p for p in products if p not in self.all_products]
        if unknown:
            raise ValueError(
                f"unknown products for {self.instrument}: {unknown}; "
                f"available: {sorted(self.all_products)}"
            )
        west, south, east, north = bbox
        if not (west < east and south < north):
            raise ValueError(f"invalid bbox (west, south, east, north): {bbox}")
        self.products = list(products)
        self.layers = [self.all_products[p].layer for p in products]
        self.bbox = bbox
        self.resolution = resolution
        self.patch_size = patch_size
        self.length = length
        self.seed = seed
        self.max_items = max_items
        self.cache = cache if cache is not None else WindowCache()

    def read(self, bbox: BBox) -> dict:
        grid = TargetGrid(bbox=bbox, width=self.patch_size, height=self.patch_size, crs=CRS)
        images, masks = [], []
        for product in self.products:
            spec = self.all_products[product]
            image, mask = self._read_layer(spec, grid)
            images.append(image)
            masks.append(mask)
        return {
            "image": torch.from_numpy(np.stack(images)).to(torch.float32),
            "mask": torch.from_numpy(np.stack(masks)).to(torch.bool),
            "layers": list(self.layers),
            "bbox": bbox,
            "crs": CRS,
            "resolution": self.resolution,
        }

    def _read_layer(self, spec: Product, grid: TargetGrid) -> tuple[np.ndarray, np.ndarray]:
        cached = self.cache.get(spec.layer, grid)
        if cached is not None:
            return cached
        hrefs = stac.find_asset_hrefs(self.collection, spec.asset, grid.bbox, self.max_items)
        image, mask = _mosaic(hrefs, grid)
        self.cache.put(spec.layer, grid, image, mask)
        return image, mask


def _mosaic(hrefs: list[str], grid: TargetGrid) -> tuple[np.ndarray, np.ndarray]:
    """Reproject each COG onto ``grid`` and fill invalid pixels from later items.

    Earlier items win where they have data; later items fill only the gaps.
    With no items, returns an all-invalid (zero) window — the mask tells the
    truth about coverage rather than fabricating data.
    """
    image = np.zeros((grid.height, grid.width), dtype=np.float32)
    valid = np.zeros((grid.height, grid.width), dtype=bool)
    for href in hrefs:
        layer_image, layer_mask = raster.read_window(href, grid)
        fill = layer_mask & ~valid
        image[fill] = layer_image[fill]
        valid |= layer_mask
    return image, valid


class KaguyaTC(InstrumentDataset):
    """Kaguya (SELENE) Terrain Camera: USGS stereo-derived DTM and orthoimage."""

    probe = "Kaguya (SELENE)"
    instrument = "Terrain Camera (USGS DTM)"
    collection = "kaguya_terrain_camera_usgs_dtms"
    all_products = {
        "dtm": Product("kaguya_tc_dtm", "dtm"),
        "ortho": Product("kaguya_tc_ortho", "orthoimage"),
    }


class KaguyaTCImagery(InstrumentDataset):
    """Kaguya (SELENE) Terrain Camera: stereoscopic radiance imagery.

    Raw Terrain Camera observations (16-bit DN scaled to radiance), distinct
    from the USGS-derived DTM products in :class:`KaguyaTC`.
    """

    probe = "Kaguya (SELENE)"
    instrument = "Terrain Camera (imagery)"
    collection = "kaguya_terrain_camera_stereoscopic_uncontrolled_observations"
    all_products = {"image": Product("kaguya_tc_image", "image")}


class IntersectionDataset(_WindowedDataset):
    """Coregistered channel stack of two datasets over their overlap.

    Owns the sampling loop: every drawn window is read from both children at
    the same bbox and the results are concatenated along the channel axis.
    Children must share ``resolution`` and ``patch_size``. Usually created
    with the ``&`` operator, which also nests: ``a & b & c``.

    Args:
        first: left dataset; its seed wins if both are set.
        second: right dataset.
    """

    def __init__(self, first: _WindowedDataset, second: _WindowedDataset) -> None:
        if (first.resolution, first.patch_size) != (second.resolution, second.patch_size):
            raise ValueError(
                "datasets must share resolution and patch_size, got "
                f"({first.resolution}, {first.patch_size}) and "
                f"({second.resolution}, {second.patch_size})"
            )
        west = max(first.bbox[0], second.bbox[0])
        south = max(first.bbox[1], second.bbox[1])
        east = min(first.bbox[2], second.bbox[2])
        north = min(first.bbox[3], second.bbox[3])
        if not (west < east and south < north):
            raise ValueError(f"bboxes do not overlap: {first.bbox} and {second.bbox}")
        self.first = first
        self.second = second
        self.bbox = (west, south, east, north)
        self.resolution = first.resolution
        self.patch_size = first.patch_size
        self.length = min(first.length, second.length)
        self.seed = first.seed if first.seed is not None else second.seed
        self.layers = first.layers + second.layers

    def read(self, bbox: BBox) -> dict:
        left = self.first.read(bbox)
        right = self.second.read(bbox)
        return {
            "image": torch.cat([left["image"], right["image"]]),
            "mask": torch.cat([left["mask"], right["mask"]]),
            "layers": left["layers"] + right["layers"],
            "bbox": bbox,
            "crs": left["crs"],
            "resolution": self.resolution,
        }
