"""Torch dataset classes over lunar instrument data.

Phase 0 scaffolding: the API surface is real, the data path is a placeholder.
Instrument datasets will be backed by the STAC sampler (Phase 1); until then
``read`` yields correctly-shaped random tensors so the training loop contract —
``for sample in KaguyaTC(...) & LROCWAC(...)`` — can be exercised end to end.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import ClassVar

import torch
from torch.utils.data import IterableDataset

BBox = tuple[float, float, float, float]
"""(west, south, east, north) in degrees, IAU 2015 Moon."""

CRS = "IAU_2015:30100"
"""Ocentric IAU 2015 Moon CRS carried by every sample."""


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

    def read(self, bbox: BBox, generator: torch.Generator | None = None) -> dict:
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
            yield self.read(self._sample_bbox(generator), generator)

    def __len__(self) -> int:
        return self.length

    def __and__(self, other: _WindowedDataset) -> IntersectionDataset:
        return IntersectionDataset(self, other)

    def _sample_bbox(self, generator: torch.Generator) -> BBox:
        west, south, east, north = self.bbox
        u, v = torch.rand(2, generator=generator).tolist()
        # Placeholder: a degenerate point bbox; Phase 1 sizes the window from
        # patch_size * resolution on the target grid.
        lon = west + u * (east - west)
        lat = south + v * (north - south)
        return (lon, lat, lon, lat)


class InstrumentDataset(_WindowedDataset):
    """Iterable dataset of patches from a single instrument.

    Samples random bounding boxes within ``bbox`` and yields sample dicts:
    ``image`` is a (C, H, W) float tensor with one channel per requested
    product, ``mask`` a same-shaped bool validity tensor (orbital swaths do
    not cover everything), plus ``layers``/``bbox``/``crs``/``resolution``
    provenance. Combine instruments with ``&`` to stack their channels over
    the overlapping region.

    Args:
        products: product names to stack, e.g. ``["dtm"]``; defaults to all
            products the instrument offers.
        bbox: region to sample from as (west, south, east, north) degrees.
        resolution: target resolution in meters per pixel.
        patch_size: output height and width in pixels.
        length: number of patches per epoch.
        seed: RNG seed for reproducible sampling.

    Example:
        >>> moondata = LROCWAC(bbox=(-60.0, 5.0, -55.0, 10.0))
        >>> for sample in moondata:
        ...     sample["image"]  # torch.Tensor (C, H, W)
    """

    probe: ClassVar[str]
    """Name of the probe (spacecraft) carrying this instrument."""

    instrument: ClassVar[str]
    """Human-readable instrument name."""

    all_products: ClassVar[dict[str, str]]
    """Product name -> layer identifier (see ``astrofetch.moon.layers``)."""

    def __init__(
        self,
        products: list[str] | None = None,
        bbox: BBox = (-180.0, -90.0, 180.0, 90.0),
        resolution: float = 100.0,
        patch_size: int = 256,
        length: int = 1000,
        seed: int | None = None,
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
        self.layers = [self.all_products[p] for p in products]
        self.bbox = bbox
        self.resolution = resolution
        self.patch_size = patch_size
        self.length = length
        self.seed = seed

    def read(self, bbox: BBox, generator: torch.Generator | None = None) -> dict:
        # Placeholder for the Phase 1 STAC sampler: query the USGS ARD
        # catalog, window-read COGs, reproject, and stack.
        image = torch.rand(len(self.layers), self.patch_size, self.patch_size, generator=generator)
        return {
            "image": image,
            "mask": torch.ones_like(image, dtype=torch.bool),
            "layers": list(self.layers),
            "bbox": bbox,
            "crs": CRS,
            "resolution": self.resolution,
        }


class KaguyaTC(InstrumentDataset):
    """Kaguya (SELENE) Terrain Camera: stereo-derived DTM and orthomosaic."""

    probe = "Kaguya (SELENE)"
    instrument = "Terrain Camera"
    all_products = {"dtm": "kaguya_tc_dtm", "ortho": "kaguya_tc_ortho"}


class LROCWAC(InstrumentDataset):
    """LRO Wide Angle Camera: global morphologic mosaic."""

    probe = "Lunar Reconnaissance Orbiter"
    instrument = "LROC Wide Angle Camera"
    all_products = {"mosaic": "lroc_wac"}


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

    def read(self, bbox: BBox, generator: torch.Generator | None = None) -> dict:
        left = self.first.read(bbox, generator)
        right = self.second.read(bbox, generator)
        return {
            "image": torch.cat([left["image"], right["image"]]),
            "mask": torch.cat([left["mask"], right["mask"]]),
            "layers": left["layers"] + right["layers"],
            "bbox": bbox,
            "crs": left["crs"],
            "resolution": self.resolution,
        }
