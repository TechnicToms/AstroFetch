"""Torch dataset classes over lunar data layers.

Phase 0 scaffolding: the API surface is real, the data path is a placeholder.
``LunarMoon`` will be backed by the STAC sampler (Phase 1); until then it
yields correctly-shaped random tensors so the training loop contract —
``for batch in LunarMoon(...)`` — can be exercised end to end.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
from torch.utils.data import IterableDataset

BBox = tuple[float, float, float, float]
"""(west, south, east, north) in degrees, IAU 2015 Moon."""


@dataclass(frozen=True)
class Patch:
    """A coregistered multichannel sample and its provenance."""

    tensor: torch.Tensor
    """(C, H, W) float tensor of physical values."""

    meta: dict = field(default_factory=dict)
    """Projection, units, and provenance per channel."""


class LunarMoon(IterableDataset):
    """Iterable dataset of coregistered lunar patches.

    Samples random bounding boxes within ``bbox`` and returns stacked layers
    as (C, H, W) tensors at the requested resolution.

    Args:
        layers: layer identifiers, e.g. ``["kaguya_tc_dtm", "lroc_wac"]``.
        bbox: region to sample from as (west, south, east, north) degrees.
        resolution: target resolution in meters per pixel.
        patch_size: output height and width in pixels.
        length: number of patches per epoch.
        seed: RNG seed for reproducible sampling.

    Example:
        >>> moondata = LunarMoon(layers=["lroc_wac"], bbox=(-60.0, 5.0, -55.0, 10.0))
        >>> for batch in moondata:
        ...     batch.tensor  # torch.Tensor (C, H, W)
    """

    def __init__(
        self,
        layers: list[str],
        bbox: BBox = (-180.0, -90.0, 180.0, 90.0),
        resolution: float = 100.0,
        patch_size: int = 256,
        length: int = 1000,
        seed: int | None = None,
    ) -> None:
        if not layers:
            raise ValueError("at least one layer is required")
        west, south, east, north = bbox
        if not (west < east and south < north):
            raise ValueError(f"invalid bbox (west, south, east, north): {bbox}")
        self.layers = list(layers)
        self.bbox = bbox
        self.resolution = resolution
        self.patch_size = patch_size
        self.length = length
        self.seed = seed

    def __iter__(self) -> Iterator[Patch]:
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(self.seed)
        for _ in range(self.length):
            sample_bbox = self._sample_bbox(generator)
            yield self._fetch(sample_bbox, generator)

    def __len__(self) -> int:
        return self.length

    def _sample_bbox(self, generator: torch.Generator) -> BBox:
        west, south, east, north = self.bbox
        u, v = torch.rand(2, generator=generator).tolist()
        # Placeholder: a degenerate point bbox; Phase 1 sizes the window from
        # patch_size * resolution on the target grid.
        lon = west + u * (east - west)
        lat = south + v * (north - south)
        return (lon, lat, lon, lat)

    def _fetch(self, bbox: BBox, generator: torch.Generator) -> Patch:
        # Placeholder for the Phase 1 STAC sampler: query the USGS ARD
        # catalog, window-read COGs, reproject, and stack.
        tensor = torch.rand(
            len(self.layers), self.patch_size, self.patch_size, generator=generator
        )
        meta = {
            "layers": self.layers,
            "bbox": bbox,
            "resolution": self.resolution,
            "crs": "IAU_2015:30100",
            "provenance": "synthetic-placeholder",
        }
        return Patch(tensor=tensor, meta=meta)
