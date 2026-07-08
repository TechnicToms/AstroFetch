# AstroFetch

PyTorch-friendly, ML-ready access to planetary science data, starting with the Moon.
Request a bounding box, receive a coregistered multichannel tensor. AstroFetch is a
thin layer over existing archive tooling (USGS STAC, Cloud Optimized GeoTIFFs),
never a mirror of any archive.

## Quickstart

```python
import astrofetch as af

moondata = af.LunarMoon(
    layers=["kaguya_tc_dtm", "lroc_wac"],
    bbox=(-60.0, 5.0, -55.0, 10.0),  # west, south, east, north (degrees)
    resolution=100,                   # meters per pixel
)

for batch in moondata:
    batch.tensor  # torch.Tensor (C, H, W)
    batch.meta    # projection, units, provenance per channel
```

> **Status:** Phase 0 scaffolding. The API surface is real; the data path currently
> yields synthetic placeholder tensors until the STAC sampler lands (Phase 1).

## Development

```bash
uv sync
uv run pytest
uv run ruff check
```
