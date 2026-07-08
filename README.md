# AstroFetch

PyTorch-friendly, ML-ready access to planetary science data, starting with the Moon.
Request a bounding box, receive a coregistered multichannel tensor.

[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12%20|%203.13%20|%203.14-blue.svg)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

AstroFetch is a thin layer over existing archive tooling — the USGS Astrogeology STAC
catalog and Cloud Optimized GeoTIFFs — that turns planetary science archives into
ML-ready tensors. It wraps existing tools rather than mirroring archives: nothing is
rehosted, everything is fetched on demand and cached. It is, roughly, what
[TorchGeo](https://github.com/microsoft/torchgeo) is for Earth observation, pointed at
the Moon (and Mars next).

> **Status: Phase 0 scaffolding.** The API surface is real and stable; the data path
> currently returns synthetic placeholder tensors until the STAC sampler lands in Phase 1.
> See the [roadmap](#roadmap).

## Install

```bash
uv add astrofetch      # or: pip install astrofetch
```

## Usage

One dataset class per instrument; combine instruments with `&` to stack their
channels over the overlapping region:

```python
import astrofetch as af

bbox = (-60.0, 5.0, -55.0, 10.0)  # west, south, east, north (degrees)

moondata = af.KaguyaTC(products=["dtm"], bbox=bbox, resolution=100) & af.LROCWAC(
    bbox=bbox, resolution=100
)

for sample in moondata:
    sample["image"]   # torch.Tensor (C, H, W), one channel per layer
    sample["mask"]    # torch.BoolTensor (C, H, W), validity (nodata gaps)
    sample["layers"]  # ["kaguya_tc_dtm", "lroc_wac"], plus bbox/crs/resolution
```

It plugs directly into a PyTorch training loop — samples are plain dicts, so
the default collation just works:

```python
from torch.utils.data import DataLoader

loader = DataLoader(moondata, batch_size=16)
for batch in loader:
    batch["image"]  # torch.Tensor (16, C, H, W)
```

Not sure what data exists? The `MOON` catalog enumerates probes, instruments,
and products, and points at the dataset classes:

```python
from astrofetch.moon import MOON

for probe in MOON.probes.values():
    for instrument in probe.instruments.values():
        print(probe.name, "/", instrument.name, "->", sorted(instrument.products))

MOON.probes["lro"].instruments["lroc_wac"].dataset  # <class 'astrofetch.moon.datasets.LROCWAC'>
```

## Design principles

- **One dataset class per instrument.** `KaguyaTC`, `LROCWAC`, … (the TorchGeo pattern);
  cross-instrument stacks are explicit compositions via `&` (`IntersectionDataset`).
  Probes and bodies are catalog metadata, never dataset boundaries.
- **Wrap, do not reimplement.** `pystac-client` queries STAC, `rasterio` reads and windows
  COGs. AstroFetch composes these rather than re-deriving archive tooling.
- **STAC + COGs are the primary source.** Windowed HTTP range reads give real physical
  values in their existing map projection. Rendered tile services are a visualization
  convenience only, never the quantitative path.
- **Cache is throwaway.** On-demand data is reproducible and disposable; nothing fetched
  is load-bearing or rehosted.
- **Body-namespaced from day one.** The Moon ships first; adding Mars is a new module,
  not a redesign.

## Roadmap

| Phase | Deliverable | Status |
|:-----:|:------------|:------:|
| 0 | Scaffolding — package, CI, docs, target API | In progress |
| 1 | STAC sampler MVP — bbox to coregistered `(C, H, W)` tensor | Planned |
| 2 | Datasets and transforms — grid-tile dataset, spatial splits, transforms | Planned |
| 3 | Release and community — PyPI, planetarypy affiliation, paper | Planned |

## Development

```bash
git clone https://github.com/TechnicToms/AstroFetch.git
cd AstroFetch
uv sync              # create the venv and install everything
uv run pytest        # run the test suite (no network calls)
uv run ruff check    # lint
```

## Contributing

Contributions are welcome, especially new body modules (Mars first) and new dataset
classes. Open an issue to discuss substantial changes, keep unit tests network-free by
mocking archive calls, and run `ruff` before pushing.

## Contributors

[![Contributors](https://contrib.rocks/image?repo=TechnicToms/AstroFetch)](https://github.com/TechnicToms/AstroFetch/graphs/contributors)

## License

Released under the [Apache 2.0](LICENSE) license.

## Acknowledgements

Built on the planetary open-source community: the
[USGS Astrogeology Science Center](https://www.usgs.gov/centers/astrogeology-science-center),
[planetarypy](https://github.com/planetarypy), [rasterio](https://github.com/rasterio/rasterio),
and [pystac-client](https://github.com/stac-utils/pystac-client), with a nod to
[TorchGeo](https://github.com/microsoft/torchgeo) for showing the way on Earth.
