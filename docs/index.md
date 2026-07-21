# AstroFetch

PyTorch-friendly, ML-ready access to planetary science data, starting with the
Moon. Request a bounding box, receive a coregistered multichannel tensor.

!!! info "Phase 1 (STAC sampler)"
    The data path is real: a request fetches the Cloud Optimized GeoTIFFs
    covering the window from the USGS ARD catalog, reprojects them onto a
    common grid, and returns physical values. See the [roadmap](roadmap.md).

## Install

```bash
uv add astrofetch      # or: pip install astrofetch
```

## Quickstart

One dataset class per instrument; combine instruments with `&` to stack their
channels over the overlapping region:

```python
import astrofetch as af

bbox = (-26.3, -50.6, -25.5, -49.7)  # west, south, east, north (degrees)

moondata = af.KaguyaTC(products=["dtm"], bbox=bbox, resolution=100) & af.KaguyaTCImagery(
    bbox=bbox, resolution=100
)

for sample in moondata:
    sample["image"]   # torch.Tensor (C, H, W), one channel per layer
    sample["mask"]    # torch.BoolTensor (C, H, W), validity (nodata gaps)
    sample["layers"]  # ["kaguya_tc_dtm", "kaguya_tc_image"], plus bbox/crs/resolution
```

Samples are plain dicts, so a `DataLoader` collates them with no custom code:

```python
from torch.utils.data import DataLoader

loader = DataLoader(moondata, batch_size=16)
for batch in loader:
    batch["image"]  # torch.Tensor (16, C, H, W)
```

## Beyond the STAC catalog

Some instruments (LROC, LOLA, ...) aren't in the USGS ARD STAC catalog at
all; those datasets search the NASA PDS Orbital Data Explorer instead, or
read a single fixed mosaic URL, behind the exact same interface:

```python
import astrofetch as af

# LROC NAC stereo DTM sites are a few hundred named sites, not global
# coverage, so sampled windows are drawn from inside a real site by default.
nac = af.LROCNACDTM(products=["dtm", "ortho"], bbox=(3.0, 25.0, 4.5, 26.5))
sample = nac[0]

# A global 100 m WAC mosaic and a global LOLA DEM, channel-stacked with `&`.
terrain = af.LROCWACMosaic(resolution=100) & af.LOLA(resolution=100)
```

## Discovering what data exists

The `MOON` catalog enumerates probes, instruments, and products, and points at
the dataset classes:

```python
from astrofetch.moon import MOON

for probe in MOON.probes.values():
    for instrument in probe.instruments.values():
        print(probe.name, "/", instrument.name, "->", sorted(instrument.products))
```

For the full design rationale and contributor guidance, see
[`AGENTS.md`](https://github.com/TechnicToms/AstroFetch/blob/main/AGENTS.md).
