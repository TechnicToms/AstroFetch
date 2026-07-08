# AstroFetch

PyTorch-friendly, ML-ready access to planetary science data, starting with the
Moon. Request a bounding box, receive a coregistered multichannel tensor.

!!! warning "Phase 0 scaffolding"
    The API surface is real and stable; the data path currently returns
    synthetic placeholder tensors until the STAC sampler lands in Phase 1.
    See the [roadmap](roadmap.md).

## Install

```bash
uv add astrofetch      # or: pip install astrofetch
```

## Quickstart

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

Samples are plain dicts, so a `DataLoader` collates them with no custom code:

```python
from torch.utils.data import DataLoader

loader = DataLoader(moondata, batch_size=16)
for batch in loader:
    batch["image"]  # torch.Tensor (16, C, H, W)
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
