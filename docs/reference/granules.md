# Raw granules (experimental)

!!! warning "Experimental — a different contract than every other dataset"
    Unlike the [instrument datasets](datasets.md), raw granule datasets are
    **not** reprojected onto a common grid: each item is one raw PDS product
    (an NAC/WAC calibrated strip, an M3 radiance cube, ...), read in its own
    native camera/instrument geometry. No reprojection, resampling, ISIS, or
    SPICE processing is applied.

    Consequences:

    - **No bbox windowing.** `__getitem__` returns a whole granule (or a row
      range, via `rows=`), not a patch cropped to a requested extent.
    - **Ragged shapes across items.** The default `DataLoader` collation will
      not work; use `batch_size=None` or a custom `collate_fn`.
    - **No `&` composition.** There is no shared grid to stack channels onto.

    `len()` is the number of PDS ODE products matching a bbox, fetched once
    and eagerly in `__init__`, so `len()` never needs a network call.

## Reading large strips

NAC/WAC strips can be gigapixel. Reading a whole granule with no `rows=`
raises once its pixel count exceeds `max_pixels` (about 512 MiB as float32
by default), naming the granule's size and suggesting a row range:

```python
import astrofetch as af

# Read only the first 512 rows of every matching strip instead of the whole
# multi-gigapixel granule.
dataset = af.LROCNACRaw(bbox=(-26.3, -50.6, -25.5, -49.7), rows=slice(0, 512))
sample = dataset[0]
sample["image"]  # (bands, 512, W) float32, physical values
sample["mask"]   # same-shaped bool validity
```

## Datasets

::: astrofetch.moon.granules.LROCNACRaw

::: astrofetch.moon.granules.LROCWACRaw

::: astrofetch.moon.granules.M3

## Base class

::: astrofetch.moon.granules.GranuleDataset
