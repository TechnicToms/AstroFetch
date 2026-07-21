# Catalog & layer registry

`MOON` is the discovery catalog — enumerate probes, instruments, and products,
and reach the dataset classes. `LAYERS` maps each layer id to its provenance
and backing source (`LayerSpec.source` is `"stac"`, `"ode"`, or `"mosaic"`).
A probe may also carry `granules`: experimental, non-map-projected dataset
classes (see [Raw granules](granules.md)) that sit outside the layer/product
contract entirely and are not part of `LAYERS`.

::: astrofetch.moon.layers.MOON
    options:
      show_root_heading: true

::: astrofetch.moon.layers.LAYERS
    options:
      show_root_heading: true

::: astrofetch.moon.layers.LayerSpec

::: astrofetch.moon.layers.Body

::: astrofetch.moon.layers.Probe

::: astrofetch.moon.layers.Instrument
