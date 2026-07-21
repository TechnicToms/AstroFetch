# Instrument datasets

One dataset class per instrument, each a map-style `torch` dataset that samples
coregistered patches. Combine instruments with `&` to stack their channels over
the overlapping region. Every instrument dataset shares the same sample-dict
contract regardless of where its products are read from — see
[Data layer](data.md) for the three sources (STAC, PDS ODE, fixed mosaics).

For raw, non-map-projected data (camera-geometry strips), see the separate,
experimental [Raw granules](granules.md) page — a deliberately different
contract, not part of the windowed-dataset family below.

## Kaguya (SELENE)

::: astrofetch.moon.datasets.KaguyaTC

::: astrofetch.moon.datasets.KaguyaTCImagery

## Lunar Reconnaissance Orbiter

::: astrofetch.moon.datasets.LROCNACDTM

::: astrofetch.moon.datasets.LROCWACMosaic

::: astrofetch.moon.datasets.LOLA

::: astrofetch.moon.datasets.SLDEM2015

::: astrofetch.moon.datasets.MiniRF

::: astrofetch.moon.datasets.DivinerGDR

::: astrofetch.moon.datasets.WACGLD100

::: astrofetch.moon.datasets.WACTiO2

::: astrofetch.moon.datasets.LROCWACGlobal

::: astrofetch.moon.datasets.LROCWACColor

::: astrofetch.moon.datasets.LROCNACROI

## Korea Pathfinder Lunar Orbiter

::: astrofetch.moon.datasets.ShadowCam

## Clementine

::: astrofetch.moon.datasets.ClementineUVVIS

::: astrofetch.moon.datasets.ClementineNIR

## Base classes

::: astrofetch.moon.datasets.InstrumentDataset

::: astrofetch.moon.datasets.ODEInstrumentDataset

::: astrofetch.moon.datasets.MosaicDataset

::: astrofetch.moon.datasets.IntersectionDataset

## Product specs

::: astrofetch.moon.datasets.Product

::: astrofetch.moon.datasets.ODEAsset

::: astrofetch.moon.datasets.MosaicAsset
