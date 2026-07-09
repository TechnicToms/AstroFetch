# Instrument datasets

One dataset class per instrument, each a map-style `torch` dataset that samples
coregistered patches. Combine instruments with `&` to stack their channels over
the overlapping region.

::: astrofetch.moon.datasets.KaguyaTC

::: astrofetch.moon.datasets.KaguyaTCImagery

::: astrofetch.moon.datasets.InstrumentDataset

::: astrofetch.moon.datasets.IntersectionDataset

::: astrofetch.moon.datasets.Product
