# Data layer

Body-agnostic building blocks shared by every instrument dataset: STAC and
PDS ODE search, windowed raster reads, the target grid, and the disposable
cache. Three sources back instrument datasets — see
[Instrument datasets](datasets.md) for which each instrument uses:

- **STAC** (`astrofetch.data.stac`): the USGS Astrogeology Analysis Ready
  Data catalog, searched via `pystac-client`.
- **PDS ODE** (`astrofetch.data.ode`): the NASA PDS Orbital Data Explorer
  REST API, for instruments (LROC, LOLA, M3, ...) the STAC catalog does not
  carry.
- **Fixed mosaics**: a handful of instruments are published as one global
  (or near-global) file rather than many searchable items; these are read
  directly from a well-known URL in `astrofetch.data.endpoints`, no search.

All three reproject through the same windowed-read path and share the same
sample-dict contract.

## Target grid

::: astrofetch.data.grid

## Raster reads

::: astrofetch.data.raster

## STAC search

::: astrofetch.data.stac

## PDS ODE search

::: astrofetch.data.ode

## Cache

::: astrofetch.data.cache

## Endpoints

::: astrofetch.data.endpoints

## Errors

::: astrofetch.errors
