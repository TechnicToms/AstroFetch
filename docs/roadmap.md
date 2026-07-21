# Roadmap

AstroFetch is built in phases. The full phased plan, design rules, and
non-goals live in [`AGENTS.md`](https://github.com/TechnicToms/AstroFetch/blob/main/AGENTS.md);
this page is the short version.

| Phase | Deliverable | Status |
|:-----:|:------------|:------:|
| 0 | Scaffolding — package, CI, docs, target API | Done |
| 1 | STAC sampler MVP — bbox to coregistered `(C, H, W)` tensor | Done |
| 2 | Datasets and transforms — new data sources, grid-tile dataset, spatial splits, transforms | In progress |
| 3 | Release and community — PyPI, planetarypy affiliation, paper | Planned |

**Current phase: Phase 2.** `InstrumentDataset.read(bbox)` fetches real COGs
from the USGS ARD catalog, reprojects them onto a common geographic grid,
applies scale/offset, mosaics overlapping items, and caches the result — the
Phase 1 exit criterion. Phase 2 has delivered new data sources beyond STAC:
the NASA PDS Orbital Data Explorer (`ODEInstrumentDataset`) now backs a wide
roster of instruments — LROC NAC stereo DTM sites and region-of-interest
mosaics, Mini-RF S-band radar, Diviner rock abundance and regolith
temperature, WAC GLD100, TiO2, tiled global morphology, and 7-color
reflectance, ShadowCam polar mosaics and DTMs, and the Clementine UVVIS and
NIR basemaps — and fixed-URL mosaics (`MosaicDataset`) add the LRO WAC
global mosaic and the LOLA and SLDEM2015 global DEMs, all behind the same
sample-dict contract and `&` composition as the STAC-backed datasets. See
[Instrument datasets](reference/datasets.md) for the full list. An
experimental, separately-contracted raw-granule path
(`astrofetch.moon.granules`) also now exists for camera-geometry NAC/WAC
strips and M3 radiance cubes; see [Raw granules](reference/granules.md).
Still open for Phase 2: `GridTileDataset`, spatial-autocorrelation-aware
train/val/test splitting, transforms, and the WMS/WMTS rendered mode.
