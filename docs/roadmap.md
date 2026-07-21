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
Phase 1 exit criterion. Phase 2 has started delivering new data sources
beyond STAC: the NASA PDS Orbital Data Explorer (`ODEInstrumentDataset`) adds
LROC NAC stereo DTM sites, and fixed-URL mosaics (`MosaicDataset`) add the
LRO WAC global mosaic and the LOLA and SLDEM2015 global DEMs — all behind
the same sample-dict contract and `&` composition as the STAC-backed
datasets. An experimental, separately-contracted raw-granule path
(`astrofetch.moon.granules`) also now exists for camera-geometry NAC/WAC
strips and M3 radiance cubes; see [Raw granules](reference/granules.md).
Still open for Phase 2: `GridTileDataset`, spatial-autocorrelation-aware
train/val/test splitting, transforms, and the WMS/WMTS rendered mode.
