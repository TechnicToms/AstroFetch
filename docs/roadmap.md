# Roadmap

AstroFetch is built in phases. The full phased plan, design rules, and
non-goals live in [`AGENTS.md`](https://github.com/TechnicToms/AstroFetch/blob/main/AGENTS.md);
this page is the short version.

| Phase | Deliverable | Status |
|:-----:|:------------|:------:|
| 0 | Scaffolding — package, CI, docs, target API | Done |
| 1 | STAC sampler MVP — bbox to coregistered `(C, H, W)` tensor | In progress |
| 2 | Datasets and transforms — grid-tile dataset, spatial splits, transforms, LRO WAC | Planned |
| 3 | Release and community — PyPI, planetarypy affiliation, paper | Planned |

**Current phase: Phase 1.** `InstrumentDataset.read(bbox)` now fetches real COGs
from the USGS ARD catalog, reprojects them onto a common geographic grid,
applies scale/offset, mosaics overlapping items, and caches the result.
