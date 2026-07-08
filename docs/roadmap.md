# Roadmap

AstroFetch is built in phases. The full phased plan, design rules, and
non-goals live in [`AGENTS.md`](https://github.com/TechnicToms/AstroFetch/blob/main/AGENTS.md);
this page is the short version.

| Phase | Deliverable | Status |
|:-----:|:------------|:------:|
| 0 | Scaffolding — package, CI, docs, target API | In progress |
| 1 | STAC sampler MVP — bbox to coregistered `(C, H, W)` tensor | Planned |
| 2 | Datasets and transforms — grid-tile dataset, spatial splits, transforms | Planned |
| 3 | Release and community — PyPI, planetarypy affiliation, paper | Planned |

**Current phase: Phase 0.** The per-instrument API surface is real and stable;
the data path returns synthetic placeholder tensors until the Phase 1 STAC
sampler replaces `InstrumentDataset.read(bbox)`.
