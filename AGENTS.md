# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Cursor, and others) working in the AstroFetch repository. CLAUDE.md in this repo points here; treat this file as the single source of truth. It covers what the project is, the rules for working in it, and the phased roadmap (see [Roadmap](#roadmap)).

## What this project is

AstroFetch is an open source, PyTorch-friendly library for ML-ready access to planetary science data, starting with the Moon. The core promise: request a bounding box, receive a coregistered multichannel tensor. The core API: one dataset class per instrument (`KaguyaTC`, `LROCWAC`, ...), each yielding TorchGeo-style sample dicts with a coregistered multichannel `"image"` tensor, a validity `"mask"`, and per-channel provenance; instruments compose with `&` (`IntersectionDataset`) to stack channels over their overlapping region. Probes and bodies are discovery-catalog metadata (`MOON`), never dataset boundaries. It is a thin composition layer over existing archive tooling, never a mirror of any archive.

## Architecture at a glance

```
src/astrofetch/
  data/
    endpoints.py      # ALL external URLs live here, nowhere else
    stac.py           # STAC queries (pystac-client) against USGS Astrogeology ARD
    raster.py         # windowed COG reads (rasterio), scale/offset -> physical values
    grid.py           # target grid definition, reprojection, channel stacking
    cache.py          # throwaway local cache, keyed by (collection, item, window, res)
    tiles.py          # secondary rendered mode: USGS WMS, Moon Trek WMTS
  moon/
    layers.py         # layer registry (name -> STAC collection + read config) + Body/Probe/Instrument catalog (MOON)
    datasets.py       # instrument dataset classes (InstrumentDataset, KaguyaTC, KaguyaTCImagery) + IntersectionDataset
tests/
  unit/               # network fully mocked, runs in CI
  live/               # hits real endpoints, manual trigger only
```

## Non-negotiable design rules

1. **Default to wrapping archive tooling; reimplement only with a measured reason.** pystac-client queries STAC, rasterio reads and windows COGs. Reach for these first — reinventing them is usually wasted effort and a maintenance burden. Two carve-outs: (a) **never** hand-roll domain-specific correctness — map-projection math, COG windowing, scale/offset conversion — the bugs there are subtle and scientific, so always defer to the established library; (b) generic plumbing (discovery helpers, small utilities) *may* be replaced when a dependency is provably a bad trade — too slow on the hot path, a heavy transitive dependency for a sliver of use, etc. Justify any such reimplementation in the PR description with the concrete reason.
2. **All external endpoint URLs go in `data/endpoints.py`.** No URL literals anywhere else in `src/`. Endpoints move (QuickMap changed domains); one module keeps that survivable.
3. **Quantitative vs rendered is a hard boundary.** The STAC/COG path returns physical values and is the only path for quantitative or ML use. The WMS/WMTS tile path returns rendered 8-bit imagery and must be labeled as such in APIs and docs. Never mix them silently.
4. **Cache is disposable.** Nothing in the cache layer may be load-bearing for correctness or reproducibility. Everything fetched on demand must be re-fetchable from the archive and safe to delete; never commit fetched data to the repo.
5. **Be polite to archive servers.** Default concurrency is low, retries use exponential backoff, and any code path that could issue many requests must go through the rate-limited session in `data/stac.py` / `data/tiles.py`. Never write a loop that hammers NASA or USGS servers.
6. **Body-namespaced layout.** Moon-specific code lives under `moon/`. Body-agnostic code (grid math, COG reads, caching) lives under `data/`. Adding Mars must be a new sibling module, not edits scattered through existing files.
7. **Samples are dicts with a fixed contract.** `"image"` is (C, H, W) float32 (physical values, channel i = `layers[i]`), `"mask"` is (C, H, W) bool validity, plus `"layers"`, `"bbox"`, `"crs"`, and `"resolution"` provenance keys. Datasets that deviate must document it. Samples must collate under the default `DataLoader` collation.

## Dev environment and commands

- Python 3.10+, src layout, uv-managed: `uv sync` installs the package plus the `dev` group (or `pip install -e .` for the package alone).
- Lint: `uv run ruff check` and `uv run ruff format`
- Types: `uv run ty check` (config in pyproject.toml; keep new code typed)
- Tests: `uv run pytest` (unit only; the `live` marker is deselected by default). Live endpoint tests: `uv run pytest tests/live -m live` (never run these in CI or in loops; they hit real government servers).
- Docs: `uv run --group docs mkdocs serve`

## Testing rules for agents

- Unit tests must not touch the network. Mock STAC responses with recorded JSON fixtures in `tests/fixtures/` and mock raster reads with small in-memory GeoTIFFs created via rasterio.
- If you add an endpoint or change a query, add or update a fixture; do not point unit tests at live services.
- When you genuinely need to verify a live endpoint, run a single targeted test from `tests/live`, once.
- Every new public API function needs a docstring with a runnable example and at least one unit test.

## Conventions

- Conventional commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Public API is what is exported in `astrofetch/__init__.py` and `astrofetch/moon/__init__.py`; everything else is private and may change freely.
- Errors: raise `astrofetch.errors.EndpointError` for remote failures with the endpoint name and a hint about `data/endpoints.py`; raise `ValueError` for bad user input with the offending value in the message.
- No print statements in library code; use the `astrofetch` logger.
- Docstrings in NumPy style. Keep examples copy-pasteable.

## General engineering rules

Widely-adopted defaults that keep the codebase consistent. When in doubt, match the surrounding code.

- **Formatting is automated, never manual.** `ruff format` is the single source of truth for layout, quotes, and line length (config in pyproject.toml). Do not hand-align or fight the formatter; run it before every commit.
- **Imports are sorted and grouped** (standard library, third-party, first-party), managed by ruff's isort rules. No unused imports, no wildcard (`from x import *`) imports.
- **Naming follows PEP 8.** `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants, a leading underscore for private names. Names describe intent, not type (`bbox`, not `tuple4`).
- **Type-hint every public function** and prefer typing internal ones too; keep `mypy src` clean.
- **Small, single-responsibility functions.** Prefer early returns over deep nesting; if a function needs a paragraph to explain, split it.
- **No magic numbers or strings.** Name them as module-level constants (see `CRS`, `BBox` in `moon/datasets.py`).
- **Fail loud, handle errors explicitly.** No bare `except:`; catch the narrowest exception and re-raise with context. Never silently swallow errors or return `None` on failure without documenting it.
- **DRY, but avoid premature abstraction.** Factor out real duplication; do not build generic machinery for a single caller.
- **Delete dead code, don't comment it out.** Git remembers history; commented-out blocks and unused helpers are noise.
- **Comments explain *why*, not *what*.** The code already says what it does; reserve comments for rationale, gotchas, and references.
- **Tests are first-class.** Change behavior, change or add a test in the same PR; never leave a failing or skipped test without an explanation.

## Domain notes agents should know

- Coordinates are IAU 2015 Moon (ocentric, longitude 0 to 360 or -180 to 180 must be normalized at the API boundary; internal convention is -180 to 180).
- Equatorial data uses equirectangular projection; polar data uses polar stereographic. `data/grid.py` owns this decision; never assume equirectangular blindly near the poles.
- COGs from the USGS ARD catalog often store 16-bit DN with scale/offset to physical units (for example Kaguya TC radiance). Always apply scale/offset in `raster.py`; downstream code assumes physical values.
- Nodata regions are common (orbital swaths do not cover everything). Every sample carries a boolean validity tensor under its `"mask"` key; do not silently zero-fill.

## What NOT to do

- Do not add dependencies casually. Core deps are: torch, rasterio, pystac-client, numpy. Anything else needs a justification in the PR description.
- Do not commit data files, fetched tiles, or notebooks with executed output containing large images.
- Do not target QuickMap's internal tile URLs; they are not a public API. Use USGS WMS or Moon Trek WMTS via `data/endpoints.py`.
- Do not "fix" scientific constants or projection parameters without a source; cite the reference in the commit message.
- Do not weaken the mocked-network rule in unit tests to make something pass.

## Roadmap

Check the current phase before proposing work; for example, do not build Phase 2 datasets and transforms while Phase 1 (STAC sampler MVP) is incomplete. Everything is a thin layer above existing archive tooling, never a mirror of any archive.

**Current phase: Phase 1 (STAC sampler). Phase 0 scaffolding is complete. `InstrumentDataset.read` now fetches the real COGs covering a window from the USGS ARD catalog, reprojects them onto a common geographic grid, applies scale/offset, mosaics overlapping items, and caches the result — no more synthetic tensors.**

### Phase 0: Scaffolding (weekend 1)

- Repo setup: pyproject.toml (hatchling or setuptools), src layout, MIT or Apache-2.0 license, CITATION.cff.
- CI: GitHub Actions running ruff, ruff format, `ty` type checking, and pytest on 3.10 through 3.14.
- Testing policy established early: unit tests mock all network calls; a separate, manually triggered "live" test suite hits real endpoints.
- Placeholder docs (mkdocs-material) and a README with the one-line pitch and the target API sketch.

Exit criteria: `pip install -e .` works, CI is green on an empty package.

### Phase 1: STAC sampler MVP (the core, 2 to 4 weekends)

The single most important deliverable. Everything else builds on it.

- `astrofetch.data.stac`: query the USGS ARD catalog root with pystac-client, filter by collection and bbox.
- `astrofetch.data.raster`: windowed COG reads via rasterio, honoring scale/offset to return physical values, resampling to a requested resolution.
- `astrofetch.data.grid`: define a common target grid (equirectangular, IAU 2015 Moon), reproject and stack layers into a (C, H, W) float tensor with a per-channel metadata record.
- The sampler plugs into `InstrumentDataset.read(bbox)`, replacing the Phase 0 synthetic placeholder. Shipped public API:

```python
import astrofetch as af
bbox = (-26.3, -50.6, -25.5, -49.7)  # a Kaguya TC USGS DTM footprint
moondata = af.KaguyaTC(products=["dtm"], bbox=bbox, resolution=100) & af.KaguyaTCImagery(
    bbox=bbox, resolution=100
)
sample = next(iter(moondata))
sample["image"]   # torch.Tensor (C, H, W), physical values
sample["mask"]    # torch.BoolTensor (C, H, W), validity (nodata gaps)
sample["layers"]  # ["kaguya_tc_dtm", "kaguya_tc_image"], plus bbox/crs/resolution
```

- Local disk cache keyed by (collection, item, window, resolution), transparent and clearable.

Exit criteria: `KaguyaTC(products=["dtm", "ortho"], bbox=...)` fetches a real, coregistered two-layer patch from the USGS ARD catalog on a clean machine (covered by `tests/live`). A plotting quickstart notebook is a nice-to-have follow-up.

### Phase 2: Datasets and transforms (2 to 3 weekends)

- `astrofetch.moon.datasets`: random-bbox sampling already ships inside the instrument datasets (Phase 0); add `GridTileDataset` (deterministic tiling of an ROI) over the same `read(bbox)` interface, plus region-list sampling for the random path.
- Samplers that respect spatial autocorrelation for train/val/test splits (block splitting, not random pixels).
- Transforms: per-channel normalization stats, nodata masking, polar/equatorial projection handling made explicit.
- Secondary access mode behind the same interface: WMS/WMTS rendered mode, clearly labeled non-quantitative.
- LRO WAC global mosaic: the USGS ARD STAC catalog has no LRO WAC collection, so add it here from a non-STAC source (a public COG mosaic or WMS/WMTS), behind the same instrument-dataset interface as a new `LROCWAC` class.

Exit criteria: `DataLoader` trains a toy model on random lunar patches without custom user code.

### Phase 3: Release and community (ongoing)

- v0.1.0 to PyPI, announcement on OpenPlanetary forum and relevant mailing lists.
- Apply for planetarypy affiliated package status.
- Short JOSS paper or arXiv preprint describing the library.
- Issue templates and a CONTRIBUTING.md that explicitly invites new body modules (Mars first) and new dataset classes.

### Deliberate non-goals for v0.x

- No mirroring or rehosting of raw PDS archives.
- No PDS granule / full-fidelity product access in v0.x; the STAC/COG path is the only quantitative source.
- No pretrained models or frozen benchmarks; AstroFetch delivers ML-ready data, you bring the model.
- No GUI or web viewer; QuickMap and Trek exist.
- No Earth support; TorchGeo owns that space.

### Risks and mitigations

- Endpoint drift (services move, as QuickMap's domain change showed): keep all endpoint URLs in one config module, cover them with the live test suite, and document last-verified dates.
- M3 data quality issues: defer the M3 dataset class until after v0.1 unless a user strictly needs it; budget preprocessing time if so.
- Server load courtesy: default to conservative request concurrency, exponential backoff, and a bulk prefetch helper so training never hammers archive servers with random access.
- Solo-maintainer bus factor: keep scope small, tests honest, and architecture boring enough that contributors can navigate it without you.
