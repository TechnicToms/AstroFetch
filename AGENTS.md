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
    ode.py            # granule mode: ODE REST + pdr (full-fidelity, later phase)
  moon/
    layers.py         # layer registry (name -> STAC collection + read config) + Body/Probe/Instrument catalog (MOON)
    datasets.py       # instrument dataset classes (InstrumentDataset, KaguyaTC, LROCWAC) + IntersectionDataset
    checksums.py      # pinned item IDs and hashes for frozen benchmarks
  models/
    mae.py            # multimodal MAE encoder + heads
    weights.py        # pretrained checkpoint registry (Hugging Face Hub)
  benchmarks/
    moon/             # frozen tasks: splits, metrics, eval harness
tests/
  unit/               # network fully mocked, runs in CI
  live/               # hits real endpoints, manual trigger only
```

## Non-negotiable design rules

1. **Never reimplement archive tooling.** pdr reads PDS products, planetarypy discovers and fetches them, pystac-client queries STAC, rasterio reads rasters. If you find yourself parsing a PDS label or writing tile math by hand, stop and use the existing library.
2. **All external endpoint URLs go in `data/endpoints.py`.** No URL literals anywhere else in `src/`. Endpoints move (QuickMap changed domains); one module keeps that survivable.
3. **Quantitative vs rendered is a hard boundary.** The STAC/COG path returns physical values and is the only path benchmarks may use. The WMS/WMTS tile path returns rendered 8-bit imagery and must be labeled as such in APIs and docs. Never mix them silently.
4. **Cache is disposable, benchmarks are frozen.** Nothing in the cache layer may be load-bearing for reproducibility. Frozen benchmark data is pinned by STAC item ID and checksum in `moon/checksums.py` and hosted externally (Hugging Face / Zenodo), never committed to the repo.
5. **Be polite to archive servers.** Default concurrency is low, retries use exponential backoff, and any code path that could issue many requests must go through the rate-limited session in `data/stac.py` / `data/tiles.py`. Never write a loop that hammers NASA or USGS servers.
6. **Body-namespaced layout.** Moon-specific code lives under `moon/`. Body-agnostic code (grid math, COG reads, caching) lives under `data/`. Adding Mars must be a new sibling module, not edits scattered through existing files.
7. **Samples are dicts with a fixed contract.** `"image"` is (C, H, W) float32 (physical values, channel i = `layers[i]`), `"mask"` is (C, H, W) bool validity, plus `"layers"`, `"bbox"`, `"crs"`, and `"resolution"` provenance keys. Datasets that deviate must document it. Samples must collate under the default `DataLoader` collation.

## Dev environment and commands

- Python 3.10+, src layout, editable install: `pip install -e ".[dev]"`
- Lint: `ruff check src tests` and `ruff format`
- Types: `mypy src` (config in pyproject.toml; keep new code typed)
- Tests: `pytest tests/unit` (CI default). Live endpoint tests: `pytest tests/live -m live` (never run these in CI or in loops; they hit real government servers).
- Docs: `mkdocs serve`

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

## Domain notes agents should know

- Coordinates are IAU 2015 Moon (ocentric, longitude 0 to 360 or -180 to 180 must be normalized at the API boundary; internal convention is -180 to 180).
- Equatorial data uses equirectangular projection; polar data uses polar stereographic. `data/grid.py` owns this decision; never assume equirectangular blindly near the poles.
- COGs from the USGS ARD catalog often store 16-bit DN with scale/offset to physical units (for example Kaguya TC radiance). Always apply scale/offset in `raster.py`; downstream code assumes physical values.
- Nodata regions are common (orbital swaths do not cover everything). Every sample carries a boolean validity tensor under its `"mask"` key; do not silently zero-fill.

## What NOT to do

- Do not add dependencies casually. Core deps are: torch, rasterio, pystac-client, numpy, pdr, planetarypy. Anything else needs a justification in the PR description.
- Do not commit data files, fetched tiles, model weights, or notebooks with executed output containing large images.
- Do not target QuickMap's internal tile URLs; they are not a public API. Use USGS WMS or Moon Trek WMTS via `data/endpoints.py`.
- Do not "fix" scientific constants, projection parameters, or checksums without a source; cite the reference in the commit message.
- Do not weaken the mocked-network rule in unit tests to make something pass.

## Roadmap

Check the current phase before proposing work; for example, do not build benchmark infrastructure while Phase 1 (STAC sampler MVP) is incomplete. Everything is a thin layer above existing archive tooling, never a mirror of any archive.

**Current phase: Phase 0 (scaffolding) — the per-instrument API surface is real and stable; the data path returns synthetic placeholder tensors until the Phase 1 STAC sampler lands.**

### Phase 0: Scaffolding (weekend 1)

- Repo setup: pyproject.toml (hatchling or setuptools), src layout, MIT or Apache-2.0 license, CITATION.cff.
- CI: GitHub Actions running ruff, mypy (lenient at first), pytest on 3.10 through 3.12.
- Testing policy established early: unit tests mock all network calls; a separate, manually triggered "live" test suite hits real endpoints.
- Placeholder docs (mkdocs-material) and a README with the one-line pitch and the target API sketch.

Exit criteria: `pip install -e .` works, CI is green on an empty package.

### Phase 1: STAC sampler MVP (the core, 2 to 4 weekends)

The single most important deliverable. Everything else builds on it.

- `astrofetch.data.stac`: query the USGS ARD catalog root with pystac-client, filter by collection and bbox.
- `astrofetch.data.raster`: windowed COG reads via rasterio, honoring scale/offset to return physical values, resampling to a requested resolution.
- `astrofetch.data.grid`: define a common target grid (equirectangular, IAU 2015 Moon), reproject and stack layers into a (C, H, W) float tensor with a per-channel metadata record.
- The sampler plugs into `InstrumentDataset.read(bbox)`, replacing the Phase 0 synthetic placeholder. Public API target (already the shipped Phase 0 surface):

```python
import astrofetch as af
bbox = (-60.0, 5.0, -55.0, 10.0)
moondata = af.KaguyaTC(products=["dtm"], bbox=bbox, resolution=100) & af.LROCWAC(
    bbox=bbox, resolution=100
)
sample = next(iter(moondata))
sample["image"]   # torch.Tensor (C, H, W), physical values
sample["mask"]    # torch.BoolTensor (C, H, W), validity (nodata gaps)
sample["layers"]  # ["kaguya_tc_dtm", "lroc_wac"], plus bbox/crs/resolution
```

- Local disk cache keyed by (collection, item, window, resolution), transparent and clearable.

Exit criteria: the quickstart notebook fetches a Reiner Gamma patch with two layers and plots it, end to end, on a clean machine.

### Phase 2: Datasets and transforms (2 to 3 weekends)

- `astrofetch.moon.datasets`: random-bbox sampling already ships inside the instrument datasets (Phase 0); add `GridTileDataset` (deterministic tiling of an ROI) over the same `read(bbox)` interface, plus region-list sampling for the random path.
- Samplers that respect spatial autocorrelation for train/val/test splits (block splitting, not random pixels).
- Transforms: per-channel normalization stats, nodata masking, polar/equatorial projection handling made explicit.
- Secondary access modes behind the same interface: WMS/WMTS rendered mode (clearly labeled non-quantitative) and an ODE REST granule mode stub for later full-fidelity work.

Exit criteria: `DataLoader` trains a toy model on random lunar patches without custom user code.

### Phase 3: Pretrained weights (1 to 2 weekends, leverages existing thesis assets)

- `astrofetch.models`: multimodal MAE encoder definition matching the thesis architecture, plus fine-tuning heads.
- `astrofetch.models.weights`: registry pattern, `af.models.load_pretrained("lunar-mae-base")`, weights hosted on Hugging Face Hub.
- Model card documenting training data (with STAC provenance), input channels, and known limitations.

Exit criteria: a user can load the pretrained encoder and extract features from a Phase 1 patch in under ten lines.

### Phase 4: First frozen benchmark (2 to 3 weekends)

- Pick one task with clear scientific value, for example swirl segmentation or crater detection on a defined region set.
- Freeze: exact STAC item IDs, windows, checksums, and numeric arrays (not rendered tiles). Host the frozen set on Hugging Face, mint a Zenodo DOI at release.
- Ship the evaluation harness: fixed metric, fixed splits, a baseline result from the Phase 3 encoder.

Exit criteria: a third party can reproduce the baseline number from a fresh clone.

### Phase 5: Release and community (ongoing)

- v0.1.0 to PyPI, announcement on OpenPlanetary forum and relevant mailing lists.
- Apply for planetarypy affiliated package status.
- Short JOSS paper or arXiv preprint describing the library and benchmark.
- Issue templates and a CONTRIBUTING.md that explicitly invites new body modules (Mars first) and new dataset classes.

### Deliberate non-goals for v0.x

- No mirroring or rehosting of raw PDS archives.
- No support for every PDS product type; only what the sampler and datasets need.
- No GUI or web viewer; QuickMap and Trek exist.
- No Earth support; TorchGeo owns that space.

### Risks and mitigations

- Endpoint drift (services move, as QuickMap's domain change showed): keep all endpoint URLs in one config module, cover them with the live test suite, and document last-verified dates.
- M3 data quality issues: defer M3 dataset class until after v0.1 unless the swirls benchmark strictly requires it; budget preprocessing time if it does.
- Server load courtesy: default to conservative request concurrency, exponential backoff, and a bulk prefetch helper so training never hammers archive servers with random access.
- Solo-maintainer bus factor: keep scope small, tests honest, and architecture boring enough that contributors can navigate it without you.
