# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Cursor, and others) working in the AstroFetch repository. CLAUDE.md in this repo points here; treat this file as the single source of truth.

## What this project is

AstroFetch is an open source, PyTorch-friendly library for ML-ready access to planetary science data, starting with the Moon. The core API: one dataset class per instrument (`KaguyaTC`, `LROCWAC`, ...), each yielding TorchGeo-style sample dicts with a coregistered multichannel `"image"` tensor, a validity `"mask"`, and per-channel provenance; instruments compose with `&` (`IntersectionDataset`) to stack channels over their overlapping region. Probes and bodies are discovery-catalog metadata (`MOON`), never dataset boundaries. It is a thin composition layer over existing archive tooling.

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

## Current phase

See BUILD_PLAN.md for the phased roadmap. Check the phase before proposing work; for example, do not build benchmark infrastructure while Phase 1 (STAC sampler MVP) is incomplete.
