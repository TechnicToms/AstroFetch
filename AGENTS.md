# AGENTS.md

Instructions for AI coding agents (Claude Code, Codex, Cursor, and others) working in the AstroFetch repository. CLAUDE.md in this repo points here; treat this file as the single source of truth. It covers what the project is, the rules for working in it, and the phased roadmap (see [Roadmap](#roadmap)).

## What this project is

AstroFetch is an open source, PyTorch-friendly library for ML-ready access to planetary science data, starting with the Moon. The core promise: request a bounding box, receive a coregistered multichannel tensor. The core API: one dataset class per instrument (`KaguyaTC`, `LROCNACDTM`, `LROCWACMosaic`, ...), each yielding TorchGeo-style sample dicts with a coregistered multichannel `"image"` tensor, a validity `"mask"`, and per-channel provenance; instruments compose with `&` (`IntersectionDataset`) to stack channels over their overlapping region. Probes and bodies are discovery-catalog metadata (`MOON`), never dataset boundaries. It is a thin composition layer over existing archive tooling, never a mirror of any archive. A separate, explicitly-marked-experimental family (`astrofetch.moon.granules`) deviates from this contract for raw, non-map-projected camera data — see rule 7 and its own module docstring.

## Architecture at a glance

```
src/astrofetch/
  data/
    endpoints.py      # ALL external URLs live here, nowhere else
    stac.py           # STAC queries (pystac-client) against USGS Astrogeology ARD
    ode.py            # PDS Orbital Data Explorer (ODE) REST queries, for instruments STAC doesn't carry
    raster.py         # windowed/full raster reads (rasterio), scale/offset -> physical values
    grid.py           # target grid definition, reprojection, channel stacking
    cache.py          # throwaway local cache, keyed by (collection, item, window, res)
    tiles.py          # secondary rendered mode: USGS WMS, Moon Trek WMTS (not yet built)
  moon/
    layers.py         # layer registry (name -> source config: STAC/ODE/mosaic) + Body/Probe/Instrument catalog (MOON)
    datasets.py       # windowed dataset classes: InstrumentDataset (STAC), ODEInstrumentDataset (PDS ODE),
                       # MosaicDataset (fixed URL), concrete instruments, IntersectionDataset
    granules.py       # EXPERIMENTAL: raw, non-map-projected granule datasets (GranuleDataset and subclasses)
tests/
  unit/               # network fully mocked, runs in CI
  live/               # hits real endpoints, manual trigger only
  fixtures/           # recorded JSON response fixtures for unit tests
```

## Non-negotiable design rules

1. **Default to wrapping archive tooling; reimplement only with a measured reason.** pystac-client queries STAC, rasterio reads and windows rasters. Reach for these first — reinventing them is usually wasted effort and a maintenance burden. Two carve-outs: (a) **never** hand-roll domain-specific correctness — map-projection math, raster windowing, scale/offset conversion — the bugs there are subtle and scientific, so always defer to the established library; (b) generic plumbing (discovery helpers, small utilities) *may* be replaced when a dependency is provably a bad trade — too slow on the hot path, a heavy transitive dependency for a sliver of use, etc. Justify any such reimplementation in the PR description with the concrete reason. Note that "defer to the library" can still require picking the *right entry point* into that library: for the LROC NAC DTM PDS4 products, opening the data file directly (GDAL's native GTiff/PDS driver) gives correct georeferencing and nodata, while opening the same product through its detached `.xml` label does not (a confirmed GDAL PDS4-driver resolution-unit parsing bug, not an astrofetch reimplementation) — verify a new source's actual behavior live before trusting either path.
2. **All external endpoint URLs go in `data/endpoints.py`.** No URL literals anywhere else in `src/`. Endpoints move (QuickMap changed domains); one module keeps that survivable.
3. **Quantitative vs rendered is a hard boundary.** The STAC/ODE/mosaic paths return physical values and are the only paths for quantitative or ML use. The WMS/WMTS tile path returns rendered 8-bit imagery and must be labeled as such in APIs and docs. Never mix them silently — this is also why a dataset offers only the quantitative products a source provides: e.g. `LROCNACDTM` deliberately excludes the SDP pipeline's color-coded slope and shaded-relief products, which are rendered 8-bit visualizations, not calibrated rasters.
4. **Cache is disposable.** Nothing in the cache layer may be load-bearing for correctness or reproducibility. Everything fetched on demand must be re-fetchable from the archive and safe to delete; never commit fetched data to the repo.
5. **Be polite to archive servers.** Default concurrency is low, retries use exponential backoff, and any code path that could issue many requests must go through the rate-limited session in `data/stac.py` / `data/ode.py` / `data/tiles.py`. Never write a loop that hammers NASA, USGS, or PDS-node servers.
6. **Body-namespaced layout.** Moon-specific code lives under `moon/`. Body-agnostic code (grid math, raster reads, caching, archive search) lives under `data/`. Adding Mars must be a new sibling module, not edits scattered through existing files.
7. **Samples are dicts with a fixed contract.** `"image"` is (C, H, W) float32 (physical values, channel i = `layers[i]`), `"mask"` is (C, H, W) bool validity, plus `"layers"`, `"bbox"`, `"crs"`, and `"resolution"` provenance keys. Datasets that deviate must document it — the one deliberate exception is `astrofetch.moon.granules`, whose module docstring documents its different contract (no bbox windowing, ragged shapes, no `&`) up front. Samples must collate under the default `DataLoader` collation; granule datasets are the one documented exception (`batch_size=None` or a custom `collate_fn`).

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

- Coordinates are IAU 2015 Moon (ocentric, longitude 0 to 360 or -180 to 180 must be normalized at the API boundary; internal convention is -180 to 180). PDS ODE's REST API wants 0-360 `westernlon`/`easternlon`; `data/ode.py` converts at that one boundary — shift the west bound into 0-360 and add back the original span, rather than taking `% 360` of each bound independently, or a full-Moon bbox like `(-180, 180)` collapses to a zero-width query.
- Equatorial data uses equirectangular projection; polar data uses polar stereographic. `data/grid.py` owns this decision; never assume equirectangular blindly near the poles.
- COGs and other rasters often store 16-bit DN (or similar) with scale/offset to physical units (for example Kaguya TC radiance). Always apply scale/offset in `raster.py`; downstream code assumes physical values.
- Nodata regions are common (orbital swaths do not cover everything). Every sample carries a boolean validity tensor under its `"mask"` key; do not silently zero-fill. Some PDS products omit a declared nodata value even though their raster does not cover its full requested extent; `raster.read_window`'s `nodata_override` exists for exactly this (see its docstring) — reach for a source's own declared nodata first, and only override when you have live-verified the product genuinely has none.
- Some instruments cover only a handful of named sites, not the whole Moon (e.g. LROC NAC stereo DTMs via PDS ODE). `ODEInstrumentDataset.footprint_sampling` draws windows from inside real product footprints for exactly this case; leave it off for globally-covered instruments.
- Some ODE product types mix the products a dataset actually wants with many unrelated ones under the same `pt` — a different parameter, a rendered visualization, or per-orbit granules that vastly outnumber the wanted product (Diviner's `GDR_L3` is ~80% per-orbit `TBOL` files; LROC's `SDWDTM` is dominated by rendered `WAC_CSHADE`). A bbox-only search then has to page through candidates in whatever order ODE returns them before ever reaching the wanted product, which no reasonable `max_products` cap reliably reaches — and paging that deep violates rule 5. `ODEAsset.product_id` (an ODE `productid` wildcard filter, e.g. `"*wac_gld100*"`) narrows the search itself server-side; reach for it whenever a new source's product type isn't cleanly single-purpose. `ODEInstrumentDataset._product_footprints` applies the same per-product `product_id` narrowing, not just `_hrefs`, so footprint-sampled instruments don't draw from an unrelated sibling instrument's sites either.

## What NOT to do

- Do not add dependencies casually. Core deps are: torch, rasterio, pystac-client, numpy, requests. Anything else needs a justification in the PR description.
- Do not commit data files, fetched tiles, or notebooks with executed output containing large images.
- Do not target QuickMap's internal tile URLs; they are not a public API. Use USGS WMS or Moon Trek WMTS via `data/endpoints.py`.
- Do not "fix" scientific constants or projection parameters without a source; cite the reference in the commit message.
- Do not weaken the mocked-network rule in unit tests to make something pass.
- Do not trust a raster driver's georeferencing just because the file opens and reads without error — a mechanically successful open/read is not proof the transform, bounds, or nodata are correct (see rule 1's PDS4-label example). Verify live against a known location before shipping a new source.

## Roadmap

Check the current phase before proposing work; for example, do not build Phase 2 datasets and transforms while Phase 1 (STAC sampler MVP) is incomplete. Everything is a thin layer above existing archive tooling, never a mirror of any archive.

**Current phase: Phase 2 (datasets and transforms). Phase 0 and Phase 1 are complete: `InstrumentDataset.read` fetches the real COGs covering a window from the USGS ARD catalog, reprojects them onto a common geographic grid, applies scale/offset, mosaics overlapping items, and caches the result. Phase 2 has delivered a wide roster of new data sources beyond STAC (see below); `GridTileDataset`, spatial-split samplers, transforms, and the WMS/WMTS rendered mode remain open.**

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

Exit criteria (met): `KaguyaTC(products=["dtm", "ortho"], bbox=...)` fetches a real, coregistered two-layer patch from the USGS ARD catalog on a clean machine (covered by `tests/live`). A plotting quickstart notebook is a nice-to-have follow-up, still open.

### Phase 2: Datasets and transforms (2 to 3 weekends)

**New data sources beyond STAC (done):** the USGS ARD STAC catalog has no LROC, LOLA, or other lunar collections beyond Kaguya TC, so growing past it required a second search backend.

- `astrofetch.data.ode`: query the NASA PDS Orbital Data Explorer (ODE) REST API by instrument host/id and product type, politely (same retry/backoff/timeout posture as `data/stac.py`), normalizing ODE's JSON quirks (single-result dict vs list, `"No Products Found"`, HTTP-200 error bodies). `query_products`/`find_file_urls` also accept a `product_id` wildcard filter for product types that mix the wanted product with many unrelated ones (see the domain note above).
- `astrofetch.moon.datasets.ODEInstrumentDataset`: the ODE-backed sibling of `InstrumentDataset`, with `footprint_sampling` for instruments that cover only named sites rather than the whole Moon. Ships the wider PDS ODE roster: `LROCNACDTM` (NAC stereo DTM sites), `LROCNACROI` (NAC region-of-interest mosaics, 5 m/20 m only — native resolution can reach ~14 GB per site), `MiniRF` (S-band radar global mosaics), `DivinerGDR` (rock abundance / regolith temperature, pinned to the most complete cumulative-mosaic date), `WACGLD100` (WAC global 100 m DTM), `WACTiO2` (WAC TiO2 abundance), `LROCWACGlobal` (WAC global morphology, tiled/searched sibling of `LROCWACMosaic`), `LROCWACColor` (WAC 7-color reflectance), `ShadowCam` (KPLO PSR mosaics and DTMs, genuine COGs), and `ClementineUVVIS`/`ClementineNIR` (5-/6-band basemaps, the project's first attached-PDS3-label source).
- `astrofetch.moon.datasets.MosaicDataset`: reads one well-known archive URL directly, for instruments published as a single global (or near-global) file. Ships `LROCWACMosaic` (the LRO WAC 100 m global mosaic — the dataset this phase originally named as its LRO WAC deliverable, shipped as `LROCWACMosaic` rather than `LROCWAC` since a raw, non-map-projected `LROCWACRaw` also now exists), `LOLA` (global gridded DEM), and `SLDEM2015` (LOLA + Kaguya TC co-registered DEM).
- `astrofetch.moon.granules` (new, experimental — see its Deliberate non-goals amendment below): raw, camera-geometry PDS granules for instruments that are not map-projected at all (`LROCNACRaw`, `LROCWACRaw`, `M3`). A deliberately different, documented sample contract; not part of the `InstrumentDataset` family.
- Roster items deliberately dropped, with reasons (do not re-propose without addressing the reason): **Kaguya MI** — its JAXA DARTS host ignores HTTP Range headers (a ranged GET returns the full ~90 MB body), so windowed remote reads are impossible without downloading whole files; **Kaguya LALT** — coarse (3 products) and superseded by `SLDEM2015`; **Diviner `TBOL`** — per-orbit bolometric temperature, not part of the cumulative-mosaic family `DivinerGDR` offers, and mosaicking single-orbit epochs would misrepresent the data; **`MDRHAP`** (Hapke-normalized WAC color), **`SDPWMG`** (monthly WAC mosaics), **Mini-RF polar `MOSCDR`**, **Clementine `MDIMG`** basemap and **`HIRES`** — redundant with what's shipped or left as follow-up.

**Still open:**

- `astrofetch.moon.datasets`: random-bbox sampling already ships inside the instrument datasets (Phase 0); add `GridTileDataset` (deterministic tiling of an ROI) over the same `read(bbox)` interface, plus region-list sampling for the random path.
- Samplers that respect spatial autocorrelation for train/val/test splits (block splitting, not random pixels).
- Transforms: per-channel normalization stats, nodata masking, polar/equatorial projection handling made explicit.
- Secondary access mode behind the same interface: WMS/WMTS rendered mode, clearly labeled non-quantitative.
- A dedicated polar target grid (`data/grid.py`): `ShadowCam` and other near-polar sources currently reproject onto the same equirectangular geographic grid as everything else, which distorts near the poles; a polar-stereographic target grid for high-latitude requests remains future work.

Exit criteria: `DataLoader` trains a toy model on random lunar patches without custom user code.

### Phase 3: Release and community (ongoing)

- v0.1.0 to PyPI, announcement on OpenPlanetary forum and relevant mailing lists.
- Apply for planetarypy affiliated package status.
- Short JOSS paper or arXiv preprint describing the library.
- Issue templates and a CONTRIBUTING.md that explicitly invites new body modules (Mars first) and new dataset classes.

### Deliberate non-goals for v0.x

- No mirroring or rehosting of raw PDS archives.
- **Amended:** raw PDS granule access was originally ruled out entirely for v0.x ("the STAC/COG path is the only quantitative source"). `astrofetch.moon.granules` now provides it, narrowly: camera-geometry data only, no map projection, no ISIS/SPICE, explicitly marked experimental with its own documented (different) sample contract, and excluded from the `LAYERS` registry. The `InstrumentDataset`/`ODEInstrumentDataset`/`MosaicDataset` windowed-and-reprojected path remains the only *quantitative, coregistered-tensor* source — that guarantee is unchanged.
- No pretrained models or frozen benchmarks; AstroFetch delivers ML-ready data, you bring the model.
- No GUI or web viewer; QuickMap and Trek exist.
- No Earth support; TorchGeo owns that space.

### Risks and mitigations

- Endpoint drift (services move, as QuickMap's domain change showed): keep all endpoint URLs in one config module, cover them with the live test suite, and document last-verified dates.
- M3 data quality: rather than deferring M3 entirely, it shipped scoped to what's verified reliable — the experimental raw-granule path (`astrofetch.moon.granules.M3`), radiance plus geolocation backplane, no map projection or further calibration claimed. A map-projected, quantitative M3 `InstrumentDataset`/`ODEInstrumentDataset` remains deferred until a user need justifies the preprocessing work.
- Archive driver quirks: a source opening and reading without error is not proof its georeferencing or nodata are correct (see design rule 1's PDS4-label example, caught via `tests/live` before shipping `LROCNACDTM`). Live-verify a new source against a known location, not just that `rasterio.open` succeeds. The same live check can also come back clean: `ClementineUVVIS`/`ClementineNIR` (attached PDS3 label, sinusoidal projection) checked out on the first try, comparing the opened raster's bounds against the product's own ODE footprint — not every new archive hides a bug, but every one needs the check.
- Not every archive quirk is a georeferencing bug: `MiniRF`'s global mosaics declare a valid `MISSING_CONSTANT` in their PDS3 label, but it's a float64 literal inside a float32 band, and GDAL's overflowing cast leaves `src.nodata` unset rather than raising — out-of-coverage pixels silently read back as a specific overflow artifact (not even the standard float32 minimum), marked "valid". Caught by reading raw pixels directly and pinning the exact observed bit pattern as `nodata_override`. The general lesson: a *declared* nodata value is not automatically a *working* one — verify what the library actually did with it, not just that the label mentions it.
- Server load courtesy: default to conservative request concurrency, exponential backoff, and a bulk prefetch helper so training never hammers archive servers with random access. Also watch for product types where a bbox-only search can't stay small on its own (Diviner's `GDR_L3`, LROC's `SDWDTM`) — `ODEAsset.product_id` narrows the search server-side instead of paging through hundreds of irrelevant candidates client-side.
- Solo-maintainer bus factor: keep scope small, tests honest, and architecture boring enough that contributors can navigate it without you.
