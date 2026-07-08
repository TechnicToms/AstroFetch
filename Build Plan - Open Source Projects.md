# AstroFetch Build Plan

AstroFetch is an open source, PyTorch-friendly library for ML-ready access to planetary science data, starting with the Moon. The core promise: request a bounding box, receive a coregistered multichannel tensor. The library is a thin layer above existing archive tooling, never a mirror of any archive.

## Guiding principles

1. Wrap, do not reimplement. pdr reads PDS products, planetarypy finds and fetches them, pystac-client queries the USGS STAC catalog, rasterio reads COGs. AstroFetch composes these.
2. The USGS Astrogeology ARD STAC catalog (Cloud Optimized GeoTIFFs) is the primary data source. It provides windowed HTTP range reads, real physical values, and existing map projection.
3. Rendered tile services (USGS WMS, Moon Trek WMTS) are a secondary convenience path for visualization and coverage, never the benchmark path.
4. Cache is throwaway, benchmarks are frozen. Anything fetched on demand is reproducible and disposable. Only frozen benchmark artifacts (checksummed, numeric, hosted on HF/Zenodo) are permanent.
5. Body-namespaced from day one. Moon ships first; the layout must make Mars a sibling module, not a redesign.

## Phase 0: Scaffolding (weekend 1)

- Repo setup: pyproject.toml (hatchling or setuptools), src layout, MIT or Apache-2.0 license, CITATION.cff.
- CI: GitHub Actions running ruff, mypy (lenient at first), pytest on 3.10 through 3.12.
- Testing policy established early: unit tests mock all network calls; a separate, manually triggered "live" test suite hits real endpoints.
- Placeholder docs (mkdocs-material) and a README with the one-line pitch and the target API sketch.

Exit criteria: `pip install -e .` works, CI is green on an empty package.

## Phase 1: STAC sampler MVP (the core, 2 to 4 weekends)

The single most important deliverable. Everything else builds on it.

- `astrofetch.data.stac`: query the USGS ARD catalog root with pystac-client, filter by collection and bbox.
- `astrofetch.data.raster`: windowed COG reads via rasterio, honoring scale/offset to return physical values, resampling to a requested resolution.
- `astrofetch.data.grid`: define a common target grid (equirectangular, IAU 2015 Moon), reproject and stack layers into a (C, H, W) float tensor with a per-channel metadata record.
- Public API target:

```python
import astrofetch as af
patch = af.moon.fetch(
    layers=["kaguya_tc_dtm", "lroc_wac"],
    bbox=(-60.0, 5.0, -55.0, 10.0),
    resolution=100,   # meters per pixel
)
patch.tensor   # torch.Tensor (C, H, W)
patch.meta     # projection, units, provenance per channel
```

- Local disk cache keyed by (collection, item, window, resolution), transparent and clearable.

Exit criteria: the quickstart notebook fetches a Reiner Gamma patch with two layers and plots it, end to end, on a clean machine.

## Phase 2: Datasets and transforms (2 to 3 weekends)

- `astrofetch.moon.datasets`: torch `Dataset` classes built on the sampler. First: `RandomRegionDataset` (sample random bboxes from a region list) and `GridTileDataset` (deterministic tiling of an ROI).
- Samplers that respect spatial autocorrelation for train/val/test splits (block splitting, not random pixels).
- Transforms: per-channel normalization stats, nodata masking, polar/equatorial projection handling made explicit.
- Secondary access modes behind the same interface: WMS/WMTS rendered mode (clearly labeled non-quantitative) and an ODE REST granule mode stub for later full-fidelity work.

Exit criteria: `DataLoader` trains a toy model on random lunar patches without custom user code.

## Phase 3: Pretrained weights (1 to 2 weekends, leverages existing thesis assets)

- `astrofetch.models`: multimodal MAE encoder definition matching the thesis architecture, plus fine-tuning heads.
- `astrofetch.models.weights`: registry pattern, `af.models.load_pretrained("lunar-mae-base")`, weights hosted on Hugging Face Hub.
- Model card documenting training data (with STAC provenance), input channels, and known limitations.

Exit criteria: a user can load the pretrained encoder and extract features from a Phase 1 patch in under ten lines.

## Phase 4: First frozen benchmark (2 to 3 weekends)

- Pick one task with clear scientific value, for example swirl segmentation or crater detection on a defined region set.
- Freeze: exact STAC item IDs, windows, checksums, and numeric arrays (not rendered tiles). Host the frozen set on Hugging Face, mint a Zenodo DOI at release.
- Ship the evaluation harness: fixed metric, fixed splits, a baseline result from the Phase 3 encoder.

Exit criteria: a third party can reproduce the baseline number from a fresh clone.

## Phase 5: Release and community (ongoing)

- v0.1.0 to PyPI, announcement on OpenPlanetary forum and relevant mailing lists.
- Apply for planetarypy affiliated package status.
- Short JOSS paper or arXiv preprint describing the library and benchmark.
- Issue templates and a CONTRIBUTING.md that explicitly invites new body modules (Mars first) and new dataset classes.

## Deliberate non-goals for v0.x

- No mirroring or rehosting of raw PDS archives.
- No support for every PDS product type; only what the sampler and datasets need.
- No GUI or web viewer; QuickMap and Trek exist.
- No Earth support; TorchGeo owns that space.

## Risks and mitigations

- Endpoint drift (services move, as QuickMap's domain change showed): keep all endpoint URLs in one config module, cover them with the live test suite, and document last-verified dates.
- M3 data quality issues: defer M3 dataset class until after v0.1 unless the swirls benchmark strictly requires it; budget preprocessing time if it does.
- Server load courtesy: default to conservative request concurrency, exponential backoff, and a bulk prefetch helper so training never hammers archive servers with random access.
- Solo-maintainer bus factor: keep scope small, tests honest, and architecture boring enough that contributors can navigate it without you.
