"""Torch dataset classes over lunar instrument data.

Each instrument dataset samples random windows inside its ``bbox`` and, for
each window, reads its product rasters, reprojects them onto a common target
grid, and stacks them into a coregistered ``(C, H, W)`` tensor. Combine
instruments with ``&`` to stack their channels over the region they share.

Three read paths back the products, chosen per instrument by which base class
it subclasses:

- :class:`InstrumentDataset` -- searches the USGS ARD STAC catalog
  (:mod:`astrofetch.data.stac`) for COG items covering a window.
- :class:`ODEInstrumentDataset` -- searches the NASA PDS Orbital Data
  Explorer (:mod:`astrofetch.data.ode`) for products covering a window, for
  instruments (LROC, LOLA, ...) the STAC catalog does not carry.
- :class:`MosaicDataset` -- reads a single well-known archive URL directly,
  for instruments published as one global (or near-global) file rather than
  many searchable items.

All three reproject through :mod:`astrofetch.data.raster` and cache through
:class:`astrofetch.data.cache.WindowCache`; the sample-dict contract and the
``&`` composition operator are identical regardless of which backs a
particular instrument.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import ClassVar, NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from astrofetch.data import endpoints, ode, raster, stac
from astrofetch.data.cache import WindowCache
from astrofetch.data.grid import GEOGRAPHIC_CRS, TargetGrid, meters_to_degrees

logger = logging.getLogger(__name__)

BBox = tuple[float, float, float, float]
"""(west, south, east, north) in degrees, IAU 2015 Moon."""

CRS = GEOGRAPHIC_CRS
"""Ocentric IAU 2015 Moon geographic CRS carried by every sample."""


class Product(NamedTuple):
    """One STAC-backed product: its sample-dict layer id, STAC asset key,
    band, and (rarely needed) nodata override."""

    layer: str
    asset: str
    band: int = 1
    nodata: float | None = None


class ODEAsset(NamedTuple):
    """One PDS-ODE-backed product: its sample-dict layer id, ODE product
    type, and a filename pattern selecting the right file within each
    product (a product bundle can contain many files -- data, browse,
    derived -- so the pattern narrows to the one that should be read)."""

    layer: str
    pt: str
    pattern: str
    band: int = 1
    nodata: float | None = None
    file_type: str = "Product"
    """Required ODE file role for ``pattern`` to match against. Almost every
    product's actual data file is typed ``"Product"``; a few (e.g. ShadowCam
    DTM confidence maps) ship their data under ``"Referenced"`` instead."""
    product_id: str | None = None
    """ODE ``productid`` wildcard filter narrowing the bbox search
    server-side, e.g. ``"*wac_gld100*"``. Needed whenever a product type
    mixes the wanted product with many unrelated ones (other parameters,
    rendered visualizations, per-orbit granules) that would otherwise
    dominate the results within any reasonable ``max_products`` cap."""


class MosaicAsset(NamedTuple):
    """One fixed-URL product: its sample-dict layer id and archive href (an
    :mod:`astrofetch.data.endpoints` constant), read directly with no
    search."""

    layer: str
    href: str
    band: int = 1
    nodata: float | None = None


def _random_window(
    bbox: BBox, patch_size: int, resolution: float, generator: torch.Generator
) -> BBox:
    """Pick one patch_size*resolution-sized window at a random position inside ``bbox``."""
    west, south, east, north = bbox
    span_lon, span_lat = east - west, north - south
    # Window ground size from patch_size * resolution, converted to degrees
    # at the region's centre latitude; clamp to the bbox so it stays inside.
    center_lat = (south + north) / 2.0
    win_lon, win_lat = meters_to_degrees(patch_size * resolution, center_lat)
    win_lon, win_lat = min(win_lon, span_lon), min(win_lat, span_lat)
    u, v = torch.rand(2, generator=generator).tolist()
    west0 = west + u * (span_lon - win_lon)
    south0 = south + v * (span_lat - win_lat)
    return (west0, south0, west0 + win_lon, south0 + win_lat)


def _bbox_area(bbox: BBox) -> float:
    west, south, east, north = bbox
    return (east - west) * (north - south)


class _WindowedDataset(Dataset[dict]):
    """Map-style dataset: index ``i`` deterministically samples one window.

    Subclasses set ``bbox``, ``resolution``, ``patch_size``, ``length``,
    ``seed``, and ``layers``, and implement ``read``. ``dataset[i]`` and
    iteration both work, and because each index maps to a fixed window the
    dataset supports ``DataLoader`` shuffling, samplers, and ``random_split``.
    """

    bbox: BBox
    resolution: float
    patch_size: int
    length: int
    seed: int | None
    layers: list[str]
    _cached_seed_base: int | None = None

    def read(self, bbox: BBox) -> dict:
        """Read one window as a sample dict.

        Returns:
            dict with keys ``image`` (C, H, W) float32, ``mask`` (C, H, W)
            bool validity, ``layers``, ``bbox``, ``crs``, and ``resolution``.
        """
        raise NotImplementedError

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += self.length
        if not 0 <= index < self.length:
            raise IndexError(f"index out of range for length {self.length}")
        return self.read(self._sample_bbox(index))

    def __iter__(self) -> Iterator[dict]:
        for index in range(self.length):
            yield self[index]

    def __len__(self) -> int:
        return self.length

    def __and__(self, other: _WindowedDataset) -> IntersectionDataset:
        return IntersectionDataset(self, other)

    @property
    def _seed_base(self) -> int:
        # Fixed per instance: an explicit seed makes samples reproducible across
        # instances; without one, each instance still gives a stable dataset[i].
        if self._cached_seed_base is None:
            if self.seed is not None:
                self._cached_seed_base = self.seed
            else:
                self._cached_seed_base = int(torch.randint(0, 2**31 - 1, (1,)).item())
        return self._cached_seed_base

    def _seeded_generator(self, index: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed((self._seed_base * 1_000_003 + index) % (2**63 - 1))
        return generator

    def _sample_bbox(self, index: int) -> BBox:
        generator = self._seeded_generator(index)
        return _random_window(self.bbox, self.patch_size, self.resolution, generator)


class _ProductDataset(_WindowedDataset):
    """Shared read/validation/cache path for every product-backed dataset.

    Subclasses provide ``all_products`` and implement ``_hrefs`` to resolve
    one product's spec and window into the source hrefs to mosaic; see
    :class:`InstrumentDataset`, :class:`ODEInstrumentDataset`, and
    :class:`MosaicDataset`.
    """

    probe: ClassVar[str]
    """Name of the probe (spacecraft) carrying this instrument."""

    instrument: ClassVar[str]
    """Human-readable instrument name."""

    all_products: ClassVar[dict[str, Product | ODEAsset | MosaicAsset]]
    """Product name -> asset spec."""

    def __init__(
        self,
        products: list[str] | None = None,
        bbox: BBox = (-180.0, -90.0, 180.0, 90.0),
        resolution: float = 100.0,
        patch_size: int = 256,
        length: int = 1000,
        seed: int | None = None,
        cache: WindowCache | None = None,
    ) -> None:
        if products is None:
            products = list(self.all_products)
        if not products:
            raise ValueError("at least one product is required")
        unknown = [p for p in products if p not in self.all_products]
        if unknown:
            raise ValueError(
                f"unknown products for {self.instrument}: {unknown}; "
                f"available: {sorted(self.all_products)}"
            )
        west, south, east, north = bbox
        if not (west < east and south < north):
            raise ValueError(f"invalid bbox (west, south, east, north): {bbox}")
        self.products = list(products)
        self.layers = [self.all_products[p].layer for p in products]
        self.bbox = bbox
        self.resolution = resolution
        self.patch_size = patch_size
        self.length = length
        self.seed = seed
        self.cache = cache if cache is not None else WindowCache()

    def read(self, bbox: BBox) -> dict:
        grid = TargetGrid(bbox=bbox, width=self.patch_size, height=self.patch_size, crs=CRS)
        images, masks = [], []
        for product in self.products:
            spec = self.all_products[product]
            image, mask = self._read_layer(spec, grid)
            images.append(image)
            masks.append(mask)
        return {
            "image": torch.from_numpy(np.stack(images)).to(torch.float32),
            "mask": torch.from_numpy(np.stack(masks)).to(torch.bool),
            "layers": list(self.layers),
            "bbox": bbox,
            "crs": CRS,
            "resolution": self.resolution,
        }

    def _read_layer(
        self, spec: Product | ODEAsset | MosaicAsset, grid: TargetGrid
    ) -> tuple[np.ndarray, np.ndarray]:
        cached = self.cache.get(spec.layer, grid)
        if cached is not None:
            return cached
        hrefs = self._hrefs(spec, grid.bbox)
        image, mask = _mosaic(hrefs, grid, band=spec.band, nodata_override=spec.nodata)
        self.cache.put(spec.layer, grid, image, mask)
        return image, mask

    def _hrefs(self, spec: Product | ODEAsset | MosaicAsset, bbox: BBox) -> list[str]:
        raise NotImplementedError


class InstrumentDataset(_ProductDataset):
    """Map-style dataset of patches from a single USGS-ARD-STAC instrument.

    Each index deterministically samples a bounding box within ``bbox`` and
    returns a sample dict: ``image`` is a (C, H, W) float tensor with one
    channel per requested product (physical values) where ``H = W =
    patch_size``, ``mask`` a same-shaped bool validity tensor (orbital swaths
    do not cover everything), plus ``layers``/``bbox``/``crs``/``resolution``
    provenance. Combine instruments with ``&`` to stack their channels over the
    overlapping region.

    Args:
        products: product names to stack, e.g. ``["dtm"]``; defaults to all
            products the instrument offers.
        bbox: region to sample from as (west, south, east, north) degrees.
        resolution: target resolution in metres per pixel.
        patch_size: output height and width in pixels.
        length: number of patches per epoch.
        seed: RNG seed for reproducible sampling.
        max_items: cap on STAC items mosaicked per layer per window.
        cache: window cache; defaults to the shared on-disk cache.

    Example:
        >>> moondata = KaguyaTC(products=["dtm"], bbox=(-26.4, -50.7, -25.4, -49.6))
        >>> for sample in moondata:  # doctest: +SKIP
        ...     sample["image"]  # torch.Tensor (C, H, W)
    """

    collection: ClassVar[str]
    """USGS ARD STAC collection id backing this instrument's products."""

    def __init__(
        self,
        products: list[str] | None = None,
        bbox: BBox = (-180.0, -90.0, 180.0, 90.0),
        resolution: float = 100.0,
        patch_size: int = 256,
        length: int = 1000,
        seed: int | None = None,
        max_items: int = 20,
        cache: WindowCache | None = None,
    ) -> None:
        super().__init__(products, bbox, resolution, patch_size, length, seed, cache)
        self.max_items = max_items

    def _hrefs(self, spec: Product | ODEAsset | MosaicAsset, bbox: BBox) -> list[str]:
        assert isinstance(spec, Product)
        return stac.find_asset_hrefs(self.collection, spec.asset, bbox, self.max_items)


class ODEInstrumentDataset(_ProductDataset):
    """Map-style dataset of patches from an instrument searched via PDS ODE.

    Same sample contract as :class:`InstrumentDataset`, but resolves products
    through the NASA PDS Orbital Data Explorer (:mod:`astrofetch.data.ode`)
    instead of the USGS ARD STAC catalog, for instruments the STAC catalog
    does not carry.

    Some instruments cover only a handful of named sites rather than the
    whole Moon (e.g. LROC NAC stereo DTMs). Set ``footprint_sampling = True``
    on such a subclass so sampled windows are drawn from inside real product
    footprints instead of uniformly over ``bbox`` -- which, for a sparse
    instrument, would draw mostly-empty windows.

    Args:
        products: product names to stack; defaults to all products offered.
        bbox: region to sample from as (west, south, east, north) degrees.
        resolution: target resolution in metres per pixel.
        patch_size: output height and width in pixels.
        length: number of patches per epoch.
        seed: RNG seed for reproducible sampling.
        max_products: cap on ODE products mosaicked per layer per window.
        footprint_sampling: override the class default; ``None`` keeps it.
        cache: window cache; defaults to the shared on-disk cache.
    """

    ihid: ClassVar[str]
    """ODE instrument host id, e.g. ``"LRO"``."""

    iid: ClassVar[str]
    """ODE instrument id, e.g. ``"LROC"``."""

    footprint_sampling: bool = False
    """Sample windows from inside real product footprints rather than
    uniformly over ``bbox``. Set ``True`` for sparse, site-based instruments.
    A plain (not ``ClassVar``) attribute: subclasses set the default, and
    ``__init__`` may shadow it per instance."""

    def __init__(
        self,
        products: list[str] | None = None,
        bbox: BBox = (-180.0, -90.0, 180.0, 90.0),
        resolution: float = 100.0,
        patch_size: int = 256,
        length: int = 1000,
        seed: int | None = None,
        max_products: int = 20,
        footprint_sampling: bool | None = None,
        cache: WindowCache | None = None,
    ) -> None:
        super().__init__(products, bbox, resolution, patch_size, length, seed, cache)
        self.max_products = max_products
        if footprint_sampling is not None:
            self.footprint_sampling = footprint_sampling
        self._footprints: list[BBox] | None = None

    def _hrefs(self, spec: Product | ODEAsset | MosaicAsset, bbox: BBox) -> list[str]:
        assert isinstance(spec, ODEAsset)
        return ode.find_file_urls(
            self.ihid,
            self.iid,
            spec.pt,
            spec.pattern,
            bbox,
            self.max_products,
            spec.file_type,
            spec.product_id,
        )

    def _sample_bbox(self, index: int) -> BBox:
        if not self.footprint_sampling:
            return super()._sample_bbox(index)
        footprints = self._product_footprints()
        generator = self._seeded_generator(index)
        if not footprints:
            logger.warning(
                "%s: no product footprints found in bbox %s; falling back to "
                "uniform sampling over the full bbox",
                self.instrument,
                self.bbox,
            )
            return _random_window(self.bbox, self.patch_size, self.resolution, generator)
        # Same seeded generator, drawn from in order: which footprint, then
        # where inside it -- so a given (seed, index) always yields the same
        # window, exactly like the uniform-sampling path.
        areas = torch.tensor([_bbox_area(fp) for fp in footprints])
        choice = int(torch.multinomial(areas, 1, generator=generator).item())
        return _random_window(footprints[choice], self.patch_size, self.resolution, generator)

    def _product_footprints(self) -> list[BBox]:
        # Fetched lazily (on first sample, not __init__) so construction never
        # touches the network -- tests can build instances hermetically.
        if self._footprints is None:
            # Grouped by (pt, product_id): a product type can mix products
            # this instrument doesn't offer (e.g. a sibling instrument's
            # products under the same pt), so footprints must go through the
            # same product_id narrowing as the file search itself, or the
            # sampling pool would include sites that never yield this
            # instrument's data.
            queries: set[tuple[str, str | None]] = set()
            for name in self.products:
                entry = self.all_products[name]
                if isinstance(entry, ODEAsset):
                    queries.add((entry.pt, entry.product_id))
            footprints: list[BBox] = []
            for pt, product_id in queries:
                products = ode.query_products(
                    self.ihid, self.iid, pt, self.bbox, max_products=500, product_id=product_id
                )
                footprints.extend(product.bbox for product in products if product.bbox is not None)
            self._footprints = footprints
        return self._footprints


class MosaicDataset(_ProductDataset):
    """Map-style dataset of patches from a fixed-URL global (or near-global)
    product, e.g. a single mosaic or DEM file.

    Same sample contract as :class:`InstrumentDataset`, but each product maps
    to one well-known archive URL (an :mod:`astrofetch.data.endpoints`
    constant) rather than being searched: there is exactly one item to read
    per layer, so no catalog lookup happens on ``read``.
    """

    def _hrefs(self, spec: Product | ODEAsset | MosaicAsset, bbox: BBox) -> list[str]:
        assert isinstance(spec, MosaicAsset)
        return [spec.href]


def _mosaic(
    hrefs: list[str],
    grid: TargetGrid,
    band: int = 1,
    nodata_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject each source onto ``grid`` and fill invalid pixels from later items.

    Earlier items win where they have data; later items fill only the gaps.
    With no items, returns an all-invalid (zero) window — the mask tells the
    truth about coverage rather than fabricating data.
    """
    image = np.zeros((grid.height, grid.width), dtype=np.float32)
    valid = np.zeros((grid.height, grid.width), dtype=bool)
    for href in hrefs:
        layer_image, layer_mask = raster.read_window(
            href, grid, band=band, nodata_override=nodata_override
        )
        fill = layer_mask & ~valid
        image[fill] = layer_image[fill]
        valid |= layer_mask
    return image, valid


class KaguyaTC(InstrumentDataset):
    """Kaguya (SELENE) Terrain Camera: USGS stereo-derived DTM and orthoimage."""

    probe = "Kaguya (SELENE)"
    instrument = "Terrain Camera (USGS DTM)"
    collection = "kaguya_terrain_camera_usgs_dtms"
    all_products = {
        "dtm": Product("kaguya_tc_dtm", "dtm"),
        "ortho": Product("kaguya_tc_ortho", "orthoimage"),
    }


class KaguyaTCImagery(InstrumentDataset):
    """Kaguya (SELENE) Terrain Camera: stereoscopic radiance imagery.

    Raw Terrain Camera observations (16-bit DN scaled to radiance), distinct
    from the USGS-derived DTM products in :class:`KaguyaTC`.
    """

    probe = "Kaguya (SELENE)"
    instrument = "Terrain Camera (imagery)"
    collection = "kaguya_terrain_camera_stereoscopic_uncontrolled_observations"
    all_products = {"image": Product("kaguya_tc_image", "image")}


class LROCNACDTM(ODEInstrumentDataset):
    """LRO LROC NAC stereo photogrammetric DTM sites: elevation, orthoimage,
    and per-pixel confidence, searched via PDS ODE (product type ``SDNDTM``).

    Coverage is a few hundred named sites (Apollo landing sites, craters and
    other features of interest) rather than the whole Moon, so
    ``footprint_sampling`` is on by default: sampled windows land inside an
    actual site instead of drawing uniformly over ``bbox`` and mostly missing.
    The color-coded slope and shaded-relief SDP products are rendered 8-bit
    visualizations, not quantitative rasters, so they are intentionally not
    offered here (AGENTS rule 3: never mix rendered and quantitative data).
    """

    probe = "Lunar Reconnaissance Orbiter"
    instrument = "LROC NAC (stereo DTM sites)"
    ihid = "LRO"
    iid = "LROC"
    footprint_sampling = True
    all_products = {
        # PDS4 .xml labels for this SDP pipeline carry a broken resolution
        # unit that GDAL's PDS4 driver mis-parses into a near-zero pixel
        # size; opening each data file directly (GTiff/raw driver) instead
        # of through its label reads correct georeferencing, scale, and
        # nodata (verified live 2026-07-20). DTM and confidence/shade/slope
        # share a stem, so DTM excludes the derived-product suffixes.
        "dtm": ODEAsset(
            "lroc_nac_dtm",
            "SDNDTM",
            r"NAC_DTM_(?:(?!_CLRDISC|_CLRGRAD|_CONF|_SHADE|_SLOPE).)+\.TIF",
        ),
        "ortho": ODEAsset("lroc_nac_ortho", "SDNDTM", r"NAC_DTM_.+_M\d+_(?:50CM|2M)\.IMG"),
        "confidence": ODEAsset("lroc_nac_confidence", "SDNDTM", r"NAC_DTM_.+_CONF\.IMG"),
    }


class LROCWACMosaic(MosaicDataset):
    """LRO LROC WAC global morphology mosaic, 100 m/px, equirectangular.

    A single fixed mosaic (see :mod:`astrofetch.data.endpoints`), not a Cloud
    Optimized GeoTIFF: it has no overviews, so prefer ``resolution=100`` (its
    native resolution) -- coarser resolutions force GDAL to read every
    source row under the requested window.
    """

    probe = "Lunar Reconnaissance Orbiter"
    instrument = "LROC WAC (global 100 m mosaic)"
    all_products = {
        "morphology": MosaicAsset("lroc_wac_mosaic", endpoints.LROC_WAC_MOSAIC_100M_URL)
    }


class LOLA(MosaicDataset):
    """LRO LOLA global gridded DEM, 128 px/degree (~237 m/px at the equator).

    Elevation in metres above the IAU 2015 Moon reference sphere. A single
    fixed global product (see :mod:`astrofetch.data.endpoints`), not searched.
    """

    probe = "Lunar Reconnaissance Orbiter"
    instrument = "LOLA (global gridded DEM)"
    all_products = {"dem": MosaicAsset("lola_dem", endpoints.LOLA_DEM_128_URL)}


class SLDEM2015(MosaicDataset):
    """SLDEM2015: LOLA + Kaguya Terrain Camera co-registered DEM, 128 px/degree.

    Elevation in metres above the IAU 2015 Moon reference sphere. Source
    coverage is 60S-60N only (not a bug): windows outside that band read back
    with ``mask`` all ``False``. A single fixed product, not searched.
    """

    probe = "Lunar Reconnaissance Orbiter"
    instrument = "SLDEM2015 (LOLA + Kaguya TC DEM)"
    all_products = {"dem": MosaicAsset("sldem2015_dem", endpoints.SLDEM2015_URL)}


_MINIRF_NODATA = -3.4028226550889045e38
"""Mini-RF global mosaics declare ``MISSING_CONSTANT = -1.7976931E+308`` (a
float64 sentinel) in a float32 band; GDAL's overflowing cast of that constant
into the band's dtype leaves ``src.nodata`` unset (rather than raising), and
out-of-coverage pixels read back as this specific overflow artifact -- not
even the standard float32 minimum -- marked "valid" (verified live
2026-07-21 by reading raw pixels directly and inspecting the exact bit
pattern; same class of issue as the LROCNACDTM PDS4-label bug -- rule 1)."""


class MiniRF(ODEInstrumentDataset):
    """LRO Mini-RF S-band bistatic radar global mosaics, 128 px/degree,
    searched via PDS ODE (product type ``MOSDDR``): circular polarization
    ratio, and same- and opposite-sense circular received power.

    Each product is itself a single global mosaic (detached PDS3 label, same
    read path as :class:`LOLA`); ODE is still searched rather than reading a
    fixed URL, matching the rest of the ODE-backed roster. Verified live
    2026-07-21.
    """

    probe = "Lunar Reconnaissance Orbiter"
    instrument = "Mini-RF (S-band radar mosaics)"
    ihid = "LRO"
    iid = "MRFLRO"
    all_products = {
        "cpr": ODEAsset(
            "lro_minirf_cpr", "MOSDDR", r"GLOBAL_CPR_128PPD_SIMP_0C\.LBL", nodata=_MINIRF_NODATA
        ),
        "sc": ODEAsset(
            "lro_minirf_sc", "MOSDDR", r"GLOBAL_SC_128PPD_SIMP_0C\.LBL", nodata=_MINIRF_NODATA
        ),
        "oc": ODEAsset(
            "lro_minirf_oc", "MOSDDR", r"GLOBAL_OC_128PPD_SIMP_0C\.LBL", nodata=_MINIRF_NODATA
        ),
    }


class DivinerGDR(ODEInstrumentDataset):
    """LRO Diviner rock abundance and regolith temperature, mission-cumulative
    global mosaics, 128 px/degree, searched via PDS ODE (product type
    ``GDR_L3``).

    Each parameter is republished periodically as a new cumulative global
    mosaic (same footprint, more orbits folded in); this pins the most
    complete date verified live, 2016-09-13, rather than a loose pattern
    that would otherwise match all ~105 dated products and mosaic redundant
    copies of the same coverage. ``GDR_L3`` also carries per-orbit
    bolometric temperature (``TBOL``), which alone outnumbers every other
    parameter combined, so a bbox-only search would need to page through
    thousands of unrelated candidates before ever reaching a dated RA or ST
    product; ``product_id`` narrows the ODE search itself to just that
    parameter (verified live 2026-07-21 -- see :func:`astrofetch.data.ode.query_products`).
    Coverage is -80 to 80 latitude (cylindrical projection, not a bug).
    ``TBOL`` is intentionally not offered here: it is not part of this
    cumulative-mosaic family and mixing single-orbit epochs into a windowed
    read would misrepresent the data.
    """

    probe = "Lunar Reconnaissance Orbiter"
    instrument = "Diviner (rock abundance / regolith temperature)"
    ihid = "LRO"
    iid = "DLRE"
    all_products = {
        "rock_abundance": ODEAsset(
            "lro_diviner_rock_abundance",
            "GDR_L3",
            r"DGDR_RA_CLC_CYL_20160913N_128_IMG\.LBL",
            product_id="*ra_clc_cyl_20160913*",
        ),
        "regolith_temp": ODEAsset(
            "lro_diviner_regolith_temp",
            "GDR_L3",
            r"DGDR_ST_CLC_CYL_20160913N_128_IMG\.LBL",
            product_id="*st_clc_cyl_20160913*",
        ),
    }


class WACGLD100(ODEInstrumentDataset):
    """LRO LROC WAC GLD100 global DTM, 100 m/px, searched via PDS ODE
    (product type ``SDWDTM``): 8 near-global quadrant tiles plus 2 polar
    caps, pinned to their 100 m native resolution (coarser 128/256 px-per-
    degree copies of the same tiles, and one separate whole-globe file at
    even coarser multi-resolution, both exist under the same product type
    and are excluded by the pattern). The same product type is dominated by
    ``WAC_CSHADE`` shaded-relief products (a rendered visualization, AGENTS
    rule 3, and far more numerous than GLD100 itself), so ``product_id``
    narrows the ODE search server-side rather than relying on the filename
    pattern alone (verified live 2026-07-21). im-ldi PDS4 archive: opens the
    data ``.IMG`` file directly, never its ``.xml`` label (rule 1's
    PDS4-label lesson).
    """

    probe = "Lunar Reconnaissance Orbiter"
    instrument = "LROC WAC GLD100 (global DTM)"
    ihid = "LRO"
    iid = "LROC"
    all_products = {
        "dtm": ODEAsset(
            "lroc_wac_gld100_dtm",
            "SDWDTM",
            r"WAC_GLD100_.+_100M\.IMG",
            product_id="*wac_gld100*",
        ),
    }


class WACTiO2(ODEInstrumentDataset):
    """LRO LROC WAC TiO2 abundance map, searched via PDS ODE (product type
    ``SDWTIO``), weight-percent TiO2 in the regolith derived from WAC
    multispectral photometry. im-ldi PDS4 archive: opens the data ``.IMG``
    file directly, never its ``.xml`` label (rule 1's PDS4-label lesson).
    """

    probe = "Lunar Reconnaissance Orbiter"
    instrument = "LROC WAC TiO2 abundance"
    ihid = "LRO"
    iid = "LROC"
    all_products = {
        "tio2": ODEAsset("lroc_wac_tio2", "SDWTIO", r"WAC_TIO2_.+\.IMG"),
    }


class LROCWACGlobal(ODEInstrumentDataset):
    """LRO LROC WAC global morphology mosaic, 100 m/px, searched via PDS ODE
    (product type ``BDRWGL``) as 8 near-global quadrant tiles rather than one
    monolithic file -- the searched sibling of the fixed-URL
    :class:`LROCWACMosaic`. Prefer this when only part of the globe is
    needed (fetches one small tile instead of opening the ~2 GB monolith);
    prefer :class:`LROCWACMosaic` for dense sampling over a wide area, since
    its cache reuses one already-open source across reads. The same product
    type also carries a whole-globe ``O``-prefixed file family at each
    resolution (an alternative to :class:`LROCWACMosaic`, not offered here);
    the pattern selects only the tiled ``E``-prefixed family (verified live
    2026-07-21).
    """

    probe = "Lunar Reconnaissance Orbiter"
    instrument = "LROC WAC global mosaic (tiled)"
    ihid = "LRO"
    iid = "LROC"
    all_products = {
        "morphology": ODEAsset(
            "lroc_wac_global_tiled", "BDRWGL", r"WAC_GLOBAL_E\d{3}[NS]\d{4}_100M\.IMG"
        ),
    }


class LROCWACColor(ODEInstrumentDataset):
    """LRO LROC WAC empirically-normalized 7-color reflectance, searched via
    PDS ODE (product type ``MDREMP``): one product per band (321, 360, 415,
    566, 604, 643, and 689 nm), each its own tiled family pinned to its
    64 px/degree tiling (other resolutions of the 643 nm band also exist
    under the same product type; the pattern excludes them). The composite
    ``3BAND`` product and the Hapke-photometrically-normalized ``MDRHAP``
    variant are not offered here. im-ldi PDS4 archive: opens the data
    ``.IMG`` file directly, never its ``.xml`` label (rule 1's PDS4-label
    lesson).
    """

    probe = "Lunar Reconnaissance Orbiter"
    instrument = "LROC WAC 7-color reflectance"
    ihid = "LRO"
    iid = "LROC"
    all_products = {
        "refl_321nm": ODEAsset("lroc_wac_refl_321nm", "MDREMP", r"WAC_EMP_321NM_.+_064P\.IMG"),
        "refl_360nm": ODEAsset("lroc_wac_refl_360nm", "MDREMP", r"WAC_EMP_360NM_.+_064P\.IMG"),
        "refl_415nm": ODEAsset("lroc_wac_refl_415nm", "MDREMP", r"WAC_EMP_415NM_.+_064P\.IMG"),
        "refl_566nm": ODEAsset("lroc_wac_refl_566nm", "MDREMP", r"WAC_EMP_566NM_.+_064P\.IMG"),
        "refl_604nm": ODEAsset("lroc_wac_refl_604nm", "MDREMP", r"WAC_EMP_604NM_.+_064P\.IMG"),
        "refl_643nm": ODEAsset("lroc_wac_refl_643nm", "MDREMP", r"WAC_EMP_643NM_.+_064P\.IMG"),
        "refl_689nm": ODEAsset("lroc_wac_refl_689nm", "MDREMP", r"WAC_EMP_689NM_.+_064P\.IMG"),
    }


class IntersectionDataset(_WindowedDataset):
    """Coregistered channel stack of two datasets over their overlap.

    Owns the sampling loop: every drawn window is read from both children at
    the same bbox and the results are concatenated along the channel axis.
    Children must share ``resolution`` and ``patch_size``. Usually created
    with the ``&`` operator, which also nests: ``a & b & c``.

    Args:
        first: left dataset; its seed wins if both are set.
        second: right dataset.
    """

    def __init__(self, first: _WindowedDataset, second: _WindowedDataset) -> None:
        if (first.resolution, first.patch_size) != (second.resolution, second.patch_size):
            raise ValueError(
                "datasets must share resolution and patch_size, got "
                f"({first.resolution}, {first.patch_size}) and "
                f"({second.resolution}, {second.patch_size})"
            )
        west = max(first.bbox[0], second.bbox[0])
        south = max(first.bbox[1], second.bbox[1])
        east = min(first.bbox[2], second.bbox[2])
        north = min(first.bbox[3], second.bbox[3])
        if not (west < east and south < north):
            raise ValueError(f"bboxes do not overlap: {first.bbox} and {second.bbox}")
        self.first = first
        self.second = second
        self.bbox = (west, south, east, north)
        self.resolution = first.resolution
        self.patch_size = first.patch_size
        self.length = min(first.length, second.length)
        self.seed = first.seed if first.seed is not None else second.seed
        self.layers = first.layers + second.layers

    def read(self, bbox: BBox) -> dict:
        left = self.first.read(bbox)
        right = self.second.read(bbox)
        return {
            "image": torch.cat([left["image"], right["image"]]),
            "mask": torch.cat([left["mask"], right["mask"]]),
            "layers": left["layers"] + right["layers"],
            "bbox": bbox,
            "crs": left["crs"],
            "resolution": self.resolution,
        }
