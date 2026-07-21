"""Layer registry and discovery catalog for lunar data.

The registry (``LAYERS``) is the single place where user-facing layer
identifiers map to their provenance and their backing source (USGS ARD STAC,
PDS ODE, or a fixed mosaic URL). The catalog (``MOON``) arranges the same
information as a Body -> Probe -> Instrument hierarchy for discovery; its
nodes hold specs and dataset *classes*, never dataset instances. A probe may
also carry ``granules``: raw, non-map-projected datasets
(:mod:`astrofetch.moon.granules`) that fall outside the layer/product
contract entirely (see that module's docstring). Flat imports
(``from astrofetch.moon import KaguyaTC``) remain the primary path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from astrofetch.moon.datasets import (
    CRS,
    LOLA,
    LROCNACDTM,
    SLDEM2015,
    KaguyaTC,
    KaguyaTCImagery,
    LROCWACMosaic,
    MosaicAsset,
    ODEAsset,
    Product,
    _ProductDataset,
)
from astrofetch.moon.granules import M3, GranuleDataset, LROCNACRaw, LROCWACRaw


@dataclass(frozen=True)
class LayerSpec:
    """Provenance and read config for one user-facing layer."""

    name: str
    """Layer identifier used in sample dicts, e.g. ``"kaguya_tc_dtm"``."""

    probe: str
    """Probe (spacecraft) that acquired the data."""

    instrument: str
    """Instrument on the probe."""

    product: str
    """Product name within the instrument, e.g. ``"dtm"``."""

    source: str
    """Where this layer is read from: ``"stac"``, ``"ode"``, or ``"mosaic"``."""

    collection: str = ""
    """USGS ARD STAC collection id (``source == "stac"``)."""

    asset: str = ""
    """STAC asset key read from each item (``source == "stac"``)."""

    ihid: str = ""
    """PDS ODE instrument host id, e.g. ``"LRO"`` (``source == "ode"``)."""

    iid: str = ""
    """PDS ODE instrument id, e.g. ``"LROC"`` (``source == "ode"``)."""

    pt: str = ""
    """PDS ODE product type, e.g. ``"SDNDTM"`` (``source == "ode"``)."""

    pattern: str = ""
    """Filename pattern selecting the file within an ODE product
    (``source == "ode"``)."""

    file_type: str = ""
    """Required ODE file role for ``pattern`` to match against
    (``source == "ode"``); almost always ``"Product"``."""

    href: str = ""
    """Fixed archive URL (``source == "mosaic"``)."""


@dataclass(frozen=True)
class Instrument:
    """Catalog node: one instrument dataset, its products, and its class."""

    name: str
    products: dict[str, LayerSpec]
    dataset: type[_ProductDataset]
    """The dataset class (not an instance); construct it on demand."""


@dataclass(frozen=True)
class Probe:
    """Catalog node: one probe, the instrument datasets it carries, and any
    experimental raw-granule datasets (see :mod:`astrofetch.moon.granules`)."""

    name: str
    instruments: dict[str, Instrument]
    granules: dict[str, type[GranuleDataset]] = field(default_factory=dict)


@dataclass(frozen=True)
class Body:
    """Catalog root: one planetary body and the probes that observed it."""

    name: str
    crs: str
    probes: dict[str, Probe]


def _spec(dataset: type[_ProductDataset], product: str) -> LayerSpec:
    entry = dataset.all_products[product]
    common = {
        "name": entry.layer,
        "probe": dataset.probe,
        "instrument": dataset.instrument,
        "product": product,
    }
    if isinstance(entry, Product):
        # dataset is an InstrumentDataset subclass here, guaranteed by
        # all_products entries always matching the dataset's own asset kind;
        # getattr keeps this function's signature at the shared base type.
        collection = getattr(dataset, "collection", "")
        return LayerSpec(**common, source="stac", collection=collection, asset=entry.asset)
    if isinstance(entry, ODEAsset):
        # dataset is an ODEInstrumentDataset subclass here, same guarantee.
        ihid = getattr(dataset, "ihid", "")
        iid = getattr(dataset, "iid", "")
        return LayerSpec(
            **common,
            source="ode",
            ihid=ihid,
            iid=iid,
            pt=entry.pt,
            pattern=entry.pattern,
            file_type=entry.file_type,
        )
    if isinstance(entry, MosaicAsset):
        return LayerSpec(**common, source="mosaic", href=entry.href)
    raise TypeError(f"unknown asset spec type for {product!r}: {type(entry)!r}")  # pragma: no cover


LAYERS: dict[str, LayerSpec] = {
    spec.name: spec
    for spec in (
        _spec(KaguyaTC, "dtm"),
        _spec(KaguyaTC, "ortho"),
        _spec(KaguyaTCImagery, "image"),
        _spec(LROCNACDTM, "dtm"),
        _spec(LROCNACDTM, "ortho"),
        _spec(LROCNACDTM, "confidence"),
        _spec(LROCWACMosaic, "morphology"),
        _spec(LOLA, "dem"),
        _spec(SLDEM2015, "dem"),
    )
}


def _instrument(dataset: type[_ProductDataset]) -> Instrument:
    return Instrument(
        name=dataset.instrument,
        products={product: LAYERS[entry.layer] for product, entry in dataset.all_products.items()},
        dataset=dataset,
    )


MOON = Body(
    name="Moon",
    crs=CRS,
    probes={
        "kaguya": Probe(
            name=KaguyaTC.probe,
            instruments={
                "tc_dtm": _instrument(KaguyaTC),
                "tc_imagery": _instrument(KaguyaTCImagery),
            },
        ),
        "lro": Probe(
            name=LROCNACDTM.probe,
            instruments={
                "nac_dtm": _instrument(LROCNACDTM),
                "wac_mosaic": _instrument(LROCWACMosaic),
                "lola": _instrument(LOLA),
                "sldem2015": _instrument(SLDEM2015),
            },
            granules={
                "nac_raw": LROCNACRaw,
                "wac_raw": LROCWACRaw,
            },
        ),
        "chandrayaan1": Probe(
            name=M3.probe,
            instruments={},
            granules={"m3": M3},
        ),
    },
)
"""Discovery catalog for the Moon: enumerate probes, instruments, products,
and (where a probe has any) experimental raw-granule datasets."""
