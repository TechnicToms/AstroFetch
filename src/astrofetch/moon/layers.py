"""Layer registry and discovery catalog for lunar data.

The registry (``LAYERS``) is the single place where user-facing layer
identifiers map to their provenance and their backing USGS ARD STAC collection
and asset. The catalog (``MOON``) arranges the same information as a
Body -> Probe -> Instrument hierarchy for discovery; its nodes hold specs and
dataset *classes*, never dataset instances. Flat imports
(``from astrofetch.moon import KaguyaTC``) remain the primary path.
"""

from __future__ import annotations

from dataclasses import dataclass

from astrofetch.moon.datasets import (
    CRS,
    InstrumentDataset,
    KaguyaTC,
    KaguyaTCImagery,
)


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

    collection: str
    """USGS ARD STAC collection id backing this layer."""

    asset: str
    """STAC asset key read from each item, e.g. ``"dtm"``."""


@dataclass(frozen=True)
class Instrument:
    """Catalog node: one instrument dataset, its products, and its class."""

    name: str
    products: dict[str, LayerSpec]
    dataset: type[InstrumentDataset]
    """The dataset class (not an instance); construct it on demand."""


@dataclass(frozen=True)
class Probe:
    """Catalog node: one probe and the instrument datasets it carries."""

    name: str
    instruments: dict[str, Instrument]


@dataclass(frozen=True)
class Body:
    """Catalog root: one planetary body and the probes that observed it."""

    name: str
    crs: str
    probes: dict[str, Probe]


def _spec(dataset: type[InstrumentDataset], product: str) -> LayerSpec:
    entry = dataset.all_products[product]
    return LayerSpec(
        name=entry.layer,
        probe=dataset.probe,
        instrument=dataset.instrument,
        product=product,
        collection=dataset.collection,
        asset=entry.asset,
    )


LAYERS: dict[str, LayerSpec] = {
    spec.name: spec
    for spec in (
        _spec(KaguyaTC, "dtm"),
        _spec(KaguyaTC, "ortho"),
        _spec(KaguyaTCImagery, "image"),
    )
}


def _instrument(dataset: type[InstrumentDataset]) -> Instrument:
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
    },
)
"""Discovery catalog for the Moon: enumerate probes, instruments, products."""
