"""Layer registry and discovery catalog for lunar data.

The registry (``LAYERS``) is the single place where user-facing layer
identifiers map to their provenance and, in Phase 1, USGS ARD STAC
collections. The catalog (``MOON``) arranges the same information as a
Body -> Probe -> Instrument hierarchy for discovery; its nodes hold specs and
dataset *classes*, never dataset instances. Flat imports
(``from astrofetch.moon import LROCWAC``) remain the primary path.
"""

from __future__ import annotations

from dataclasses import dataclass

from astrofetch.moon.datasets import CRS, LROCWAC, InstrumentDataset, KaguyaTC


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
    """USGS ARD STAC collection id; placeholder until the Phase 1 sampler."""


@dataclass(frozen=True)
class Instrument:
    """Catalog node: one instrument, its products, and its dataset class."""

    name: str
    products: dict[str, LayerSpec]
    dataset: type[InstrumentDataset]
    """The dataset class (not an instance); construct it on demand."""


@dataclass(frozen=True)
class Probe:
    """Catalog node: one probe and the instruments it carries."""

    name: str
    instruments: dict[str, Instrument]


@dataclass(frozen=True)
class Body:
    """Catalog root: one planetary body and the probes that observed it."""

    name: str
    crs: str
    probes: dict[str, Probe]


def _spec(dataset: type[InstrumentDataset], product: str, collection: str) -> LayerSpec:
    return LayerSpec(
        name=dataset.all_products[product],
        probe=dataset.probe,
        instrument=dataset.instrument,
        product=product,
        collection=collection,
    )


# Collection ids are placeholders until Phase 1 pins the real USGS ARD
# collections; endpoint URLs will live in data/endpoints.py, never here.
LAYERS: dict[str, LayerSpec] = {
    spec.name: spec
    for spec in (
        _spec(KaguyaTC, "dtm", "placeholder:kaguya_tc_dtm"),
        _spec(KaguyaTC, "ortho", "placeholder:kaguya_tc_ortho"),
        _spec(LROCWAC, "mosaic", "placeholder:lroc_wac_mosaic"),
    )
}


def _instrument(dataset: type[InstrumentDataset]) -> Instrument:
    return Instrument(
        name=dataset.instrument,
        products={product: LAYERS[layer] for product, layer in dataset.all_products.items()},
        dataset=dataset,
    )


MOON = Body(
    name="Moon",
    crs=CRS,
    probes={
        "kaguya": Probe(name=KaguyaTC.probe, instruments={"tc": _instrument(KaguyaTC)}),
        "lro": Probe(name=LROCWAC.probe, instruments={"lroc_wac": _instrument(LROCWAC)}),
    },
)
"""Discovery catalog for the Moon: enumerate probes, instruments, products."""
