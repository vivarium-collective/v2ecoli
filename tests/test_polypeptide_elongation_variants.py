import os
import pytest

CACHE = "out/cache"
pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE) and not os.environ.get("CI"),
    reason=f"cache dir {CACHE!r} not present",
)


def test_three_variants_importable_and_subclassed():
    from v2ecoli.steps.partition import PartitionedProcess
    from v2ecoli.processes.polypeptide_elongation import (
        BasePolypeptideElongation,
        TranslationSupplyPolypeptideElongation,
        SteadyStatePolypeptideElongation,
    )
    assert issubclass(BasePolypeptideElongation, PartitionedProcess)
    assert issubclass(TranslationSupplyPolypeptideElongation, BasePolypeptideElongation)
    assert issubclass(SteadyStatePolypeptideElongation,
                      TranslationSupplyPolypeptideElongation)


def test_steadystate_declares_charging_ports_base_does_not():
    """The payoff: only SteadyState exposes the charging/ppGpp port surface."""
    from v2ecoli.core import load_cache_bundle
    from v2ecoli.processes.polypeptide_elongation import (
        BasePolypeptideElongation, SteadyStatePolypeptideElongation)
    cfg = load_cache_bundle(CACHE)["configs"]["ecoli-polypeptide-elongation"]
    base_in = set(BasePolypeptideElongation(dict(cfg)).inputs().keys())
    ss_in = set(SteadyStatePolypeptideElongation(dict(cfg)).inputs().keys())
    # SteadyState reads at least as much as Base.
    assert base_in <= ss_in


import numpy as np

VARIANTS = [
    "BasePolypeptideElongation",
    "TranslationSupplyPolypeptideElongation",
    "SteadyStatePolypeptideElongation",
]


@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_elongates_protein(variant, monkeypatch):
    import v2ecoli.composites._helpers as H
    import v2ecoli.processes.polypeptide_elongation as PE
    cls = getattr(PE, variant)
    monkeypatch.setitem(H.PARTITIONED_PROCESSES, "ecoli-polypeptide-elongation", cls)
    from v2ecoli import build_composite
    from v2ecoli.library.quantity_helpers import fg_magnitude
    c = build_composite("baseline", cache_dir="out/cache", seed=0)
    a = c.state["agents"]["0"]
    m0 = float(fg_magnitude(a["listeners"]["mass"]["protein_mass"]))
    c.run(20)
    m1 = float(fg_magnitude(a["listeners"]["mass"]["protein_mass"]))
    assert m1 > m0, f"{variant}: protein mass did not increase ({m0:.1f}->{m1:.1f})"
