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
