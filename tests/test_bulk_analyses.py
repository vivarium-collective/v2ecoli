"""Registration tests for the bulk-ported native Analysis subclasses.

These assert each ported analysis is registered in ANALYSIS_REGISTRY, is an
``Analysis`` subclass, and carries the expected ``scale``.  They are NOT gated
on external parquet/sim_data fixtures, so they run in CI; the data-dependent
behaviour is exercised by the smoke tests in the porting workflow.
"""

import pytest

from v2ecoli.workflow.analyses import _wholecell_compat  # noqa: F401


def _registry():
    # Importing the package registers every ported analysis.
    import v2ecoli.workflow.analyses  # noqa: F401
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, Analysis
    return ANALYSIS_REGISTRY, Analysis


# (registry name, expected scale) for every analysis ported in this effort.
PORTED = [
    ("ptools_rna_multigeneration", "multigeneration"),
    ("ptools_rxns_multigeneration", "multigeneration"),
    ("ptools_proteins_multigeneration", "multigeneration"),
    ("dummy", "multivariant"),
    ("mass_fraction_summary_view", "single"),
    ("replication", "multigeneration"),
]


@pytest.mark.parametrize("name,scale", PORTED)
def test_ported_analysis_registered(name, scale):
    registry, Analysis = _registry()
    assert name in registry, f"{name} not registered"
    cls = registry[name]
    assert issubclass(cls, Analysis)
    assert cls.scale == scale
