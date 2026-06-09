"""Registration tests for the bulk-ported native Analysis subclasses.

These assert each ported analysis is registered in ANALYSIS_REGISTRY, is an
``Analysis`` subclass, and carries the expected ``scale``.  They are NOT gated
on external parquet/sim_data fixtures, so they run in CI; the data-dependent
behaviour is exercised by the smoke tests in the porting workflow.
"""

import numpy as np
import pandas as pd
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
    ("ptools_rna_multiseed", "multiseed"),
    ("ptools_rxns_multiseed", "multiseed"),
    ("ptools_proteins_multiseed", "multiseed"),
    ("dummy", "multivariant"),
    ("mass_fraction_summary_view", "single"),
    ("replication", "multigeneration"),
    ("cell_mass", "multivariant"),
    ("doubling_time_hist", "multivariant"),
    ("doubling_time_line", "multivariant"),
    ("ribosome_production", "multigeneration"),
    ("ribosome_components", "multigeneration"),
    ("ribosome_usage", "multigeneration"),
    ("new_gene_counts", "multigeneration"),
    ("average_monomer_counts", "multivariant"),
    ("subgenerational_expression_table", "multiseed"),
]


@pytest.mark.parametrize("name,scale", PORTED)
def test_ported_analysis_registered(name, scale):
    registry, Analysis = _registry()
    assert name in registry, f"{name} not registered"
    cls = registry[name]
    assert issubclass(cls, Analysis)
    assert cls.scale == scale


# ---------------------------------------------------------------------------
# Unit test for cross-seed collapse helper
# ---------------------------------------------------------------------------

def test_collapse_cross_seed_elementwise():
    """Two 'seeds' at the same time: bulk__count should be element-wise summed,
    bulk__id should be first-preserved, and active_ribosome (scalar) should
    be plain-summed.
    """
    from v2ecoli.workflow.analyses._helpers import collapse_cross_seed

    df = pd.DataFrame({
        "time": [1.0, 1.0],
        "bulk__id": [["A", "B", "C"], ["A", "B", "C"]],
        "bulk__count": [[1, 2, 3], [10, 20, 30]],
        "active_ribosome": [5, 7],
    })

    result = collapse_cross_seed(df, id_cols=frozenset({"bulk__id"}))

    assert len(result) == 1, "should have exactly one row per unique time"
    assert list(result["bulk__id"].iloc[0]) == ["A", "B", "C"], "id col should be first-preserved"
    np.testing.assert_array_equal(
        result["bulk__count"].iloc[0],
        [11, 22, 33],
        err_msg="bulk__count should be element-wise summed across seeds",
    )
    assert result["active_ribosome"].iloc[0] == 12, "scalar col should be plain-summed"


def test_collapse_cross_seed_shape_mismatch_raises():
    """Mismatched array shapes must raise ValueError, never silently pad/truncate."""
    from v2ecoli.workflow.analyses._helpers import collapse_cross_seed

    df = pd.DataFrame({
        "time": [1.0, 1.0],
        "bulk__count": [[1, 2, 3], [10, 20]],  # different lengths
    })
    with pytest.raises(ValueError, match="Mismatched array shapes"):
        collapse_cross_seed(df)


def test_collapse_cross_seed_id_length_mismatch_raises():
    """id_cols with mismatched lengths must raise ValueError."""
    from v2ecoli.workflow.analyses._helpers import collapse_cross_seed

    df = pd.DataFrame({
        "time": [1.0, 1.0],
        "bulk__id": [["A", "B"], ["A", "B", "C"]],
    })
    with pytest.raises(ValueError, match="Mismatched length in id column"):
        collapse_cross_seed(df, id_cols=frozenset({"bulk__id"}))
