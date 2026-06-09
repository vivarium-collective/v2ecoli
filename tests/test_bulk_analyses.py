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
    ("protein_counts_validation", "multiseed"),
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


# ---------------------------------------------------------------------------
# Unit tests for validation_data loader
# ---------------------------------------------------------------------------

class _FakeMonomerData:
    """Minimal sim_data.process.translation.monomer_data stub for loader tests."""

    def __init__(self, rows):
        self.struct_array = np.array(
            rows,
            dtype=[("id", "U80"), ("cistron_id", "U80")],
        )

    def __getitem__(self, key):
        return self.struct_array[key]


class _FakeSimData:
    class _process:
        class translation:
            pass
    def __init__(self, rows):
        self.process = self._process()
        self.process.translation = self._process.translation()
        self.process.translation.monomer_data = _FakeMonomerData(rows)


def _make_fake_sim_data():
    """Build a minimal fake sim_data that covers the first few genes in each TSV."""
    # These gene IDs appear in both TSVs (first few rows).
    gene_rows = [
        ("ALANINE-RACEMASE-BIOSYNTHETIC-MONOMER[c]", "EG10001_RNA"),  # alr
        ("YAAAD-MONOMER[c]", "EG10011_RNA"),                           # yaaA
        ("CYDC-MONOMER[c]", "EG10012_RNA"),                            # cydC
        ("MODAB-MONOMER[p]", "EG10002_RNA"),                           # modB (wisniewski)
        ("CYSZ-MONOMER[c]", "EG10003_RNA"),                            # cysZ (wisniewski)
    ]
    return _FakeSimData(gene_rows)


def test_validation_data_loader_schmidt_non_empty():
    """Loader produces non-empty schmidt2015Data with required keys."""
    from v2ecoli.library.validation_data import build_validation_data

    sd = _make_fake_sim_data()
    vd = build_validation_data(sd)

    assert hasattr(vd, "protein")
    assert hasattr(vd.protein, "schmidt2015Data")
    s = vd.protein.schmidt2015Data
    assert len(s) > 0, "schmidt2015Data should be non-empty"
    assert "monomerId" in s.dtype.names
    assert "glucoseCounts" in s.dtype.names
    # All returned monomer IDs should be non-empty strings
    assert all(m != "" for m in s["monomerId"])


def test_validation_data_loader_wisniewski_non_empty():
    """Loader produces non-empty wisniewski2014Data with required keys."""
    from v2ecoli.library.validation_data import build_validation_data

    sd = _make_fake_sim_data()
    vd = build_validation_data(sd)

    assert hasattr(vd.protein, "wisniewski2014Data")
    w = vd.protein.wisniewski2014Data
    assert len(w) > 0, "wisniewski2014Data should be non-empty"
    assert "monomerId" in w.dtype.names
    assert "avgCounts" in w.dtype.names
    assert all(m != "" for m in w["monomerId"])


def test_validation_data_loader_wisniewski_averages_replicates():
    """avgCounts is the mean of the three replicate columns."""
    from v2ecoli.library.validation_data import build_validation_data, _read_tsv

    sd = _make_fake_sim_data()
    vd = build_validation_data(sd)

    # Look up one gene we know is in the fake sim_data (EG10001 = alr)
    w = vd.protein.wisniewski2014Data
    gene_monomer = "ALANINE-RACEMASE-BIOSYNTHETIC-MONOMER[c]"
    hits = np.where(w["monomerId"] == gene_monomer)[0]
    if len(hits) == 0:
        pytest.skip("alr not in wisniewski TSV (gene may be absent)")
    idx = hits[0]

    # Verify against raw TSV
    rows = _read_tsv("wisniewski2014_supp2.tsv")
    for row in rows:
        if row["EcoCycID"] == "EG10001":
            expected = (float(row["rep1"]) + float(row["rep2"]) + float(row["rep3"])) / 3
            np.testing.assert_almost_equal(
                w["avgCounts"][idx], expected, decimal=5,
                err_msg="wisniewski avgCounts should be mean(rep1, rep2, rep3)"
            )
            break
