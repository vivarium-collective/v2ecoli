import pytest
from bigraph_schema import allocate_core

from v2ecoli.workflow.analysis import (
    AnalysisStep, ANALYSIS_SCALES, MassFractionSummary)


def _core():
    return allocate_core()


def test_five_scales_registered():
    assert set(ANALYSIS_SCALES) == {
        "single", "multidaughter", "multigeneration", "multiseed", "multivariant"}


def test_mass_fraction_summary_is_single_scale():
    assert MassFractionSummary.scale == "single"
    assert issubclass(MassFractionSummary, AnalysisStep)


def test_mass_fraction_summary_computes_fractions():
    # core=None is rejected by bigraph_schema.Edge; use a real TypeSystem instead.
    step = MassFractionSummary({}, core=_core())
    rows = [
        {"listeners": {"mass": {"dry_mass": 100.0, "protein_mass": 60.0,
                                "rRna_mass": 20.0, "dna_mass": 20.0}}},
        {"listeners": {"mass": {"dry_mass": 200.0, "protein_mass": 120.0,
                                "rRna_mass": 40.0, "dna_mass": 40.0}}},
    ]
    out = step.analyze(rows)
    assert abs(out["protein_fraction_mean"] - 0.6) < 1e-9
    assert abs(out["rRna_fraction_mean"] - 0.2) < 1e-9
    assert out["n_rows"] == 2


def test_update_passes_results_through_to_analyze():
    step = MassFractionSummary({}, core=allocate_core())
    rows = [{"listeners": {"mass": {"dry_mass": 100.0, "protein_mass": 50.0,
                                    "rRna_mass": 10.0, "dna_mass": 5.0}}}]
    out = step.update({"results": rows})
    assert "analysis" in out
    assert out["analysis"]["n_valid_rows"] == 1
    assert abs(out["analysis"]["protein_fraction_mean"] - 0.5) < 1e-9


def test_empty_rows_uniform_contract():
    step = MassFractionSummary({}, core=allocate_core())
    out = step.analyze([])
    assert out == {"n_rows": 0, "n_valid_rows": 0,
                   "protein_fraction_mean": 0.0,
                   "rRna_fraction_mean": 0.0, "dna_fraction_mean": 0.0}


def test_zero_dry_mass_rows_excluded_from_valid_count():
    step = MassFractionSummary({}, core=allocate_core())
    rows = [
        {"listeners": {"mass": {"dry_mass": 0.0, "protein_mass": 0.0}}},
        {"listeners": {"mass": {"dry_mass": 100.0, "protein_mass": 40.0,
                                "rRna_mass": 0.0, "dna_mass": 0.0}}},
    ]
    out = step.analyze(rows)
    assert out["n_rows"] == 2
    assert out["n_valid_rows"] == 1
    assert abs(out["protein_fraction_mean"] - 0.4) < 1e-9


def test_unimplemented_analyze_propagates():
    # A subclass that doesn't implement analyze must raise (not silently
    # return {}) when invoked.
    from v2ecoli.workflow.analysis import AnalysisStep
    class Incomplete(AnalysisStep):
        name = "incomplete_analysis"
        scale = "single"
    step = Incomplete({}, core=allocate_core())
    with pytest.raises(NotImplementedError):
        step.invoke({"results": []})
