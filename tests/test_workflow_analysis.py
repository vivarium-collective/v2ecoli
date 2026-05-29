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
