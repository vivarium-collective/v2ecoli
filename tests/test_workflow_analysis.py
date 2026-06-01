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


def test_analysis_registry_maps_names_to_steps():
    from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY, MassFractionSummary
    assert ANALYSIS_REGISTRY["mass_fraction_summary"] is MassFractionSummary
    from v2ecoli.workflow.analysis import AnalysisStep
    for name, cls in ANALYSIS_REGISTRY.items():
        assert issubclass(cls, AnalysisStep)
        assert cls.name == name


def test_daughter_mass_symmetry():
    from v2ecoli.workflow.analysis import DaughterMassSymmetry
    from bigraph_schema import allocate_core
    step = DaughterMassSymmetry({}, core=allocate_core())
    out = step.analyze([{"newborn_dry_mass": 300.0}, {"newborn_dry_mass": 360.0}])
    assert out["n_sisters"] == 2
    assert abs(out["mass_asymmetry"] - (60.0 / 660.0)) < 1e-9
    one = step.analyze([{"newborn_dry_mass": 300.0}])
    assert one["n_sisters"] == 1 and "skipped" in one


def test_mass_growth_across_generations():
    from v2ecoli.workflow.analysis import MassGrowthAcrossGenerations
    from bigraph_schema import allocate_core
    step = MassGrowthAcrossGenerations({}, core=allocate_core())
    rows = [
        {"generation": 1, "newborn_dry_mass": 350.0, "final_dry_mass": 700.0, "division_time": 2500.0},
        {"generation": 0, "newborn_dry_mass": 380.0, "final_dry_mass": 702.0, "division_time": 2400.0},
    ]
    out = step.analyze(rows)
    assert out["n_generations"] == 2
    assert [g["generation"] for g in out["per_generation"]] == [0, 1]
    assert abs(out["per_generation"][0]["fold_change"] - (702.0 / 380.0)) < 1e-9
    assert abs(out["mean_division_time"] - 2450.0) < 1e-9


def test_doubling_time_distribution():
    from v2ecoli.workflow.analysis import DoublingTimeDistribution
    from bigraph_schema import allocate_core
    step = DoublingTimeDistribution({}, core=allocate_core())
    rows = [
        {"divided": True, "division_time": 2400.0, "final_dry_mass": 700.0},
        {"divided": True, "division_time": 2600.0, "final_dry_mass": 720.0},
        {"divided": False, "division_time": 4000.0, "final_dry_mass": 500.0},
    ]
    out = step.analyze(rows)
    assert out["n_cells"] == 3 and out["n_divided"] == 2
    assert abs(out["doubling_time_mean"] - 2500.0) < 1e-9
    assert out["doubling_time_std"] > 0
    assert abs(out["final_dry_mass_mean"] - (700.0 + 720.0 + 500.0) / 3) < 1e-6


def test_metric_across_variants():
    from v2ecoli.workflow.analysis import MetricAcrossVariants
    from bigraph_schema import allocate_core
    step = MetricAcrossVariants({}, core=allocate_core())
    rows = [
        {"variant": 0, "divided": True, "division_time": 2400.0, "final_dry_mass": 700.0},
        {"variant": 0, "divided": True, "division_time": 2600.0, "final_dry_mass": 720.0},
        {"variant": 1, "divided": True, "division_time": 3000.0, "final_dry_mass": 650.0},
    ]
    pv = {d["variant"]: d for d in step.analyze(rows)["per_variant"]}
    assert pv[0]["n_cells"] == 2 and abs(pv[0]["mean_division_time"] - 2500.0) < 1e-9
    assert pv[1]["n_cells"] == 1 and abs(pv[1]["mean_final_dry_mass"] - 650.0) < 1e-9
