"""General declarable-observable hook: a study can surface ANY genuine-vEcoli
listener leaf as a measurement (dotted "group.leaf" under listeners) with no
code change — the general counterpart to observable_bulk_ids / exchange_fluxes.
"""
from v2ecoli.library.vivarium_ecoli_engine import (
    _select_observables, _merge_listener_schema, _deep_merge)


def test_select_reads_nested_listener_leaves():
    lst = {"rna_synth_prob": {"total_rna_init": 42.0}, "mass": {"cell_mass": 100.0}}
    out = _select_observables(lst, ["rna_synth_prob.total_rna_init", "listeners.mass.cell_mass"])
    assert out == {"rna_synth_prob": {"total_rna_init": 42.0}, "mass": {"cell_mass": 100.0}}


def test_select_missing_path_is_zero():
    assert _select_observables({"mass": {}}, ["mass.cell_mass", "nope.x"]) == {
        "mass": {"cell_mass": 0.0}, "nope": {"x": 0.0}}


def test_select_empty():
    assert _select_observables({"a": {"b": 1.0}}, []) == {}
    assert _select_observables(None, ["a.b"]) == {"a": {"b": 0.0}}


def test_merge_listener_schema_declares_overwrite_leaves():
    sch = {"mass": {"cell_mass": "overwrite[float]"}}
    _merge_listener_schema(sch, ["rna_synth_prob.total_rna_init", "mass.dry_mass"])
    assert sch["rna_synth_prob"]["total_rna_init"] == "overwrite[float]"
    assert sch["mass"]["dry_mass"] == "overwrite[float]"        # merges, keeps cell_mass
    assert sch["mass"]["cell_mass"] == "overwrite[float]"


def test_deep_merge():
    dst = {"listeners": {"mass": {"cell_mass": 1.0}}}
    _deep_merge(dst, {"listeners": {"rna_synth_prob": {"x": 2.0}}})
    assert dst["listeners"]["mass"]["cell_mass"] == 1.0
    assert dst["listeners"]["rna_synth_prob"]["x"] == 2.0
