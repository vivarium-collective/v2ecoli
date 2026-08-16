"""The equivalence grade step joins a pinned v1 reference to a v2 candidate.

Until this script existed, ``grade_card`` was called only from tests and nothing
read ``vecoli_reference.json``, so the committed verdict was hand-emitted and
drifted from the reference beside it undetected. These tests pin the properties
that make that drift impossible to repeat: the verdict records which reference it
graded, it refuses to grade across conditions, and it fails loudly rather than
silently when an input is missing.

Inputs are synthesized, so this runs in a second rather than needing a real sweep.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

_GRADE = pathlib.Path(__file__).resolve().parents[1] / "scripts" / \
    "grade_vecoli_equivalence.py"


@pytest.fixture(scope="module")
def grade():
    spec = importlib.util.spec_from_file_location("_grade_vecoli_eq", _GRADE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A repeating pattern so any multiple-of-3 sample has EXACTLY this mean — the two
# sides then share a mean by construction rather than approximately, which keeps
# the within_tol assertion about the grader and not about fixture arithmetic.
_BASE = [0.0178, 0.0180, 0.0182]          # mean 0.0180
_REF_N, _CAND_N = 111, 21                 # both multiples of 3


def _values(n):
    return [_BASE[i % 3] for i in range(n)]


def _reference(condition="basal", n=_REF_N):
    """A minimal pinned reference: one ttest axis, with a stimulus stamp."""
    return {
        "title": f"Basal-condition population phenotype ({condition})",
        "status": "populated",
        "stimulus": {
            "reference_model": "vEcoli (v1)",
            "measured_model": "v2ecoli (v2)",
            "blessed_model_ref": "6a2d3402",
            "ensemble": f"vEcoli ensemble, {n} cells",
            "condition": condition,
            "generation_lower_bound": 3,
        },
        "axes": {
            "composition.dna_fraction": {
                "group": "Composition",
                "label": "DNA / dry weight",
                "units": "g/gDW",
                "criterion": {"type": "ttest", "p_min": 0.05,
                              "within_pct": 0.05, "mismatch_pct": 0.10,
                              "ref_values": _values(n)},
            },
        },
    }


def _analysis(values):
    """An analysis.json holding one multiseed population_phenotype_basal result."""
    import statistics
    n = len(values)
    return {"multiseed": {"population_phenotype_basal": {"variant=0": {
        "n_cells": n,
        "generation_lower_bound": 3,
        "sim_health": {"n_total": n, "n_divided": n, "n_failed": 0},
        "composition": {"dna_fraction": {
            "values": list(values),
            "mean": sum(values) / n,
            "std": statistics.pstdev(values) if n > 1 else 0.0,
            "cv": 0.0, "n": n}},
    }}}}


def _write(tmp_path, condition="basal", cand=None, ref=None):
    sweep = tmp_path / "sweep"
    sweep.mkdir(parents=True, exist_ok=True)
    cand = cand if cand is not None else _values(_CAND_N)
    (sweep / "analysis.json").write_text(json.dumps(_analysis(cand)))
    rpath = tmp_path / "ref.json"
    rpath.write_text(json.dumps(ref if ref is not None else _reference(condition)))
    return sweep, rpath


def _run(grade, monkeypatch, sweep, rpath, out, condition="basal", extra=()):
    argv = ["grade", "--sweep-dir", str(sweep), "--condition", condition,
            "--reference", str(rpath), "--model-ref", "cafe1234",
            "--gen-lb", "3", "--out", str(out), "--skip-vectors", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    grade.main()
    return json.loads(pathlib.Path(out).read_text(encoding="utf-8"))


def test_default_paths_follow_the_condition(grade):
    """Reference read and verdict written both live in the condition's card dir,
    parallel to pin_vecoli_equivalence_reference's own default."""
    for cond in ["basal", "acetate", "succinate", "no_oxygen", "with_aa"]:
        assert grade._default_reference(cond) == (
            f"docs/report_cards/population_phenotype_{cond}/vs_vecoli/"
            f"vecoli_reference.json")
        assert grade._default_out(cond) == (
            f"docs/report_cards/population_phenotype_{cond}/vs_vecoli/"
            f"report_card_verdict.json")


def test_verdict_records_the_reference_it_graded_against(
        grade, tmp_path, monkeypatch):
    """The defect this script exists to prevent.

    The pre-existing verdict carried only `reference_model: "vEcoli (v1)"` — no
    reference commit, sweep or n — so it could describe a different ensemble than
    the reference file beside it with nothing able to surface the disagreement.
    """
    sweep, rpath = _write(tmp_path)
    v = _run(grade, monkeypatch, sweep, rpath, tmp_path / "v.json")
    p = v["provenance"]
    assert p["reference_model_ref"] == "6a2d3402"
    assert p["reference_ensemble"] == f"vEcoli ensemble, {_REF_N} cells"
    assert p["reference_condition"] == "basal"
    assert p["reference_n_by_axis"]["composition.dna_fraction"] == _REF_N
    assert p["reference_n_min"] == _REF_N
    # and the candidate side, so both halves of the comparison are stamped
    assert p["candidate_n_cells"] == _CAND_N
    assert p["candidate_generation_lower_bound"] == 3
    assert p["candidate_sim_health"]["n_divided"] == _CAND_N


def test_refuses_to_grade_a_candidate_against_another_conditions_reference(
        grade, tmp_path, monkeypatch):
    """Two different stimuli are not an equivalence comparison. A silent
    cross-condition grade would look exactly like a real verdict."""
    sweep, rpath = _write(tmp_path, ref=_reference("acetate"))
    with pytest.raises(SystemExit, match="pinned for condition"):
        _run(grade, monkeypatch, sweep, rpath, tmp_path / "v.json",
             condition="basal")


def test_missing_analysis_names_the_cause_rather_than_KeyError(
        grade, tmp_path, monkeypatch):
    """Raw parquet does not carry the per-cell scalar axes; say so."""
    sweep = tmp_path / "empty"
    sweep.mkdir()
    rpath = tmp_path / "ref.json"
    rpath.write_text(json.dumps(_reference()))
    with pytest.raises(SystemExit, match="analysis runner"):
        _run(grade, monkeypatch, sweep, rpath, tmp_path / "v.json")


def test_missing_reference_points_at_the_pin_script(grade, tmp_path, monkeypatch):
    sweep, _ = _write(tmp_path)
    with pytest.raises(SystemExit, match="pin_vecoli_equivalence_reference"):
        _run(grade, monkeypatch, sweep, tmp_path / "absent.json",
             tmp_path / "v.json")


def test_grades_the_candidate_and_emits_the_v1_verdict_schema(
        grade, tmp_path, monkeypatch):
    """A candidate matching the reference grades within_tol; a shifted one does
    not — so the script is actually grading, not just transcribing."""
    sweep, rpath = _write(tmp_path)
    same = _run(grade, monkeypatch, sweep, rpath, tmp_path / "same.json")
    assert same["schema"] == "report_card_verdict/v1"
    assert same["model_ref"] == "cafe1234"
    assert same["condition"] == "basal"
    assert same["groups"]["composition"]["verdict"] == "within_tol"
    assert same["overall"] == "within_tol"

    shifted = [v * 1.5 for v in _values(_CAND_N)]
    sweep2, rpath2 = _write(tmp_path / "b", cand=shifted)
    bad = _run(grade, monkeypatch, sweep2, rpath2, tmp_path / "bad.json")
    assert bad["groups"]["composition"]["verdict"] == "mismatch"
    assert bad["overall"] == "mismatch"


def test_skip_vectors_leaves_vector_axes_ungraded_rather_than_absent(
        grade, tmp_path, monkeypatch):
    """--skip-vectors is a plumbing check: the omics/flux axes must read as
    ungraded (honest), never silently vanish from the verdict."""
    ref = _reference()
    ref["axes"]["omics.transcriptome"] = {
        "group": "Gene expression", "label": "Transcriptome",
        "criterion": {"type": "r2", "r2_min": 0.99, "r2_drift": 0.95,
                      "ref_vector": [1.0, 2.0, 3.0]},
    }
    sweep, rpath = _write(tmp_path, ref=ref)
    v = _run(grade, monkeypatch, sweep, rpath, tmp_path / "v.json")
    ge = v["groups"]["gene_expression"]
    assert [a["id"] for a in ge["axes"]] == ["omics.transcriptome"]
    assert ge["verdict"] == "ungraded"
