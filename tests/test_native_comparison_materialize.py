"""Phase B, Task 2: the workbench-NATIVE single-study-per-config materializer.

Re-models the comparison from PAIRED studies (two per config, Task 4 of the
earlier phase) to the workbench-native model: ONE study per config, with the
candidate (``ecoli_baseline``) as the study's ``baseline`` and the reference
(``vecoli``) as a ``variants`` entry in the SAME study. The candidate's run
registers with ``sim_name = <config>`` (the v4 slug convention); the
variant's run registers with ``sim_name = "reference"``.

Hermetic: no engines run, no ParCa cache, no upstream vEcoli fork checkout --
spec/wiring assertions only, same style as
``test_comparison_investigation_materialize.py`` (the OLD paired path, which
this file does NOT touch or replace -- ``to_study_specs``/``matrix_analysis_entry``
stay intact for parallel-safety, per the task brief).
"""
from __future__ import annotations

import v2ecoli.workflow.analyses  # noqa: F401 -- registers comparison_cards/comparison_matrix
from v2ecoli.workflow.comparison_materialize import (
    CANDIDATE_COMPOSITE, PARCA_PREP_COMPOSITE, REFERENCE_COMPOSITE,
    REFERENCE_VARIANT_NAME, materialize_comparison,
    to_native_investigation_analyses, to_native_study_specs)
from v2ecoli.workflow.parca_study import PARCA_STUDY_NAME


def _comparison_block(ve_cache: str, configs=None) -> dict:
    return {
        "reference": {"repo": "/fake/vecoli-fork", "kind": "vecoli"},
        "v2_cache": "out/cache_full",
        "ve_cache": ve_cache,
        "defaults": {"seeds": 1, "generations": 1, "cards": ["summary", "standard"]},
        "configs": configs or [{"name": "basal", "condition": "basal"}],
    }


def _materialize(tmp_path, configs=None):
    ve_cache = tmp_path / "vecoli_parca"
    ve_cache.mkdir()
    (ve_cache / "simData.cPickle").write_bytes(b"fake-simdata")
    block = _comparison_block(str(ve_cache), configs=configs)
    materialized = materialize_comparison(block, invest_name="test-comparison")
    return materialized, str(ve_cache)


# --- the parca study --------------------------------------------------------

def test_parca_study_uses_parca_prep_with_real_cache_dirs(tmp_path):
    materialized, ve_cache = _materialize(tmp_path)
    specs = to_native_study_specs(materialized)

    parca_spec = specs[PARCA_STUDY_NAME]
    assert parca_spec["name"] == PARCA_STUDY_NAME
    assert len(parca_spec["baseline"]) == 1
    baseline = parca_spec["baseline"][0]
    assert baseline["name"] == PARCA_STUDY_NAME
    assert baseline["composite"] == PARCA_PREP_COMPOSITE
    # candidate_cache_dir must be the REAL cache dir, not left empty.
    assert baseline["params"]["candidate_cache_dir"] == "out/cache_full"
    assert baseline["params"]["candidate_cache_dir"]
    assert baseline["params"]["reference_cache_dir"] == ve_cache
    assert baseline["params"]["reference_repo"] == "/fake/vecoli-fork"

    # The parca study is the root of the DAG -- no pipeline_gate of its own.
    assert "pipeline_gate" not in parca_spec


# --- one study per config ----------------------------------------------------

def test_one_study_per_config_keyed_by_config_name(tmp_path):
    materialized, _ = _materialize(
        tmp_path, configs=[{"name": "basal", "condition": "basal"},
                           {"name": "with_aa", "condition": "with_aa"}])
    specs = to_native_study_specs(materialized)

    assert set(specs) == {PARCA_STUDY_NAME, "basal", "with_aa"}


def test_study_baseline_is_candidate_ecoli_baseline(tmp_path):
    materialized, ve_cache = _materialize(tmp_path)
    specs = to_native_study_specs(materialized)
    study = specs["basal"]

    assert study["name"] == "basal"
    assert len(study["baseline"]) == 1
    baseline = study["baseline"][0]
    assert baseline["name"] == "basal"
    assert baseline["composite"] == CANDIDATE_COMPOSITE
    # Reused verbatim from the existing paired emission -- no science change.
    assert baseline["params"]["match_condition"] == "basal"
    assert baseline["params"]["match_simdata"] == f"{ve_cache}/simData.cPickle"
    assert baseline["params"]["cache_dir"] == "out/cache_full"
    assert "seed" in baseline["params"]


def test_study_variant_is_reference_vecoli(tmp_path):
    materialized, ve_cache = _materialize(tmp_path)
    specs = to_native_study_specs(materialized)
    study = specs["basal"]

    assert len(study["variants"]) == 1
    variant = study["variants"][0]
    assert variant["name"] == "reference"
    assert variant["composite"] == REFERENCE_COMPOSITE
    # Reused verbatim from the existing paired emission -- no science change.
    assert variant["params"]["reference_repo"] == "/fake/vecoli-fork"
    assert variant["params"]["condition"] == "basal"
    assert variant["params"]["cache_dir"] == ve_cache
    assert "seed" in variant["params"]
    assert "fork_config" in variant["params"]


def test_with_aa_config_still_threads_its_own_condition(tmp_path):
    """Regression guard carried over from the paired-materializer tests: a
    non-basal config's candidate/variant must carry ITS OWN condition, not a
    hardcoded 'basal'."""
    materialized, _ = _materialize(
        tmp_path, configs=[{"name": "with_aa", "condition": "with_aa"}])
    specs = to_native_study_specs(materialized)
    study = specs["with_aa"]

    assert study["baseline"][0]["params"]["match_condition"] == "with_aa"
    assert study["variants"][0]["params"]["condition"] == "with_aa"


# --- per-study analyses: comparison_cards -----------------------------------

def test_study_analyses_wire_comparison_cards_to_config_and_reference_sim_names(tmp_path):
    materialized, _ = _materialize(tmp_path)
    specs = to_native_study_specs(materialized)
    study = specs["basal"]

    assert len(study["analyses"]) == 1
    entry = study["analyses"][0]
    assert entry["name"] == "comparison_cards"
    # candidate_run/reference_run MUST be the sim_names the substrate assigns:
    # baseline = the study slug (<config>), variant = "reference".
    assert entry["params"]["candidate_run"] == "basal"
    assert entry["params"]["reference_run"] == REFERENCE_VARIANT_NAME
    assert entry["params"]["reference_run"] == "reference"
    assert entry["params"]["seeds"] == 1
    assert entry["params"]["cards"] == ["summary", "standard"]


# --- pipeline_gate -----------------------------------------------------------

def test_study_pipeline_gate_depends_on_parca(tmp_path):
    materialized, _ = _materialize(
        tmp_path, configs=[{"name": "basal", "condition": "basal"},
                           {"name": "with_aa", "condition": "with_aa"}])
    specs = to_native_study_specs(materialized)

    for name in ("basal", "with_aa"):
        prereqs = specs[name]["pipeline_gate"]["prerequisites"]
        assert prereqs == [{"study": PARCA_STUDY_NAME, "relation": "leads-to"}]


# --- comparative_visualizations ----------------------------------------------

def test_comparative_visualizations_reference_config_and_reference_sim_names(tmp_path):
    materialized, _ = _materialize(tmp_path)
    specs = to_native_study_specs(materialized)
    study = specs["basal"]

    cvs = study["comparative_visualizations"]
    assert len(cvs) >= 1
    for cv in cvs:
        assert cv["name"]
        assert cv["observable_path"]
        runs = cv["runs"]
        sim_names = {r["sim_name"] for r in runs}
        assert sim_names == {"basal", "reference"}
        for r in runs:
            assert r["label"]


def test_comparative_visualizations_differ_per_config_by_sim_name(tmp_path):
    materialized, _ = _materialize(
        tmp_path, configs=[{"name": "basal", "condition": "basal"},
                           {"name": "with_aa", "condition": "with_aa"}])
    specs = to_native_study_specs(materialized)

    basal_names = {r["sim_name"] for cv in specs["basal"]["comparative_visualizations"]
                   for r in cv["runs"]}
    with_aa_names = {r["sim_name"] for cv in specs["with_aa"]["comparative_visualizations"]
                     for r in cv["runs"]}
    assert basal_names == {"basal", "reference"}
    assert with_aa_names == {"with_aa", "reference"}


# --- investigation-level comparison_matrix analysis --------------------------

def test_native_investigation_analyses_names_config_slugs_not_run_tokens(tmp_path):
    materialized, _ = _materialize(
        tmp_path, configs=[{"name": "basal", "condition": "basal"},
                           {"name": "with_aa", "condition": "with_aa"}])
    entries = to_native_investigation_analyses(materialized)

    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "comparison_matrix"
    assert entry["name"] in __import__(
        "v2ecoli.workflow.analysis", fromlist=["ANALYSIS_REGISTRY"]).ANALYSIS_REGISTRY
    # The config slug LIST -- not the old "<candidate_run>::comparison_cards"
    # token map (Task 4 reads verdicts from disk by these slugs).
    assert entry["params"]["config_studies"] == ["basal", "with_aa"]
    for slug in entry["params"]["config_studies"]:
        assert "::" not in slug
