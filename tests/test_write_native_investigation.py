"""Phase B, Task 3: write the native materialization (Task 2) to actual
workspace files -- ``<workspace>/studies/<name>/study.yaml`` per config (plus
the ``parca`` study) and ``<workspace>/investigations/<invest_slug>/
investigation.yaml`` -- so the investigation-as-composite substrate
(``vivarium_workbench.lib.investigation_execution``) can run it.

Hermetic: writes to ``tmp_path``, reloads with ``yaml.safe_load``, and
asserts the round-tripped shape. No engines run, no ParCa cache, no upstream
vEcoli fork checkout -- same style as ``test_native_comparison_materialize.py``
(Task 2), which this file builds on (reuses its fixture pattern) without
touching it.
"""
from __future__ import annotations

import yaml

import v2ecoli.workflow.analyses  # noqa: F401 -- registers comparison_cards/comparison_matrix
from v2ecoli.workflow.comparison_materialize import (
    CANDIDATE_COMPOSITE, PARCA_PREP_COMPOSITE, REFERENCE_COMPOSITE,
    REFERENCE_VARIANT_NAME, materialize_comparison, write_native_investigation)
from v2ecoli.workflow.parca_study import PARCA_STUDY_NAME

INVEST_SLUG = "wcm-comparison"


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


def _write(tmp_path, configs=None):
    materialized, ve_cache = _materialize(tmp_path, configs=configs)
    workspace = tmp_path / "workspace"
    result = write_native_investigation(materialized, workspace, INVEST_SLUG)
    return result, workspace, ve_cache


# --- returned paths exist -----------------------------------------------------

def test_returned_paths_exist(tmp_path):
    result, workspace, _ = _write(tmp_path)

    assert result["investigation_path"].exists()
    assert set(result["study_paths"]) == {PARCA_STUDY_NAME, "basal"}
    for path in result["study_paths"].values():
        assert path.exists()

    assert result["investigation_path"] == (
        workspace / "investigations" / INVEST_SLUG / "investigation.yaml")
    assert result["study_paths"]["basal"] == workspace / "studies" / "basal" / "study.yaml"
    assert result["study_paths"][PARCA_STUDY_NAME] == (
        workspace / "studies" / PARCA_STUDY_NAME / "study.yaml")


# --- per-study study.yaml round-trips -----------------------------------------

def test_parca_study_yaml_round_trips(tmp_path):
    result, _, ve_cache = _write(tmp_path)
    spec = yaml.safe_load(result["study_paths"][PARCA_STUDY_NAME].read_text())

    assert spec["name"] == PARCA_STUDY_NAME
    baseline = spec["baseline"][0]
    assert baseline["composite"] == PARCA_PREP_COMPOSITE
    assert baseline["params"]["candidate_cache_dir"] == "out/cache_full"
    assert baseline["params"]["reference_cache_dir"] == ve_cache
    assert "pipeline_gate" not in spec


def test_config_study_yaml_round_trips(tmp_path):
    result, _, ve_cache = _write(tmp_path)
    spec = yaml.safe_load(result["study_paths"]["basal"].read_text())

    assert spec["name"] == "basal"
    baseline = spec["baseline"][0]
    assert baseline["name"] == "basal"
    assert baseline["composite"] == CANDIDATE_COMPOSITE
    assert baseline["params"]["match_simdata"] == f"{ve_cache}/simData.cPickle"

    variant = spec["variants"][0]
    assert variant["name"] == REFERENCE_VARIANT_NAME
    assert variant["composite"] == REFERENCE_COMPOSITE

    assert len(spec["comparative_visualizations"]) >= 1

    analyses = spec["analyses"]
    assert len(analyses) == 1
    assert analyses[0]["name"] == "comparison_cards"
    assert analyses[0]["params"]["candidate_run"] == "basal"
    assert analyses[0]["params"]["reference_run"] == REFERENCE_VARIANT_NAME

    prereqs = spec["pipeline_gate"]["prerequisites"]
    assert prereqs == [{"study": PARCA_STUDY_NAME, "relation": "leads-to"}]


# --- investigation.yaml round-trips -------------------------------------------

def test_investigation_yaml_round_trips_members_and_analyses(tmp_path):
    result, _, _ = _write(
        tmp_path, configs=[{"name": "basal", "condition": "basal"},
                           {"name": "with_aa", "condition": "with_aa"}])
    spec = yaml.safe_load(result["investigation_path"].read_text())

    assert spec["schema_version"] == 4
    assert spec["name"] == INVEST_SLUG
    # parca FIRST, then configs in materialization order.
    assert spec["studies"] == [PARCA_STUDY_NAME, "basal", "with_aa"]

    analyses = spec["analyses"]
    assert len(analyses) == 1
    assert analyses[0]["name"] == "comparison_matrix"
    assert analyses[0]["params"]["config_studies"] == ["basal", "with_aa"]


def test_investigation_studies_key_matches_written_study_paths(tmp_path):
    """The substrate's investigation_member_slugs reads `studies:` -- every
    slug listed there must correspond to an actual written study.yaml."""
    result, _, _ = _write(tmp_path)
    spec = yaml.safe_load(result["investigation_path"].read_text())

    for slug in spec["studies"]:
        assert slug in result["study_paths"]
        assert result["study_paths"][slug].exists()
