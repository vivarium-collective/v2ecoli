"""Comparison convergence Phase 2, Task 4: materializing a ``comparison:``
block into paired candidate ('ecoli_baseline') + reference ('vecoli') run
specs the general vivarium-workbench runner runs, with the comparison
Analyses (Task 2 ``comparison_cards``, Task 3 ``comparison_matrix``) wired to
the materialized runs.

Hermetic: no engines run, no ParCa cache, no upstream vEcoli fork checkout —
spec/wiring assertions only (heavy e2e is Task 5, gated).

The key regression this file guards: Task 1's ``ecoli_baseline.
_apply_match_simdata`` used to hardcode ``condition="basal"``, so a non-basal
config's candidate would silently overlay BASAL reference state. The
``with_aa`` config below exists specifically to catch that — if the
materializer (or the ``match_condition`` fix it depends on) regresses back to
a hardcoded "basal", ``test_with_aa_config_candidate_carries_with_aa_condition_not_basal``
fails.
"""
from __future__ import annotations

import v2ecoli.composites.ecoli_baseline as eb
from v2ecoli.workflow.analysis import ANALYSIS_REGISTRY
import v2ecoli.workflow.analyses  # noqa: F401 -- registers comparison_cards/comparison_matrix
from v2ecoli.workflow.comparison_materialize import (
    CANDIDATE_COMPOSITE, REFERENCE_COMPOSITE, ComparisonPair,
    MaterializedInvestigation, RunSpec, materialize_comparison,
    matrix_analysis_entry, to_study_specs)


def _comparison_block(ve_cache: str) -> dict:
    """A small comparison: block -- candidate ecoli_baseline (implicit),
    reference vecoli (a fake fork repo path), TWO configs: one basal, one
    NON-basal (with_aa) -- proving per-config condition threading, not just
    the default."""
    return {
        "reference": {"repo": "/fake/vecoli-fork", "kind": "vecoli"},
        "v2_cache": "out/cache_full",
        "ve_cache": ve_cache,
        "defaults": {"seeds": 1, "generations": 1, "cards": ["summary", "standard"]},
        "configs": [
            {"name": "basal", "condition": "basal"},
            {"name": "with_aa", "condition": "with_aa"},
        ],
    }


def _materialize(tmp_path):
    ve_cache = tmp_path / "vecoli_parca"
    ve_cache.mkdir()
    (ve_cache / "simData.cPickle").write_bytes(b"fake-simdata")
    block = _comparison_block(str(ve_cache))
    materialized = materialize_comparison(block, invest_name="test-comparison")
    return materialized, str(ve_cache)


# --- materialization shape -------------------------------------------------

def test_materialize_yields_one_pair_per_config(tmp_path):
    materialized, _ = _materialize(tmp_path)
    assert isinstance(materialized, MaterializedInvestigation)
    assert [p.config for p in materialized.pairs] == ["basal", "with_aa"]


def test_candidate_run_spec_is_ecoli_baseline_with_match_simdata(tmp_path):
    materialized, ve_cache = _materialize(tmp_path)
    basal_pair = materialized.pairs[0]
    cand = basal_pair.candidate
    assert isinstance(cand, RunSpec)
    assert cand.composite == CANDIDATE_COMPOSITE
    assert cand.name == "basal-candidate"
    # match_simdata resolves to the reference cache's simData.cPickle -- the
    # SAME path resolution the reference vecoli composite itself performs
    # (v2ecoli.composites.vecoli._resolve_sim_data_path), so candidate and
    # reference agree on which simData they're paired against.
    assert cand.params["match_simdata"] == f"{ve_cache}/simData.cPickle"


def test_with_aa_config_candidate_carries_with_aa_condition_not_basal(tmp_path):
    """THE regression test: the with_aa config's candidate RunSpec must carry
    match_condition='with_aa' -- catching Task 1's old hardcoded
    condition="basal" in _apply_match_simdata. Without per-config condition
    threading, this would silently be 'basal' regardless of the config."""
    materialized, _ = _materialize(tmp_path)
    with_aa_pair = next(p for p in materialized.pairs if p.config == "with_aa")
    assert with_aa_pair.condition == "with_aa"
    assert with_aa_pair.candidate.params["match_condition"] == "with_aa"
    assert with_aa_pair.candidate.params["match_condition"] != "basal"


def test_basal_config_candidate_still_carries_basal_condition(tmp_path):
    materialized, _ = _materialize(tmp_path)
    basal_pair = next(p for p in materialized.pairs if p.config == "basal")
    assert basal_pair.candidate.params["match_condition"] == "basal"


def test_reference_run_spec_is_vecoli_with_repo_and_right_condition(tmp_path):
    materialized, _ = _materialize(tmp_path)
    with_aa_pair = next(p for p in materialized.pairs if p.config == "with_aa")
    ref = with_aa_pair.reference
    assert isinstance(ref, RunSpec)
    assert ref.composite == REFERENCE_COMPOSITE
    assert ref.name == "with_aa-reference"
    assert ref.params["reference_repo"] == "/fake/vecoli-fork"
    assert ref.params["condition"] == "with_aa"

    basal_pair = next(p for p in materialized.pairs if p.config == "basal")
    assert basal_pair.reference.params["condition"] == "basal"
    assert basal_pair.reference.params["reference_repo"] == "/fake/vecoli-fork"


def test_candidate_and_reference_conditions_always_match_within_a_pair(tmp_path):
    """The whole point of per-config threading: within ONE pair, candidate's
    match_condition and reference's condition must be identical -- otherwise
    the two engines aren't being compared on the same biology."""
    materialized, _ = _materialize(tmp_path)
    for pair in materialized.pairs:
        assert pair.candidate.params["match_condition"] == pair.reference.params["condition"]
        assert pair.candidate.params["match_condition"] == pair.condition


# --- Analysis wiring ---------------------------------------------------

def test_comparison_cards_and_matrix_are_registered_analyses():
    assert "comparison_cards" in ANALYSIS_REGISTRY
    assert "comparison_matrix" in ANALYSIS_REGISTRY


def test_each_pair_wires_comparison_cards_to_its_own_two_runs(tmp_path):
    materialized, _ = _materialize(tmp_path)
    for pair in materialized.pairs:
        assert len(pair.analyses) == 1
        entry = pair.analyses[0]
        assert entry["name"] == "comparison_cards"
        assert entry["name"] in ANALYSIS_REGISTRY
        assert entry["params"]["candidate_run"] == pair.candidate.name
        assert entry["params"]["reference_run"] == pair.reference.name


def test_matrix_analysis_wires_every_config_to_its_pair(tmp_path):
    materialized, _ = _materialize(tmp_path)
    matrix = materialized.matrix_analysis
    assert matrix["name"] == "comparison_matrix"
    assert matrix["name"] in ANALYSIS_REGISTRY
    config_verdicts = matrix["params"]["config_verdicts"]
    assert set(config_verdicts) == {"basal", "with_aa"}
    for pair in materialized.pairs:
        assert config_verdicts[pair.config] == f"{pair.candidate.name}::comparison_cards"


def test_matrix_analysis_entry_is_a_pure_function_of_pairs():
    pairs = [
        ComparisonPair(config="c1", condition="basal",
                       candidate=RunSpec(name="c1-candidate", composite=CANDIDATE_COMPOSITE),
                       reference=RunSpec(name="c1-reference", composite=REFERENCE_COMPOSITE))
    ]
    entry = matrix_analysis_entry(pairs)
    assert entry == {
        "name": "comparison_matrix",
        "params": {"config_verdicts": {"c1": "c1-candidate::comparison_cards"}},
    }


# --- study-spec rendering (conditions.baseline + analyses:) -----------

def test_to_study_specs_shape_matches_study_yaml_conditions_baseline(tmp_path):
    materialized, _ = _materialize(tmp_path)
    specs = to_study_specs(materialized)

    assert set(specs) == {
        "basal-candidate", "basal-reference",
        "with_aa-candidate", "with_aa-reference",
    }
    cand = specs["with_aa-candidate"]
    assert cand["conditions"]["baseline"]["composite"] == CANDIDATE_COMPOSITE
    assert cand["conditions"]["baseline"]["params"]["match_condition"] == "with_aa"
    assert cand["analyses"][0]["name"] == "comparison_cards"

    ref = specs["with_aa-reference"]
    assert ref["conditions"]["baseline"]["composite"] == REFERENCE_COMPOSITE
    assert ref["conditions"]["baseline"]["params"]["condition"] == "with_aa"
    # The comparison_cards Analysis lives on the candidate study only -- not
    # duplicated onto the reference study (avoids running it twice per pair).
    assert ref["analyses"] == []


# --- ecoli_baseline declares match_condition (sanity against Task 1's fix) -

def test_ecoli_baseline_declares_match_condition_param():
    entry = eb.baseline._composite_generator_entry
    assert "match_condition" in entry.parameters
    decl = entry.parameters["match_condition"]
    assert decl["default"] == "basal"
