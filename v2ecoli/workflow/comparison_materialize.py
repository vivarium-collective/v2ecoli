"""Comparison convergence Phase 2, Task 4: materialize a ``comparison:`` block
into paired candidate+reference study specs the GENERAL ``vivarium-workbench``
runner runs, with the post-sim comparison Analyses (Task 2's ``comparison_cards``,
Task 3's ``comparison_matrix``) wired to read exactly those runs.

Per spec §3 ("A comparison is an investigation of paired Composite studies"):
per config, materialize a **candidate** study (``ecoli_baseline``) and a
**reference** study (``vecoli``), with matched inputs (``condition``,
``seed``, and matched-initial-state) — so ``vivarium-workbench run
investigation <slug>`` / ``prepare-investigation --investigation <slug>``
runs both engines per config and the comparison Analyses render the cards +
matrix, entirely through general capabilities.

Reuses ``scripts/_compare/study_spec.py``'s ``specs_from_configs`` READ-ONLY
(not modified, not reimplemented) to parse the ``comparison.configs[]`` list
into ``StudySpec`` objects — the exact schema the ``comparison:`` block
already follows (candidate/reference/defaults/configs) — so this module and
the legacy ``v2e-compare`` runner agree on one config parser. Also reuses
``v2ecoli.composites.vecoli._resolve_sim_data_path`` (the SAME simData
resolution the reference ``vecoli`` composite itself performs) to compute the
candidate's ``match_simdata`` path from the reference's cache dir, so the
paired runs are guaranteed to agree on which simData.cPickle they compare
against.

**Condition threading (Task 1 CARRY):** ``ecoli_baseline._apply_match_simdata``
used to hardcode ``condition="basal"`` when building the matched-init overlay
— silently wrong for any non-basal config (e.g. ``with_aa``), which would
overlay basal-media reference state onto a with_aa candidate. Fixed by adding
``ecoli_baseline``'s ``match_condition`` param (threaded into
``_vecoli_reference_state(..., condition=...)``, default "basal" for
back-compat — see ``v2ecoli/composites/ecoli_baseline.py`` and
``tests/test_matched_initial_state_param.py``). This module is what actually
supplies a non-default value: every candidate ``RunSpec`` sets
``match_condition=<the config's own condition>``.

**Analysis wiring.** Each ``ComparisonPair`` carries an ``analyses`` list in
the exact shape ``vivarium_workbench.lib.study_run_post.build_analysis_options``
consumes from a study spec's own ``analyses:`` key (``[{"name": <
ANALYSIS_REGISTRY key>, "params": {...}}, ...]`` — verified against the
installed ``vivarium_workbench`` package, not guessed): one
``comparison_cards`` entry per pair, with ``candidate_run``/``reference_run``
params set to that pair's two materialized run names — the same
``candidate_run``/``reference_run`` contract
``v2ecoli.workflow.analyses.comparison_cards.ComparisonCards.config_schema``
declares. ``to_study_specs()`` renders every materialized run as a
``{"conditions": {"baseline": {"composite", "params"}}, "analyses": [...]}``
dict — the same ``conditions.baseline.{composite,params}`` shape existing
study.yaml files already use (see e.g.
``workspace/studies/param-uq-03-growth-stratified/study.yaml``) — ready to be
written to a study.yaml or handed to the study-run engine directly.

The investigation-level cross-config ``comparison_matrix`` (Task 3) is wired
via ``matrix_analysis_entry()``: its ``config_verdicts`` param maps each
config name to a ``"<candidate_run>::comparison_cards"`` reference token
identifying which pair's ``comparison_cards`` output that config's verdict
comes from. Resolving those tokens into real verdict dicts at run time (after
each pair's ``comparison_cards`` Analysis has actually run) is the general
runner's job, exercised end-to-end only in Task 5's gated e2e — this module's
scope is the hermetic materialization + wiring shape, not running engines.

Does NOT modify ``scripts/_compare/*`` (parallel-safe with ``v2e-compare``,
per plan Global Constraints) — ``study_spec.py`` is read-only, imported and
called exactly as investigations that still drive the legacy runner do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from scripts._compare.study_spec import StudySpec, specs_from_configs
from scripts._compare.reference import ReferenceEngine
from v2ecoli.composites.vecoli import _resolve_sim_data_path

CANDIDATE_COMPOSITE = "v2ecoli.composites.ecoli_baseline.ecoli_baseline"
REFERENCE_COMPOSITE = "v2ecoli.composites.vecoli.vecoli"

COMPARISON_CARDS_ANALYSIS = "comparison_cards"
COMPARISON_MATRIX_ANALYSIS = "comparison_matrix"

# Mirrors scripts/_compare/study_spec.py's own defaults so a `comparison:`
# block that omits v2_cache/ve_cache resolves identically here and there.
_DEFAULT_V2_CACHE = "out/cache_full"
_DEFAULT_VE_CACHE = "out/compare_harness/vecoli_parca"


@dataclass
class RunSpec:
    """One materialized engine run: a workbench study's ``conditions.baseline``
    (composite + params), keyed by its study/run name."""
    name: str
    composite: str
    params: dict = field(default_factory=dict)


@dataclass
class ComparisonPair:
    """One config's candidate + reference paired runs, plus the
    ``comparison_cards`` Analysis entry wired to read exactly those two runs."""
    config: str
    condition: str
    candidate: RunSpec
    reference: RunSpec
    analyses: list = field(default_factory=list)


@dataclass
class MaterializedInvestigation:
    invest_name: str
    pairs: list  # [ComparisonPair, ...]
    matrix_analysis: dict  # {"name": "comparison_matrix", "params": {...}}


def _run_name(config_name: str, role: str) -> str:
    return f"{config_name}-{role}"


def _build_ctx(comparison: dict, invest_name: str) -> dict:
    """Adapt a raw ``comparison:`` block dict into the ``ctx`` shape
    ``scripts._compare.study_spec.specs_from_configs`` expects — the same
    fields ``study_spec._context`` derives from an investigation.yaml's
    ``comparison`` block, built here directly from the block instead of
    reading a file off disk (this module's callers may materialize from an
    in-memory block, e.g. before it's ever written to investigation.yaml)."""
    defaults = comparison.get("defaults") or {}
    return {
        "invest_name": invest_name,
        "reference": ReferenceEngine.from_spec(comparison.get("reference") or {}),
        "configs": comparison.get("configs") or [],
        "v2_cache": comparison.get("v2_cache", _DEFAULT_V2_CACHE),
        "ve_cache": comparison.get("ve_cache", _DEFAULT_VE_CACHE),
        "defaults": defaults,
    }


def _fork_config_for(sp: StudySpec) -> str:
    """``sp.config`` is a condition name (no swap) OR a reference-config path
    driving a process swap on BOTH engines (see study_spec.py's module
    docstring). A bare name equal to the config's own ``name`` means no swap;
    anything else is passed through as the reference engine's ``fork_config``."""
    if sp.config and sp.config != sp.name:
        return sp.config
    return ""


def materialize_comparison(comparison: dict, invest_name: str = "comparison",
                           seed: int = 0) -> MaterializedInvestigation:
    """Materialize a ``comparison:`` block (candidate, reference, defaults,
    configs[]) into paired candidate (``ecoli_baseline``) + reference
    (``vecoli``) ``RunSpec``s per config, threading each config's OWN
    ``condition`` into both sides — the candidate via ``match_condition``
    (Task 1's fix), the reference via its native ``condition`` param — and
    wiring the ``comparison_cards`` Analysis to each pair's two run names.

    ``comparison`` is the raw block dict (as it appears under an
    investigation.yaml's ``comparison:`` key — candidate/reference/defaults/
    configs[], per ``scripts/_compare/study_spec.py``'s schema, read-only).
    """
    ctx = _build_ctx(comparison, invest_name)
    specs = specs_from_configs(ctx)

    pairs: list[ComparisonPair] = []
    for sp in specs:
        match_simdata = _resolve_sim_data_path(sp.ve_cache)
        candidate = RunSpec(
            name=_run_name(sp.name, "candidate"),
            composite=CANDIDATE_COMPOSITE,
            params={
                "cache_dir": sp.v2_cache,
                "seed": seed,
                "match_simdata": match_simdata,
                "match_condition": sp.condition,
            },
        )
        reference = RunSpec(
            name=_run_name(sp.name, "reference"),
            composite=REFERENCE_COMPOSITE,
            params={
                "reference_repo": sp.reference.repo if sp.reference else "",
                "condition": sp.condition,
                "seed": seed,
                "fork_config": _fork_config_for(sp),
                "cache_dir": sp.ve_cache,
            },
        )
        analyses = [{
            "name": COMPARISON_CARDS_ANALYSIS,
            "params": {
                "candidate_run": candidate.name,
                "reference_run": reference.name,
                "seeds": sp.seeds,
                "cards": list(sp.cards),
            },
        }]
        pairs.append(ComparisonPair(config=sp.name, condition=sp.condition,
                                    candidate=candidate, reference=reference,
                                    analyses=analyses))

    matrix_analysis = matrix_analysis_entry(pairs)
    return MaterializedInvestigation(invest_name=invest_name, pairs=pairs,
                                     matrix_analysis=matrix_analysis)


def matrix_analysis_entry(pairs: "list[ComparisonPair]") -> dict:
    """The investigation-level ``comparison_matrix`` Analysis entry (Task 3),
    ``config_verdicts`` mapping each config name to a
    ``"<candidate_run>::comparison_cards"`` token identifying which pair's
    ``comparison_cards`` verdict output feeds that config's column. Resolving
    the token into a real verdict dict at run time is the general runner's
    job (Task 5's gated e2e); this is the wiring shape only."""
    config_verdicts = {
        pair.config: f"{pair.candidate.name}::{COMPARISON_CARDS_ANALYSIS}"
        for pair in pairs
    }
    return {
        "name": COMPARISON_MATRIX_ANALYSIS,
        "params": {"config_verdicts": config_verdicts},
    }


def to_study_specs(materialized: MaterializedInvestigation) -> "dict[str, dict]":
    """``{study_name: study-spec dict}`` for every materialized run, in the
    ``conditions.baseline.{composite,params}`` + top-level ``analyses:``
    shape existing study.yaml files use and
    ``vivarium_workbench.lib.study_run_post.build_analysis_options`` reads
    (``spec.get("analyses")`` -> ``[{"name", "params"}, ...]``) — ready to
    write to ``workspace/studies/<name>/study.yaml`` or hand to the study-run
    engine directly. The reference run carries no ``analyses`` entry of its
    own (the pair's ``comparison_cards`` entry is attached to the CANDIDATE
    study only, avoiding a double-run of the same Analysis when both studies
    in a pair execute)."""
    out: dict[str, dict] = {}
    for pair in materialized.pairs:
        out[pair.candidate.name] = {
            "name": pair.candidate.name,
            "conditions": {"baseline": {"composite": pair.candidate.composite,
                                        "params": pair.candidate.params}},
            "analyses": list(pair.analyses),
        }
        out[pair.reference.name] = {
            "name": pair.reference.name,
            "conditions": {"baseline": {"composite": pair.reference.composite,
                                        "params": pair.reference.params}},
            "analyses": [],
        }
    return out
