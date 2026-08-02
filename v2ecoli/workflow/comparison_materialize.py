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
from pathlib import Path
from typing import Any

import yaml

from scripts._compare.study_spec import StudySpec, specs_from_configs
from scripts._compare.reference import ReferenceEngine
from v2ecoli.composites.vecoli import _resolve_sim_data_path
from v2ecoli.workflow.parca_study import (
    CANDIDATE_ENGINE, PARCA_STUDY_NAME, REFERENCE_ENGINE, prerequisite_edge)

try:
    # The substrate's single-source atomic write (write-to-.tmp then
    # os.replace) -- reused so a concurrent reader (e.g. the dashboard) never
    # observes a half-written study.yaml/investigation.yaml. Optional import:
    # this module must still work if the substrate isn't on PYTHONPATH (e.g.
    # a hermetic unit test importing only v2ecoli), falling back to a plain
    # write below.
    from vivarium_workbench.lib.atomic_io import atomic_write_text
except ImportError:  # pragma: no cover - exercised when substrate absent
    atomic_write_text = None

CANDIDATE_COMPOSITE = "v2ecoli.composites.ecoli_baseline.ecoli_baseline"
REFERENCE_COMPOSITE = "v2ecoli.composites.vecoli.vecoli"
#: Task 1's ParCa pull-or-compute composite (Gap 1) -- the ``parca`` native
#: study's ``baseline`` composite (see ``to_native_study_specs``).
PARCA_PREP_COMPOSITE = "v2ecoli.composites.parca_prep.parca_prep"

COMPARISON_CARDS_ANALYSIS = "comparison_cards"
COMPARISON_MATRIX_ANALYSIS = "comparison_matrix"

#: The variant name every per-config native study's reference run registers
#: under (the v4 workbench sim_name convention: baseline's sim_name is the
#: study slug itself, i.e. the config name -- see ``to_native_study_specs``).
REFERENCE_VARIANT_NAME = "reference"

# Mirrors scripts/_compare/study_spec.py's own defaults so a `comparison:`
# block that omits v2_cache/ve_cache resolves identically here and there.
_DEFAULT_V2_CACHE = "out/cache_full"
_DEFAULT_VE_CACHE = "out/compare_harness/vecoli_parca"


@dataclass
class RunSpec:
    """One materialized engine run: a workbench study's ``conditions.baseline``
    (composite + params), keyed by its study/run name.

    ``prerequisites``: study names this run's ``pipeline_gate.prerequisites``
    (Task: ParCa pull-or-compute) must list — every candidate/reference
    ``RunSpec`` this module materializes carries ``[PARCA_STUDY_NAME]`` so the
    per-config study DEPENDS ON the ParCa study, expressed via the same
    ``{study, relation}`` DAG-edge shape ``vivarium_workbench.lib.study_seed``
    already writes elsewhere (see ``parca_study.prerequisite_edge``)."""
    name: str
    composite: str
    params: dict = field(default_factory=dict)
    prerequisites: list = field(default_factory=list)


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
class ParcaPrerequisite:
    """The ParCa study: a first-class member of the comparison investigation
    that both engines' per-config studies depend on (``pipeline_gate.
    prerequisites``). Not a Composite sim -- a pull-or-compute check+build
    step (``v2ecoli.workflow.parca_study.resolve_or_build_parca``) per engine,
    resolved BEFORE any candidate/reference study runs.

    ``candidate_cache_dir``/``reference_cache_dir`` are the SINGLE source
    every per-config candidate/reference ``RunSpec``'s ``cache_dir``/
    ``match_simdata`` params are derived from (see ``materialize_comparison``)
    -- not independently re-read from the raw ``comparison:`` block per
    config, so a candidate/reference pair and the ParCa study can never
    silently disagree on which cache dir is "the" cache dir."""
    name: str = PARCA_STUDY_NAME
    candidate_cache_dir: str = ""
    reference_cache_dir: str = ""
    reference_repo: str = ""


@dataclass
class MaterializedInvestigation:
    invest_name: str
    pairs: list  # [ComparisonPair, ...]
    matrix_analysis: dict  # {"name": "comparison_matrix", "params": {...}}
    parca: ParcaPrerequisite = field(default_factory=ParcaPrerequisite)


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

    Every candidate/reference ``RunSpec`` DEPENDS ON the ParCa study
    (``prerequisites=[PARCA_STUDY_NAME]``) and its ``cache_dir``/
    ``match_simdata`` params are derived from the single
    ``ParcaPrerequisite`` built here (``parca.candidate_cache_dir``/
    ``parca.reference_cache_dir``) rather than re-reading ``sp.v2_cache``/
    ``sp.ve_cache`` per config — see ``ParcaPrerequisite``'s docstring for why
    that single-source routing matters.
    """
    ctx = _build_ctx(comparison, invest_name)
    specs = specs_from_configs(ctx)

    parca = ParcaPrerequisite(
        name=PARCA_STUDY_NAME,
        candidate_cache_dir=ctx["v2_cache"],
        reference_cache_dir=ctx["ve_cache"],
        reference_repo=ctx["reference"].repo if ctx["reference"] else "",
    )
    match_simdata = _resolve_sim_data_path(parca.reference_cache_dir)

    pairs: list[ComparisonPair] = []
    for sp in specs:
        candidate = RunSpec(
            name=_run_name(sp.name, "candidate"),
            composite=CANDIDATE_COMPOSITE,
            params={
                "cache_dir": parca.candidate_cache_dir,
                "seed": seed,
                "match_simdata": match_simdata,
                "match_condition": sp.condition,
            },
            prerequisites=[parca.name],
        )
        reference = RunSpec(
            name=_run_name(sp.name, "reference"),
            composite=REFERENCE_COMPOSITE,
            params={
                "reference_repo": sp.reference.repo if sp.reference else "",
                "condition": sp.condition,
                "seed": seed,
                "fork_config": _fork_config_for(sp),
                "cache_dir": parca.reference_cache_dir,
            },
            prerequisites=[parca.name],
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
                                     matrix_analysis=matrix_analysis, parca=parca)


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


def parca_study_spec(parca: ParcaPrerequisite) -> dict:
    """The ParCa study's own entry: not a Composite sim -- a pull-or-compute
    check+build step (``v2ecoli.workflow.parca_study.resolve_or_build_parca``)
    run once per engine, ahead of every candidate/reference study that
    depends on it. ``kind: parca_prerequisite`` distinguishes this from an
    ordinary ``conditions.baseline``-shaped study spec; ``engines`` carries
    exactly the kwargs ``resolve_or_build_parca(engine, cache_dir,
    reference_repo=...)`` needs per engine."""
    return {
        "name": parca.name,
        "kind": "parca_prerequisite",
        "engines": {
            CANDIDATE_ENGINE: {"cache_dir": parca.candidate_cache_dir},
            REFERENCE_ENGINE: {"cache_dir": parca.reference_cache_dir,
                               "reference_repo": parca.reference_repo},
        },
    }


def _pipeline_gate(run: RunSpec) -> dict:
    return {"prerequisites": [prerequisite_edge(name) for name in run.prerequisites]}


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
    in a pair execute).

    Includes the ParCa study itself (keyed by ``PARCA_STUDY_NAME``, see
    ``parca_study_spec``) and stamps every candidate/reference entry's
    ``pipeline_gate.prerequisites`` with an edge back to it -- the general
    runner's existing DAG machinery (``investigation_graph_views.py``'s
    ``pipeline_gate.prerequisites`` edge reader) is what makes that a real,
    renderable dependency rather than a convention only this module knows
    about."""
    out: dict[str, dict] = {materialized.parca.name: parca_study_spec(materialized.parca)}
    for pair in materialized.pairs:
        out[pair.candidate.name] = {
            "name": pair.candidate.name,
            "conditions": {"baseline": {"composite": pair.candidate.composite,
                                        "params": pair.candidate.params}},
            "analyses": list(pair.analyses),
            "pipeline_gate": _pipeline_gate(pair.candidate),
        }
        out[pair.reference.name] = {
            "name": pair.reference.name,
            "conditions": {"baseline": {"composite": pair.reference.composite,
                                        "params": pair.reference.params}},
            "analyses": [],
            "pipeline_gate": _pipeline_gate(pair.reference),
        }
    return out


# --- Phase B: workbench-NATIVE single-study-per-config materializer --------
#
# Re-models the paired emission above (two studies per config) into the
# workbench-native model: ONE study per config, with the candidate as the
# study's `baseline` and the reference as a `variants` entry in the SAME
# study (see docs/superpowers/specs/2026-08-02-phase-b-comparison-on-
# composite-substrate-design.md, "Re-model: paired studies -> one native
# study per config"). `to_study_specs`/`matrix_analysis_entry` above are left
# untouched (parallel-safe with any other in-flight work against the paired
# path); everything below is additive.
#
# Convention this enforces: the candidate's run registers with
# `sim_name = <config>` (the v4 slug convention -- a study's baseline run
# takes the study's own slug), and the reference's run registers with
# `sim_name = "reference"` (`REFERENCE_VARIANT_NAME`). `comparison_cards`'s
# `candidate_run`/`reference_run` params and `comparative_visualizations`'
# `runs[].sim_name` entries below all reference exactly those two sim_names
# -- a mismatch would silently render empty overlays / missing verdicts (the
# design doc's own risk note).

#: Key observables overlaid per config's `comparative_visualizations` --
#: cell/dry/protein/RNA mass + instantaneous growth rate, the same
#: observable set `scripts/compare_matched_trajectories.py`'s `OBSERVABLES`
#: names (its `cell_mass`/`dry_mass`/`protein_mass`/`rna_mass`/
#: `instantaneous_growth_rate`), mapped onto their dotted state-tree paths
#: under `listeners.mass.*` (confirmed against `v2ecoli/workflow/analyses/
#: growth_overlay.py` and `dummy.py`'s observable-name catalogue -- not
#: guessed). A reasonable default set per the task brief; refining this list
#: (e.g. more listeners) is a cheap follow-up, not a shape change.
_COMPARATIVE_OBSERVABLES: "list[dict]" = [
    {"name": "cell_mass", "title": "Cell mass over time",
     "observable_path": "listeners.mass.cell_mass", "y_label": "Cell mass (fg)"},
    {"name": "dry_mass", "title": "Dry mass over time",
     "observable_path": "listeners.mass.dry_mass", "y_label": "Dry mass (fg)"},
    {"name": "protein_mass", "title": "Protein mass over time",
     "observable_path": "listeners.mass.protein_mass", "y_label": "Protein mass (fg)"},
    {"name": "rna_mass", "title": "RNA mass over time",
     "observable_path": "listeners.mass.rna_mass", "y_label": "RNA mass (fg)"},
    {"name": "growth_rate", "title": "Instantaneous growth rate over time",
     "observable_path": "listeners.mass.instantaneous_growth_rate",
     "y_label": "Growth rate (1/s)"},
]


def _comparative_visualizations(config_name: str) -> "list[dict]":
    """One ``comparative_visualizations`` entry per key observable, each
    overlaying the config's two runs -- ``sim_name=<config_name>`` (the
    candidate/baseline) and ``sim_name=REFERENCE_VARIANT_NAME`` (the
    reference/variant) -- in the exact shape
    ``vivarium_workbench.lib.comparative_runs.
    render_investigation_comparative_visualisations`` reads from a study.yaml
    (verified against the installed package's own docstring/schema, not
    guessed): ``{name, title, observable_path, y_label, runs: [{sim_name,
    label}, ...]}``."""
    return [
        {
            "name": obs["name"],
            "title": obs["title"],
            "observable_path": obs["observable_path"],
            "y_label": obs["y_label"],
            "runs": [
                {"sim_name": config_name, "label": "candidate (v2ecoli)"},
                {"sim_name": REFERENCE_VARIANT_NAME, "label": "reference (vEcoli)"},
            ],
        }
        for obs in _COMPARATIVE_OBSERVABLES
    ]


def _native_parca_study_spec(parca: ParcaPrerequisite) -> dict:
    """The ``parca`` native study: an ORDINARY study (no special ``kind``,
    Gap 1) whose single ``baseline`` run is the ``parca_prep``
    ``@composite_generator`` (Task 1) wrapping ``resolve_or_build_parca``'s
    pull-or-compute contract for both engines. Cache dirs/repo are wired
    straight from ``parca`` -- the SAME ``ParcaPrerequisite`` every
    candidate/reference ``RunSpec``'s own ``cache_dir``/``match_simdata``
    params are derived from (see ``materialize_comparison``), so this study
    and the per-config studies can never disagree on which cache dir is
    "the" cache dir. No ``pipeline_gate`` -- this study is the root of the
    comparison investigation's dependency graph, not a dependent."""
    return {
        "name": parca.name,
        "baseline": [{
            "name": parca.name,
            "composite": PARCA_PREP_COMPOSITE,
            "params": {
                "candidate_cache_dir": parca.candidate_cache_dir,
                "reference_cache_dir": parca.reference_cache_dir,
                "reference_repo": parca.reference_repo,
            },
        }],
    }


def to_native_study_specs(materialized: MaterializedInvestigation) -> "dict[str, dict]":
    """``{study_name: study-spec dict}`` in the workbench-NATIVE
    single-study-per-config shape (Phase B's re-model): the ``parca`` study
    (see ``_native_parca_study_spec``) plus, per config, ONE study keyed by
    the config's own name --

        {"name": <config>,
         "baseline": [{"name": <config>, "composite": CANDIDATE_COMPOSITE,
                       "params": {...match_simdata, match_condition, cache_dir, seed}}],
         "variants": [{"name": "reference", "composite": REFERENCE_COMPOSITE,
                       "params": {...reference_repo, condition, seed, fork_config, cache_dir}}],
         "comparative_visualizations": [...],
         "analyses": [{"name": "comparison_cards",
                       "params": {"candidate_run": <config>, "reference_run": "reference",
                                  "seeds": ..., "cards": [...]}}],
         "pipeline_gate": {"prerequisites": [{"study": "parca", "relation": "leads-to"}]}}

    Every candidate/reference ``params`` dict is reused VERBATIM from the
    existing paired emission (``pair.candidate.params``/``pair.reference.
    params``, built by ``materialize_comparison``) -- no science change, only
    the study-shape re-model. Only ``comparison_cards``'s ``candidate_run``/
    ``reference_run`` are overridden (from the old ``<config>-candidate``/
    ``<config>-reference`` run names to the native sim_names ``<config>``/
    ``"reference"``), and ``seeds``/``cards`` are pulled through from the
    same already-computed ``pair.analyses`` entry.

    Does NOT touch/replace ``to_study_specs`` (the old paired emission) --
    additive, parallel-safe."""
    out: dict[str, dict] = {materialized.parca.name: _native_parca_study_spec(materialized.parca)}
    for pair in materialized.pairs:
        old_cards_entry = pair.analyses[0]["params"]
        out[pair.config] = {
            "name": pair.config,
            "baseline": [{
                "name": pair.config,
                "composite": pair.candidate.composite,
                "params": dict(pair.candidate.params),
            }],
            "variants": [{
                "name": REFERENCE_VARIANT_NAME,
                "composite": pair.reference.composite,
                "params": dict(pair.reference.params),
            }],
            "comparative_visualizations": _comparative_visualizations(pair.config),
            "analyses": [{
                "name": COMPARISON_CARDS_ANALYSIS,
                "params": {
                    "candidate_run": pair.config,
                    "reference_run": REFERENCE_VARIANT_NAME,
                    "seeds": old_cards_entry["seeds"],
                    "cards": list(old_cards_entry["cards"]),
                },
            }],
            "pipeline_gate": {"prerequisites": [prerequisite_edge(materialized.parca.name)]},
        }
    return out


def to_native_investigation_analyses(materialized: MaterializedInvestigation) -> "list[dict]":
    """The investigation-level ``comparison_matrix`` Analysis entry for the
    NATIVE model: ``params.config_studies`` is the member config SLUG LIST
    (``[pair.config, ...]``), not the old paired-model's
    ``"<candidate_run>::comparison_cards"`` token map (``matrix_analysis_entry``
    above, kept for the paired path). Gap 2 (a later task) makes
    ``comparison_matrix`` read each config study's persisted
    ``comparison_cards`` verdict from disk BY these slugs -- this function's
    scope is only the wiring shape, matching ``to_native_study_specs``'s
    per-config study keys 1:1."""
    return [{
        "name": COMPARISON_MATRIX_ANALYSIS,
        "params": {"config_studies": [pair.config for pair in materialized.pairs]},
    }]


def _write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path``, creating parent dirs first. Uses the
    substrate's ``atomic_write_text`` (write-to-``.tmp`` then ``os.replace``)
    when available (see the module-level optional import), else a plain
    ``Path.write_text`` -- this module must still work hermetically when the
    substrate isn't on ``PYTHONPATH``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if atomic_write_text is not None:
        atomic_write_text(path, text)
    else:
        path.write_text(text, encoding="utf-8")


def write_native_investigation(materialized: MaterializedInvestigation,
                                workspace: "str | Path",
                                invest_slug: str) -> dict:
    """Write the NATIVE materialization (Phase B's re-model) to actual
    workspace files the investigation-as-composite substrate runs:

    - Every ``to_native_study_specs(materialized)`` entry (the ``parca``
      study + one study per config) -> ``<workspace>/studies/<name>/
      study.yaml``.
    - The investigation itself -> ``<workspace>/investigations/<invest_slug>/
      investigation.yaml``, with ``studies:`` (the member-slug list the
      substrate's ``investigation_member_slugs`` reads -- ``parca`` FIRST,
      then each config in materialization order; real ordering is still
      enforced by each config study's ``pipeline_gate.prerequisites``, this
      list is just declared-order hygiene) and ``analyses:``
      (``to_native_investigation_analyses(materialized)``, the
      investigation-level ``comparison_matrix`` entry).

    Returns ``{"investigation_path": Path, "study_paths": {name: Path}}`` --
    every path written, for a caller (or this module's own tests) to load
    back and verify.
    """
    workspace = Path(workspace)

    study_specs = to_native_study_specs(materialized)
    study_paths: dict[str, Path] = {}
    for name, spec in study_specs.items():
        study_path = workspace / "studies" / name / "study.yaml"
        _write_text(study_path, yaml.safe_dump(spec, sort_keys=False))
        study_paths[name] = study_path

    members = [materialized.parca.name] + [pair.config for pair in materialized.pairs]
    investigation_spec = {
        "schema_version": 4,
        "name": invest_slug,
        "studies": members,
        "analyses": to_native_investigation_analyses(materialized),
    }
    # MaterializedInvestigation doesn't currently retain the source
    # `comparison:` block (see its dataclass fields) -- if a future caller
    # attaches one, preserve it verbatim rather than silently dropping it.
    source_comparison = getattr(materialized, "comparison", None)
    if source_comparison:
        investigation_spec["comparison"] = source_comparison
    investigation_path = workspace / "investigations" / invest_slug / "investigation.yaml"
    _write_text(investigation_path, yaml.safe_dump(investigation_spec, sort_keys=False))

    return {"investigation_path": investigation_path, "study_paths": study_paths}
