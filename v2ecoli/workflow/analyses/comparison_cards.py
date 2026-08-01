"""Comparison convergence Phase 2, Task 2: a full-card comparison Analysis
that renders the WHOLE per-config report-card set (not just Phase 1's
``summary``) from two workbench runs -- a candidate and a reference --
reusing the existing comparison-report-card builders + verdict/tolerance
logic (``scripts/_compare``) read-only. No new science, no new tolerances,
no hand-rolled renderers.

Extends ``comparison_summary.py`` (Phase 1 Task 4): same run-loading path
(``scripts._compare.run_store_adapter.load_run_observables`` /
``resolve_run_store``), same matched-timepoint stat
(``scripts.compare_matched_trajectories.{matched_stats,grade,_gen1_window}``),
same verdict shape (``report_card_verdict/v1`` via
``scripts._compare.verdict``), but drives every card builder it can from a
single candidate-run vs single reference-run pairing instead of only
``summary``.

WIRED cards (single-run-vs-single-run is sufficient evidence for these):
  - ``summary``      -- ``scripts._compare.report_cards.summary.
                         build_summary_html``, fed a verdict assembled from
                         the other wired cards' groups via
                         ``scripts._compare.verdict.build_condition_verdict``
                         (the exact function the real harness uses to write
                         a study's ``report_card_verdict.json``).
  - ``standard``      -- ``update_standard_report_card`` (its
                         ``__pb_wrapped__`` -- see below), fed matched-time
                         per_obs/plot_trajs built from the two loaded runs.
  - ``parca``         -- ``update_parca_report_card``, same per_obs/plot_trajs.
  - ``trajectory``    -- ``update_trajectory_report_card``, fed via a
                         throwaway seed00-named symlink directory (see
                         ``_render_zarr_cards``). Descriptive only (no axes).
  - ``distribution``  -- ``update_distribution_report_card``, same symlink
                         bridge. Its own ``grade_axis`` ttest branch already
                         reports "ungraded" honestly for n<2 per side (a
                         single-run pairing has n=1) -- see
                         ``v2ecoli/library/card_criteria.py``.
  - ``metabolism`` / ``composition`` -- same symlink bridge; graded via
                         ``rel_tol`` (a real scalar-vs-scalar comparison, no
                         ensemble needed), so single-run pairing grades for
                         real, not just "ungraded".

Each card's own ``as_step``-decorated ``update_*`` function is invoked via
its ``__pb_wrapped__`` attribute (``process_bigraph.composite.as_step``
stashes the original plain function there -- see that decorator's
docstring: "Does NOT register into any core") rather than through
``core.link_registry``, so this module needs no ``build_core()``/process-
bigraph Step instantiation at all -- pure reuse of the pre-existing pure
functions, called with a plain state dict exactly as
``comparison_report_card.assemble_from_studies``/the harness would.

DEFERRED cards (do NOT wire from a single-run pairing):
  - ``statistical`` -- the gold-standard multi-seed card: a Welch t-test
    (``v2ecoli/library/card_criteria.py``'s ``ttest`` branch) over >=4 seeds
    per ``scripts/_compare/study_spec.py``'s ``GRADED`` set. Its builder
    (``_statistical_html_axes`` in ``scripts/_compare/report_cards/
    statistical.py``) pools ``ve_mean``/``v2_mean`` across a STUDY's full
    per-seed stat list (``comparison_report_card.build()``'s multi-seed
    ``per_obs``), which this Analysis does not have (candidate_run/
    reference_run name exactly ONE zarr store each). Rather than fabricate
    an n=1 "ensemble" to force it through, requesting ``statistical`` is
    honestly reported in the ``deferred`` output key with an explanatory
    note and NO ``card_html`` is emitted for it. Wiring it for real needs a
    later task that collects per-seed runs into a multi-seed ensemble (see
    plan Task 2 self-review note + Task 4's investigation-level wiring).
  - anything else not in ``WIRED_CARDS`` (e.g. ``config``/``config_diff``,
    which read S3/Nextflow-only artifacts out of scope for this task) is
    likewise reported in ``deferred``, never silently dropped.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY  # noqa: F401

# Module-level (not deferred) so tests can monkeypatch
# ``comparison_cards.load_run_observables`` directly, same rationale as
# comparison_summary.py.
from scripts._compare.run_store_adapter import load_run_observables, resolve_run_store
from scripts._compare.report_cards.summary import build_summary_html
from scripts._compare.report_cards.standard import update_standard_report_card
from scripts._compare.report_cards.parca import update_parca_report_card
from scripts._compare.report_cards.trajectory import update_trajectory_report_card
from scripts._compare.report_cards.distribution import update_distribution_report_card
from scripts._compare.report_cards.metabolism import update_metabolism_report_card
from scripts._compare.report_cards.composition import update_composition_report_card
from scripts._compare.report_cards.trajectory import _gen_bounds
from scripts._compare.verdict import build_condition_verdict, worst
from scripts.compare_matched_trajectories import OBSERVABLES, _gen1_window, matched_stats

# {card name -> its as_step-wrapped update function}. Called via
# ``.__pb_wrapped__`` (the plain function ``as_step`` stashes) -- see module
# docstring.
_PER_OBS_CARD_STEPS = {
    "standard": update_standard_report_card,
    "parca": update_parca_report_card,
}
_ZARR_CARD_STEPS = {
    "trajectory": update_trajectory_report_card,
    "distribution": update_distribution_report_card,
    "metabolism": update_metabolism_report_card,
    "composition": update_composition_report_card,
}
WIRED_CARDS = ("summary", "standard", "parca") + tuple(_ZARR_CARD_STEPS)
DEFAULT_CARDS = list(WIRED_CARDS)

DEFERRED_CARDS = {
    "statistical": (
        "requires a MULTI-SEED ensemble (Welch t-test over >=4 seeds, per "
        "scripts/_compare/study_spec.py GRADED + v2ecoli/library/"
        "card_criteria.py's ttest branch); a single candidate-run vs single "
        "reference-run pairing has n=1 per side. grade_axis's own ttest "
        "branch already reports that honestly as 'ungraded' rather than a "
        "real test, so comparison_cards does not fabricate one -- deferred "
        "to a later task that collects per-seed runs into an ensemble."
    ),
}


def _matched_per_obs_and_trajs(cand_obs: dict, ref_obs: dict, obs_list: list
                               ) -> tuple[dict, dict, list]:
    """Build the ``per_obs``/``plot_trajs``/``v2_bounds`` shapes the
    standard/parca card builders (``scripts/comparison_report_card.py``'s
    ``eval_section``/``runs_section``/``parca_section``) consume, from the
    two ALREADY-LOADED single-run observable dicts -- one "seed" (this run
    pair) per observable, using the exact matched-timepoint stat
    (``matched_stats``/``_gen1_window``) ``comparison_summary.py`` already
    uses for its single-observable-group verdict.
    """
    cand_gen, ref_gen = cand_obs.get("_generation"), ref_obs.get("_generation")
    per_obs: dict[str, list] = {}
    plot_trajs: dict[str, dict] = {}
    for obs in obs_list:
        if obs not in cand_obs or obs not in ref_obs:
            continue
        cand_full, ref_full = cand_obs[obs], ref_obs[obs]
        cand_g1 = _gen1_window(cand_full, cand_gen) if cand_gen else cand_full
        ref_g1 = _gen1_window(ref_full, ref_gen) if ref_gen else ref_full
        stat = matched_stats(cand_g1, ref_g1)
        per_obs[obs] = [dict(stat, seed=0)] if stat else []
        # FULL trajectory (all generations), not windowed -- matches
        # comparison_report_card.py's plot_trajs contract (used for the
        # overlay plots in the `standard` card's "runs" section).
        plot_trajs[obs] = {"v2": [cand_full], "ve": [ref_full]}
    v2_bounds = _gen_bounds(cand_gen) if cand_gen else []
    return per_obs, plot_trajs, v2_bounds


def _render_per_obs_cards(name: str, per_obs: dict, plot_trajs: dict,
                          v2_bounds: list, requested: set) -> tuple[dict, dict]:
    """Render the requested subset of {standard, parca} via their own
    as_step-wrapped update functions, unmodified."""
    cards_html: dict[str, str] = {}
    card_verdicts: dict[str, dict] = {}
    if not requested:
        return cards_html, card_verdicts
    state = {"name": name, "observables": per_obs, "plot_trajs": plot_trajs,
             "v2_bounds": v2_bounds}
    for card in requested:
        func = _PER_OBS_CARD_STEPS[card].__pb_wrapped__
        out = func(state)
        cards_html[card] = out.get("card_html", "")
        card_verdicts[card] = {"verdict": out.get("verdict", "ungraded"),
                               "axes": out.get("axes") or []}
    return cards_html, card_verdicts


def _render_zarr_cards(candidate_store, reference_store, *, name: str,
                       requested: set) -> tuple[dict, dict]:
    """Render the requested subset of {trajectory, distribution, metabolism,
    composition} via their own as_step-wrapped update functions, unmodified.

    Those cards read ``state["v2_dir"]``/``state["ve_dir"]`` + a
    ``<prefix>_seed{NN}.zarr`` naming convention (see
    ``scripts/_compare/run_store_adapter.py``'s docstring) rather than an
    already-loaded observables dict. A resolved workbench run's zarr may
    not itself follow that naming convention (e.g. the vwb XArrayEmitter's
    ``store.zarr``), so this bridges via a throwaway directory holding
    ``v2ecoli_seed00.zarr``/``vecoli_seed00.zarr`` SYMLINKS onto the two
    already-resolved stores -- pure path plumbing, no new science; the card
    builders read through the symlinks exactly as they would a real
    per-seed run directory.
    """
    cards_html: dict[str, str] = {}
    card_verdicts: dict[str, dict] = {}
    if not requested:
        return cards_html, card_verdicts
    with tempfile.TemporaryDirectory(prefix="comparison_cards_") as tmp:
        os.symlink(str(Path(candidate_store).resolve()),
                  os.path.join(tmp, "v2ecoli_seed00.zarr"))
        os.symlink(str(Path(reference_store).resolve()),
                  os.path.join(tmp, "vecoli_seed00.zarr"))
        state = {"name": name, "v2_dir": tmp, "ve_dir": tmp, "seeds": 1}
        for card in requested:
            func = _ZARR_CARD_STEPS[card].__pb_wrapped__
            out = func(state)
            cards_html[card] = out.get("card_html", "")
            card_verdicts[card] = {"verdict": out.get("verdict", "ungraded"),
                                   "axes": out.get("axes") or []}
    return cards_html, card_verdicts


def comparison_cards(candidate_run, reference_run, config=None, *,
                     seeds: int = 1, observables=None, cards=None,
                     study_dir=None, runs_db=None) -> dict:
    """Compare a candidate run against a reference run and render the
    requested report cards (default: every card wired in this task --
    ``WIRED_CARDS``).

    ``candidate_run``/``reference_run`` are workbench run references (see
    ``run_store_adapter``). ``config`` is accepted for the Analysis-signature
    contract but unused. ``cards`` defaults to ``DEFAULT_CARDS``; any
    requested card not in ``WIRED_CARDS`` (``statistical``, or an unknown
    name) is reported in the returned ``deferred`` dict instead of being
    silently dropped or faked.

    Returns ``{"cards": {name: html}, "verdict": <report_card_verdict/v1
    dict, groups keyed by rendered card name>, "deferred": {name: reason}}``.
    """
    obs_list = list(observables) if observables is not None else list(OBSERVABLES)
    requested = list(cards) if cards is not None else list(DEFAULT_CARDS)
    name = str(config) if config else "comparison"

    deferred: dict[str, str] = {}
    wired = []
    for c in requested:
        if c in WIRED_CARDS:
            wired.append(c)
        elif c in DEFERRED_CARDS:
            deferred[c] = DEFERRED_CARDS[c]
        else:
            deferred[c] = f"unknown/unwired card {c!r} (not in WIRED_CARDS)"

    cand_obs = load_run_observables(candidate_run, obs_list,
                                    study_dir=study_dir, runs_db=runs_db)
    ref_obs = load_run_observables(reference_run, obs_list,
                                   study_dir=study_dir, runs_db=runs_db)
    per_obs, plot_trajs, v2_bounds = _matched_per_obs_and_trajs(
        cand_obs, ref_obs, obs_list)

    cards_html: dict[str, str] = {}
    card_verdicts: dict[str, dict] = {}

    per_obs_requested = {c for c in wired if c in _PER_OBS_CARD_STEPS}
    html, verdicts = _render_per_obs_cards(name, per_obs, plot_trajs,
                                           v2_bounds, per_obs_requested)
    cards_html.update(html)
    card_verdicts.update(verdicts)

    zarr_requested = {c for c in wired if c in _ZARR_CARD_STEPS}
    if zarr_requested:
        cand_store = resolve_run_store(candidate_run, study_dir=study_dir,
                                       runs_db=runs_db)
        ref_store = resolve_run_store(reference_run, study_dir=study_dir,
                                      runs_db=runs_db)
        html, verdicts = _render_zarr_cards(cand_store, ref_store, name=name,
                                            requested=zarr_requested)
        cards_html.update(html)
        card_verdicts.update(verdicts)

    # `verdict` is assembled from every OTHER rendered card's group (worst-of
    # rollup via build_condition_verdict -- the same function the real
    # harness uses to write report_card_verdict.json), so `summary` reflects
    # exactly what was actually graded here (never `statistical`, since it's
    # deferred, never faked).
    verdict = build_condition_verdict(name, card_verdicts)
    verdict["model_ref"] = str(candidate_run)
    verdict["reference_model"] = str(reference_run)

    if "summary" in wired:
        cards_html["summary"] = build_summary_html(verdict, seeds)

    return {"cards": cards_html, "verdict": verdict, "deferred": deferred}


class ComparisonCards(Analysis):
    """Registered wrapper around :func:`comparison_cards`.

    ``candidate_run``/``reference_run``/``seeds``/``observables``/``cards``
    are Step config (this Analysis's inputs, not sim-history state), so they
    are declared in ``config_schema`` and read off ``self.config`` in
    ``analyze``/``update`` -- same pattern as ``ComparisonSummary``.
    """

    name = "comparison_cards"
    scale = "single"
    config_schema = {
        "candidate_run": {"_type": "maybe[string]", "_default": None},
        "reference_run": {"_type": "maybe[string]", "_default": None},
        "seeds": {"_type": "integer", "_default": 1},
        "observables": {"_type": "maybe[list[string]]", "_default": None},
        "cards": {"_type": "maybe[list[string]]", "_default": None},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {"cards": "any", "verdict": "any", "deferred": "any"}

    def analyze(self, **ctx) -> dict:
        cfg = self.config
        return comparison_cards(
            cfg.get("candidate_run"), cfg.get("reference_run"),
            seeds=cfg.get("seeds") or 1, observables=cfg.get("observables"),
            cards=cfg.get("cards"))

    def update(self, state=None, interval=None):
        return self.analyze()
