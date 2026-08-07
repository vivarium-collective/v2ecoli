"""Comparison convergence Phase 1, Task 4: a minimal comparison Analysis that
renders ONE report card (`summary`) from two workbench runs — a candidate and
a reference. Proves the comparison rendering can run as a general workbench
Analysis, reusing the existing comparison code (``scripts/_compare``)
read-only rather than reinventing tolerance/verdict logic.

Unlike the SQL/DuckDB-history Analyses in this package (``growth_overlay``,
``scorecard``, ``dummy`` — one composite run's ``history_sql``), this Analysis
compares TWO separate runs, so it does not fit ``Analysis``'s conn-based
``inputs()``/``outputs()`` contract. It keeps the same *registration*
mechanism (``Analysis`` subclass + ``name`` -> ``ANALYSIS_REGISTRY`` /
``register_post_sim`` via ``__init_subclass__``) but overrides ``inputs()``,
``outputs()``, and ``update()`` for its own {candidate_run, reference_run,
config, seeds} -> {card_html, verdict} contract.

Pipeline (all reused, read-only, from ``scripts/_compare`` /
``scripts/compare_matched_trajectories`` / ``scripts/comparison_report_card``
— no new science):

  1. ``scripts._compare.run_store_adapter.load_run_observables`` — load each
     run's ``{obs: (times, values)}`` observables (Task 3's adapter), called
     once per run (candidate, reference).
  2. ``scripts.compare_matched_trajectories.matched_stats`` /
     ``grade`` / ``_gen1_window`` — the same matched-time median-|Δ| stat and
     5%/10% tolerance bands ``comparison_report_card.build()``/``eval_section``
     use per seed, applied here to the one candidate/reference trajectory pair
     per observable.
  3. ``scripts._compare.verdict.worst`` + ``scripts._compare.report_cards.
     summary.build_summary_html`` — fold the per-observable verdicts into the
     standard ``report_card_verdict/v1`` shape and render the `summary` card.
"""
from __future__ import annotations

from typing import Any

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY  # noqa: F401

# Module-level (not deferred) so tests can monkeypatch
# ``comparison_summary.load_run_observables`` directly -- these are thin,
# already-imported-elsewhere modules (no heavy optional deps like altair).
from scripts._compare.run_store_adapter import load_run_observables
from scripts._compare.report_cards.summary import build_summary_html
from scripts._compare.verdict import worst
from scripts.compare_matched_trajectories import (
    OBSERVABLES, _gen1_window, grade, matched_stats)


def comparison_summary(candidate_run, reference_run, config=None, *,
                       seeds: int = 1, observables=None,
                       study_dir=None, runs_db=None) -> dict:
    """Compare a candidate run against a reference run and render the
    `summary` report card.

    ``candidate_run`` / ``reference_run`` are workbench run references (see
    ``run_store_adapter.load_run_observables`` — a ``.zarr`` path, a run
    directory, a bare run id resolved via ``study_dir``/``runs_db``, or a
    mapping). ``config`` is accepted for the Analysis-signature contract but
    unused (no comparison behavior is currently config-driven beyond
    ``observables``/``seeds``). ``observables`` defaults to the standard
    comparison set (``scripts.compare_matched_trajectories.OBSERVABLES``).

    Returns ``{"card_html": <html str>, "verdict": <report_card_verdict/v1
    dict>}``.
    """
    obs_list = list(observables) if observables is not None else list(OBSERVABLES)
    cand = load_run_observables(candidate_run, obs_list,
                                study_dir=study_dir, runs_db=runs_db)
    ref = load_run_observables(reference_run, obs_list,
                               study_dir=study_dir, runs_db=runs_db)
    cand_gen, ref_gen = cand.get("_generation"), ref.get("_generation")

    axes: list[dict[str, Any]] = []
    for obs in obs_list:
        if obs not in cand or obs not in ref:
            axes.append({"id": f"observables.{obs}", "label": obs,
                        "verdict": "ungraded", "detail": {}})
            continue
        cand_traj = _gen1_window(cand[obs], cand_gen) if cand_gen else cand[obs]
        ref_traj = _gen1_window(ref[obs], ref_gen) if ref_gen else ref[obs]
        stat = matched_stats(cand_traj, ref_traj)
        if stat is None:
            axes.append({"id": f"observables.{obs}", "label": obs,
                        "verdict": "ungraded", "detail": {}})
            continue
        axes.append({
            "id": f"observables.{obs}", "label": obs,
            "verdict": grade(stat["median_rel"]),
            "value": stat["median_rel"],
            "detail": {"median_rel": stat["median_rel"], "max_rel": stat["max_rel"],
                      "init_v2": stat["init_v2"], "init_ve": stat["init_ve"]},
        })

    group_verdict = worst(ax["verdict"] for ax in axes)
    verdict = {
        "schema": "report_card_verdict/v1",
        "model_ref": str(candidate_run), "reference_model": str(reference_run),
        "overall": group_verdict,
        "groups": {"observables": {"verdict": group_verdict, "axes": axes}},
    }
    html = build_summary_html(verdict, seeds)
    return {"card_html": html, "verdict": verdict}


class ComparisonSummary(Analysis):
    """Registered wrapper around :func:`comparison_summary`.

    ``candidate_run``/``reference_run``/``seeds``/``observables`` are Step
    config (this Analysis's inputs, not the sim-history state the conn-based
    Analyses read), so they are declared in ``config_schema`` and read off
    ``self.config`` in ``analyze``/``update``.
    """

    name = "comparison_summary"
    scale = "single"
    config_schema = {
        "candidate_run": {"_type": "maybe[string]", "_default": None},
        "reference_run": {"_type": "maybe[string]", "_default": None},
        "seeds": {"_type": "integer", "_default": 1},
        "observables": {"_type": "maybe[list[string]]", "_default": None},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {"card_html": "string", "verdict": "any"}

    def analyze(self, **ctx) -> dict:
        cfg = self.config
        return comparison_summary(
            cfg.get("candidate_run"), cfg.get("reference_run"),
            seeds=cfg.get("seeds") or 1, observables=cfg.get("observables"))

    def update(self, state=None, interval=None):
        return self.analyze()
