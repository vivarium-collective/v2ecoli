"""`statistical` card — graded equivalence (violin/strip). as_step Step."""
from __future__ import annotations

from process_bigraph.composite import as_step

from scripts._compare.report_cards import (
    CARD_INPUTS, CARD_OUTPUTS, REPORT_CARD_STEPS)
from scripts._compare.report_card_section import build_report_card


def _statistical_html_axes(name, per_obs, variant):
    from scripts.comparison_report_card import OBSERVABLES, CARD_KEY, EXTRA_AXES, TOL
    left, right = {}, {}
    for obs in OBSERVABLES:
        ck = CARD_KEY.get(obs, obs)
        left[ck] = [s["ve_mean"] for s in per_obs.get(obs, [])]
        right[ck] = [s["v2_mean"] for s in per_obs.get(obs, [])]
    vjson, html = build_report_card(left, right, extra_axes=EXTRA_AXES,
                                    model_ref=f"v2ecoli @ {name} variant {variant}", tol_rel=TOL)
    axes = [ax for g in (vjson.get("groups") or {}).values() for ax in (g.get("axes") or [])]
    return html, vjson.get("overall"), axes


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="statistical_report_card",
         aliases=["statistical"])
def update_statistical_report_card(state):
    html, verdict, axes = _statistical_html_axes(
        state["name"], state["observables"], state["variant"])
    return {"card_html": html, "verdict": verdict or "ungraded", "axes": axes}


REPORT_CARD_STEPS["statistical_report_card"] = update_statistical_report_card
