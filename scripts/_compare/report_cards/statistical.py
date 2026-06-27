"""`statistical` card — Chris Long's graded equivalence card (violin/strip +
<details> dropdown viz bars + within_tol/drift/mismatch pills). Mirrors the
per_obs -> build_report_card mapping in comparison_report_card.report_card()."""
from __future__ import annotations

from scripts._compare.report_cards import report_card, CardContext, Section
from scripts._compare.report_card_section import build_report_card


@report_card("statistical")
def statistical_card(ctx: CardContext) -> Section:
    from scripts.comparison_report_card import OBSERVABLES, CARD_KEY, EXTRA_AXES, TOL
    left: dict = {}   # vEcoli (reference)
    right: dict = {}  # v2ecoli (measured)
    for obs in OBSERVABLES:
        ck = CARD_KEY.get(obs, obs)
        left[ck] = [s["ve_mean"] for s in ctx.per_obs.get(obs, [])]
        right[ck] = [s["v2_mean"] for s in ctx.per_obs.get(obs, [])]
    vjson, html = build_report_card(
        left, right, extra_axes=EXTRA_AXES,
        model_ref=f"v2ecoli @ {ctx.config_name} variant {ctx.variant}", tol_rel=TOL)
    return {"title": f"{ctx.config_name} — statistical equivalence",
            "kind": "content", "anchor": f"{ctx.config_name}-statistical",
            "html": html, "verdict": vjson.get("overall")}
