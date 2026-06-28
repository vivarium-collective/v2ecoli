"""`parca` card — ParCa / initial-state match, graded on per-mass t~0 |Δ|.
as_step Step invoked via core.access('parca_report_card')."""
from __future__ import annotations

from process_bigraph.composite import as_step

from scripts._compare.report_cards import (
    report_card, CardContext, Section, CARD_INPUTS, CARD_OUTPUTS,
    _sections_to_html, REPORT_CARD_STEPS)
from scripts._compare.verdict import worst


def _parca_section_and_axes(name, per_obs, plot_trajs, v2_bounds):
    from scripts.comparison_report_card import parca_section
    sec = parca_section({name: (per_obs, plot_trajs, v2_bounds)})
    sec["anchor"] = f"{name}-parca"
    sec["title"] = f"{name} — ParCa / initial-state match"
    axes = []
    for row in sec.get("rows", []):
        if "median_rel" not in row:
            continue
        v = row.get("verdict", "ungraded")
        if v == "not_compared":
            v = "ungraded"
        axes.append({"id": f"parca.{row['label']}", "label": row["label"], "verdict": v,
                     "value": row.get("median_rel"), "meter": row.get("reason", ""),
                     "detail": {"init_rel": row.get("median_rel")}})
    return sec, axes


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="parca_report_card",
         aliases=["parca"])
def update_parca_report_card(state):
    sec, axes = _parca_section_and_axes(
        state["name"], state["observables"], state["plot_trajs"], state["v2_bounds"])
    return {"card_html": _sections_to_html([sec]),
            "verdict": worst(a["verdict"] for a in axes), "axes": axes}


REPORT_CARD_STEPS["parca_report_card"] = update_parca_report_card


@report_card("parca")
def parca_card(ctx: CardContext) -> Section:
    sec, axes = _parca_section_and_axes(
        ctx.config_name, ctx.per_obs, ctx.plot_trajs, ctx.v2_bounds)
    sec["verdict"] = worst(a["verdict"] for a in axes)
    sec["verdict_axes"] = axes
    return sec
