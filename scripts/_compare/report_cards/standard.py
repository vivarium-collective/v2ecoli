"""`standard` card — matched-time runs + per-observable evaluation. as_step Step;
the harness invokes it via core.access('standard_report_card')."""
from __future__ import annotations

from process_bigraph.composite import as_step

from scripts._compare.report_cards import (
    report_card, CardContext, Section, CARD_INPUTS, CARD_OUTPUTS,
    _sections_to_html, REPORT_CARD_STEPS)
from scripts._compare.verdict import worst


def _standard_sections_and_axes(name, per_obs, plot_trajs, v2_bounds):
    from scripts.comparison_report_card import runs_section, eval_section
    runs = runs_section(name, per_obs, plot_trajs, v2_bounds)
    ev = eval_section(name, per_obs)
    axes = []
    for row in ev.get("rows", []):
        v = row.get("verdict", "ungraded")
        if v == "not_compared":
            v = "ungraded"
        axes.append({"id": f"standard.{row.get('label', '')}", "label": row.get("label", ""),
                     "verdict": v, "value": row.get("median_rel"),
                     "meter": row.get("reason", ""),
                     "detail": {"median_rel": row.get("median_rel"),
                                "max_rel": row.get("max_rel")}})
    return [runs, ev], axes


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="standard_report_card",
         aliases=["standard"])
def update_standard_report_card(state):
    sections, axes = _standard_sections_and_axes(
        state["name"], state["observables"], state["plot_trajs"], state["v2_bounds"])
    return {"card_html": _sections_to_html(sections),
            "verdict": worst(a["verdict"] for a in axes), "axes": axes}


REPORT_CARD_STEPS["standard_report_card"] = update_standard_report_card


# --- transitional: keep the old registry wrapper until the Task 5 cutover ---
@report_card("standard")
def standard_card(ctx: CardContext) -> list[Section]:
    sections, axes = _standard_sections_and_axes(
        ctx.config_name, ctx.per_obs, ctx.plot_trajs, ctx.v2_bounds)
    ev = sections[1]
    ev["verdict"] = worst(a["verdict"] for a in axes)
    ev["verdict_axes"] = axes
    return sections
