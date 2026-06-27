"""`standard` card — matched-time run trajectories + evaluation (the lighter
card). Thin wrapper over comparison_report_card.runs_section / eval_section.
The evaluation section also carries a card-level verdict + axes (one per
observable) so the comparison can gate on it via report_card_axis."""
from __future__ import annotations

from scripts._compare.report_cards import report_card, CardContext, Section
from scripts._compare.verdict import worst


@report_card("standard")
def standard_card(ctx: CardContext) -> list[Section]:
    # Imported lazily: comparison_report_card imports heavy deps; importing it at
    # module load would slow registry import and risk a cycle.
    from scripts.comparison_report_card import runs_section, eval_section
    runs = runs_section(ctx.config_name, ctx.per_obs, ctx.plot_trajs, ctx.v2_bounds)
    ev = eval_section(ctx.config_name, ctx.per_obs)
    axes = []
    for row in ev.get("rows", []):
        v = row.get("verdict", "ungraded")
        if v == "not_compared":
            v = "ungraded"
        axes.append({
            "id": f"standard.{row['label']}",
            "label": row["label"],
            "verdict": v,
            "value": row.get("median_rel"),
            "meter": row.get("reason", ""),
            "detail": {"median_rel": row.get("median_rel"),
                       "max_rel": row.get("max_rel")},
        })
    ev["verdict"] = worst(a["verdict"] for a in axes)
    ev["verdict_axes"] = axes
    return [runs, ev]
