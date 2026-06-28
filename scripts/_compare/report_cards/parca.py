"""`parca` card — ParCa / initial-state match for one config.

Graded: the t~0 (first matched emit) initial masses must agree between v2ecoli
and vEcoli — both sims start from their engine's ParCa fit, so this is the
same-initial-state evidence the per-condition dynamics build on. Each mass
observable's initial |Δ| is graded (5% within / 10% drift); the worst is the
card verdict, so a ParCa study can gate its follow-ups.
"""
from __future__ import annotations

from scripts._compare.report_cards import report_card, CardContext, Section
from scripts._compare.verdict import worst


@report_card("parca")
def parca_card(ctx: CardContext) -> Section:
    from scripts.comparison_report_card import parca_section
    # parca_section takes the cond_data map; build a single-cond slice.
    sec = parca_section({ctx.config_name: (ctx.per_obs, ctx.plot_trajs, ctx.v2_bounds)})
    sec["anchor"] = f"{ctx.config_name}-parca"
    sec["title"] = f"{ctx.config_name} — ParCa / initial-state match"
    # Grade the per-mass initial-state agreement. parca_section already grades
    # each mass row (_grade over the init |Δ|); expose those as verdict_axes so
    # the card produces a verdict the framework can gate on. Rows without a
    # graded value (condition headers, missing observables) are skipped.
    axes = []
    for row in sec.get("rows", []):
        if "median_rel" not in row:
            continue
        v = row.get("verdict", "ungraded")
        if v == "not_compared":
            v = "ungraded"
        axes.append({
            "id": f"parca.{row['label']}",
            "label": row["label"],
            "verdict": v,
            "value": row.get("median_rel"),
            "meter": row.get("reason", ""),
            "detail": {"init_rel": row.get("median_rel")},
        })
    sec["verdict"] = worst(a["verdict"] for a in axes)
    sec["verdict_axes"] = axes
    return sec
