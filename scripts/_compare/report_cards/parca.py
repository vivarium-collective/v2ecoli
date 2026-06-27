"""`parca` card — ParCa / initial-state match for one config."""
from __future__ import annotations

from scripts._compare.report_cards import report_card, CardContext, Section


@report_card("parca")
def parca_card(ctx: CardContext) -> Section:
    from scripts.comparison_report_card import parca_section
    # parca_section takes the cond_data map; build a single-cond slice.
    sec = parca_section({ctx.config_name: (ctx.per_obs, ctx.plot_trajs, ctx.v2_bounds)})
    sec["anchor"] = f"{ctx.config_name}-parca"
    return sec
