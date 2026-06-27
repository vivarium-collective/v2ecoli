"""`standard` card — matched-time run trajectories + evaluation (the lighter
card). Thin wrapper over comparison_report_card.runs_section / eval_section."""
from __future__ import annotations

from scripts._compare.report_cards import report_card, CardContext, Section


@report_card("standard")
def standard_card(ctx: CardContext) -> list[Section]:
    # Imported lazily: comparison_report_card imports heavy deps; importing it at
    # module load would slow registry import and risk a cycle.
    from scripts.comparison_report_card import runs_section, eval_section
    return [
        runs_section(ctx.config_name, ctx.per_obs, ctx.plot_trajs, ctx.v2_bounds),
        eval_section(ctx.config_name, ctx.per_obs),
    ]
