"""`config_diff` card — vEcoli vs v2ecoli config comparison for one config."""
from __future__ import annotations

from scripts._compare.report_cards import report_card, CardContext, Section


@report_card("config_diff")
def config_diff_card(ctx: CardContext) -> list[Section]:
    from scripts.comparison_report_card import config_sections_for
    return config_sections_for(ctx.config_name, ctx.v2_dir, ctx.ve_dir)
