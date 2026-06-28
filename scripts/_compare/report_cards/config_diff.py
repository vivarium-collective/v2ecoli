"""`config_diff` card — vEcoli vs v2ecoli config comparison (S3/Nextflow). as_step Step."""
from __future__ import annotations

from process_bigraph.composite import as_step

from scripts._compare.report_cards import (
    report_card, CardContext, Section, CARD_INPUTS, CARD_OUTPUTS,
    _sections_to_html, REPORT_CARD_STEPS)


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="config_diff_report_card",
         aliases=["config_diff"])
def update_config_diff_report_card(state):
    from scripts.comparison_report_card import config_sections_for
    secs = config_sections_for(state["name"], state["v2_dir"], state["ve_dir"])
    return {"card_html": _sections_to_html(secs), "verdict": "ungraded", "axes": []}


REPORT_CARD_STEPS["config_diff_report_card"] = update_config_diff_report_card


@report_card("config_diff")
def config_diff_card(ctx: CardContext) -> list[Section]:
    from scripts.comparison_report_card import config_sections_for
    return config_sections_for(ctx.config_name, ctx.v2_dir, ctx.ve_dir)
