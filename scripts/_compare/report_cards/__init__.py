"""Registry of modular comparison report cards (as_step Steps).

Cards are process-bigraph Steps registered in REPORT_CARD_STEPS and
accessible via core.link_registry['<name>_report_card']. Each card's
update function returns {card_html, verdict, axes}.
"""
from __future__ import annotations

import html as _html

Section = dict  # {title, kind, html, anchor, verdict?}

# Typed contract for a report-card Step. Pragmatic: structural fields typed;
# the per-seed stat records under `observables` stay loose. Strings pinned in
# Task 1 Step 1 — substitute the validated forms if any printed BAD.
CARD_INPUTS = {
    "name": "string", "condition": "string",
    "seeds": "integer", "generations": "integer", "variant": "integer",
    "observables": "tree[list[map]]", "plot_trajs": "tree[map]",
    "v2_bounds": "list[float]", "config": "tree[map]",
    "v2_dir": "string", "ve_dir": "string",
}
CARD_OUTPUTS = {
    "card_html": "overwrite[string]",
    "verdict": "overwrite[string]",
    "axes": "overwrite[list[map]]",
}

REPORT_CARD_STEPS: dict[str, type] = {}   # {name: StepCls}; populated by the card modules


def _row_table(rows: list) -> str:
    cells = []
    for r in rows:
        label = _html.escape(str(r.get("label", "")))
        left = _html.escape(str(r.get("left", "")))
        right = _html.escape(str(r.get("right", "")))
        verdict = _html.escape(str(r.get("verdict", "")))
        reason = _html.escape(str(r.get("reason", "")))
        cells.append(
            f'<tr><td style="padding:2px 10px">{label}</td>'
            f'<td style="padding:2px 10px">{left}</td>'
            f'<td style="padding:2px 10px">{right}</td>'
            f'<td style="padding:2px 10px">{verdict}</td>'
            f'<td style="padding:2px 10px;color:#6b7280">{reason}</td></tr>')
    return ('<table style="border-collapse:collapse;font-size:13px">'
            '<thead><tr style="text-align:left">'
            '<th style="padding:2px 10px">observable</th><th>vEcoli</th>'
            '<th>v2ecoli</th><th>verdict</th><th>note</th></tr></thead><tbody>'
            + "".join(cells) + "</tbody></table>")


def _sections_to_html(sections: list) -> str:
    """Render a card's section dicts into one HTML fragment. A section with an
    `html` field is emitted as-is; a section with `rows` is rendered as a
    table (eval_section / parca_section produce rows)."""
    parts = []
    for sec in sections:
        if sec.get("title"):
            parts.append(f'<h3 style="margin:14px 0 6px">{_html.escape(str(sec["title"]))}</h3>')
        if sec.get("desc"):
            parts.append(f'<p style="color:#6b7280;font-size:12px">{_html.escape(str(sec["desc"]))}</p>')
        if sec.get("html"):
            parts.append(sec["html"])
        elif sec.get("rows"):
            parts.append(_row_table(sec["rows"]))
    return "".join(parts)


# Built-in card modules register themselves into REPORT_CARD_STEPS on import.
from scripts._compare.report_cards import standard, statistical  # noqa: E402,F401
from scripts._compare.report_cards import parca, config_diff, config  # noqa: E402,F401
from scripts._compare.report_cards import trajectory  # noqa: E402,F401
