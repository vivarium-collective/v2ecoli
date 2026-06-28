"""Registry of modular comparison report cards.

A card is Callable[[CardContext], Section | list[Section]]. Register with the
@report_card("name") decorator; assign by name in the comparison manifest.
Cards are thin wrappers over the existing section functions and Chris Long's
on-main card library (v2ecoli.library.report_card).
"""
from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Any, Callable

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


@dataclass
class CardContext:
    config_name: str
    variant: int
    v2_dir: str
    ve_dir: str
    seeds: int
    gens: int
    per_obs: dict = field(default_factory=dict)
    plot_trajs: dict = field(default_factory=dict)
    v2_bounds: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


Card = Callable[[CardContext], "Section | list[Section]"]
REGISTRY: dict[str, Card] = {}


def report_card(name: str) -> Callable[[Card], Card]:
    def deco(fn: Card) -> Card:
        REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> Card:
    if name not in REGISTRY:
        raise KeyError(f"unknown report card {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def all_names() -> list[str]:
    return sorted(REGISTRY)


def render(name: str, ctx: CardContext) -> list[Section]:
    out = get(name)(ctx)
    return out if isinstance(out, list) else [out]


# Built-in card modules (standard/statistical/parca/config_diff) are imported
# here once they exist (Tasks 4–5) so importing this package registers them.
from scripts._compare.report_cards import standard, statistical  # noqa: E402,F401
from scripts._compare.report_cards import parca, config_diff, config  # noqa: E402,F401
