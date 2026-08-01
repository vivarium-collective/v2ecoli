"""`summary` card — study-level at-a-glance scorecard: verdict pill strip +
per-observable |Δ| status heat row + seed count + gate status. Placed FIRST
in the default card ordering (see study_spec._DEFAULT_CARDS) so a reader sees
the objective bottom line before any per-card detail.

Purely informational: it carries no measure of its own and is deliberately
NOT in study_spec.GRADED, so it never gates a study by itself. Its "gate
status" line reports whether the cards that DO gate (GRADED = {parca,
statistical}) are within tolerance.

``build_summary_html`` is a pure function (no I/O, no process-bigraph deps)
so it is unit-testable in isolation -- see tests/compare/test_summary_card.py.
The Step wrapper (as_step, registered as ``summary_report_card`` via
core.link_registry['summary_report_card']) assembles the verdict dict live
from the same underlying section builders the parca/statistical cards use,
so summary is self-sufficient regardless of where it sits in a study's card
list (in particular, first -- before those cards would otherwise have run).
"""
from __future__ import annotations

import html as _html

from process_bigraph.composite import as_step

from scripts._compare.report_cards import CARD_INPUTS, CARD_OUTPUTS, REPORT_CARD_STEPS
from scripts._compare.study_spec import GRADED
from scripts._compare.theme import STATUS
from scripts._compare.verdict import worst

_UNKNOWN = {"color": "#6b7280", "glyph": "?", "label": "Ungraded"}


def _status(verdict: str) -> dict:
    return STATUS.get(verdict, _UNKNOWN)


def _pill(label: str, verdict: str) -> str:
    entry = _status(verdict)
    return (
        '<span style="display:inline-flex;align-items:center;gap:5px;'
        f'padding:3px 10px;border-radius:999px;background:{entry["color"]}22;'
        f'color:{entry["color"]};font-weight:600;font-size:12px" '
        f'title="{_html.escape(label)}: {_html.escape(entry["label"])} ({_html.escape(verdict)})">'
        f'{_html.escape(label)}: {entry["glyph"]} {_html.escape(entry["label"])}</span>'
    )


def _heat_cell(axis: dict) -> str:
    label = _html.escape(str(axis.get("label", "")))
    verdict = axis.get("verdict", "ungraded")
    entry = _status(verdict)
    median_rel = (axis.get("detail") or {}).get("median_rel")
    value = f"{median_rel * 100:.1f}%" if isinstance(median_rel, (int, float)) else "--"
    return (
        '<div style="display:flex;flex-direction:column;align-items:center;gap:2px;'
        f'padding:6px 12px;border-radius:6px;background:{entry["color"]}1a;min-width:76px">'
        f'<div style="font-size:12px;color:#6b7280">{label}</div>'
        f'<div style="font-size:16px;line-height:1">{entry["glyph"]}</div>'
        f'<div style="font-size:12px;font-weight:600">{value}</div>'
        f'<div style="font-size:10px;color:#6b7280">{_html.escape(entry["label"])} '
        f'({_html.escape(verdict)})</div>'
        '</div>'
    )


def _legend() -> str:
    items = "".join(
        f'<span style="margin-right:14px">{entry["glyph"]} '
        f'{_html.escape(entry["label"])} ({_html.escape(key)})</span>'
        for key, entry in STATUS.items()
    )
    return f'<div style="font-size:11px;color:#6b7280;margin-top:8px">{items}</div>'


def _gate_verdict(groups: dict) -> str:
    """Worst verdict among the GATING groups only (parca/statistical)."""
    return worst(d.get("verdict", "ungraded") for g, d in groups.items() if g in GRADED)


def build_summary_html(verdict: dict, seeds: int) -> str:
    """Pure builder: an at-a-glance scorecard for one study.

    `verdict` is the standard report_card_verdict/v1 shape: {"overall": str,
    "groups": {group_name: {"verdict": str, "axes": [{"label", "verdict",
    "detail": {"median_rel": float}}, ...]}}}. `seeds` is the study's seed
    count. Returns a self-contained HTML fragment. Status is always conveyed
    by glyph + label together, never color alone.
    """
    verdict = verdict or {}
    groups = verdict.get("groups") or {}
    overall = verdict.get("overall", "ungraded")
    gate_verdict = _gate_verdict(groups)

    axes = [ax for g in groups.values() for ax in (g.get("axes") or [])]

    pills = _pill("Overall", overall) + " " + _pill("Gate", gate_verdict)
    heat_row = "".join(_heat_cell(ax) for ax in axes) if axes else (
        '<p style="color:#6b7280;font-size:12px">no graded observables</p>')

    return (
        '<div class="summary-card">'
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">{pills}'
        f'<span style="font-size:12px;color:#6b7280">{seeds} seeds</span></div>'
        f'<p style="font-size:12px;color:#6b7280;margin:8px 0 4px">'
        f'gate status: <strong>{_html.escape(gate_verdict)}</strong> '
        f'(graded cards: {", ".join(sorted(g for g in groups if g in GRADED)) or "none"})</p>'
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:6px">{heat_row}</div>'
        f'{_legend()}'
        '</div>'
    )


@as_step(inputs=CARD_INPUTS, outputs=CARD_OUTPUTS, name="summary_report_card",
         aliases=["summary"])
def update_summary_report_card(state):
    # Self-sufficient: reuse the parca/statistical section builders directly
    # rather than reading their already-rendered verdicts, since `summary`
    # renders FIRST in the default card ordering -- before those cards would
    # otherwise have run in a naive left-to-right pass over spec.cards.
    from scripts._compare.report_cards.parca import _parca_section_and_axes
    from scripts._compare.report_cards.statistical import _statistical_html_axes

    groups = {}
    try:
        _, parca_axes = _parca_section_and_axes(
            state["name"], state["observables"], state["plot_trajs"], state["v2_bounds"])
        groups["parca"] = {"verdict": worst(a["verdict"] for a in parca_axes), "axes": parca_axes}
    except Exception:
        pass
    try:
        _, stat_verdict, stat_axes = _statistical_html_axes(
            state["name"], state["observables"], state["variant"])
        groups["statistical"] = {"verdict": stat_verdict or "ungraded", "axes": stat_axes}
    except Exception:
        pass

    overall = worst(g["verdict"] for g in groups.values())
    html = build_summary_html({"overall": overall, "groups": groups}, state.get("seeds") or 1)
    # Informational: no measure of its own, never a gate.
    return {"card_html": html, "verdict": "ungraded", "axes": []}


REPORT_CARD_STEPS["summary_report_card"] = update_summary_report_card
