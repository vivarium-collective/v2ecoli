"""Comparison convergence Phase 2, Task 3: an investigation-level cross-config
matrix Analysis -- configs x observables verdicts, reusing the existing
``reports/_summary`` matrix builder (``aggregate.py``'s ``matrix`` structure +
``render.py``'s ``_matrix_table``) read-only, rather than hand-rolling a new
table renderer.

Task 2's ``comparison_cards``/``comparison_summary`` Analyses each produce ONE
config's verdict dict (``report_card_verdict/v1``: ``{"overall", "groups":
{group: {"verdict", "axes": [{"label", "verdict", "detail": {"median_rel",
...}}]}}}``). This module fans those per-config verdicts OUT across configs
into the same ``{"columns": [...], "rows": [{"study": ..., "cells": {label:
verdict}}]}`` matrix shape ``reports/_summary/aggregate.py``'s ``aggregate()``
builds for one investigation's per-STUDY axes -- here "study" is repurposed as
"config": an investigation-level view across configs rather than one study's
cards.

``render._matrix_table`` renders that shape as a verdict-colored HTML table
(glyph+label per cell, ``theme.STATUS``-driven CSS classes), which is exactly
the "existing renderer" this task calls for. Its cell template, however,
carries no per-axis numeric value -- elsewhere in this codebase
(``report_cards/summary.py``'s ``_heat_cell``) the median-|delta| magnitude is
additionally surfaced next to the glyph, and this Analysis is asked to do the
same for the matrix. Rather than reinvent ``_matrix_table``'s table markup /
CSS / escaping / glyph+label lookup, ``comparison_matrix`` calls the REAL
function to get the verdict-graded skeleton, then does a minimal,
order-matched regex splice to append each graded cell's "Delta N.N%" (from
``detail.median_rel``) -- the one thing ``_matrix_table`` doesn't already do.
Table markup, CSS classes, colors, glyphs, and labels are 100% reused from
``render.py``/``theme.py``; only the delta annotation is new, thin, and
mechanical (an adapter, not a re-render).
"""
from __future__ import annotations

import re
from typing import Any

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY  # noqa: F401

# Module-level (not deferred) so tests can reach the same objects this module
# uses, same rationale as comparison_summary.py/comparison_cards.py.
from reports._summary import render as _render
from scripts._compare import theme

# Matches exactly the graded `<td>` cells `_matrix_table` emits (verdict-none
# cells -- a config missing that observable -- deliberately excluded, same as
# the delta lookup below).
_CELL_RE = re.compile(r'(<td class="verdict-(?:within_tol|drift|mismatch)">[^<]*)</td>')

_EXTRA_CSS = (
    ".comparison-matrix .delta{display:block;font-size:0.75em;opacity:0.85}"
    ".comparison-matrix .matrix-legend{font-size:0.8em;"
    "color:var(--muted,#6b7280);margin:6px 0 14px}"
    ".comparison-matrix .matrix-legend .legend-item{margin-right:14px}"
)


def _fmt_delta(median_rel: Any) -> str | None:
    if not isinstance(median_rel, (int, float)):
        return None
    return f"{abs(median_rel) * 100:.1f}%"


def _config_verdicts_to_matrix(config_verdicts: dict[str, dict]) -> dict[str, Any]:
    """Adapter: ``{config_name: report_card_verdict/v1}`` -> the
    ``{"columns": [...], "rows": [{"study", "cells"}]}`` shape
    ``reports/_summary``'s ``aggregate()``/``render()`` expect, plus a
    parallel per-cell delta annotation (``_deltas``, popped before the shape
    is handed to ``_matrix_table``) that ``_inject_deltas`` uses to decorate
    the reused table. One row per config (the config name fills the `study`
    slot); one column per observable label seen across ANY config's axes, in
    first-seen order.
    """
    columns: list[str] = []
    rows: list[dict[str, Any]] = []
    for config_name, verdict in (config_verdicts or {}).items():
        cells: dict[str, str] = {}
        deltas: dict[str, str] = {}
        for group in ((verdict or {}).get("groups") or {}).values():
            for axis in group.get("axes", []) or []:
                label = axis.get("label")
                if not label:
                    continue
                if label not in columns:
                    columns.append(label)
                cells[label] = axis.get("verdict")
                delta = _fmt_delta((axis.get("detail") or {}).get("median_rel"))
                if delta is not None:
                    deltas[label] = delta
        rows.append({"study": config_name, "cells": cells, "_deltas": deltas})
    return {"columns": columns, "rows": rows}


def _inject_deltas(table_html: str, matrix: dict[str, Any]) -> str:
    """Splice each graded cell's "Delta N.N%" into HTML `_matrix_table`
    already produced from `matrix` (columns/rows, ignoring the `_deltas`
    key). `_matrix_table` walks rows then columns in that exact order when
    emitting `<td>`s, so the deltas stream in the same order -- matched only
    to cells that actually got a `verdict-{within_tol,drift,mismatch}` class
    (an ungraded or missing observable is left as `_matrix_table` rendered
    it, untouched)."""
    deltas_in_order: list[str | None] = []
    for row in matrix["rows"]:
        for col in matrix["columns"]:
            v = row["cells"].get(col)
            if v in ("within_tol", "drift", "mismatch"):
                deltas_in_order.append(row["_deltas"].get(col))
    it = iter(deltas_in_order)

    def _repl(m: re.Match) -> str:
        delta = next(it, None)
        if not delta:
            return m.group(0)
        return f'{m.group(1)}<br><span class="delta">Δ {delta}</span></td>'

    return _CELL_RE.sub(_repl, table_html)


def _legend_html() -> str:
    """Glyph+label legend for every graded status, sourced from
    ``theme.STATUS`` -- the same shared, dataviz-validated palette the
    report cards use (never color alone)."""
    items = "".join(
        f'<span class="legend-item">{entry["glyph"]} {entry["label"]} ({key})</span>'
        for key, entry in theme.STATUS.items()
    )
    return f'<div class="matrix-legend">{items}</div>'


def comparison_matrix(config_verdicts: dict[str, dict]) -> dict[str, str]:
    """Render the configs x observables verdict matrix from Task 2's
    per-config verdict dicts (``comparison_summary``/``comparison_cards``'s
    ``verdict`` output, one per config), reusing
    ``reports/_summary.render._matrix_table`` for the table/CSS/glyph/label
    machinery -- see module docstring for the delta-annotation adapter.

    Returns ``{"matrix_html": <self-contained HTML fragment>}``.
    """
    matrix = _config_verdicts_to_matrix(config_verdicts)
    summary_matrix = {
        "columns": matrix["columns"],
        "rows": [{"study": r["study"], "cells": r["cells"]} for r in matrix["rows"]],
    }
    table_html = _render._matrix_table({"matrix": summary_matrix})
    table_html = _inject_deltas(table_html, matrix)
    html = (
        f'<div class="comparison-matrix"><style>{_render._BASE_CSS}\n{_EXTRA_CSS}</style>'
        f"{table_html}{_legend_html()}</div>"
    )
    return {"matrix_html": html}


class ComparisonMatrix(Analysis):
    """Registered wrapper around :func:`comparison_matrix`.

    ``config_verdicts`` (config name -> ``report_card_verdict/v1`` dict) is
    Step config (this Analysis's input, not sim-history state, and not
    scoped to a single run the way ``comparison_summary``/``comparison_cards``
    are) -- an investigation-level Analysis consuming several configs' worth
    of already-graded verdicts, not raw run data.
    """

    name = "comparison_matrix"
    scale = "single"
    config_schema = {
        "config_verdicts": {"_type": "maybe[map]", "_default": None},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {"matrix_html": "string"}

    def analyze(self, **ctx) -> dict:
        return comparison_matrix(self.config.get("config_verdicts") or {})

    def update(self, state=None, interval=None):
        return self.analyze()
