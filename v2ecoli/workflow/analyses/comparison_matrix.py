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
function to get the verdict-graded skeleton, then decorates it with each
graded cell's "Delta N.N%" (from ``detail.median_rel``) -- the one thing
``_matrix_table`` doesn't already do. Table markup, CSS classes, colors,
glyphs, and labels are 100% reused from ``render.py``/``theme.py``; only the
delta annotation is new.

Delta injection is KEYED, not positional (fix-round 1): an earlier version
spliced deltas in by a flat, order-assumed pass over every graded `<td>`,
which a future `_matrix_table` change that reorders cells (resorted columns,
an inserted row) but keeps the same classes/count could silently mislabel --
no crash, just the wrong |Delta| on the wrong (config, observable) cell.
``_inject_deltas`` instead re-derives cell identity from the table's OWN
structure: it reads the header's column order (``_parse_header_columns``)
and, per body row, the row's config identity from its
``<td class="study">`` cell (``_parse_body_rows``), then zips each row's
cells against the header's column order to look up that (config, observable)
pair's delta from ``matrix``'s per-config ``_deltas``. This survives a
`_matrix_table` change that resorts columns (header and row cells move
together, from the same ``cols`` loop) and FAILS LOUD
(``_MatrixStructureError``) the moment that structure breaks -- a row whose
study cell isn't a known config, or whose cell count doesn't match the
header -- rather than guessing.
"""
from __future__ import annotations

import html as _html
import re
from typing import Any

from v2ecoli.workflow.analysis import Analysis, ANALYSIS_REGISTRY  # noqa: F401

# Module-level (not deferred) so tests can reach the same objects this module
# uses, same rationale as comparison_summary.py/comparison_cards.py.
from reports._summary import render as _render
from scripts._compare import theme

# Structural parsing of `_matrix_table`'s own (fixed) markup -- see module
# docstring "Delta injection is KEYED, not positional".
_HEADER_ROW_RE = re.compile(r"<thead><tr>(.*?)</tr></thead>", re.DOTALL)
_TH_RE = re.compile(r"<th[^>]*>(.*?)</th>")
_TBODY_RE = re.compile(r"<tbody>(.*)</tbody>", re.DOTALL)
_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
_STUDY_CELL_RE = re.compile(r'<td class="study">(.*?)</td>')
_DATA_CELL_RE = re.compile(r'<td class="([\w-]+)">(.*?)</td>', re.DOTALL)

_GRADED_CLASSES = ("verdict-within_tol", "verdict-drift", "verdict-mismatch")


class _MatrixStructureError(ValueError):
    """The reused ``_matrix_table`` output no longer has the row=config /
    per-row-cells=header-columns structure ``_inject_deltas`` needs to place
    a delta on the correct (config, observable) cell. Raised instead of
    guessing -- see module docstring."""


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


def _parse_header_columns(table_html: str) -> list[str]:
    """The header's observable-column order, AS ACTUALLY RENDERED (not
    assumed from our own input) -- `_matrix_table`'s header and each row's
    cells are built from the same `cols` loop, so reading the header back is
    what lets `_inject_deltas` follow a resorted-columns change instead of
    silently mislabeling against a stale assumed order."""
    m = _HEADER_ROW_RE.search(table_html)
    if not m:
        raise _MatrixStructureError(
            "comparison_matrix: no <thead><tr>...</tr></thead> header row "
            "found in the reused _matrix_table output")
    ths = _TH_RE.findall(m.group(1))
    if not ths or ths[0] != "study":
        raise _MatrixStructureError(
            f"comparison_matrix: expected the header's first column to be "
            f"'study', got {ths[:1]!r} -- _matrix_table's markup changed in "
            "a way this adapter can no longer safely map deltas onto")
    return [_html.unescape(h) for h in ths[1:]]


def _parse_body_rows(table_html: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """[(config_name, [(css_class, cell_text), ...]), ...] -- one entry per
    body row, config identity read from that row's OWN
    ``<td class="study">`` cell (never assumed from row position), cells in
    the order `_matrix_table` emitted them for that row."""
    m = _TBODY_RE.search(table_html)
    if not m:
        raise _MatrixStructureError(
            "comparison_matrix: no <tbody>...</tbody> body found in the "
            "reused _matrix_table output")
    rows: list[tuple[str, list[tuple[str, str]]]] = []
    for row_html in _ROW_RE.findall(m.group(1)):
        study_m = _STUDY_CELL_RE.search(row_html)
        if not study_m:
            raise _MatrixStructureError(
                'comparison_matrix: a matrix row has no <td class="study">'
                "...</td> identifying its config -- cannot map deltas onto "
                "it without guessing")
        config_name = _html.unescape(study_m.group(1))
        rest = row_html[study_m.end():]
        cells = [(klass, text) for klass, text in _DATA_CELL_RE.findall(rest)]
        rows.append((config_name, cells))
    return rows


def _inject_deltas(table_html: str, matrix: dict[str, Any]) -> str:
    """Decorate each graded cell of the REUSED `_matrix_table` HTML with its
    "Delta N.N%" (from `matrix`'s per-config `_deltas`), keyed by (config,
    observable) identity parsed back out of the table's own header + each
    row's study cell -- see module docstring "Delta injection is KEYED, not
    positional". Raises `_MatrixStructureError` (fails loud) rather than
    guessing if that structure doesn't line up. A table with no columns (no
    configs graded any observable) is passed through unchanged -- there is
    nothing to key against."""
    if not table_html:
        return table_html

    header_columns = _parse_header_columns(table_html)
    deltas_by_config = {row["study"]: row["_deltas"] for row in matrix["rows"]}
    known_configs = set(deltas_by_config)

    rebuilt_rows: list[str] = []
    for config_name, cells in _parse_body_rows(table_html):
        if config_name not in known_configs:
            raise _MatrixStructureError(
                f"comparison_matrix: matrix row study-cell {config_name!r} "
                f"doesn't match any known config {sorted(known_configs)} -- "
                "refusing to guess which config's deltas belong on this row "
                "(e.g. a transposed table, or a non-config row)")
        if len(cells) != len(header_columns):
            raise _MatrixStructureError(
                f"comparison_matrix: row {config_name!r} has {len(cells)} "
                f"cell(s) but the header has {len(header_columns)} "
                "column(s) -- refusing to guess the (config, observable) "
                "mapping")
        row_deltas = deltas_by_config[config_name]
        rebuilt_cells = [f'<td class="study">{_html.escape(config_name)}</td>']
        for column_label, (klass, text) in zip(header_columns, cells):
            cell_html = f'<td class="{klass}">{text}'
            if klass in _GRADED_CLASSES:
                delta = row_deltas.get(column_label)
                if delta:
                    cell_html += f'<br><span class="delta">Δ {delta}</span>'
            rebuilt_cells.append(cell_html + "</td>")
        rebuilt_rows.append(f"<tr>{''.join(rebuilt_cells)}</tr>")

    new_body = "".join(rebuilt_rows)
    return _TBODY_RE.sub(lambda _m: f"<tbody>{new_body}</tbody>", table_html)


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
