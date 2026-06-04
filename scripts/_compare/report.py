"""Self-contained two-column HTML report renderer.

Renders a list of comparison *sections* (vEcoli left, v2ecoli right) into a
single self-contained HTML page: a summary banner with per-verdict counts, a
legend, a sticky section nav, and one card per section. Rows may carry an
optional ``group`` key to cluster them under sub-headings within a section,
and an optional ``reason``/``max_rel`` for extra context. The output embeds
no external assets (no http/https links), so the file is portable.
"""
from __future__ import annotations

import html as _html
from typing import Any

# verdict -> (human label, glyph, css token). The css token is also used as
# the row class `verdict-<token>` (kept stable for tests + styling).
_VERDICTS = {
    "within_tol": ("within tolerance", "✓", "within_tol"),
    "drift": ("drift", "≈", "drift"),
    "mismatch": ("mismatch", "✗", "mismatch"),
    "not_compared": ("not compared", "–", "not_compared"),
}
_VERDICT_ORDER = ["within_tol", "drift", "mismatch", "not_compared"]

_CSS = """
:root {
  --green:#2e7d32; --amber:#ef6c00; --red:#c62828; --grey:#757575;
  --bg:#f6f7f9; --card:#fff; --ink:#1a1d21; --muted:#6b7280; --line:#e5e7eb;
}
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin:0; background:var(--bg); color:var(--ink); line-height:1.45; }
header.top { background:linear-gradient(135deg,#1f2937,#111827); color:#fff;
  padding:22px 28px; }
header.top h1 { margin:0; font-size:20px; font-weight:650; letter-spacing:.2px; }
header.top .sub { margin-top:4px; color:#cbd5e1; font-size:13px; }
.legend { display:flex; gap:14px; flex-wrap:wrap; margin-top:14px; }
.legend .item { display:flex; align-items:center; gap:6px; font-size:12px;
  color:#e5e7eb; }
.dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.dot.within_tol{background:var(--green)} .dot.drift{background:var(--amber)}
.dot.mismatch{background:var(--red)} .dot.not_compared{background:var(--grey)}
nav.sticky { position:sticky; top:0; z-index:5; background:rgba(255,255,255,.95);
  backdrop-filter:saturate(1.4) blur(6px); border-bottom:1px solid var(--line);
  padding:10px 28px; display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
nav.sticky a { text-decoration:none; color:var(--ink); font-size:13px;
  padding:5px 10px; border-radius:999px; border:1px solid var(--line);
  display:inline-flex; align-items:center; gap:7px; }
nav.sticky a:hover { background:var(--bg); }
.minicount { font-size:11px; color:var(--muted); }
main { padding:24px 28px 64px; max-width:1180px; margin:0 auto; }
.card { background:var(--card); border:1px solid var(--line); border-radius:12px;
  margin:0 0 22px; overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.card > .head { padding:16px 18px; border-bottom:1px solid var(--line);
  display:flex; align-items:center; justify-content:space-between; gap:12px;
  flex-wrap:wrap; }
.card > .head h2 { margin:0; font-size:16px; font-weight:620; }
.chips { display:flex; gap:7px; flex-wrap:wrap; }
.chip { font-size:12px; font-weight:600; padding:3px 9px; border-radius:999px;
  color:#fff; display:inline-flex; gap:5px; align-items:center; }
.chip.within_tol{background:var(--green)} .chip.drift{background:var(--amber)}
.chip.mismatch{background:var(--red)} .chip.not_compared{background:var(--grey)}
.chip.zero{ background:#eceff1; color:#90a4ae; }
table { border-collapse:collapse; width:100%; }
thead th { text-align:left; font-size:11px; text-transform:uppercase;
  letter-spacing:.5px; color:var(--muted); padding:9px 18px;
  border-bottom:1px solid var(--line); background:#fafbfc; }
tbody td { padding:9px 18px; border-bottom:1px solid #f1f3f5;
  vertical-align:top; font-size:13px; }
tbody tr:hover { background:#fbfcfe; }
.grouprow td { background:#f3f4f6; font-weight:650; font-size:11px;
  text-transform:uppercase; letter-spacing:.6px; color:var(--muted); }
.metric { font-weight:560; }
.val { font-family:"SF Mono",ui-monospace,Menlo,Consolas,monospace;
  font-size:12px; color:#374151; white-space:pre-wrap; word-break:break-word; }
.col-left, .col-right { width:34%; }
.badge { font-size:10.5px; font-weight:700; padding:2px 7px; border-radius:6px;
  color:#fff; margin-left:8px; white-space:nowrap; }
.verdict-within_tol .badge{background:var(--green)}
.verdict-drift .badge{background:var(--amber)}
.verdict-mismatch .badge{background:var(--red)}
.verdict-not_compared .badge{background:var(--grey)}
.verdict-within_tol td:first-child{box-shadow:inset 3px 0 0 var(--green)}
.verdict-drift td:first-child{box-shadow:inset 3px 0 0 var(--amber)}
.verdict-mismatch td:first-child{box-shadow:inset 3px 0 0 var(--red)}
.verdict-not_compared td:first-child{box-shadow:inset 3px 0 0 var(--grey)}
.reason { color:var(--muted); font-size:11.5px; margin-top:3px; }
.metersmall { font-size:11px; color:var(--muted); margin-top:2px; }
footer { color:var(--muted); font-size:12px; padding:0 28px 40px;
  max-width:1180px; margin:0 auto; }
"""


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.lower())


def _e(x: Any) -> str:
    return _html.escape(str(x))


def _counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    c = {k: 0 for k in _VERDICT_ORDER}
    for r in rows:
        v = r.get("verdict", "not_compared")
        c[v] = c.get(v, 0) + 1
    return c


def _chips(counts: dict[str, int]) -> str:
    out = []
    for v in _VERDICT_ORDER:
        n = counts.get(v, 0)
        label, glyph, tok = _VERDICTS[v]
        cls = tok if n else "zero"
        out.append(f'<span class="chip {cls}">{glyph} {n} {_e(label)}</span>')
    return f'<div class="chips">{"".join(out)}</div>'


def _row_html(row: dict[str, Any]) -> str:
    verdict = row.get("verdict", "not_compared")
    _, glyph, tok = _VERDICTS.get(verdict, _VERDICTS["not_compared"])
    label = _e(row.get("label", ""))
    left = _e(row.get("left", ""))
    right = _e(row.get("right", ""))
    reason = row.get("reason", "")
    meter = ""
    median_rel = row.get("median_rel", None)
    max_rel = row.get("max_rel", None)
    if isinstance(median_rel, (int, float)):
        bits = [f"median rel Δ = {median_rel:.3g}"]
        if isinstance(max_rel, (int, float)):
            bits.append(f"max = {max_rel:.3g}")
        fw = row.get("frac_within", None)
        if isinstance(fw, (int, float)):
            bits.append(f"{fw * 100:.0f}% within tol")
        meter = f'<div class="metersmall">{" · ".join(bits)}</div>'
    elif isinstance(max_rel, (int, float)):
        meter = f'<div class="metersmall">max rel Δ = {max_rel:.3g}</div>'
    reason_html = f'<div class="reason">{_e(reason)}</div>' if reason else ""
    return (
        f'<tr class="verdict-{tok}">'
        f'<td class="metric">{label}'
        f'<span class="badge">{glyph} {tok}</span>{reason_html}{meter}</td>'
        f'<td class="val col-left">{left}</td>'
        f'<td class="val col-right">{right}</td>'
        f'</tr>'
    )


def _rows_block(rows: list[dict[str, Any]]) -> str:
    """Render rows, clustering by optional ``group`` key (insertion order)."""
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    index: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        g = r.get("group", "")
        if g not in index:
            index[g] = []
            groups.append((g, index[g]))
        index[g].append(r)
    has_named = any(g for g, _ in groups)
    out = []
    for gname, grows in groups:
        if gname and has_named:
            out.append(
                f'<tr class="grouprow"><td colspan="3">{_e(gname)}</td></tr>')
        out.extend(_row_html(r) for r in grows)
    return "".join(out)


def _section_html(section: dict[str, Any]) -> str:
    title = section["title"]
    rows = section.get("rows", [])
    extra = section.get("html", "")  # embedded plots (data: URIs)
    body = _rows_block(rows)
    return (
        f'<section class="card" id="{_slug(title)}">'
        f'<div class="head"><h2>{_e(title)}</h2>{_chips(_counts(rows))}</div>'
        f'<table><thead><tr>'
        f'<th>metric</th>'
        f'<th class="col-left">vEcoli</th>'
        f'<th class="col-right">v2ecoli</th>'
        f'</tr></thead><tbody>{body}</tbody></table>{extra}</section>'
    )


def _legend() -> str:
    items = "".join(
        f'<span class="item"><span class="dot {_VERDICTS[k][2]}"></span>'
        f'{_e(_VERDICTS[k][0])}</span>'
        for k in _VERDICT_ORDER
    )
    return f'<div class="legend">{items}</div>'


def _nav(sections: list[dict[str, Any]]) -> str:
    links = []
    for s in sections:
        c = _counts(s.get("rows", []))
        n = sum(c.values())
        off = c.get("mismatch", 0) + c.get("drift", 0)
        mini = f'<span class="minicount">{n - off}/{n} ok</span>' if n else ""
        links.append(
            f'<a href="#{_slug(s["title"])}">{_e(s["title"])} {mini}</a>')
    return f'<nav class="sticky">{"".join(links)}</nav>'


def render_report(sections: list[dict[str, Any]], *, title: str) -> str:
    """Render sections to a single self-contained, organized HTML string.

    Each section is ``{"title": str, "rows": [...], "html"?: str}``; each row
    is ``{"label","left","right","verdict"}`` plus optional
    ``reason``/``max_rel``/``group``. ``group`` clusters rows under
    sub-headings; verdict counts are summarized per section and overall.
    """
    total = _counts([r for s in sections for r in s.get("rows", [])])
    body = "".join(_section_html(s) for s in sections)
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>{_e(title)}</title><style>{_CSS}</style></head><body>'
        f'<header class="top"><h1>{_e(title)}</h1>'
        f'<div class="sub">vEcoli (left) vs v2ecoli (right) · '
        f'{sum(total.values())} metrics compared</div>'
        f'{_chips(total)}{_legend()}</header>'
        f'{_nav(sections)}'
        f'<main>{body}</main>'
        f'<footer>Generated by the vEcoli&#8596;v2ecoli comparison harness '
        f'(scripts/compare_harness.py). Self-contained; no external assets.'
        f'</footer></body></html>'
    )
