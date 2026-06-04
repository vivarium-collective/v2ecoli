"""Self-contained two-column HTML report renderer."""
from __future__ import annotations

import html as _html
from typing import Any

_CSS = """
body { font-family: -apple-system, sans-serif; margin: 0; }
nav { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ccc;
      padding: 8px; }
nav a { margin-right: 12px; }
section { padding: 16px; border-bottom: 1px solid #eee; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #f0f0f0;
         vertical-align: top; }
.col-left { width: 40%; } .col-right { width: 40%; }
.badge { padding: 1px 6px; border-radius: 4px; font-size: 11px; color: #fff; }
.verdict-within_tol .badge { background: #2e7d32; }
.verdict-drift .badge { background: #ef6c00; }
.verdict-mismatch .badge { background: #c62828; }
.verdict-not_compared .badge { background: #757575; }
"""


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.lower())


def _row_html(row: dict[str, Any]) -> str:
    verdict = row.get("verdict", "not_compared")
    reason = row.get("reason", "")
    label = _html.escape(str(row.get("label", "")))
    left = _html.escape(str(row.get("left", "")))
    right = _html.escape(str(row.get("right", "")))
    reason_html = (f'<div class="reason">{_html.escape(reason)}</div>'
                   if reason else "")
    return (
        f'<tr class="verdict-{verdict}">'
        f'<td>{label}<span class="badge">{verdict}</span>{reason_html}</td>'
        f'<td class="col-left">{left}</td>'
        f'<td class="col-right">{right}</td>'
        f'</tr>'
    )


def _section_html(section: dict[str, Any]) -> str:
    title = section["title"]
    rows = "".join(_row_html(r) for r in section.get("rows", []))
    extra = section.get("html", "")  # for embedded plots (data: URIs)
    return (
        f'<section id="{_slug(title)}">'
        f'<h2>{_html.escape(title)}</h2>'
        f'<table><thead><tr><th>metric</th>'
        f'<th class="col-left">vEcoli</th>'
        f'<th class="col-right">v2ecoli</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>{extra}</section>'
    )


def render_report(sections: list[dict[str, Any]], *, title: str) -> str:
    """Render a list of sections to a single self-contained HTML string."""
    nav = "".join(
        f'<a href="#{_slug(s["title"])}">{_html.escape(s["title"])}</a>'
        for s in sections
    )
    body = "".join(_section_html(s) for s in sections)
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<title>{_html.escape(title)}</title><style>{_CSS}</style></head>'
        f'<body><nav>{nav}</nav><h1>{_html.escape(title)}</h1>'
        f'{body}</body></html>'
    )
