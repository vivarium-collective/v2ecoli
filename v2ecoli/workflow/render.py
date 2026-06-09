"""Render helpers for Analysis ``view`` outputs.

``fig_to_html`` turns a matplotlib Figure into a self-contained HTML fragment
with an inline SVG, so plot-style analyses port near-verbatim from vEcoli (swap
``fig.savefig(path)`` for ``return {"view": fig_to_html(fig)}``).
"""

from __future__ import annotations

import io


def fig_to_html(fig, title: str = "") -> str:
    """Serialize a matplotlib Figure to an HTML fragment with inline SVG."""
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    svg = buf.getvalue()
    # Strip the XML/doctype preamble so the <svg> embeds cleanly in HTML.
    idx = svg.find("<svg")
    svg = svg[idx:] if idx != -1 else svg
    heading = f"<h3>{title}</h3>" if title else ""
    return f'<div class="analysis-view">{heading}{svg}</div>'
