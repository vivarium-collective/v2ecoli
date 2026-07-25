"""Render the Units Atlas: a styled, self-contained catalog of every
unit-bearing readout in the E. coli whole-cell model, grouped by physical
dimension, with magnitudes sampled from a run when available."""
from __future__ import annotations

import html as _html
from pathlib import Path

from viva_superpowers.visualization import Visualization
from v2ecoli.library.units_atlas import (
    build_atlas,
    dimension_description,
    format_magnitude,
)

# Accent color per dimension (header bar + chip).
_DIM_COLOR = {
    "mass": "#7b3fa0",
    "concentration": "#1f77b4",
    "rate": "#d6610a",
    "time": "#2c8c5a",
    "count": "#b0860b",
    "volume": "#9467bd",
    "length": "#8c564b",
    "other": "#6b7280",
}

_CSS = """
<style>
.units-atlas { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
    Roboto, Helvetica, Arial, sans-serif; color: #1f2933; max-width: 1000px;
    margin: 0 auto; line-height: 1.5; }
.units-atlas .lead { font-size: 15px; color: #3e4c59; }
.units-atlas .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 14px 0 24px; }
.units-atlas .chip { display: inline-flex; align-items: baseline; gap: 6px;
    padding: 4px 11px; border-radius: 999px; color: #fff; font-size: 13px;
    font-weight: 600; }
.units-atlas .chip .n { font-size: 12px; opacity: .85; font-weight: 500; }
.units-atlas section { margin: 0 0 26px; border: 1px solid #e4e7eb;
    border-radius: 8px; overflow: hidden; }
.units-atlas .dim-head { padding: 10px 14px; color: #fff; }
.units-atlas .dim-head h3 { margin: 0; font-size: 16px; }
.units-atlas .dim-head .desc { margin: 4px 0 0; font-size: 13px; opacity: .92; }
.units-atlas table { width: 100%; border-collapse: collapse; font-size: 13px; }
.units-atlas th, .units-atlas td { text-align: left; padding: 7px 14px;
    border-bottom: 1px solid #eef1f4; }
.units-atlas thead th { background: #f5f7fa; font-size: 11px; text-transform:
    uppercase; letter-spacing: .04em; color: #616e7c; }
.units-atlas td.path { font-family: ui-monospace, "SF Mono", Menlo, monospace;
    font-size: 12px; color: #1f2933; }
.units-atlas td.unit { font-family: ui-monospace, Menlo, monospace; color: #52606d; }
.units-atlas td.num { text-align: right; font-variant-numeric: tabular-nums;
    color: #3e4c59; }
.units-atlas tr:last-child td { border-bottom: none; }
.units-atlas .foot { font-size: 12px; color: #7b8794; margin-top: 8px;
    border-top: 1px solid #e4e7eb; padding-top: 12px; }
.units-atlas .flags { background: #fff8e6; border: 1px solid #f0d58a;
    border-radius: 8px; padding: 10px 14px; font-size: 13px; }
</style>
"""

_DIM_ORDER = ["mass", "concentration", "rate", "count", "time", "volume",
              "length", "other"]


class UnitsAtlasVisualization(Visualization):
    """Descriptive catalog of every unit-bearing readout, grouped by dimension."""

    def inputs(self):
        # Schema-derived; an optional run path may be wired for magnitudes.
        return {"run_dir": "string"}

    def accumulate(self, state):
        self._run_dir = (state or {}).get("run_dir")

    def render(self):
        cfg = getattr(self, "config", {}) or {}
        title = cfg.get("title") or "Units Atlas"
        run_dir = getattr(self, "_run_dir", None)
        atlas = build_atlas(run_dir)

        dims = [d for d in atlas if not d.startswith("_")]
        ordered = [d for d in _DIM_ORDER if d in dims] + \
                  sorted(d for d in dims if d not in _DIM_ORDER)
        total = sum(len(atlas[d]) for d in dims)
        sampled = any(r.get("example") is not None
                      for d in dims for r in atlas[d])
        provenance = (
            f"Example, min, and max magnitudes are sampled from "
            f"<code>{_html.escape(Path(str(run_dir)).name)}</code>."
            if run_dir and sampled else
            "Magnitudes are blank — render with a parquet run wired to "
            "<code>run_dir</code> to populate them."
        )

        parts = [_CSS, '<div class="units-atlas">']
        parts.append(f"<h2 style='margin:0 0 6px'>{_html.escape(title)}</h2>")
        parts.append(
            "<p class='lead'>Every quantity the E. coli whole-cell model "
            "declares on its listener and process ports, resolved <b>live "
            "from the typed port schema</b> (no run required) and grouped by "
            f"physical dimension. <b>{total} unit-bearing readouts</b> across "
            f"<b>{len(dims)} dimensions</b>. These are the same declared units "
            "that now auto-label plot axes across every v2ecoli visualization. "
            f"{provenance}</p>"
        )

        # Dimension chips (overview).
        parts.append('<div class="chips">')
        for d in ordered:
            color = _DIM_COLOR.get(d, "#6b7280")
            parts.append(
                f'<span class="chip" style="background:{color}">{_html.escape(d)}'
                f'<span class="n">{len(atlas[d])}</span></span>'
            )
        parts.append('</div>')

        # Per-dimension sections.
        for d in ordered:
            rows = atlas[d]
            color = _DIM_COLOR.get(d, "#6b7280")
            units_here = sorted({r["unit"] for r in rows})
            parts.append("<section>")
            parts.append(f'<div class="dim-head" style="background:{color}">')
            parts.append(
                f"<h3>{_html.escape(d)} · {len(rows)} readout"
                f"{'s' if len(rows) != 1 else ''} · "
                f"<span style='font-weight:500'>{_html.escape(', '.join(units_here))}</span></h3>"
            )
            desc = dimension_description(d)
            if desc:
                parts.append(f"<p class='desc'>{_html.escape(desc)}</p>")
            parts.append("</div>")
            parts.append("<table><thead><tr>"
                         "<th>readout</th><th>unit</th>"
                         "<th style='text-align:right'>example</th>"
                         "<th style='text-align:right'>min</th>"
                         "<th style='text-align:right'>max</th>"
                         "</tr></thead><tbody>")
            for r in rows:
                parts.append(
                    "<tr>"
                    f"<td class='path'>{_html.escape(r['path'])}</td>"
                    f"<td class='unit'>{_html.escape(r['unit'])}</td>"
                    f"<td class='num'>{format_magnitude(r['example'])}</td>"
                    f"<td class='num'>{format_magnitude(r['min'])}</td>"
                    f"<td class='num'>{format_magnitude(r['max'])}</td>"
                    "</tr>"
                )
            parts.append("</tbody></table>")
            parts.append("</section>")

        flags = atlas.get("_flags") or []
        if flags:
            parts.append("<div class='flags'><b>Flagged — dimensionless / "
                         "missing unit:</b><ul>")
            parts.extend(f"<li>{_html.escape(str(f))}</li>" for f in flags)
            parts.append("</ul></div>")

        parts.append(
            "<p class='foot'>Built by "
            "<code>v2ecoli.library.units_atlas.build_atlas</code>, which walks "
            "the baseline composite's declared <code>quantity[...]</code> / "
            "<code>float[unit]</code> port types via "
            "<code>units_resolver.build_units_index</code>. Descriptive "
            "reference — no pass/fail gate.</p>"
        )
        parts.append("</div>")
        return "\n".join(parts)
