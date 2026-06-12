"""Render the units atlas as a grouped HTML table (one section per dimension)."""
from __future__ import annotations

import html as _html

from pbg_superpowers.visualization import Visualization
from v2ecoli.library.units_atlas import build_atlas


class UnitsAtlasVisualization(Visualization):
    """Descriptive catalog of every unit-bearing readout, grouped by dimension."""

    def inputs(self):
        # Schema-derived; an optional run path may be wired for magnitudes.
        return {"run_dir": "string"}

    def accumulate(self, state):
        self._run_dir = (state or {}).get("run_dir")

    def render(self):
        title = (getattr(self, "config", {}) or {}).get("title") or "Units Atlas"
        atlas = build_atlas(getattr(self, "_run_dir", None))
        parts = [f"<h2>{_html.escape(title)}</h2>"]
        for dim in sorted(k for k in atlas if not k.startswith("_")):
            rows = atlas[dim]
            parts.append(f"<h3>{_html.escape(dim)} ({len(rows)})</h3>")
            parts.append("<table border='1' cellpadding='4' "
                         "style='border-collapse:collapse'>")
            parts.append("<tr><th>readout</th><th>unit</th>"
                         "<th>example</th><th>min</th><th>max</th></tr>")
            for r in rows:
                parts.append(
                    "<tr>"
                    f"<td>{_html.escape(r['path'])}</td>"
                    f"<td>{_html.escape(r['unit'])}</td>"
                    f"<td>{'' if r['example'] is None else r['example']}</td>"
                    f"<td>{'' if r['min'] is None else r['min']}</td>"
                    f"<td>{'' if r['max'] is None else r['max']}</td>"
                    "</tr>"
                )
            parts.append("</table>")
        flags = atlas.get("_flags") or []
        if flags:
            parts.append("<h3>flags — dimensionless / missing unit</h3><ul>")
            parts.extend(f"<li>{_html.escape(str(f))}</li>" for f in flags)
            parts.append("</ul>")
        return "\n".join(parts)
