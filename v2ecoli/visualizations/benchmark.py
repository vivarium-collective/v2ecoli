"""BenchmarkVisualization — v2ecoli vs vEcoli composite benchmark comparison.

The legacy reports/benchmark_report.py on current main is stdout-only. This
Step produces a fresh HTML side-by-side comparison from two trajectory
inputs (one per engine). The wrapper at reports/benchmark_report.py
handles the subprocess invocations + trajectory collection.
"""

from __future__ import annotations
from html import escape
from typing import Any

from pbg_superpowers.visualization import Visualization

from v2ecoli.visualizations._helpers import render_document

_TD = 'style="padding:6px 12px;border:1px solid #e2e8f0;"'
_TH = 'style="padding:8px 12px;text-align:left;border:1px solid #e2e8f0;background:#f1f5f9;"'


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def _colored_cell(v: Any, other: Any, lower_is_better: bool) -> str:
    """Return a <td> with optional green/red coloring for numeric comparisons."""
    text = _fmt(v)
    if v is None or other is None:
        return f"<td {_TD}>{escape(text)}</td>"
    try:
        v_num = float(v)
        o_num = float(other)
        if lower_is_better:
            color = "#16a34a" if v_num <= o_num else "#dc2626"
        else:
            color = "#16a34a" if v_num >= o_num else "#dc2626"
        return (
            f'<td {_TD} style="padding:6px 12px;border:1px solid #e2e8f0;'
            f'color:{color};font-weight:600">{escape(text)}</td>'
        )
    except (TypeError, ValueError):
        return f"<td {_TD}>{escape(text)}</td>"


class BenchmarkVisualization(Visualization):
    """Render v2ecoli vs vEcoli side-by-side benchmark.

    Inputs:
      - history_v2ecoli: list[map[node]] — trajectory rows from v2ecoli run
      - history_vecoli:  list[map[node]] — trajectory rows from vEcoli run
      - metadata:        map[node]       — run metadata (seed, duration, etc.)
    """

    config_schema = {
        **Visualization.config_schema,
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "history_v2ecoli": "list[map[node]]",
            "history_vecoli":  "list[map[node]]",
            "metadata":        "map[node]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        h_v2 = state.get("history_v2ecoli") or []
        h_ve = state.get("history_vecoli") or []
        meta = state.get("metadata") or {}
        title = self.config.get("title") or "v2ecoli vs vEcoli benchmark"
        body = self._render_body(h_v2, h_ve, meta, title)
        html = render_document(title=title, body_html=body, include_banner=True)
        return {"html": html}

    def _render_body(self, h_v2: list, h_ve: list, meta: dict, title: str) -> str:
        """Render a side-by-side comparison table and summary metrics."""
        last_v2 = h_v2[-1] if h_v2 else {}
        last_ve = h_ve[-1] if h_ve else {}

        # Build comparison rows: (label, v2_value, ve_value, lower_is_better|None)
        # None means neutral (no coloring).
        rows: list[tuple[str, Any, Any, bool | None]] = [
            ("rows in trajectory", len(h_v2), len(h_ve), None),
        ]
        for key, label, lower in (
            ("load",      "load time (s)",     True),
            ("run",       "run time (s)",       True),
            ("dry_mass",  "dry mass (fg)",      False),
            ("cell_mass", "cell mass (fg)",     False),
            ("mass",      "final mass (fg)",    False),
            ("elapsed_sec", "wall elapsed (s)", True),
        ):
            v2_val = last_v2.get(key)
            ve_val = last_ve.get(key)
            if v2_val is not None or ve_val is not None:
                rows.append((label, v2_val, ve_val, lower))

        table_rows = ""
        for label, v2_val, ve_val, lower in rows:
            label_cell = f"<td {_TD}><strong>{escape(label)}</strong></td>"
            if lower is None:
                v2_cell = f"<td {_TD}>{escape(_fmt(v2_val))}</td>"
                ve_cell = f"<td {_TD}>{escape(_fmt(ve_val))}</td>"
            else:
                v2_cell = _colored_cell(v2_val, ve_val, lower_is_better=lower)
                ve_cell = _colored_cell(ve_val, v2_val, lower_is_better=lower)
            table_rows += f"<tr>{label_cell}{v2_cell}{ve_cell}</tr>"

        # Performance ratio verdict
        ratio_row = ""
        v2_run = last_v2.get("run")
        ve_run = last_ve.get("run")
        if v2_run is not None and ve_run is not None:
            try:
                ratio = float(v2_run) / float(ve_run)
                ratio_color = (
                    "#16a34a" if ratio <= 1.2
                    else "#f59e0b" if ratio <= 1.5
                    else "#dc2626"
                )
                verdict = (
                    "EXCELLENT: v2ecoli matches vEcoli performance" if ratio <= 1.2
                    else "GOOD: within 1.5x of vEcoli" if ratio <= 1.5
                    else "OK: within 2x of vEcoli" if ratio <= 2.0
                    else f"SLOW: v2ecoli is {ratio:.1f}x slower"
                )
                ratio_row = (
                    '<p style="margin-top:12px;font-size:0.9em;">'
                    "<strong>v2ecoli / vEcoli run-time ratio: "
                    f'<span style="color:{ratio_color}">{ratio:.2f}x</span></strong>'
                    f" — {escape(verdict)}</p>"
                )
            except (TypeError, ValueError):
                pass

        meta_lines = "".join(
            f"<li><strong>{escape(str(k))}</strong>: {escape(str(v))}</li>"
            for k, v in (meta or {}).items()
        )

        return (
            '<div style="padding: 20px; font-family: -apple-system, sans-serif; '
            'max-width: 900px; margin: 0 auto;">'
            f'<h1 style="font-size:1.6em;margin-bottom:6px;">{escape(title)}</h1>'
            '<p style="color:#64748b;font-size:0.9em;">v2ecoli (new, partitioned) '
            "vs vEcoli composite — side-by-side comparison</p>"
            '<table style="border-collapse:collapse;width:100%;margin-top:16px;">'
            "<thead><tr>"
            f"<th {_TH}>Metric</th>"
            f"<th {_TH}>v2ecoli</th>"
            f"<th {_TH}>vEcoli</th>"
            "</tr></thead>"
            f'<tbody style="font-size:0.9em;">{table_rows}</tbody>'
            "</table>"
            f"{ratio_row}"
            '<h2 style="margin-top:24px;font-size:1.1em;">Run Metadata</h2>'
            f'<ul style="font-size:0.9em;line-height:1.8;">{meta_lines}</ul>'
            "</div>"
        )
