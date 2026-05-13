"""CompareVisualization — side-by-side Cytoscape diagrams for the three
v2ecoli architectures.

v0: structural-only. Each input ``composite_spec`` is rendered as a Cytoscape
diagram (via _helpers.render_cytoscape_html) and the three are laid out
side-by-side using an iframe srcdoc pattern: each spec's full Cytoscape HTML
goes into its own <iframe srcdoc="..."> element. This isolates each diagram's
Cytoscape.js instance (and its global ``cy`` variable) from the others.

Time-series comparison (mass trajectories per architecture) is a follow-up.
The legacy reports/compare_report.py produces matplotlib mass-trajectory
plots, divergence analysis, and molecule divergence tables alongside the
interactive network viewers. Those are deferred — reports/compare_report.py
keeps the 3-architecture parallel build and dispatches to this Step for the
structural-only HTML; the time-series rendering lives in the wrapper.

Migrated from reports/compare_report.py (structural portion).
"""

from __future__ import annotations

import json
from html import escape
from typing import Any

from pbg_superpowers.visualization import Visualization

from v2ecoli.visualizations._helpers import (
    render_cytoscape_html,
    render_document,
)

# Architecture display metadata — used by the wrapper and by the Step body.
ARCH_META: dict[str, dict[str, str]] = {
    "baseline": {
        "label": "Baseline (Partitioned)",
        "color": "#2563eb",
    },
    "departitioned": {
        "label": "Departitioned",
        "color": "#dc2626",
    },
    "reconciled": {
        "label": "Reconciled",
        "color": "#16a34a",
    },
}


class CompareVisualization(Visualization):
    """Render three architectures' Cytoscape diagrams side-by-side.

    Each ``composite_spec`` dict must have the keys that ``build_graph``
    returns — ``nodes``, ``edges``, ``layers``, ``legend`` — plus an
    ``architecture`` key (``"baseline"``, ``"departitioned"``, or
    ``"reconciled"``) that labels the panel.

    The three diagrams are embedded as ``<iframe srcdoc="...">`` elements.
    This isolates each diagram's Cytoscape.js instance so there is no
    global-variable collision between the three panels.

    Inputs
    ------
    composite_specs : list of graph-data dicts, one per architecture.
        Each dict: ``{architecture, nodes, edges, layers, legend}``.
        Typically produced by calling ``build_graph(composite, layers)``
        and merging in the ``architecture`` key.

    Outputs
    -------
    html : complete HTML document string.
    """

    config_schema = {
        **Visualization.config_schema,
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "composite_specs": "list[map[any]]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        specs = state.get("composite_specs") or []
        title = self.config.get("title") or "v2ecoli architecture compare"
        body = self._render_body(specs, title)
        html = render_document(title=title, body_html=body, include_banner=True)
        return {"html": html}

    # ------------------------------------------------------------------
    # Private rendering
    # ------------------------------------------------------------------

    def _render_body(self, specs: list[dict], title: str) -> str:
        """Build the three-panel side-by-side HTML body.

        Each spec gets its own ``<iframe srcdoc="...">`` containing a full
        Cytoscape.js document so the diagrams don't share JS state.
        """
        if not specs:
            return "<p style='padding:20px;'>No architecture specs provided.</p>"

        panels: list[str] = []
        for spec in specs:
            arch = str(spec.get("architecture", "unknown"))
            arch_label = ARCH_META.get(arch, {}).get("label", arch)
            arch_color = ARCH_META.get(arch, {}).get("color", "#333")

            # Build a graph-data dict compatible with render_cytoscape_html.
            # The spec may already be in the right shape (from build_graph),
            # or it may be a synthetic test dict — either way we pass it.
            graph_data = {
                "nodes": spec.get("nodes", []),
                "edges": spec.get("edges", []),
                "layers": spec.get("layers", []),
                "legend": spec.get("legend", []),
            }

            n_proc = sum(
                1 for n in graph_data["nodes"]
                if (n.get("data") or n).get("kind") == "process"
            )
            n_store = sum(
                1 for n in graph_data["nodes"]
                if (n.get("data") or n).get("kind") == "store"
            )
            n_edges = len(graph_data["edges"])

            subtitle = f"{n_proc} processes · {n_store} stores · {n_edges} edges"

            # render_cytoscape_html returns a complete <!doctype html> document.
            sub_html = render_cytoscape_html(
                graph_data,
                title=arch_label,
                subtitle=subtitle,
            )

            # Embed as srcdoc. Double-quotes inside the srcdoc content must be
            # replaced with &quot; so the outer HTML attribute isn't broken.
            srcdoc = sub_html.replace("&", "&amp;").replace('"', "&quot;")

            panel = (
                f'<div style="'
                f'display:flex;flex-direction:column;flex:1;min-width:0;">'
                f'<h2 style="'
                f'margin:0 0 6px 0;font-size:1.05em;color:{escape(arch_color)};'
                f'border-bottom:3px solid {escape(arch_color)};padding-bottom:4px;">'
                f'{escape(arch_label)}</h2>'
                f'<iframe srcdoc="{srcdoc}" '
                f'style="border:1px solid #ddd;border-radius:6px;'
                f'width:100%;flex:1;min-height:600px;" '
                f'title="Composition diagram — {escape(arch_label)}" '
                f'loading="lazy"></iframe>'
                f'</div>'
            )
            panels.append(panel)

        body = (
            f'<div style="'
            f'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
            f'padding:20px;background:#f8fafc;min-height:100vh;">'
            f'<h1 style="margin:0 0 16px;color:#0f172a;">{escape(title)}</h1>'
            f'<div style="'
            f'display:flex;gap:16px;align-items:stretch;height:calc(100vh - 100px);">'
            f'{"".join(panels)}'
            f'</div>'
            f'</div>'
        )
        return body
