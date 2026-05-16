"""NetworkVisualization — interactive Cytoscape.js diagram of one v2ecoli
architecture's composition.

The Step takes a ``composite_spec`` describing the architecture (nodes,
edges, layers, legend) and returns a complete HTML document. The wrapper
at reports/network_report.py is responsible for:
  - building the composite via v2ecoli.build_composite(name)
  - extracting nodes/edges/layers via _helpers.build_graph
  - passing the resulting spec to this Step's update().
"""

from __future__ import annotations
from typing import Any

from pbg_superpowers.visualization import Visualization

from v2ecoli.visualizations._helpers import render_cytoscape_html


class NetworkVisualization(Visualization):
    """Render one architecture's Cytoscape network diagram."""

    config_schema = {
        **Visualization.config_schema,
        "subtitle": {"_type": "string", "_default": ""},
    }

    def inputs(self) -> dict[str, Any]:
        return {
            "composite_spec": "map[node]",
        }

    def update(self, state: dict[str, Any]) -> dict:
        spec = state.get("composite_spec") or {}
        title = self.config.get("title") or "v2ecoli network"
        subtitle = self.config.get("subtitle") or spec.get("architecture", "")
        html = render_cytoscape_html(spec, title=title, subtitle=subtitle)
        return {"html": html}
