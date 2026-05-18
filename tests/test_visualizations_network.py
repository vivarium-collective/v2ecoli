"""Unit tests for v2ecoli.visualizations.network.NetworkVisualization."""

import pytest


@pytest.mark.fast
def test_network_visualization_is_visualization_subclass():
    from v2ecoli.visualizations.network import NetworkVisualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(NetworkVisualization, Visualization)


@pytest.mark.fast
def test_network_visualization_inputs_has_composite_spec():
    from v2ecoli.visualizations.network import NetworkVisualization
    from bigraph_schema import allocate_core
    viz = NetworkVisualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "composite_spec" in inputs


@pytest.mark.fast
def test_network_visualization_outputs_html():
    from v2ecoli.visualizations.network import NetworkVisualization
    from bigraph_schema import allocate_core
    viz = NetworkVisualization(config={"title": "test"}, core=allocate_core())
    outputs = viz.outputs()
    assert outputs == {"html": "string"}


@pytest.mark.fast
def test_network_visualization_renders_synthetic_spec():
    """Construct with a minimal synthetic composite_spec; assert HTML output."""
    from v2ecoli.visualizations.network import NetworkVisualization
    from bigraph_schema import allocate_core
    viz = NetworkVisualization(
        config={"title": "Synthetic Network"},
        core=allocate_core(),
    )
    result = viz.update({
        "composite_spec": {
            "architecture": "baseline",
            "nodes": [{"id": "step1", "label": "Step 1"}],
            "edges": [],
            "layers": [["step1"]],
            "legend": [],
        },
    })
    assert "html" in result
    html = result["html"]
    assert "<html" in html and "</html>" in html
    assert "Synthetic Network" in html
