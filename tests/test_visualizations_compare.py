"""Unit tests for v2ecoli.visualizations.compare.CompareVisualization."""

import pytest


@pytest.mark.fast
def test_compare_subclass():
    from v2ecoli.visualizations.compare import CompareVisualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(CompareVisualization, Visualization)


@pytest.mark.fast
def test_compare_inputs_has_composite_specs():
    from v2ecoli.visualizations.compare import CompareVisualization
    from bigraph_schema import allocate_core
    viz = CompareVisualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "composite_specs" in inputs


@pytest.mark.fast
def test_compare_outputs_html():
    from v2ecoli.visualizations.compare import CompareVisualization
    from bigraph_schema import allocate_core
    viz = CompareVisualization(config={"title": "test"}, core=allocate_core())
    assert viz.outputs() == {"html": "string"}


@pytest.mark.fast
def test_compare_renders_three_synthetic_specs():
    from v2ecoli.visualizations.compare import CompareVisualization
    from bigraph_schema import allocate_core
    viz = CompareVisualization(
        config={"title": "Architecture compare"},
        core=allocate_core(),
    )
    result = viz.update({
        "composite_specs": [
            {"architecture": "baseline",
             "nodes": [{"id": "n1", "label": "Step 1"}],
             "edges": [], "layers": [["n1"]], "legend": []},
            {"architecture": "departitioned",
             "nodes": [{"id": "n2", "label": "Step 2"}],
             "edges": [], "layers": [["n2"]], "legend": []},
            {"architecture": "reconciled",
             "nodes": [{"id": "n3", "label": "Step 3"}],
             "edges": [], "layers": [["n3"]], "legend": []},
        ],
    })
    assert "html" in result
    html = result["html"]
    assert "<html" in html and "</html>" in html
    assert "Architecture compare" in html
