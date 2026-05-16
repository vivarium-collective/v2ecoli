"""Unit tests for v2ecoli.visualizations.multigeneration.MultigenerationVisualization."""

import pytest


@pytest.mark.fast
def test_multigeneration_subclass():
    from v2ecoli.visualizations.multigeneration import MultigenerationVisualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(MultigenerationVisualization, Visualization)


@pytest.mark.fast
def test_multigeneration_inputs():
    from v2ecoli.visualizations.multigeneration import MultigenerationVisualization
    from bigraph_schema import allocate_core
    viz = MultigenerationVisualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "history" in inputs
    assert "metadata" in inputs


@pytest.mark.fast
def test_multigeneration_outputs_html():
    from v2ecoli.visualizations.multigeneration import MultigenerationVisualization
    from bigraph_schema import allocate_core
    viz = MultigenerationVisualization(config={"title": "test"}, core=allocate_core())
    assert viz.outputs() == {"html": "string"}


@pytest.mark.fast
def test_multigeneration_renders_synthetic_two_generations():
    from v2ecoli.visualizations.multigeneration import MultigenerationVisualization
    from bigraph_schema import allocate_core
    viz = MultigenerationVisualization(
        config={"title": "Multi-gen test"},
        core=allocate_core(),
    )
    result = viz.update({
        "history": [
            {"generation": 1, "time": 0.0, "mass": 380.0},
            {"generation": 1, "time": 1200.0, "mass": 500.0},
            {"generation": 1, "time": 2520.0, "mass": 702.0},
            {"generation": 2, "time": 2520.0, "mass": 351.0},
            {"generation": 2, "time": 3720.0, "mass": 460.0},
            {"generation": 2, "time": 5040.0, "mass": 640.0},
        ],
        "metadata": {"n_generations": 2, "seed": 0},
    })
    assert "html" in result
    html = result["html"]
    assert "<html" in html and "</html>" in html
    assert "Multi-gen test" in html
