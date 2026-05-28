"""Unit tests for v2ecoli.visualizations.colony.ColonyVisualization."""

import pytest


@pytest.mark.fast
def test_colony_subclass():
    from v2ecoli.visualizations.colony import ColonyVisualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(ColonyVisualization, Visualization)


@pytest.mark.fast
def test_colony_inputs():
    from v2ecoli.visualizations.colony import ColonyVisualization
    from bigraph_schema import allocate_core
    viz = ColonyVisualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "history" in inputs
    assert "metadata" in inputs


@pytest.mark.fast
def test_colony_outputs_html():
    from v2ecoli.visualizations.colony import ColonyVisualization
    from bigraph_schema import allocate_core
    viz = ColonyVisualization(config={"title": "test"}, core=allocate_core())
    assert viz.outputs() == {"html": "string"}


@pytest.mark.fast
def test_colony_renders_synthetic_colony_state():
    from v2ecoli.visualizations.colony import ColonyVisualization
    from bigraph_schema import allocate_core
    viz = ColonyVisualization(
        config={"title": "Colony test"},
        core=allocate_core(),
    )
    result = viz.update({
        "history": [
            {"time": 0.0, "agent_id": "0",  "x": 0.0, "y": 0.0, "length": 1.0, "mass": 380.0},
            {"time": 0.0, "agent_id": "01", "x": 1.0, "y": 0.0, "length": 1.0, "mass": 380.0},
            {"time": 60.0, "agent_id": "0",  "x": 0.0, "y": 0.1, "length": 1.05, "mass": 390.0},
            {"time": 60.0, "agent_id": "01", "x": 1.0, "y": 0.0, "length": 1.05, "mass": 390.0},
        ],
        "metadata": {"colony_size": 2, "seed": 0},
    })
    assert "html" in result
    html = result["html"]
    assert "<html" in html and "</html>" in html
    assert "Colony test" in html
