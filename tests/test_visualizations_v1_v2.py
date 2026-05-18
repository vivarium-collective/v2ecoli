"""Unit tests for v2ecoli.visualizations.v1_v2.V1V2Visualization."""

import pytest


@pytest.mark.fast
def test_v1_v2_visualization_subclass():
    from v2ecoli.visualizations.v1_v2 import V1V2Visualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(V1V2Visualization, Visualization)


@pytest.mark.fast
def test_v1_v2_inputs_has_three_history_ports():
    from v2ecoli.visualizations.v1_v2 import V1V2Visualization
    from bigraph_schema import allocate_core
    viz = V1V2Visualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "history_v1" in inputs
    assert "history_v2" in inputs
    assert "history_v2ecoli" in inputs


@pytest.mark.fast
def test_v1_v2_outputs_html():
    from v2ecoli.visualizations.v1_v2 import V1V2Visualization
    from bigraph_schema import allocate_core
    viz = V1V2Visualization(config={"title": "test"}, core=allocate_core())
    assert viz.outputs() == {"html": "string"}


@pytest.mark.fast
def test_v1_v2_renders_synthetic_inputs():
    from v2ecoli.visualizations.v1_v2 import V1V2Visualization
    from bigraph_schema import allocate_core
    viz = V1V2Visualization(
        config={"title": "v1 vs v2 vs v2ecoli"},
        core=allocate_core(),
    )
    result = viz.update({
        "history_v1":      [{"time": 0.0, "mass": 380.0}, {"time": 60.0, "mass": 390.0}],
        "history_v2":      [{"time": 0.0, "mass": 380.0}, {"time": 60.0, "mass": 391.0}],
        "history_v2ecoli": [{"time": 0.0, "mass": 380.0}, {"time": 60.0, "mass": 392.0}],
        "metadata":        {"seed": 0, "duration_sec": 60},
    })
    assert "html" in result
    html = result["html"]
    assert "<html" in html and "</html>" in html
    assert "v1 vs v2 vs v2ecoli" in html
