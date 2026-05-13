"""Unit tests for v2ecoli.visualizations.workflow.WorkflowVisualization."""

import pytest


@pytest.mark.fast
def test_workflow_subclass():
    from v2ecoli.visualizations.workflow import WorkflowVisualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(WorkflowVisualization, Visualization)


@pytest.mark.fast
def test_workflow_inputs():
    from v2ecoli.visualizations.workflow import WorkflowVisualization
    from bigraph_schema import allocate_core
    viz = WorkflowVisualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "history" in inputs
    assert "metadata" in inputs


@pytest.mark.fast
def test_workflow_outputs_html():
    from v2ecoli.visualizations.workflow import WorkflowVisualization
    from bigraph_schema import allocate_core
    viz = WorkflowVisualization(config={"title": "test"}, core=allocate_core())
    assert viz.outputs() == {"html": "string"}


@pytest.mark.fast
def test_workflow_renders_synthetic_trajectory():
    from v2ecoli.visualizations.workflow import WorkflowVisualization
    from bigraph_schema import allocate_core
    viz = WorkflowVisualization(
        config={"title": "Workflow test"},
        core=allocate_core(),
    )
    result = viz.update({
        "history": [
            {"time": 0.0,    "mass": 380.0, "dry_mass": 280.0, "growth_rate": 0.0},
            {"time": 600.0,  "mass": 410.0, "dry_mass": 300.0, "growth_rate": 0.001},
            {"time": 1200.0, "mass": 450.0, "dry_mass": 330.0, "growth_rate": 0.001},
            {"time": 2500.0, "mass": 700.0, "dry_mass": 510.0, "growth_rate": 0.001},
        ],
        "metadata": {"duration": 2520, "seed": 0},
    })
    assert "html" in result
    html = result["html"]
    assert "<html" in html and "</html>" in html
    assert "Workflow test" in html
