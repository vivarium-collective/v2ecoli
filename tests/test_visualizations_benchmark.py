"""Unit tests for v2ecoli.visualizations.benchmark.BenchmarkVisualization."""

import pytest


@pytest.mark.fast
def test_benchmark_visualization_subclass():
    from v2ecoli.visualizations.benchmark import BenchmarkVisualization
    from pbg_superpowers.visualization import Visualization
    assert issubclass(BenchmarkVisualization, Visualization)


@pytest.mark.fast
def test_benchmark_inputs_has_two_history_ports():
    from v2ecoli.visualizations.benchmark import BenchmarkVisualization
    from bigraph_schema import allocate_core
    viz = BenchmarkVisualization(config={"title": "test"}, core=allocate_core())
    inputs = viz.inputs()
    assert "history_v2ecoli" in inputs
    assert "history_vecoli" in inputs


@pytest.mark.fast
def test_benchmark_outputs_html():
    from v2ecoli.visualizations.benchmark import BenchmarkVisualization
    from bigraph_schema import allocate_core
    viz = BenchmarkVisualization(config={"title": "test"}, core=allocate_core())
    assert viz.outputs() == {"html": "string"}


@pytest.mark.fast
def test_benchmark_renders_synthetic_inputs():
    """Construct with two synthetic trajectories; assert HTML output."""
    from v2ecoli.visualizations.benchmark import BenchmarkVisualization
    from bigraph_schema import allocate_core
    viz = BenchmarkVisualization(
        config={"title": "Benchmark Test"},
        core=allocate_core(),
    )
    result = viz.update({
        "history_v2ecoli": [
            {"time": 0.0,   "mass": 380.0, "elapsed_sec": 0.0},
            {"time": 60.0,  "mass": 395.0, "elapsed_sec": 5.0},
        ],
        "history_vecoli": [
            {"time": 0.0,  "mass": 380.0, "elapsed_sec": 0.0},
            {"time": 60.0, "mass": 396.0, "elapsed_sec": 6.0},
        ],
        "metadata": {"seed": 0, "duration_sec": 60},
    })
    assert "html" in result
    html = result["html"]
    assert "<html" in html and "</html>" in html
    assert "Benchmark Test" in html
