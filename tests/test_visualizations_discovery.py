"""Smoke test: all 6 Visualization Steps register into core.link_registry."""

import pytest


EXPECTED_VISUALIZATION_CLASSES = {
    "NetworkVisualization",
    "WorkflowVisualization",
    "MultigenerationVisualization",
    "ColonyVisualization",
    "BenchmarkVisualization",
    "V1V2Visualization",
}


@pytest.mark.fast
def test_all_visualizations_discoverable():
    import v2ecoli  # noqa: F401 — forces discovery side-effects
    from bigraph_schema import allocate_core
    core = allocate_core()
    found = {
        k.rsplit(".", 1)[-1]
        for k in core.link_registry
        if k.startswith("v2ecoli.visualizations.")
    }
    missing = EXPECTED_VISUALIZATION_CLASSES - found
    assert not missing, f"missing visualizations in link_registry: {missing}"


@pytest.mark.fast
def test_visualizations_are_visualization_subclasses():
    """Each registered Visualization class is a pbg_superpowers Visualization subclass."""
    import v2ecoli  # noqa
    from bigraph_schema import allocate_core
    from pbg_superpowers.visualization import Visualization
    core = allocate_core()
    for k, cls in core.link_registry.items():
        if not k.startswith("v2ecoli.visualizations."):
            continue
        assert issubclass(cls, Visualization), f"{k} is not a Visualization subclass"
