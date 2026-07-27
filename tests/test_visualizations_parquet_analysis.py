"""Unit tests for v2ecoli.visualizations.parquet_analysis.ParquetAnalysisView.

These cover the adapter's contract + graceful degradation without needing a real
parquet sweep or a ParCa sim_data pickle: any missing prerequisite must return
an explanatory ``{"html": ...}`` panel, never raise (a broken figure must not
sink the whole Visualizations tab).
"""

import pytest


@pytest.mark.fast
def test_parquet_analysis_view_is_visualization_subclass():
    from v2ecoli.visualizations.parquet_analysis import ParquetAnalysisView
    from viva_superpowers.visualization import Visualization
    assert issubclass(ParquetAnalysisView, Visualization)


@pytest.mark.fast
def test_parquet_analysis_view_self_contained_ports():
    from v2ecoli.visualizations.parquet_analysis import ParquetAnalysisView
    from bigraph_schema import allocate_core
    viz = ParquetAnalysisView(config={"analysis": "cell_mass"},
                              core=allocate_core())
    # Self-contained: reads the parquet sweep itself, wires no ports.
    assert viz.inputs() == {}
    assert viz.outputs() == {}


@pytest.mark.fast
def test_missing_analysis_name_returns_note():
    from v2ecoli.visualizations.parquet_analysis import ParquetAnalysisView
    from bigraph_schema import allocate_core
    viz = ParquetAnalysisView(config={"title": "Empty"}, core=allocate_core())
    out = viz.update({})
    assert isinstance(out, dict) and isinstance(out.get("html"), str)
    assert "No analysis name" in out["html"]


@pytest.mark.fast
def test_missing_sweep_dir_returns_note_not_error():
    from v2ecoli.visualizations.parquet_analysis import ParquetAnalysisView
    from bigraph_schema import allocate_core
    viz = ParquetAnalysisView(
        config={"title": "Mass", "analysis": "mass_fraction_summary_view"},
        core=allocate_core(),
    )
    out = viz.update({})
    assert isinstance(out.get("html"), str)
    # No sweep dir yet → prompt to run, not a traceback.
    assert "Run this composite" in out["html"]


@pytest.mark.fast
def test_bogus_sweep_dir_is_caught_as_note(tmp_path):
    from v2ecoli.visualizations.parquet_analysis import ParquetAnalysisView
    from bigraph_schema import allocate_core
    viz = ParquetAnalysisView(
        config={"title": "Mass", "analysis": "mass_fraction_summary_view",
                "sweep_dir": str(tmp_path / "does_not_exist")},
        core=allocate_core(),
    )
    out = viz.update({})
    # An empty/absent sweep dir must degrade to a panel, never raise.
    assert isinstance(out.get("html"), str)
    assert "Could not render" in out["html"] or "Run this composite" in out["html"]
