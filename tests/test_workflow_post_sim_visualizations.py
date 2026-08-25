"""Unit tests for v2ecoli.workflow.post_sim_visualizations.EmitterHistorySummary
(backlog item 88's "Analysis flush" for a multi-node process-bigraph composite
dispatch, e.g. colony)."""

import json

import pytest


@pytest.mark.fast
def test_emitter_history_summary_is_a_post_sim_visualization_step_not_the_accumulate_render_family():
    """The load-bearing design decision this whole module exists to get right:
    unlike every OTHER v2ecoli visualization (ColonyVisualization/
    ColonyGrowthGif/etc, all viva_superpowers.visualization.Visualization —
    the process-bigraph-native, live-in-composite family run_flush() cannot
    see), this must be a v2ecoli.workflow.post_sim.Visualization
    (= VisualizationStep, the POST_SIM_REGISTRY family run_flush()'s
    iter_post_sim("visualization") actually discovers)."""
    from v2ecoli.workflow.post_sim_visualizations import EmitterHistorySummary
    from v2ecoli.workflow.post_sim import Visualization as PostSimVisualization
    from viva_superpowers.visualization import Visualization as LiveStepVisualization

    assert issubclass(EmitterHistorySummary, PostSimVisualization)
    assert not issubclass(EmitterHistorySummary, LiveStepVisualization)


@pytest.mark.fast
def test_emitter_history_summary_registered_in_post_sim_registry_end_to_end():
    """Proves the actual wiring (__init_subclass__ registration on import),
    not just the class's own isolated behavior -- import the module exactly
    as scripts/run_multi_node_analysis.py's own explicit
    `import v2ecoli.workflow.post_sim_visualizations` call site does before
    calling run_flush (plain `import v2ecoli` alone does NOT cascade into
    this module -- see its own docstring for why, mirroring workflow.analyses/
    workflow.report_cards' identical explicit-import convention)."""
    import v2ecoli.workflow.post_sim_visualizations  # noqa: F401 -- triggers registration side effects
    from v2ecoli.workflow.post_sim_visualizations import EmitterHistorySummary
    from v2ecoli.workflow.post_sim import iter_post_sim

    names_and_classes = dict(iter_post_sim("visualization"))
    assert names_and_classes.get("emitter_history_summary") is EmitterHistorySummary


@pytest.mark.fast
def test_emitter_history_summary_inputs_declares_out_dir():
    from bigraph_schema import allocate_core

    from v2ecoli.workflow.post_sim_visualizations import EmitterHistorySummary

    viz = EmitterHistorySummary({}, core=allocate_core())
    assert viz.inputs() == {"out_dir": "string"}


@pytest.mark.fast
def test_emitter_history_summary_renders_from_history_and_final_state(tmp_path):
    from bigraph_schema import allocate_core

    from v2ecoli.workflow.post_sim_visualizations import EmitterHistorySummary

    (tmp_path / "emitter_history.json").write_text(
        json.dumps(
            {
                "emitter": [[0.0, {"x": 1}], [60.0, {"x": 2}], [120.0, {"x": 3}]],
                "nested.path": [{"time": 5.0, "y": 1}],
            }
        )
    )
    (tmp_path / "final_state.json").write_text(json.dumps({"cells": {}, "global_time": 120.0}))

    viz = EmitterHistorySummary({}, core=allocate_core())
    result = viz.render(str(tmp_path))

    assert result is not None
    html, data = result
    assert "<h2>Emitter history summary</h2>" in html
    assert "emitter" in html and "nested.path" in html
    assert data["emitters"]["emitter"] == {"n_records": 3, "t_start": 0.0, "t_end": 120.0}
    assert data["emitters"]["nested.path"] == {"n_records": 1, "t_start": 5.0, "t_end": 5.0}
    assert data["final_state_keys"] == ["cells", "global_time"]
    assert "cells" in html and "global_time" in html


@pytest.mark.fast
def test_emitter_history_summary_renders_from_final_state_only(tmp_path):
    """No emitter_history.json (e.g. _persist_emitter_history's own
    best-effort gather found nothing, or never ran) -- must still render a
    real report from final_state.json alone, not degrade to nothing."""
    from bigraph_schema import allocate_core

    from v2ecoli.workflow.post_sim_visualizations import EmitterHistorySummary

    (tmp_path / "final_state.json").write_text(json.dumps({"agents": {"a": {}, "b": {}}}))

    viz = EmitterHistorySummary({}, core=allocate_core())
    result = viz.render(str(tmp_path))

    assert result is not None
    html, data = result
    assert "No in-memory emitter history was captured" in html
    assert data["emitters"] == {}
    assert data["final_state_keys"] == ["agents"]


@pytest.mark.fast
def test_emitter_history_summary_returns_none_when_neither_file_present(tmp_path):
    from bigraph_schema import allocate_core

    from v2ecoli.workflow.post_sim_visualizations import EmitterHistorySummary

    viz = EmitterHistorySummary({}, core=allocate_core())
    assert viz.render(str(tmp_path)) is None


@pytest.mark.fast
def test_emitter_history_summary_degrades_on_malformed_json(tmp_path):
    """A malformed emitter_history.json (e.g. a truncated upload) must not
    raise -- degrades to 'not present', falling back to final_state.json."""
    from bigraph_schema import allocate_core

    from v2ecoli.workflow.post_sim_visualizations import EmitterHistorySummary

    (tmp_path / "emitter_history.json").write_text("{not valid json")
    (tmp_path / "final_state.json").write_text(json.dumps({"x": 1}))

    viz = EmitterHistorySummary({}, core=allocate_core())
    result = viz.render(str(tmp_path))

    assert result is not None
    html, data = result
    assert data["emitters"] == {}
    assert data["final_state_keys"] == ["x"]


@pytest.mark.fast
def test_emitter_history_summary_update_reads_out_dir_from_state_not_study(tmp_path):
    """The exact real bug this class's own update() override exists to avoid:
    VisualizationStep's base update() hardcodes state["study"] -> render(study).
    Exercised via the SAME calling convention run_flush._run_one_step actually
    uses (a state dict built from this step's own declared inputs())."""
    from bigraph_schema import allocate_core

    from v2ecoli.workflow.post_sim_visualizations import EmitterHistorySummary

    (tmp_path / "final_state.json").write_text(json.dumps({"k": "v"}))

    viz = EmitterHistorySummary({}, core=allocate_core())
    out = viz.update({"out_dir": str(tmp_path)})

    assert out["view"] != ""
    assert "k" in out["view"]
    assert out["data"]["final_state_keys"] == ["k"]


@pytest.mark.fast
def test_emitter_history_summary_update_degrades_to_empty_view_without_out_dir():
    from bigraph_schema import allocate_core

    from v2ecoli.workflow.post_sim_visualizations import EmitterHistorySummary

    viz = EmitterHistorySummary({}, core=allocate_core())
    assert viz.update({}) == {"view": "", "data": {}}


@pytest.mark.fast
def test_summarize_entries_handles_tuple_list_and_dict_shapes():
    from v2ecoli.workflow.post_sim_visualizations import _summarize_entries

    assert _summarize_entries([[0.0, {"a": 1}], [10.0, {"a": 2}]]) == {
        "n_records": 2,
        "t_start": 0.0,
        "t_end": 10.0,
    }
    assert _summarize_entries([{"time": 3.0}, {"time": 7.0}]) == {
        "n_records": 2,
        "t_start": 3.0,
        "t_end": 7.0,
    }
    assert _summarize_entries("not-a-list") == {"n_records": 0, "t_start": None, "t_end": None}
    assert _summarize_entries([]) == {"n_records": 0, "t_start": None, "t_end": None}
