"""scripts/run_multi_node_analysis.py -- backlog item 88's "Analysis flush"
entrypoint for a completed multi-node process-bigraph composite dispatch
(e.g. a colony composite spread across N Ray-cluster nodes). Proves the
renderer-resolution-by-composite-id contract, the emitter-history flattening
that bridges viva-api's generic ``run_pbg.py`` persistence to
``ColonyVisualization``'s expected shape, and the manifest contract
``GET /analyses/{id}/status`` actually reads (``written``/``errors``)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_multi_node_analysis import (
    _derive_env_size,
    _flatten_agent_history,
    _render_colony,
    _unpack_entry,
    run,
)


def _fake_aws_cp(monkeypatch, downloads: dict[str, str], written: dict[str, str]):
    """Stub _aws_cp: 'download' (src is s3://) writes pre-seeded content to
    the requested local dest; 'upload' (dst is s3://) records dst -> local
    file content. Mirrors test_standalone_analysis.py's own identical
    pattern for the sibling entrypoint script."""
    import scripts.run_multi_node_analysis as mod

    def fake(src: str, dst: str) -> None:
        if src.startswith("s3://"):
            if src not in downloads:
                raise __import__("subprocess").CalledProcessError(
                    1, ["aws"], stderr=b"NoSuchKey"
                )
            Path(dst).write_text(downloads[src])
        else:
            written[dst] = Path(src).read_text()

    monkeypatch.setattr(mod, "_aws_cp", fake)


def test_unpack_entry_handles_tuple_and_dict_shapes():
    t, data = _unpack_entry((5.0, {"agents": {"a": {}}}))
    assert t == 5.0
    assert data == {"agents": {"a": {}}}

    t, data = _unpack_entry({"time": 10.0, "agents": {"b": {}}})
    assert t == 10.0
    assert data["agents"] == {"b": {}}

    t, data = _unpack_entry("garbage")
    assert (t, data) == (0.0, {})


def test_flatten_agent_history_from_synthetic_emitter_entries():
    entries = [
        (
            0.0,
            {
                "agents": {
                    "cell-1": {"location": [1.0, 2.0], "length": 1.5, "mass": 380.0}
                }
            },
        ),
        {
            "time": 60.0,
            "agents": {
                "cell-1": {"location": (1.1, 2.1), "length": 1.6, "mass": 400.0}
            },
        },
    ]
    rows = _flatten_agent_history(entries)
    assert len(rows) == 2
    assert rows[0] == {
        "agent_id": "cell-1",
        "time": 0.0,
        "x": 1.0,
        "y": 2.0,
        "length": 1.5,
        "mass": 380.0,
    }
    assert rows[1]["time"] == 60.0
    assert rows[1]["mass"] == 400.0


def test_flatten_agent_history_skips_non_dict_entries():
    entries = [(0.0, {"agents": {"c1": {"location": [0.0, 0.0]}, "c2": "not-a-dict"}})]
    rows = _flatten_agent_history(entries)
    assert len(rows) == 1
    assert rows[0]["agent_id"] == "c1"


def test_derive_env_size_from_positions_and_default_when_empty():
    rows = [{"x": 3.0, "y": -4.0}, {"x": 1.0, "y": 1.0}]
    # extent = max(|x|,|y|) across rows = 4.0 -> 4*2 + margin(5) = 13.0
    assert _derive_env_size(rows) == 13.0
    assert _derive_env_size([]) == 40.0  # default, no data to derive from


def test_render_colony_produces_html_with_history(tmp_path):
    emitter_history = {
        "emitter": [
            (
                0.0,
                {
                    "agents": {
                        "a": {"location": [0.0, 0.0], "length": 1.0, "mass": 380.0}
                    }
                },
            ),
            (
                60.0,
                {
                    "agents": {
                        "a": {"location": [0.1, 0.0], "length": 1.1, "mass": 400.0}
                    }
                },
            ),
        ]
    }
    final_state = {"cells": {"a": {}}}
    html = _render_colony(emitter_history, final_state, tmp_path)
    assert "<html" in html and "</html>" in html
    assert "E. coli Colony Simulation" in html


def test_render_colony_degrades_gracefully_without_emitter_history(tmp_path):
    """No history captured (e.g. run_pbg's best-effort gather found nothing)
    -- must still render a real report from final_state alone, not raise."""
    final_state = {"cells": {"a": {}, "b": {}}}
    html = _render_colony(None, final_state, tmp_path)
    assert "<html" in html and "</html>" in html


def test_run_writes_report_and_manifest_when_history_available(tmp_path, monkeypatch):
    written: dict[str, str] = {}
    emitter_history = {
        "emitter": [
            (
                0.0,
                {
                    "agents": {
                        "a": {"location": [0.0, 0.0], "length": 1.0, "mass": 380.0}
                    }
                },
            ),
        ]
    }
    _fake_aws_cp(
        monkeypatch,
        {
            "s3://bucket/exp/emitter_history.json": json.dumps(emitter_history),
            "s3://bucket/exp/final_state.json": json.dumps({"cells": {"a": {}}}),
        },
        written,
    )

    manifest = run(
        composite_id="v2ecoli.composites.ecoli_colony.ecoli_colony",
        history_uri="s3://bucket/exp",
        out_uri="s3://bucket/exp/analyses/test-analysis",
        experiment_id="exp",
        tmp=tmp_path,
    )

    assert manifest["status"] == "done"
    assert manifest["written"] == ["s3://bucket/exp/analyses/test-analysis/report.html"]
    assert manifest["errors"] == []
    assert "<html" in written["s3://bucket/exp/analyses/test-analysis/report.html"]
    manifest_written = json.loads(
        written["s3://bucket/exp/analyses/test-analysis/_manifest.json"]
    )
    assert manifest_written == manifest


def test_run_records_error_for_unregistered_composite_id(tmp_path, monkeypatch):
    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {}, written)

    manifest = run(
        composite_id="some_workspace.composites.unregistered_composite",
        history_uri="s3://bucket/exp",
        out_uri="s3://bucket/exp/analyses/test-analysis",
        experiment_id="exp",
        tmp=tmp_path,
    )

    assert manifest["status"] == "failed"
    assert manifest["written"] == []
    assert (
        "no registered multi-node analysis renderer" in manifest["errors"][0]["error"]
    )


def test_run_records_error_when_neither_history_nor_final_state_exists(
    tmp_path, monkeypatch
):
    written: dict[str, str] = {}
    _fake_aws_cp(monkeypatch, {}, written)  # every download 404s

    manifest = run(
        composite_id="v2ecoli.composites.ecoli_colony.ecoli_colony",
        history_uri="s3://bucket/exp",
        out_uri="s3://bucket/exp/analyses/test-analysis",
        experiment_id="exp",
        tmp=tmp_path,
    )

    assert manifest["status"] == "failed"
    assert manifest["written"] == []
    assert (
        "neither emitter_history.json nor final_state.json"
        in manifest["errors"][0]["error"]
    )
