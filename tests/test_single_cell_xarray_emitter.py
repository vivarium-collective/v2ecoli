"""Unit tests for ``_single_cell_xarray_config`` (Task 2 of the single-cell
XArray emitter plan).

Uses a FAKE in-memory cell dict (no composite build) — the helper is a pure
config builder, no IO, no simulation needed to exercise it.

NOTE on the ``bulk`` view root: the brief's draft test snippet asserted
``("bulk",) in roots``, matching the brief's draft pseudocode (``root=
("bulk",)``, leaf ``path=()``). Task 1's spike (see
``.superpowers/sdd/2026-08-10-single-cell-xarray-emitter/task-1-report.md``,
section 2) found that draft is wrong two ways: it computes the wrong read
path (``("bulk","bulk")``, which doesn't exist in the cell state), and
``LeafView.path`` (the OUTPUT variable name) must be non-empty, so
``path=()`` raises ``TypeError``. The validated, working shape is
``root=()`` with a leaf named ``"bulk"`` in ``variables`` (read path ==
``() + ("bulk",) == ("bulk",)``, matching ``cell["bulk"]``'s actual
top-level location). This test asserts the CORRECT (validated) shape rather
than the brief's draft snippet, per the parent task's explicit instruction
to honor Task 1's load-bearing decisions over the brief's pseudocode.
"""
from v2ecoli.composites.ecoli_baseline import _single_cell_xarray_config


def _fake_cell():
    return {"global_time": 0.0,
            "bulk": [0, 1, 2, 3],
            "listeners": {"mass": {"cell_mass": 1.0, "dry_mass": 0.3}}}


def test_single_cell_xarray_config_is_flat_and_covers_bulk_and_listeners(tmp_path):
    cfg = _single_cell_xarray_config(_fake_cell(), out_uri=str(tmp_path / "s.zarr"))
    assert cfg["strategy"] == "flat" and cfg["emit_root"] == []
    assert cfg["out_uri"].endswith("s.zarr")
    # streaming, bounded: a small transducer buffer, not an unbounded history
    assert cfg["transducer"]["buffer"]["size"] >= 1

    by_root = {tuple(entry["root"]): entry["variables"] for entry in cfg["view"]}
    assert ("listeners",) in by_root
    assert "mass" in by_root[("listeners",)]

    # bulk: root must be () (agent-relative top-level leaf), not ("bulk",) —
    # see module docstring above / Task 1 report section 2.
    assert () in by_root
    assert "bulk" in by_root[()]

    # metadata must be non-empty (Task 1 gotcha #1 — an empty dict silently
    # skips XArrayEmitter's partition setup and crashes on the first update()).
    assert cfg["metadata"]

    # sanity: output_metadata / writer / emit keys present and pure dict shapes.
    assert isinstance(cfg["output_metadata"], dict)
    assert cfg["writer"]["backend"] == "zarr"


# ---------------------------------------------------------------------------
# Task 3: wiring _single_cell_xarray_config into the emitter=="xarray" branch
# of ecoli_baseline.baseline(). These tests build a REAL ecoli_baseline
# composite (heavy — ParCa cache load, ~minutes) so they exercise the actual
# document-building branch, not a fake cell dict.
# ---------------------------------------------------------------------------
_CACHE_DIR = "/Users/eranagmon/code/v2ecoli/out/cache"


def test_xarray_build_has_in_document_emitter(tmp_path):
    from v2ecoli import build_composite
    comp = build_composite("ecoli_baseline",
                           cache_dir=_CACHE_DIR,
                           out_dir=str(tmp_path),
                           emitter="xarray")
    emitter_step = comp.state["agents"]["0"]["emitter"]
    inst = emitter_step["instance"] if isinstance(emitter_step, dict) else emitter_step[0]
    assert type(inst).__name__ == "XArrayEmitter"
    # agent-relative wiring resolves bulk -> agents/0/bulk (not top-level)
    wires = emitter_step["inputs"] if isinstance(emitter_step, dict) else {}
    assert "bulk" in wires and "listeners" in wires


def test_parquet_default_still_parquet(tmp_path):
    from v2ecoli import build_composite
    comp = build_composite("ecoli_baseline", cache_dir=_CACHE_DIR)
    step = comp.state["agents"]["0"]["emitter"]
    inst = step["instance"] if isinstance(step, dict) else step[0]
    assert "Parquet" in type(inst).__name__ or "RAM" in type(inst).__name__  # unchanged default path
