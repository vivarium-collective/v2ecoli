"""Tests for the single-cell in-document XArray emitter.

Three layers:

  * ``_single_cell_xarray_config`` — a pure builder of the STATIC XArrayEmitter
    config skeleton (no IO, no simulation). Since Task 4 the ``view`` and
    ``output_metadata`` are NOT built here — they are discovered lazily from the
    REALIZED composite state by ``SingleCellXArrayEmitter`` at run time (see its
    docstring / Task 4 C2), so this helper no longer inspects ``cell_state``.
  * ``build_composite("ecoli_baseline", emitter="xarray")`` — the real
    document-building branch (heavy: ParCa cache load, ~minutes). Verifies the
    in-document lazy emitter is wired and the parquet default is unchanged.
  * a ``slow`` integration test that actually runs 5 ticks and asserts a
    non-empty zarr store for ``bulk`` AND a substantial listener set.
"""
import glob
import os

import pytest

from v2ecoli.composites.ecoli_baseline import _single_cell_xarray_config

_CACHE_DIR = os.environ.get(
    "V2ECOLI_CACHE_DIR", "/Users/eranagmon/code/v2ecoli/out/cache")

# The build/run tests below load the ParCa cache (heavy, ~minutes) from a
# machine-specific absolute path. Skip them cleanly when the cache is absent so
# a bare ``pytest`` on CI / another machine collects and runs only the fast
# pure-config tests instead of failing. They are also marked ``slow``.
_needs_cache = pytest.mark.skipif(
    not os.path.isdir(_CACHE_DIR),
    reason=f"ParCa cache not found at {_CACHE_DIR} (set V2ECOLI_CACHE_DIR)")


def test_single_cell_xarray_config_is_flat_static_skeleton(tmp_path):
    cfg = _single_cell_xarray_config(
        out_uri=str(tmp_path / "s.zarr"),
        metadata={"experiment_id": "t", "variant": 0, "lineage_seed": 0})
    assert cfg["strategy"] == "flat" and cfg["emit_root"] == []
    assert cfg["out_uri"].endswith("s.zarr")
    # streaming, bounded: a small transducer buffer, not an unbounded history
    assert cfg["transducer"]["buffer"]["size"] >= 1
    assert cfg["writer"]["backend"] == "zarr"
    # metadata must be non-empty (Task 1 gotcha #1 — an empty dict silently
    # skips XArrayEmitter's partition setup and crashes on the first update()).
    assert cfg["metadata"]
    # view/output_metadata are discovered lazily at run time — NOT baked here.
    assert "view" not in cfg
    assert "output_metadata" not in cfg


def test_single_cell_xarray_config_metadata_defaults_nonempty(tmp_path):
    # Omitting metadata still yields a non-empty placeholder (never {}).
    cfg = _single_cell_xarray_config(out_uri=str(tmp_path / "s.zarr"))
    assert cfg["metadata"]


# ---------------------------------------------------------------------------
# Real ecoli_baseline document-building branch (heavy — ParCa cache, ~minutes).
# ---------------------------------------------------------------------------


@pytest.mark.slow
@_needs_cache
def test_xarray_build_has_in_document_emitter(tmp_path):
    from v2ecoli import build_composite
    comp = build_composite("ecoli_baseline",
                           cache_dir=_CACHE_DIR,
                           out_dir=str(tmp_path),
                           emitter="xarray")
    emitter_step = comp.state["agents"]["0"]["emitter"]
    inst = emitter_step["instance"] if isinstance(emitter_step, dict) else emitter_step[0]
    # In-document lazy wrapper around a real XArrayEmitter (built at run time).
    assert type(inst).__name__ == "SingleCellXArrayEmitter"
    # agent-relative wiring resolves bulk -> agents/0/bulk (not top-level)
    wires = emitter_step["inputs"] if isinstance(emitter_step, dict) else {}
    assert "bulk" in wires and "listeners" in wires


@pytest.mark.slow
@_needs_cache
def test_parquet_default_still_parquet(tmp_path):
    from v2ecoli import build_composite
    comp = build_composite("ecoli_baseline", cache_dir=_CACHE_DIR)
    step = comp.state["agents"]["0"]["emitter"]
    inst = step["instance"] if isinstance(step, dict) else step[0]
    assert "Parquet" in type(inst).__name__ or "RAM" in type(inst).__name__  # unchanged default path


@pytest.mark.slow
@_needs_cache
def test_xarray_run_captures_bulk_and_listeners(tmp_path):
    """The crux: a real 5-tick single-cell run streams non-empty bulk +
    a substantial listener set to zarr."""
    import xarray as xr
    from v2ecoli import build_composite

    comp = build_composite("ecoli_baseline",
                           cache_dir=_CACHE_DIR,
                           emitter="xarray",
                           out_dir=str(tmp_path))
    comp.run(5)  # build_composite's run-wrap flushes the trailing buffer

    stores = glob.glob(str(tmp_path / "**" / "*.zarr"), recursive=True)
    assert stores, "no zarr store written"
    dt = xr.open_datatree(stores[0], engine="zarr")

    # collect non-empty observable groups
    nonempty = {}
    for g in dt.groups:
        for v in dt[g].ds.data_vars:
            sz = int(dt[g].ds[v].size)
            if sz > 0:
                nonempty.setdefault(g.rsplit("/", 1)[-1], 0)
                nonempty[g.rsplit("/", 1)[-1]] += sz
    assert nonempty, "zarr store has no observable data"
    # bulk must be captured non-empty (C1: structured record array projected).
    assert nonempty.get("bulk", 0) > 0, f"bulk empty/missing; groups={list(nonempty)}"
    # a substantial listener set must be captured (C2: realized-state view),
    # not the ~4 leaves a pre-realize view would yield.
    listener_leaves = [k for k in nonempty if k != "bulk"
                       and not k.startswith("time_gen")]
    assert len(listener_leaves) >= 30, (
        f"too few listener leaves captured: {len(listener_leaves)} "
        f"({sorted(listener_leaves)})")

    # F3: the run-end flush hook finalized the writer, so a SECOND run() must
    # fail loudly (single-run sink) rather than silently drive a closed writer.
    with pytest.raises(RuntimeError, match="already closed"):
        comp.run(5)
