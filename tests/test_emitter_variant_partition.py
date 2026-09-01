"""Regression tests for the parquet hive ``variant`` partition column.

Guards the P0-10 fix (pipeline audit §3.2): the generator-declared *default*
ParquetEmitter path (``_build_declared_emitter``) previously threaded only
``out_dir`` and ``experiment_id`` into the vEcoli-shaped preset, so every cell
in a multivariant sweep wrote hive partition ``variant=0``. A multivariant KPI
analysis grouping by ``variant`` then saw all variants as one — meaningless
results.

These tests assert that the emitter config each cell carries reflects that
cell's own variant index (and that a declared ``experiment_id`` actually
reaches the emitter instead of silently defaulting to ``"default"``).
"""

from __future__ import annotations

import pytest

from v2ecoli.composites._helpers import _build_declared_emitter


LISTENERS_SCHEMA = {"mass": {"cell_mass": "float", "dry_mass": "float"}}


def _emitter_cfg(core, *, variant=None, experiment_id=None, lineage_seed=None,
                 agent_id=None, out_dir="/tmp/pq_variant_test"):
    """Build the declared-default ParquetEmitter for one cell and return its
    resolved config dict (what actually drives the hive partitioning)."""
    cfg = {"out_dir": out_dir}
    if variant is not None:
        cfg["variant"] = variant
    if experiment_id is not None:
        cfg["experiment_id"] = experiment_id
    if lineage_seed is not None:
        cfg["lineage_seed"] = lineage_seed
    if agent_id is not None:
        cfg["agent_id"] = agent_id
    decl = {"address": "local:ParquetEmitter", "config": cfg}
    instance, _topo = _build_declared_emitter(decl, LISTENERS_SCHEMA, core)
    if type(instance).__name__ != "ParquetEmitter":
        pytest.skip("[parquet] extra (viva_emitters) not installed; declared "
                    "default fell back to RAMEmitter")
    return instance.config


@pytest.mark.fast
def test_two_variant_fanout_writes_distinct_partitions(core):
    """Two cells fanned out at variant 0 and 1 carry distinct ``variant``
    partition metadata — not both ``variant=0``."""
    cfg0 = _emitter_cfg(core, variant=0, experiment_id="run4")
    cfg1 = _emitter_cfg(core, variant=1, experiment_id="run4")

    assert cfg0["metadata"]["variant"] == 0
    assert cfg1["metadata"]["variant"] == 1
    # ``variant`` is a hive partition key, so distinct values land in
    # distinct partitions the multivariant KPI analysis can group on.
    assert "variant" in cfg0["partitioning_keys"]
    assert cfg0["metadata"]["variant"] != cfg1["metadata"]["variant"]


@pytest.mark.fast
def test_variant_defaults_to_zero_when_absent(core):
    """A single (baseline) arm with no declared variant still writes
    ``variant=0`` — back-compat for the non-sweep path."""
    cfg = _emitter_cfg(core, experiment_id="baseline")
    assert cfg["metadata"]["variant"] == 0


@pytest.mark.fast
def test_declared_experiment_id_reaches_emitter(core):
    """A declared ``experiment_id`` reaches the emitter metadata (previously it
    silently defaulted to ``"default"`` on this path)."""
    cfg = _emitter_cfg(core, variant=2, experiment_id="my-run-4")
    # experiment_id is URL-quoted by the preset; assert it is NOT the fallback.
    assert cfg["metadata"]["experiment_id"] == "my-run-4"
    assert cfg["metadata"]["experiment_id"] != "default"


@pytest.mark.fast
def test_all_identity_fields_thread_through(core):
    """variant / lineage_seed / agent_id all reach the partition metadata."""
    cfg = _emitter_cfg(core, variant=5, experiment_id="run4",
                       lineage_seed=3, agent_id="0")
    md = cfg["metadata"]
    assert md["variant"] == 5
    assert md["lineage_seed"] == 3
    assert md["agent_id"] == "0"
