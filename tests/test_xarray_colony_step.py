"""Task 3: drive XArrayEmitter as a process-bigraph Step with colony config.

The emitter implementation moved to pbg-emitters >=0.2.0 where ``XArrayEmitter``
is a *generic* Step: ``emit_root`` defaults to ``()`` (flat wired state) and
``strategy`` to ``"flat"`` (degenerate single partition, ``generation == 1``).
v2ecoli's multi-cell/lineage zarr layout is preserved ONLY when the emitter is
configured with the colony partition config — ``strategy="colony"`` plus
``emit_root=["agents", <agent_id>]`` — which this module asserts the v2ecoli
config builder now emits, and that the resulting Step reproduces the colony
storage layout (``experiment_id=/variant=/lineage_seed=`` partition dirs +
``generation=N`` data-vars keyed by ``len(agent_id)``).
"""
from __future__ import annotations

import os

import pytest

from v2ecoli.library.xarray_run import build_emitter_config


# ----- unit: the config builder emits the colony partition config -----------


def test_build_emitter_config_sets_colony_strategy_and_emit_root():
    cfg = build_emitter_config(
        store_path="/tmp/colony.zarr",
        view=[{"root": ("listeners", "mass"),
               "variables": {"dry_mass": [{"path": "dry_mass", "dtype": "<f4"}]}}],
        metadata_base={"experiment_id": "t", "variant": 0, "lineage_seed": 0},
        generation=2, agent_id="00",
    )
    # colony partition config (the heart of the Task-3 migration)
    assert cfg["strategy"] == "colony"
    assert cfg["emit_root"] == ["agents", "00"]
    # emit_root is keyed to THIS generation's lineage id
    cfg_gen1 = build_emitter_config(
        store_path="/tmp/colony.zarr",
        view=[{"root": ("listeners", "mass"),
               "variables": {"dry_mass": [{"path": "dry_mass", "dtype": "<f4"}]}}],
        metadata_base={"experiment_id": "t"}, generation=1, agent_id="0")
    assert cfg_gen1["emit_root"] == ["agents", "0"]
    # metadata still carries agent_id/generation for the colony partition
    assert cfg["metadata"]["agent_id"] == "00"
    assert cfg["metadata"]["generation"] == 2


# ----- structural: the Step reproduces the colony zarr layout ---------------


def _emit_one_generation(tmp_path, agent_id, generation, values):
    """Build a colony XArrayEmitter Step and drive a few update() ticks for a
    single cell of the lineage, returning the shared store path."""
    pytest.importorskip("xarray")
    pytest.importorskip("zarr")
    from v2ecoli.core import build_core
    from v2ecoli.library.xarray_emitter import XArrayEmitter

    store = str(tmp_path / "colony.zarr")
    view = [{"root": ("listeners", "mass"),
             "variables": {"dry_mass": [{"path": "dry_mass", "dtype": "<f8"}]}}]
    cfg = build_emitter_config(
        store_path=store, view=view,
        metadata_base={"experiment_id": "expA", "variant": 0, "lineage_seed": 7},
        generation=generation, agent_id=agent_id, buffer_size=3)
    em = XArrayEmitter(config=cfg, core=build_core())
    for t, v in enumerate(values, start=1):
        # The Step receives the v2ecoli colony envelope; emit_root strips it.
        em.update({
            "time": float(t), "global_time": float(t),
            "agents": {agent_id: {"listeners": {"mass": {"dry_mass": float(v)}}}},
        })
    em.close(success=True)
    return store


def test_colony_step_reproduces_lineage_zarr_layout(tmp_path):
    import xarray as xr

    # Two generations of one lineage: mother "0" (gen 1) then daughter "00"
    # (gen 2), written into the SAME store (the multigen colony pattern).
    # 4 ticks (not a multiple of buffer_size=3) to dodge the pbg-emitters
    # final-flush ``assert not include_static`` when the buffer is exactly full
    # at close — the same reason run_multigen_xarray defaults buffer_size=3.
    store = _emit_one_generation(tmp_path, "0", 1, [10.0, 11.0, 12.0, 13.0])
    _emit_one_generation(tmp_path, "00", 2, [20.0, 21.0, 22.0, 23.0])

    # 1. partition directories: experiment_id=/variant=/lineage_seed=
    parts = os.path.join(store, "experiment_id=expA", "variant=0", "lineage_seed=7")
    assert os.path.isdir(parts), f"colony partition dir missing under {store}: " \
        f"{sorted(os.listdir(store))}"

    # 2. observable leaf node (dry_mass) present under the partition
    dt = xr.open_datatree(store, engine="zarr")
    groups = list(dt.groups)
    leaf_group = next((g for g in groups if g.endswith("dry_mass")), None)
    assert leaf_group is not None, groups

    # 3. both generations land as ``generation=N`` data-vars (keyed by
    #    len(agent_id)) inside the SAME observable node — the colony layout.
    ds = dt[leaf_group].to_dataset()
    assert "generation=1" in ds.data_vars, list(ds.data_vars)
    assert "generation=2" in ds.data_vars, list(ds.data_vars)

    # 4. values round-trip per generation
    assert ds["generation=1"].values.ravel().tolist() == [10.0, 11.0, 12.0, 13.0]
    assert ds["generation=2"].values.ravel().tolist() == [20.0, 21.0, 22.0, 23.0]
