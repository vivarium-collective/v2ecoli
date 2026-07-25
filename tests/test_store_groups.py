"""Biological store-grouping annotation (distilled from the retired `biological`
composite). Guards the contract that this is annotation ONLY: baseline's emitted
store paths stay flat so downstream analyses keep working.
"""
from __future__ import annotations

from viva_superpowers.composite_generator import _REGISTRY

import v2ecoli.composites  # noqa: F401 — force generator registration
from v2ecoli.composites.store_groups import STORE_GROUPS, group_of


def test_store_groups_shape():
    # Every value is a biological group path (tuple of segments), first segment
    # is one of the four top-level compartments.
    top = {"cell", "environment", "machinery", "clock"}
    assert STORE_GROUPS["bulk"] == ("cell", "molecules")
    assert STORE_GROUPS["listeners"] == ("cell", "observables")
    assert STORE_GROUPS["boundary"] == ("environment", "boundary")
    for store, path in STORE_GROUPS.items():
        assert isinstance(path, tuple) and path, store
        assert path[0] in top, (store, path)


def test_group_of_helper():
    assert group_of("bulk") == ("cell", "molecules")
    assert group_of("not_a_store") is None


def test_baseline_reexports_store_groups():
    # baseline "carries" its grouping: importable from the baseline module.
    from v2ecoli.composites.baseline import STORE_GROUPS as baseline_groups
    assert baseline_groups is STORE_GROUPS


def test_baseline_emit_paths_unchanged():
    # THE ANALYSES CONTRACT: the grouping is annotation only — baseline's default
    # emitter still emits the FLAT paths, never the biological ones. If this
    # breaks, downstream analyses that read global_time/bulk/listeners break.
    entry = _REGISTRY["v2ecoli.composites.baseline"]
    parquet = [e for e in (entry.emitters or [])
               if "ParquetEmitter" in str(e.get("address", ""))]
    assert parquet, "baseline must ship its default ParquetEmitter"
    assert parquet[0]["paths"] == ["global_time", "bulk", "listeners"]
    # and the biological group labels must NOT have leaked into the emit paths
    assert not any("/" in p or p.startswith(("cell", "environment", "machinery"))
                   for p in parquet[0]["paths"])
