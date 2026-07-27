"""Run harness — build a geometry+tier Composite, run, sample, extract.

Samples composite.state['cells'] directly each tick (emit_cells=False) to
avoid the outer-emitter RAM leak, mirroring the colonies-02 perf pattern.
"""
from __future__ import annotations
from typing import Any

from process_bigraph import Composite
from viva_munk import core_import

from v2ecoli.colony_bench import geometries
from v2ecoli.colony_bench.phenotypes import phenotype_extractor

_GEOMETRIES = {
    "free_colony": geometries.free_colony,
    "mother_machine": geometries.mother_machine,
    "daughter_machine": geometries.daughter_machine,
}


def build_bench_core(tier: str):
    core = core_import()  # registers PymunkProcess, GrowDivide, AdderGrowDivide
    if tier == "wcm":
        from v2ecoli.bridge import EcoliWCM
        from v2ecoli.types import ECOLI_TYPES
        core.register_types(ECOLI_TYPES)
        core.register_link("EcoliWCM", EcoliWCM)
    return core


def _sample(state) -> dict[str, dict[str, Any]]:
    out = {}
    for cid, cell in state.get("cells", {}).items():
        rec = {"mass": float(cell.get("mass", 0.0)),
               "length": float(cell.get("length", 0.0))}
        if "volume" in cell:
            rec["volume"] = float(cell.get("volume", 0.0))
        if isinstance(cell.get("exchange"), dict) and cell["exchange"]:
            rec["exchange"] = dict(cell["exchange"])
        out[cid] = rec
    return out


def run_bench(geometry: str, tier: str, *, n_ticks: int, dt: float = 1.0,
              sample_every: int = 1, seed: int = 0,
              builder_kwargs: dict | None = None) -> dict[str, Any]:
    if geometry not in _GEOMETRIES:
        raise ValueError(f"unknown geometry: {geometry!r}")
    core = build_bench_core(tier)
    doc = _GEOMETRIES[geometry](tier, seed=seed, **(builder_kwargs or {}))
    comp = Composite({"state": doc}, core=core)

    trajectory = []
    for tick in range(n_ticks):
        comp.run(dt)
        if tick % sample_every == 0:
            trajectory.append({
                "time": float(comp.state.get("global_time", (tick + 1) * dt)),
                "cells": _sample(comp.state),
            })
    return {
        "trajectory": trajectory,
        "phenotypes": phenotype_extractor(trajectory),
        "n_final": len(comp.state.get("cells", {})),
    }
