"""Biologically-organized E. coli whole-cell composite.

Identical simulation to ``baseline()`` — same processes, same update math —
but the store hierarchy is relabeled into cellular compartments / molecular
classes via a pure path-remap (see _remap.py and
docs/superpowers/specs/2026-06-13-biological-composite-design.md).

Phase 1: relabel only -> bit-identical to baseline (see
tests/test_biological_equivalence.py). Phase 2 (not built here) splits the
monolithic pools and adds unit-bearing schemas.
"""
from __future__ import annotations

from typing import Any

from viva_superpowers.composite_generator import composite_generator

from v2ecoli.composites.baseline import baseline
from v2ecoli.composites._remap import remap_cell_state


@composite_generator(
    name="biological",
    description=(
        "Biologically-organized whole-cell E. coli model. Runs the exact same "
        "55-process simulation as baseline (bit-identical results) but relabels the "
        "store hierarchy into cellular compartments and molecular classes — "
        "cell/molecules, cell/observables, environment/…, machinery/…, clock/… — so "
        "the bigraph reads as biology rather than plumbing."
    ),
    emitters=[
        {
            "address": "local:ParquetEmitter",
            "config": {},
            # Remapped emit paths (baseline used global_time/bulk/listeners).
            "paths": ["clock/global_time", "cell/molecules", "cell/observables"],
        },
    ],
)
def biological(core: Any = None, **kwargs) -> dict:
    """Build the biological composite document.

    All keyword arguments are forwarded verbatim to :func:`baseline`
    (seed, cache_dir, emitter, feature toggles, bundle, …). The finished
    baseline document is then relabeled in place at ``state.agents.<id>``.
    """
    doc = baseline(core=core, **kwargs)
    agents = doc['state']['agents']
    for agent_id, cell_state in list(agents.items()):
        agents[agent_id] = remap_cell_state(cell_state)
    return doc
