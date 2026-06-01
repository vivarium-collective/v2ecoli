"""Lean composite — a simpler wiring of the same whole-cell model.

A deliberately simplified sibling of ``baseline`` that trades bit-for-bit
parity for structural clarity. It reuses every biology component unchanged —
the same process/step classes, the same ``update()`` bodies, the same
``description``s — and simplifies the partition/scheduling scaffolding:

  * **One global allocator.** Baseline runs the partition machinery in two
    staged passes (``allocator_2`` for rna-degradation, ``allocator_3`` for the
    two elongation processes). Lean collapses them into a single ``allocator``
    pass covering all three partitioned processes at once.

  * **No unique-molecule flush barriers.** Baseline interleaves 11
    ``unique_update`` steps that drain the UniqueNumpyUpdater buffer between
    layers. Lean drops them; unique-molecule updates reconcile through the
    composite's normal apply path. This is the parity-relaxing change — flush
    timing differs, so the trajectory is NOT bit-identical to baseline.

What lean deliberately KEEPS (the experiments showed these are load-bearing,
not scaffolding):

  * **The requester → allocate → evolve split, and its flow-token ordering.**
    A fully-fused, allocator-free variant was prototyped and HANGS in the
    tRNA-charging ODE (``lsoda``): polypeptide elongation only integrates on a
    *bounded* (partitioned) resource share, so the allocator is a biological
    precondition. And dropping the flow-token barrier (pure interval+priority
    scheduling) left the rna-deg evolver reading ``bulk=None`` — the
    request→allocate→evolve ORDER is a hard dependency the token chain enforces.

Net: baseline's ~45 steps → ~34 edges (one allocator instead of two, the 11
flush steps removed), same biology, same ordering guarantees where they matter.

NOT a drop-in for baseline (not vEcoli-parity). A research/teaching variant:
the partition flow with the staging + flush scaffolding pared back to the
minimum that still runs. Validate by tolerance (growth, division time, mass),
not byte-identical parity.
"""
from __future__ import annotations

import copy
from typing import Any

from pbg_superpowers.composite_generator import composite_generator

from v2ecoli.core import build_core
from v2ecoli.composites import baseline as _baseline
from v2ecoli.composites._helpers import DEFAULT_SINGLE_CELL_VISUALIZATIONS


def _lean_layers(raw):
    """Simplify a baseline execution-layer list: collapse the two staged
    allocators into a single global ``allocator`` and drop the flush steps.

    ``raw`` is the already-built baseline layer list (passed in to avoid
    re-entering the monkeypatched ``build_execution_layers``). Keeps the
    flow-token ordering that the request→allocate→evolve flow depends on.
    """
    out = []
    seen_allocator = False
    for layer in raw:
        if not isinstance(layer, list) or not layer:
            continue
        steps = [s for s in layer if not s.startswith("unique_update")]
        if not steps:
            continue  # drop pure-flush layers
        # rewrite any staged allocator (allocator_2 / allocator_3) to a single
        # global 'allocator'; keep only the first occurrence.
        rewritten = []
        for s in steps:
            if s.startswith("allocator"):
                if seen_allocator:
                    continue
                seen_allocator = True
                rewritten.append("allocator")
            else:
                rewritten.append(s)
        if rewritten:
            out.append(rewritten)
    return out


@composite_generator(
    name="lean",
    description=(
        "Lean whole-cell E. coli — same biology as baseline with the partition "
        "scaffolding pared back: ONE global allocator instead of two staged "
        "passes, and the 11 unique-molecule flush barriers removed. The "
        "requester/allocator/evolver split and its flow-token ordering are kept "
        "(both proven load-bearing: the allocator bounds the tRNA-charging ODE, "
        "and the order is a hard dependency). Not vEcoli-parity — validate by "
        "tolerance, not bit-equality."
    ),
    parameters={
        "seed": {"type": "integer", "default": 0,
                 "description": "RNG seed for stochastic initialization."},
        "cache_dir": {"type": "string", "default": "out/cache",
                      "description": "Path to the ParCa cache directory."},
    },
    visualizations=DEFAULT_SINGLE_CELL_VISUALIZATIONS,
    emitters=[{
        "address": "local:ParquetEmitter",
        "config": {},
        "paths": ["global_time", "bulk", "listeners"],
    }],
)
def lean(core: Any = None, *, seed: int = 0, cache_dir: str = "out/cache",
         bundle: dict | None = None) -> dict:
    """Build the lean composite document.

    Monkeypatches baseline's ``build_execution_layers`` to the simplified
    lean layering for the duration of the build (collapsed allocator, no
    flushes), then delegates to ``baseline.baseline`` so all step-config and
    flow-token wiring is reused unchanged. The single ``allocator`` step is
    built by baseline's existing allocator path, which already defaults
    ``process_names`` to all partitioned processes when unscoped.
    """
    if core is None:
        core = build_core()

    _orig = _baseline.build_execution_layers
    _baseline.build_execution_layers = lambda features=None: _lean_layers(_orig(features))
    try:
        doc = _baseline.baseline(core=core, seed=seed, cache_dir=cache_dir,
                                 bundle=bundle)
    finally:
        _baseline.build_execution_layers = _orig
    return doc
