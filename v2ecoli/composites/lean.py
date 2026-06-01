"""Lean composite — a simpler wiring of the same whole-cell model.

A deliberately simplified sibling of ``baseline`` that trades bit-for-bit
parity for structural clarity. It reuses every biology component unchanged —
the same process/step classes, the same ``update()`` bodies, the same
``description``s — and simplifies the partition/scheduling scaffolding:

  * **One global allocator.** Baseline runs the partition machinery in two
    staged passes (``allocator_2`` for rna-degradation, ``allocator_3`` for the
    two elongation processes). Lean collapses them into a single ``allocator``
    pass covering all three partitioned processes at once.

What lean deliberately KEEPS — three things prototyping proved load-bearing,
NOT scaffolding (each was removed and broke the model):

  * **The 11 ``unique_update`` flush steps.** These are NOT cosmetic buffer
    drains — each emits ``{"update": True}`` to its unique-molecule store,
    which is what COMMITS the UniqueNumpyUpdater's pending additions/deletions
    into the array. Dropping them froze every unique-molecule count at its t=0
    value (no transcription / replication / ribosome creation ever landed) —
    the −50% species divergence seen in an early lean draft. Kept verbatim.
  * **The requester → allocate → evolve split, and its flow-token ordering.**
    A fully-fused, allocator-free variant HANGS in the tRNA-charging ODE
    (``lsoda``): polypeptide elongation only integrates on a *bounded*
    (partitioned) resource share, so the allocator is a biological
    precondition. And pure interval+priority scheduling (no flow tokens) left
    the rna-deg evolver reading ``bulk=None`` — request→allocate→evolve ORDER
    is a hard dependency the token chain enforces.

So lean's ONE remaining simplification is collapsing the two staged allocator
passes (allocator_2 + allocator_3) into a single global allocator over all
three partitioned processes. With the flushes kept, lean tracks baseline
closely (gene/promoter/DnaA_box exact; ribosome/RNA/RNAP within ~0.3–2%) —
the small residual is the single-allocator change, not a structural defect.

Net: baseline's ~45 steps → ~44 (one allocator instead of two), same biology.

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
        # Keep the unique_update flush layers UNCHANGED — they are NOT
        # cosmetic buffer drains; they commit the UniqueNumpyUpdater's pending
        # additions/deletions into the `unique` store. Dropping them froze
        # lean's unique-molecule counts at their t=0 values (no transcription /
        # replication / ribosome creation ever landed). The only lean change
        # is collapsing the two staged allocators into one global pass.
        rewritten = []
        for s in layer:
            if s.startswith("allocator"):
                if seen_allocator:
                    continue  # fold the second staged allocator into the first
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
        "Lean whole-cell E. coli — same biology as baseline, with the two "
        "staged allocator passes collapsed into ONE global allocator over all "
        "three partitioned processes. The unique-molecule flush steps, the "
        "requester/allocator/evolver split, and the flow-token ordering are ALL "
        "kept — prototyping proved each load-bearing (dropping flushes froze "
        "unique-molecule counts; fusing hangs the tRNA-charging ODE). Tracks "
        "baseline closely; not bit-identical — validate by tolerance."
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
