# colonies-02-parallel-multigen-perf

Redo the colony simulation from **one** whole-cell agent, let it divide
**naturally** (no forced division), keep both daughters, and follow the
colony as it doubles **1 → 2 → 4 → 8** — tracking the per-cell compute
hit generation-by-generation, sequentially and under the updated
**process-bigraph Ray protocol**.

## Why

colonies-01 PASSED but left two open threads this study closes:

- **F-06** — RSS climbed ~5 MB/sim-s in a long single-cell run, invisible
  in the static 60-tick N-sweep windows. Leak, or amortized startup?
- **PASS-narrow** — two-generation division was validated only by *forcing*
  both mothers to divide on the same tick, not a natural timeline.

It also folds in the `gil-aware-engine-research` follow-up: run the growing
colony under `ray:EcoliWCM` (sharded actor pool, `parallel_processes=True`)
to see whether process-level parallelism lifts the single-process ~13-cell
GIL ceiling colonies-01 measured.

## Build phase (must land before the long runs)

1. **jitter / mass diagnostic** — the colony.gif shows excessive movement.
   `colony.py` runs `jitter_per_second=0.5` (~5000× viva-munk's `1e-4`
   default), and the WCM `dry_mass` (~380 fg, raw) → `body.mass` coupling
   may not land correctly against `build_microbe`'s density-0.02 body
   (~0.056). Diagnose, then lower jitter and/or fix the mass→pymunk scale.
2. **Ray wiring** — switch cells (and the daughters added by
   `bridge._handle_division`) to the `ray:EcoliWCM` address with per-actor
   ParCa cache loading.

## Deliverable

A detailed compute-requirements profile (per-cell wall + RSS by generation,
total-wall-vs-realtime sequential vs Ray, cells/core at realtime) that seeds
**colonies-03-hpc-deployment**.

Status: **Design** — scaffolded, sims not yet built/run. Long runs target the
Mac mini (`mct`).
