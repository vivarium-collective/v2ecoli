# Biological Composite — design

**Date:** 2026-06-13
**Branch:** `feat/biological-composite`
**Status:** approved design, Phase 1 to be implemented

## Goal

Build a new E. coli whole-cell composite for v2ecoli whose **store hierarchy is
organized by biology** (cellular compartments → molecular classes) rather than by
*representation* (`bulk` / `unique` / `listeners`). Reuse the existing process
implementations wherever possible, make the wiring cleaner and the state tree
interpretable, and **prove equivalence against the baseline composite**.

A second axis — hardening units and schemas so the model is more biological,
interpretable, and compositional — is sequenced as Phase 2, after Phase 1's
re-wiring is proven equivalent.

## Core insight

Today's top-level stores (`bulk`, `unique`, `listeners`, plus coordination stores)
are organized by *how state is represented and applied*, not by *what it is
biologically*. process-bigraph cleanly separates **process update math** from
**store routing**: every process declares ports and a module-level `TOPOLOGY`
(port → store-path), and the baseline builder funnels all wiring through a single
choke point (`make_edge`).

Therefore the reorganization can be a **pure relabel of store paths**, not a
rewrite of any process. If no store's internals are split and no update math
changes, the simulated trajectory is **provably bit-identical** to the baseline.
This is what makes a strong equivalence claim possible and is the backbone of
Phase 1.

## Two-phase plan

- **Phase 1 — re-wire (bit-identical).** Reorganize store *paths* into a biological
  hierarchy using a uniform path-remap. Leaf types are unchanged. Target: trajectories
  numerically identical to baseline. **This is what we plan and build now.**
- **Phase 2 — harden units & schemas (statistical equivalence).** Split the
  monolithic pools into biological sub-pools, add unit-bearing / self-documenting
  schemas, distribute observables into their subsystems. Target: biological markers
  within tolerance of baseline. **Documented as a sketch; gated behind Phase 1.**

Phasing isolates "did the reorganization change anything" (Phase 1, exact) from
"did units hardening change anything" (Phase 2, tolerant) — each is independently
debuggable.

## Architecture

### Phase 1 mechanism — the path-remap layer

A new builder `biological()` reuses baseline's exact process *instances* and applies
one `REMAP` (old store-path → new biological path) uniformly to three things:

1. the pre-created store tree,
2. the seeded initial state (from the ParCa cache bundle),
3. every edge's input/output topology (the port → store-path mappings).

Because process instances and update math are untouched and only routing labels
change, the trajectory is identical to baseline.

### Re-pathing granularity (an honest constraint)

What is safely re-pathable in Phase 1 depends on how each store is structured:

- **`unique.*`** — although these are per-type leaf arrays (`unique/active_RNAP`, …),
  **they cannot be split in Phase 1.** Implementation discovered three consumers
  (`division` and the two mass listeners) wire the *whole* `unique` map through a
  single `InPlaceDict` / `map[node]` port and index molecules internally
  (`division.py` reads `states['unique']` wholesale). Relocating any molecule out of
  a shared `unique` store hands those consumers a partial map → divergence. So
  `unique` **stays one store, relocated whole** to `cell/unique_molecules` in Phase 1
  — exactly the constraint that applies to `bulk`. The per-molecule split across
  `chromosome` / `transcription` / `translation` (the rename targets are kept in
  `UNIQUE_REMAP_PHASE2`) is **Phase 2**, gated behind the same aggregator-view shim
  as the `bulk` split.
- **`bulk`** — a single performance-critical monolithic structured array; every
  process does global index lookups against it through one port. It **stays one
  physical store** in Phase 1 (relocated whole), plus a biological *manifest*
  (metadata grouping molecule names → metabolites / proteins / energy / bulk-RNA)
  for interpretability. Physically splitting it is **Phase 2** (needs an index-remap
  shim).
- **`listeners`** — a single deep-merge (`DerivedProperties`) store; processes mostly
  target it whole or via leaf sub-paths. **Relocated whole** in Phase 1; distributed
  into subsystems in Phase 2.
- **coordination stores** (`process`, `allocator_rng`, `process_state`,
  `next_update_time`, `request`, `allocate`) and **clock stores** (`global_time`,
  `timestep`, `divide`, `division_threshold`) — relocated whole under `machinery/`
  and `clock/` so the biological view is not cluttered by plumbing.

### Target hierarchy (Phase 1)

As-built Phase 1 (bit-identical to baseline):

```
cell/
  molecules/           ← bulk (whole monolith)                              [split in Phase 2]
  unique_molecules/    ← unique (whole — all 11 molecule types together)    [split in Phase 2]
  observables/         ← listeners (whole)                                  [distributed in Phase 2]
  regulation/          ← ppgpp_state, attenuation_config
environment/           ← boundary/external, media metadata, exchange
machinery/             ← process, allocator_rng, process_state, next_update_time,
                              request, allocate
clock/                 ← global_time, timestep, divide, division_threshold
```

`machinery/` and `clock/` deliberately pull bookkeeping out of `cell/` so the
biological subtree reads as biology, not plumbing.

The richer compartment layout below is the **Phase-2 target** — it requires the
aggregator-view shim (so whole-map consumers still see all of `unique` / `bulk`
while readers see split compartments), which trades bit-identity for statistical
equivalence:

```
cell/
  chromosome/          ← unique: full_chromosome, chromosome_domain, oriC,
                              DnaA_box, chromosomal_segment, gene, active_replisome
  transcription/       ← unique: active_RNAP→rna_polymerases, RNA→transcripts,
                              promoter→promoters
  translation/         ← unique: active_ribosome→ribosomes
  metabolism/          ← bulk: metabolites, energy carriers
  proteins/            ← bulk: monomers, complexes
membrane/              ← placeholder seam (ties into the autopoiesis membrane v2ecoli lacks)
```

`membrane/` is an intentional Phase-2 seam for the autopoiesis-style self-produced
membrane.

## Equivalence harness

### Phase 1 — bit-identical assertion

A `REMAP`-aware diff runs `baseline()` and `biological()` from the same seed and
cache bundle and asserts the trajectories are identical after translating the
biological state back through `REMAP⁻¹`:

- **Structural:** the remapped biological state tree, projected back through
  `REMAP⁻¹`, is key-for-key equal to baseline's tree (catches any wiring miss).
- **Numerical:** per-step, every leaf array (`bulk` counts, each `unique` array,
  every listener leaf) is `np.array_equal` to baseline. A single mismatch fails
  loudly with the offending path.

This runs as a pytest gate and reuses `scripts/compare_harness.py` plumbing, with
the bar tightened from "within tolerance" to "identical."

### Phase 2 — biological-marker equivalence

Once units/schemas change, exact equality no longer holds, so the bar drops to
statistical bands on biological observables: growth-rate curve, division time,
dry-mass fractions (protein / RNA / DNA / water), replication-initiation timing,
final molecule composition. Reported side-by-side (baseline vs biological) in an
HTML comparison, in the style of the existing 3-way composite report.

## Phase 2 — units & schema hardening (sketch only)

Layered on the now-biological hierarchy:

- **Split `cell/molecules`** into `cell/metabolism/metabolites`,
  `cell/proteins/monomers`, `cell/proteins/complexes`, `cell/transcription/bulk_rna`,
  `cell/metabolism/energy`, via an index-remap shim so existing processes still see
  one logical pool. This change forces the statistical bar.
- **Unit-bearing leaf schemas** (`quantity[float, fg]`, `count`, `mM`, …) so plot
  axes auto-label and dimensional mismatches are catchable — building on the existing
  units-propagation work.
- **`describe()`-backed schemas** so each store is self-documenting.
- **Distribute `observables`** into subsystem homes (`fba_results`→metabolism,
  `rnap_data`→transcription, `ribosome_data`→translation,
  `replication_data`→chromosome, etc.).

Phase 2 stays a sketch here; scope is revisited only after Phase 1 passes.

## Deliverables (Phase 1)

- **Worktree + branch:** `feat/biological-composite` (isolated checkout).
- `v2ecoli/composites/biological.py` — `biological()` `@composite_generator` + the
  `REMAP` table.
- `v2ecoli/composites/_remap.py` — the reusable path-remap transform (store tree,
  initial state, topology) and its inverse `REMAP⁻¹`.
- `tests/test_biological_equivalence.py` — the bit-identical pytest gate.
- `scripts/compare_biological.py` — biological-marker comparison + HTML report
  (Phase 2-ready).
- This spec.

## Scope discipline

- Phase 1 only (re-wire + bit-identical proof) is planned and built now.
- Phase 2 (pool splitting + units) is documented but gated behind Phase 1 success.
- No process *internals* are modified in Phase 1 — only routing/topology and the
  store tree shape. Any change that would touch a process's update math belongs to
  Phase 2.

## Open questions / risks

- **`make_edge` choke point:** the remap assumes topology application is centralized
  through `make_edge` in `composites/baseline.py` / `composites/_helpers.py`. Confirm
  during planning that no process bypasses it with hard-coded absolute paths.
- **Listener leaf-targeting processes:** confirm processes that read listener
  sub-paths (e.g. metabolism reads `listeners/mass`) are remapped consistently when
  `listeners` relocates whole.
- **Type resolution dispatches:** the `_resolve_*` functions in `types/__init__.py`
  key on store semantics, not paths, so relocation should be transparent — verify no
  dispatch is path-sensitive.
