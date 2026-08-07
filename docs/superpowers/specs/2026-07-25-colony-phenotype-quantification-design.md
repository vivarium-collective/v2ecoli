# Colony & Microfluidic Phenotype Quantification — Design

**Date:** 2026-07-25
**Status:** Approved (brainstorming), pending spec review
**Investigation:** `colonies` (pivot from "HPC Scaling Readiness")

## Summary

Pivot the `colonies` investigation from *HPC scaling readiness* to *colony &
microfluidic phenotype quantification*. The goal is a benchmark that runs the
**same physical device geometries** with **three tiers of cell model** and
extracts a **common phenotype panel**, so cell models can be compared
apples-to-apples against each other and (eventually) against real
mother-machine data.

The organizing idea is a matrix:

|                    | simple agent | surrogate (growth-only) | full WCM (`EcoliWCM`) |
|--------------------|:---:|:---:|:---:|
| **mother machine** | ✓ | ✓ | ✓ |
| **daughter machine** | ✓ | ✓ | ✓ |
| **free colony**    | ✓ | ✓ | ✓ (across media) |

Studies 01–03 of the current investigation are **retained** as *Part A: Compute
Foundation* (the composite works; per-cell wall is flat; the native RSS leak
bounds run length). The new work is *Part B: Phenotype Quantification*.

## Motivation

The existing `colonies` investigation established that the colony composite is
single-machine-correct (growth + natural division) but that a native/C RSS leak
blocks unbounded HPC scaling. Rather than remain blocked on the leak, we pivot
to the scientific payload the composite was built for: **quantifying colony /
microfluidic phenotypes**. This reframing:

- turns the RSS-leak constraint into a stated caveat (short-window WCM runs)
  rather than a hard blocker;
- reuses cheap cell models (simple agents, a growth surrogate) to validate the
  measurement pipeline before spending WCM compute;
- sets up a real-data comparison (mother-machine adder statistics) as the
  eventual ground truth.

## Key decisions (from brainstorming)

1. **Pivot the existing `colonies` investigation** (not a new one; not nested).
   Keep studies 01–03 as backstory; add Part B on top.
2. **Growth-only surrogate tier.** Wrap the existing linear growth emulator from
   the (complete) `surrogate-modeling` investigation as a process-bigraph
   `SurrogateProcess` exposing the same `mass`/`length`/`volume`/`agents`
   interface as `EcoliWCM`. It participates in growth/size/division-timing
   quantification only — **not** exchange/media, which the surrogate
   investigation proved unlearnable from the observable view.
3. **Real mother-machine data comparison is PLANNED, pending data.** Define the
   loader interface + phenotype contract now; wire a real dataset later.
4. **Condition axes = media/nutrients × device geometry.** Media {minimal
   glucose, minimal+AA, rich} exercises the WCM's exchange/media response;
   geometry {mother machine, daughter machine, free colony} tests how physical
   confinement shapes size/division statistics.

## Architecture

### Core factorization: geometry × cell-model tier

Today geometry and cell model are tangled: the viva-munk device documents
(`viva_munk/experiments/documents/mother_machine.py`,
`daughter_machine.py`) hardcode `grow_divide` simple agents, and
`v2ecoli/colony.py` hardcodes `EcoliWCM`. To run any tier in any geometry we
factor them apart (**Approach A**, chosen over duplicating 9 builders):

- **`cell_factory(tier, ...)`** — returns a cell-body dict for a chosen tier
  with a **uniform port contract** so geometries and the extractor are
  tier-agnostic. Tiers:
  - `simple` — viva-munk `grow_divide` / `add_adder_grow_divide_to_agents`
    (adder or exponential growth; `mass`/`length` deltas; threshold division).
  - `surrogate` — new `SurrogateProcess` wrapping the linear growth emulator;
    same outward interface as `wcm` for the growth/size subset.
  - `wcm` — existing `EcoliWCM` bridge (full 55-process cell).
- **Geometry builders** refactored to place N cells via an injected factory:
  - `mother_machine(cell_factory, ...)` — dead-end channels + flow (from
    viva-munk `mother_machine_document`, generalized).
  - `daughter_machine(cell_factory, ...)` — chamber + absorbing wall (from
    viva-munk `daughter_machine_document`, generalized).
  - `free_colony(cell_factory, ...)` — unconfined pymunk box (generalize
    `v2ecoli/colony.py::make_colony_document`).

### Uniform cell-body port contract

Every tier's cell body exposes the ports the geometry (`PymunkProcess`) and the
extractor rely on:

- `mass` (fg), `length` (µm), `volume` (fL) — physical state driving the pymunk
  body and read by the extractor.
- `location`, `angle`, `id` — placement / lineage identity.
- `agents` wire (`['..','..','cells']`) — where a division update writes
  `{_remove, _add}`, matching what `EcoliWCM._handle_division` and viva-munk
  `grow_divide` already produce.
- `exchange` — WCM only; other tiers omit it and the extractor treats it as
  absent for those tiers.

### `SurrogateProcess`

New process wrapping the linear growth emulator (`surrogate-modeling`
deliverable). Inputs: current growth-relevant state. Outputs: `mass`/`length`/
`volume` deltas + division `agents` update at threshold, mirroring the WCM
subset. It does **not** emit `exchange`. Trained-artifact path pinned during
the plan phase.

### `phenotype_extractor`

The reusable analysis unit — tier-agnostic, reads one emitted run shape and
computes the panel:

- **Lineage reconstruction** — track cell-ID appearance/disappearance across
  emitted ticks to build the mother→daughter tree and detect division events.
- **Growth rate** — per-cell mass/length exponential fit within a cell cycle.
- **Size distribution at division** — length/mass at each division event.
- **Added length (adder plot)** — Δlength per cycle vs birth length.
- **Time between daughter divisions** — inter-division interval per lineage.
- **Exchange fluxes / media response** — WCM only; per-condition summaries.

Extractor output is a structured per-run phenotype record consumable by study
report cards and, later, the real-data comparison.

## Studies (Part B)

- **colonies-04 — Device harness + simple-agent baseline** *(buildable now)*.
  Bring mother & daughter machines in from viva-munk; run with simple agents;
  stand up `cell_factory` + geometry builders + `phenotype_extractor`; produce
  the full phenotype panel cheaply. Validates the measurement pipeline before
  expensive cells.
- **colonies-05 — Surrogate agent tier** *(buildable)*. Implement
  `SurrogateProcess`; run it in all geometries; compare growth/size/
  division-timing against the simple-agent baseline. Scope caveat: no
  exchange/media.
- **colonies-06 — v2ecoli WCM across media × geometry** *(buildable,
  compute-bounded)*. Full `EcoliWCM` colonies across media {minimal glucose,
  +AA, rich} × geometry {mother machine, daughter machine, free colony};
  quantify the full panel incl. exchange & media response. Run length bounded
  by the Part-A native RSS leak (short / few-generation windows) — a stated
  caveat, not a blocker.
- **colonies-07 — Real mother-machine data comparison** *(PLANNED, pending
  data)*. Loader interface + phenotype contract; compare simulated vs
  experimental adder statistics (size-at-division, added-length, inter-division
  time). Buildable once a dataset is provided.

## Data flow

```
cell_factory(tier) ──┐
                     ├─► geometry builder (mother/daughter/free) ─► Composite
media/condition ─────┘                                                │
                                                                      ▼
                                                          run (bounded window)
                                                                      │
                                                          emitted cell trajectory
                                                                      ▼
                                                      phenotype_extractor
                                                                      │
                                    ┌─────────────────────────────────┤
                                    ▼                                 ▼
                          study report cards            colonies-07 real-data compare
```

## Testing

- **Unit — `cell_factory`:** each tier yields a runnable 1-cell document with
  the required ports.
- **Unit — `phenotype_extractor`:** synthetic lineage with known division
  events → asserted growth rate, size-at-division, added-length, inter-division
  time.
- **Smoke — geometries:** fast per-geometry run with simple agents in CI
  (cheap, deterministic).
- **WCM runs:** excluded from CI (too heavy); gated behind a pytest marker and
  run on the mini.

## Scope / caveats

- Surrogate tier is growth/size/division-timing only; exchange & media
  phenotypes come exclusively from the WCM tier.
- WCM colony run length is bounded by the unresolved native RSS leak (Part A);
  colonies-06 uses short windows and states this.
- colonies-07 is specified but not built until a real dataset is provided.
- Single-machine, small-N measurements; cross-node HPC remains out of scope
  (was the old colonies-04, now deprioritized by the pivot).

## Non-goals

- Fixing the native RSS leak (Part A's open blocker) — treated as a bound, not
  a task here.
- Re-opening the surrogate investigation's negative result to build a richer
  (exchange-capable) surrogate.
- Cross-node / HPC deployment.
