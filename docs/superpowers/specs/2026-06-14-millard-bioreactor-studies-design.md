# Design — Millard kinetic metabolism in the bioreactor + WCM mass-conservation gate

**Investigation:** multiscale-bioprocess (v2ecoli ↔ bioreactor coupling, PR #69)
**Date:** 2026-06-14
**Status:** design (awaiting review → writing-plans)

## 1. Motivation

The multiscale-bioprocess investigation couples v2ecoli's whole-cell model (WCM)
to the BiRD bioreactor and compares it to the Beulig 2025 (mSystems; Palsson lab)
high-density batch run. mbp-05's current finding is that the WCM's FBA metabolism,
under reactor conditions, tracks the published batch trajectories only coarsely.

Open question: **does a kinetic central-carbon metabolism adapt to the bioreactor's
time-varying environment (glucose depletion, dissolved-O₂ limitation) better than
the WCM's constraint-based kFBA?** This design adds the Millard 2017 kinetic ODE as
the central-carbon engine *inside* the WCM (growth still comes from the WCM), couples
that variant to the same reactor, and runs it head-to-head against the plain WCM.

It also adds a reusable **WCM mass-conservation gate** — the metabolism swap is
exactly where mass can be silently created/destroyed, and the user wants
"no mass created/destroyed in the whole-cell model" as a general investigation
invariant.

## 2. Architecture

### 2.1 Millard-as-central-carbon (the engine swap)

Starting point: `v2ecoli/composites/millard_pdmp_baseline.py` already removes the
`ecoli-metabolism` Step and slots the Millard 2017 ODE (`CopasiUTCProcess`) +
`FBABridge` into the same execution-layer slot across all ~54 WCM processes. Three
gaps to close, all documented in that composite's own docstring as out-of-scope for
its "runs end-to-end" milestone:

1. **Real bulk writeback.** Today `FBABridge` writes to a *parallel plain-dict*
   store (`central_metabolite_counts`), so downstream WCM processes see unchanged
   `bulk` — metabolism does not drive growth. Fix: write deltas into the real
   structured `bulk` array via the already-stubbed `millard-bulk-indexer` Step,
   direction `bidirectional` with v2ecoli authoritative. The species map
   (`v2ecoli/data/millard_v2ecoli_species_map.yaml`, 21 shared + 9 millard-only)
   defines the translated pool. This is what makes "growth from the WCM, fed by
   Millard's kinetics" real.
2. **Environment responsiveness.** Feed the cell's external glucose + dissolved O₂
   into the Millard ODE each tick. `CopasiUTCProcess.update` already overwrites
   COPASI state from its `species_concentrations` / `species_counts` inputs (via
   `_set_initial_concentrations` → `setInitialConcentration`, then
   `run_time_course(update_model=True)`). So responsiveness is wiring the cell's
   external concentrations (by SBML ID) into the Millard process's input port —
   **only the boundary species** (glucose, O₂), not the full pool (feeding the whole
   reconciled pool was the cause of the original `inputs: {}` NaN avoidance).
3. **Drop the LQR controller.** `millard_pdmp_baseline` carries a pdmp-specific
   `LQRControllerMultiState` modulating `PTS_4.kF` toward a setpoint. For bioreactor
   *adaptation* we want environment-driven kinetics, not setpoint control. The
   Millard study uses the metabolism swap **without** the LQR.

Deliverable: a `baseline_millard` composite (WCM with kinetic central-carbon
metabolism), parallel to `baseline.py`, that emits the same cell-side contract
surface the plain WCM does — `listeners.fba_results.external_exchange_fluxes` in
mmol/(gDW·h) (`v2ecoli/processes/metabolism.py` is the reference; the Millard variant
must publish the same port so it is a drop-in under the reactor coupler).

### 2.2 Reactor coupling

`reactor_bird_coupled` does **not exist yet** — it is mbp-03's deliverable (the TODO
at `mbp-03-bird-reactor-coupling/study.yaml`), now unblocked by the upstream
`pbg-bioreactor-transport-fork` merge (pbg-bioreactordesign `f40f82f`, 43 tests
green). The Millard apply/compare study (mbp-09) consumes that composite and swaps
the cell engine `baseline` → `baseline_millard`, demonstrating exactly the
substitutability the `cell_side_interface_contract` promises.

**Known upstream caveat:** pbg-bioreactordesign#2 — dissolved-O₂ saturation has the
wrong temperature sign (masked at 298 K; mbp-03/04/09 target 310 K → c*(O₂) ~25%
high). Surfaced as a divergence input to mbp-06, not fixed here.

## 3. Study decomposition (the Millard sub-arc)

Three new studies, ordered via `pipeline_gate.prerequisites` (NOT by renumbering
mbp-01..06 — renaming existing studies is the FK rename-drift hazard Chris flagged
and mbp-06 already logs as `study-yaml-rename-walks-internal-fks`). The DAG is
computed from prerequisites, so the narrative reads correctly without renames.

| Study | Phase | Scope | Prerequisites |
|---|---|---|---|
| `mbp-07-millard-kinetic-metabolism-swap` | Build | `baseline_millard` composite: real bulk writeback, env-responsive Millard input, LQR removed, growth from WCM | (none new) |
| `mbp-08-millard-swap-validation` | Build/test | WCM mass-conservation gate, growth-recovered-vs-baseline, kinetic responsiveness (uptake falls as glucose depletes), bridge round-trip fidelity | mbp-07 |
| `mbp-09-millard-reactor-comparison` | Evaluate | Couple `baseline_millard` to `reactor_bird_coupled`; head-to-head vs plain WCM against Beulig batch; report-card with categorized divergences (mirrors mbp-05) | mbp-03, mbp-08 |

**Resulting narrative arc** (via prerequisites + `at_a_glance` order):
v2ecoli enhancements (mbp-01 env, mbp-02 population, **mbp-07 kinetic metabolism,
mbp-08 validation**) → reactor coupling (mbp-03, mbp-04) → comparisons (mbp-05
WCM↔Beulig, **mbp-09 Millard↔WCM↔Beulig**) → gap synthesis (mbp-06).

## 4. Mass-conservation gate — a native behavior test (no custom evaluator)

### 4.1 Evaluator architecture (clarification)

Study tests are already evaluated through one unified loop
(`pbg_superpowers.study_evaluator.evaluate_study` → `evaluate_test`), dispatching on
`measure.kind` in a single resolution order:

```
native run-data kind (built-in DSL)  →  workspace-registered evaluator (by kind)  →  agent bucket
```

- **Native DSL kinds** (`derived`, `generation_average`, `range_check_per_generation`,
  …) are code-evaluated from `measure.formula/path` + `pass_if.op` (`in_range`,
  `<=`, `cv_below`, …). The `formula` is an expression over emitted-observable tokens.
- **Registered evaluators** (`pbg_v2ecoli/evaluators.py` `register_evaluators`) are
  the escape hatch for checks that need data/logic outside run-series — e.g.
  `report_card_axis` reads an external `report_card_verdict.json`.
- **Anything else falls to the agent bucket** (not auto-gated).

They are therefore already unified at dispatch; no merge work is needed. The real
cleanup is that some *authored* kinds aren't in the native set and silently degrade
to the agent bucket (see §4.3).

### 4.2 Mass conservation as a native test

The closed-balance definition (chosen): over the run, the change in total cell dry
mass equals the net of tracked import/export exchange (within tolerance) — caught as
a cumulative ratio reduced by `in_range`:

```yaml
- name: wcm-mass-conservation-closes
  measure:
    kind: derived
    formula: "cumulative_cell_mass_delta_fg / cumulative_net_exchange_mass_fg"
    window: full_lineage_from_gen_0
  pass_if: {op: in_range, low: 0.99, high: 1.01}
  cites: [agmon2022]
```

This is a **regular behavior test** — same list, same loop, renders in the study
report one row after the others (each tagged `evaluated_by`). No `register_evaluators`
entry. A custom evaluator would only be warranted later if we need a reduction the
DSL lacks (a true per-step time-integral, a worst-case `max|residual|`, or per-element
C/N stoichiometric balance) — a deliberate, separate decision.

**Requirement:** the formula tokens must resolve to *emitted observables*. If the WCM
does not already emit a cell-mass-delta and net-exchange-mass series, add a small
mass-balance listener that emits the cumulative numerator/denominator (or the per-step
residual). This listener is the only new code the gate needs.

Homes: primary in **mbp-08**; also added to **mbp-04** (multigen WCM run) so the
invariant guards the plain WCM, not just the Millard swap.

### 4.3 Bonus fix

mbp-01's existing `cumulative-mass-balance-closes` uses
`measure.kind: derived_ratio` with `numerator`/`denominator` — `derived_ratio` is
NOT a native kind and is not registered, so it currently degrades to the agent
bucket (not auto-gated). Re-author it to the native `derived` + `formula` shape so it
actually code-evaluates, reusing the same residual observables.

## 5. Components & files

- `v2ecoli/composites/baseline_millard.py` (new) — WCM + Millard central-carbon swap.
  Derived from `millard_pdmp_baseline.py` minus LQR, with real bulk writeback.
- `v2ecoli/steps/fba_bridge.py` (edit) — enable bidirectional v2ecoli-authoritative
  writeback into structured `bulk` (via `millard-bulk-indexer`).
- Millard env input wiring — map cell external glucose/O₂ → Millard `species_concentrations`.
- WCM mass-balance listener (new, small) — emits cumulative cell-mass-delta +
  net-exchange-mass (or per-step residual) for the native gate.
- `reactor_bird_coupled` (mbp-03 deliverable; consumed, not built here) + a
  Millard-cell variant wiring for mbp-09.
- New study dirs: `mbp-07-…`, `mbp-08-…`, `mbp-09-…` with `study.yaml`.
- Re-author mbp-01 `cumulative-mass-balance-closes` to native `derived`.

## 6. Testing strategy

- **mbp-07:** composite builds + runs N steps; `external_exchange_fluxes` populated;
  growth (cell_mass) increases; Millard fluxes non-trivial. Plumbing/mechanism only.
- **mbp-08:** `wcm-mass-conservation-closes` (native, in_range 0.99–1.01);
  growth-rate within tolerance of `baseline`; uptake falls monotonically as external
  glucose depletes (kinetic responsiveness); bridge mM↔count round-trip residual
  small. All native code-evaluated where possible.
- **mbp-09:** report-card overlays vs Beulig batch (glucose, biomass, acetate, dO₂);
  divergences categorized (report_card_axis); explicitly execute-and-report (no
  tuning), per the §10 reframe.

## 7. Risks / open items

- **COPASI continuity:** `run_time_course(start_time=0.0, update_model=True)` restarts
  from current initial values each tick; overwriting only boundary species should
  preserve internal-metabolite continuity, but verify no integrator reset artifacts
  across coupling ticks.
- **Bridge mass closure:** the 21-species shared pool is a subset of central carbon;
  the mass-conservation gate must account for un-bridged flux (or scope the residual
  to the bridged pool) to avoid a false mass-balance failure.
- **Yield/growth coupling:** growth comes from the WCM consuming the Millard-fed bulk;
  confirm the bridged pool actually drives the biomass-producing processes.
- **O₂ temp-sign bug (pbg-bioreactordesign#2):** biases mbp-09 dO₂ at 310 K; report as
  a divergence, do not fix here.

## 8. Out of scope

- Fed-batch operations (deferred upstream; mbp-06 gap).
- Tuning Millard parameters to match Beulig (evaluation phase = execute-and-report).
- A both-daughters population runner (separate mbp-06 candidate-future-study).
- Per-element atomic mass balance (possible future custom evaluator).
