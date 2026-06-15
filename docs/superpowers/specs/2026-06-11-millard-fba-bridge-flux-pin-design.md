# Millard FBA-bridge: kinetic-constrained (flux-pin) integration — design

**Date:** 2026-06-11
**Study:** pdmp-01-metabolism-ode (v2ecoli-pdmp investigation)
**Status:** design approved; ready for implementation plan

## Goal

Integrate the Millard 2017 kinetic ODE of central-carbon metabolism into the
v2ecoli WCM so the **kinetic model mechanistically determines central-carbon
flux**, while v2ecoli's multi-objective FBA (`modular_fba`) solves the rest of
the network (biomass, peripheral metabolism, homeostatic targets, NGAM). The
seam is the **FBA bridge**, extended from a namespace translator into a
**flux-pinning** coupling.

## Current state

`v2ecoli/steps/fba_bridge.py` today is a **concentration translator only**: it
maps Millard metabolite concentrations (mM) ↔ v2ecoli bulk counts via
`v2ecoli/data/millard_v2ecoli_species_map.yaml` and
`count = conc_mM · 1e-3 · V_cell_L · N_avogadro`, logging diagnostics. 8 unit
tests pass. Two things are explicitly deferred in its docstring and are exactly
this project's scope:
1. coupling to the WCM `Metabolism` process, and
2. flux-based coupling (the bridge does not yet route any flux).

## Chosen approach (decisions locked)

- **Coupling semantics:** flux-pin (kinetic-constrained FBA). Each tick the ODE
  exports central-carbon reaction fluxes; the bridge pins the mapped v2ecoli FBA
  reactions to those fluxes (`v_lb = v_ub = v_ODE`) before the LP solve.
- **Infeasibility handling:** soft-target fallback. If the hard pins make the LP
  infeasible, the offending reaction(s) are relaxed from hard bounds to soft
  kinetic-targets (the existing `USE_KINETICS` objective) so the LP always
  solves; the relaxed reactions and their flux residuals are logged.
- **Sequencing:** validate on a focused Metabolism+bridge harness first, then
  integrate into the full WCM.

## Why this is tractable: the pin mechanism already exists

`v2ecoli/processes/metabolism.py` already pins reactions exactly this way for
NGAM and translation-GTP:

```python
self.fba.setReactionFluxBounds(reactionID, lowerBounds=flux, upperBounds=flux)
```

and exposes `set_reaction_bounds(...)`, `set_reaction_targets(...)`, and
`getKineticTargetFluxNames()`. `USE_KINETICS = True` already enables a
kinetic-target objective. Flux-pinning the Millard reactions reuses this
established machinery rather than modifying the LP formulation.

## Components (each independently testable)

### 1. Reaction map — `v2ecoli/data/millard_v2ecoli_reaction_map.yaml` (NEW)
Maps each Millard reaction id (e.g. `PTS_4`, `PFK`, `PYK`, `PGI`, `PGK`, `ENO`,
`PDH`, `CS`, …) to one or more v2ecoli FBA reaction ids (BioCyc), with a sign
convention and an optional stoichiometric scale factor when a Millard lumped
reaction maps to several FBA reactions. Curated for the ~20–30 central-carbon
reactions where Millard and the v2ecoli FBA network overlap. Reactions with no
clean v2ecoli counterpart are listed under `millard_only` and not pinned.

Interface: `load_reaction_map(path) -> {millard_rxn: [(fba_rxn_id, scale)]}`.

### 2. Millard flux exporter (extend `millard_pdmp_metabolism.py`)
The COPASI/basico process currently emits only concentrations. Add a
`reaction_fluxes` output: after each integration step, read per-reaction fluxes
from COPASI (basico exposes reaction rates / `get_reaction_fluxes`) and emit them
in a `central_fluxes` store keyed by Millard reaction id, in mM/s.

Interface: new output port `central_fluxes: {millard_rxn -> float(mM/s)}`.

### 3. Flux unit converter (in the bridge)
Convert Millard reaction flux (mM/s) to the v2ecoli FBA bound basis. The WCM
sets bounds in `CONC_UNITS` magnitude using a `coefficient` that converts
`mmol/gDCW/hr → mM basis` (see `set_reaction_bounds`); the bridge mirrors that
conversion so pinned bounds are dimensionally consistent with the FBA's other
bounds. Conversion is pure and unit-tested against hand-worked examples.

Interface: `millard_flux_to_fba_bound(flux_mM_per_s, coefficient) -> float`.

### 4. Flux-coupling step (extend `fba_bridge.py` or a sibling `FBAFluxCoupler`)
Each tick, reads `central_fluxes`, applies the reaction map + unit conversion,
and writes a `pinned_flux_targets` store: `{fba_rxn_id -> flux_bound}`. Keeps
the existing concentration translation. Writes diagnostics (which reactions
pinned, converted values).

Interface: input `central_fluxes`, output `pinned_flux_targets` + `bridge_diagnostics`.

### 5. Metabolism consumption + infeasibility handler (extend `Metabolism`)
Before the LP solve, `Metabolism` reads `pinned_flux_targets` and calls
`setReactionFluxBounds(rxn, lb=v, ub=v)` for each. It then solves; on
GLP_NOFEAS it relaxes the most-recently-pinned reactions to soft kinetic-targets
(via `set_reaction_targets` / the kinetic objective) one batch at a time until
feasible, recording the relaxed set and residuals in
`listeners.fba_bridge.relaxed_reactions`.

## Data flow (per WCM tick)
1. Millard COPASI process advances the ODE for the tick, emits
   `central_metabolites` (mM, existing) **and** `central_fluxes` (mM/s, new).
2. Flux-coupling step maps + converts → `pinned_flux_targets` (FBA-unit bounds).
3. `Metabolism` (runs last, deriver-style) pins the mapped reactions, solves the
   LP, relaxes-to-soft-target any infeasible pins, applies the resulting flux →
   bulk count deltas as today.
4. Bridge concentration translation keeps the shared central-metabolite pools
   consistent between namespaces (existing behavior).

## v1 validation harness (sequencing milestone 1)
A focused composite: Millard COPASI process + flux-coupling bridge + `Metabolism`
(with the WCM context — cache/bulk/unique/boundary loaded, but NOT the full
55-process composite). Drive M9-glucose. Acceptance:
- **Pin fidelity:** for pinned reactions, the FBA's realized flux equals the
  ODE flux within solver tolerance (or the reaction is logged as relaxed).
- **Feasibility:** the LP solves every tick (hard or via soft fallback); the
  fraction of ticks needing relaxation and which reactions are reported.
- **Viability:** central-metabolite concentrations and growth stay biologically
  plausible (no NaN/zero-pool collapse) over ≥600 s.
- **Provenance:** a committed diagnostics figure (pin map, per-reaction
  ODE-vs-FBA flux, relaxation rate).

## Milestones
1. Reaction map + Millard flux exporter + flux converter (each unit-tested).
2. Flux-coupling step + Metabolism consumption + soft-target fallback.
3. v1 harness + validation on M9-glucose (acceptance above).
4. Drop into the full WCM composite; re-validate viability + central fluxes.
5. (Stretch) 3-condition interface validation vs the Phase-0 reference (the
   pdmp-01 gate), and the causal/teleonomic parameter partition for every
   pinned reaction.

## Out of scope (this spec)
- Replacing the `consumption_matched` ref-growth driver (separate mechanism;
  the bridge is the principled alternative but the driver stays until the bridge
  is validated in the full WCM).
- LQR control of the kinetic parameters (separate pdmp-01 thread; already fixed
  the LQR degeneracy).
- Performance/compilation (Phase 4).

## Risks
- **Reaction-network mismatch:** Millard's lumped reactions may not map cleanly
  to v2ecoli's BioCyc reactions → some reactions unpinnable (listed
  `millard_only`); acceptable, the overlap core is what matters.
- **High relaxation rate:** if many pins are infeasible, the mechanistic claim
  weakens. The harness measures this explicitly before the full WCM.
- **Flux sign/stoichiometry conventions:** Millard vs BioCyc directionality must
  be reconciled in the map (captured by the per-mapping sign/scale).
