# Consensus Elongation Model — Audit & Phase Plan

**Branch**: `consensus_elongation` (forked 2026-06-26 from `trna_charging_final` HEAD `5ffb76de`).
**Design spec**: `v2ecoli_consensus_model.md` (repo root).
**Status**: Phase 3 in progress. P3a (scaffold) + P3b-i (plumbing) + P3b-ii (ODE merge) landed. Binding ODE-merge proof passing.

This audit is a fresh investigation of the current branch state. A prior attempt on `consensus_elongation_model` (now deleted) reached "Phase 6 complete" — its findings inform but do not constrain this plan. Every claim below is grounded in current-state file reads.

**Revision 2 note**: Revision 1 of this audit framed the consensus as "port two ppGpp methods + fix one bug" and dismissed the spec's "merge two ODE systems" warning. That framing was wrong on a substantive mechanistic point — the AA synthesis/import/export terms are part of the SteadyState ODE's RHS and are absent from the kinetic ODE's RHS. The merge is real. This revision corrects the scope. See §2 for the side-by-side ODE diff.

---

## 1. The Three Components and How They Couple

The design spec asks for three behaviors to live on the kinetic-charging path: ppGpp regulation, AA synthesis/import/export, and the `aa_count_diff` feedback to metabolism. Each of these couples to the charging ODE differently, and the coupling pattern dictates how invasive the integration is.

| Component | Coupling to charging ODE | Implementation pattern in SteadyState | Status on kinetic class |
|---|---|---|---|
| Codon-aware reading kinetics | inside ODE | n/a (SteadyState doesn't have it) | ✅ already present |
| Charged/uncharged tRNA balance | inside ODE | inside `dcdt` (`kinetics.py:330-355`) | ✅ already present |
| AA pool depletion from charging | inside ODE | `daa = -v_charging + supply` | ✅ already present (depletion half) |
| **AA synthesis / import / export** | **inside ODE RHS** | `supply(aa_conc) → (v_syn, v_imp, v_exp)` closure evaluated at every RK45 step (`kinetics.py:345-353`); pool replenishment is part of `daa` | **❌ missing — see §2** |
| ppGpp synthesis / degradation | OUTSIDE ODE (single-shot) | `ppgpp_metabolite_changes(...)` at `kinetics.py:30`; called once per tick from `_ppgpp_request` (predict) and `_ppgpp_evolve` (realize) | **❌ missing — see §3** |
| ppGpp → elongation-rate inhibition | applied BEFORE the solve | sets `max_elong_rate` from `elong_rate_by_ppgpp(ppgpp_conc)` (`polypeptide_elongation.py:1000-1012`) | **❌ missing — see §3** |
| `aa_count_diff` → metabolism homeostatic FBA | computed AFTER the solve | `aa_supply − aa_used_trna` from accumulators (sign: positive = over-supplied → raise target) | **❌ broken — see §4** |

Two of these are real ODE-RHS extensions on the kinetic side (AA supply). Two are decoupled outer-loop hooks (ppGpp request/evolve, `aa_count_diff` write). One is a pre-solve scalar (ppGpp inhibition of elongation rate). The plan in §6 follows this stratification.

---

## 2. The ODE Merge: AA Supply Terms

### The diff that defines the work

**SteadyState ODE state vector** (`v2ecoli/processes/polypeptide/kinetics.py:330-355`):
```
[uncharged_trna, charged_trna, AA, total_synthesis, total_import, total_export]
```
RHS body (lines 345-355):
```python
v_synthesis, v_import, v_export = supply(unit_conversion * aa_conc)
v_supply = v_synthesis + v_import - v_export
daa[mask] = v_supply[mask] - v_charging
return np.hstack((-dtrna, dtrna, daa, v_synthesis, v_import, v_export))
```

**Kinetic ODE state vector** (`v2ecoli/processes/polypeptide/kinetic_charging.py:888-898`):
```
[free_trnas, charged_trnas, AA, charging_counter, reading_counter, codons_to_trnas_counter]
```
RHS body (line 891):
```python
dx_dt[self.slice_amino_acids] = -(self.trnas_to_amino_acids @ charging_rate)
```

The kinetic ODE has no synthesis/import/export terms in the AA balance, and no accumulators for them in the state vector. AAs are a one-way drain during the timestep.

### Why this matters mechanistically

As AA pools drain inside the kinetic timestep, the `adjusted_amino_acid_saturation` factor (`kinetic_charging.py:857`) throttles `charging_rate`. So the kinetic model already captures AA-pool-dependent charging — but only the *depletion* half. The kinetic model cannot capture rich-media replenishment happening on the same timescale as consumption, because supply doesn't enter the RHS. AAs always trend down inside a timestep; the only thing that resets them is the next tick's metabolism update arriving via the partition merge.

For the consensus to capture spec §4's "media richness sustains translation" (rich media keeps AA pools full → fast charging → high elongation), the kinetic ODE has to **see supply during the solve**, not just before/after.

### What needs to change

1. **Extend the kinetic ODE state vector** to include three accumulator slots: `total_synthesis`, `total_import`, `total_export`. These are zero at `t=0` and integrate the time-integrated flux of each over the timestep.
2. **Add slice constants** alongside `self.slice_free_trnas`, `self.slice_charged_trnas`, `self.slice_amino_acids`, etc. — `self.slice_total_synthesis`, `self.slice_total_import`, `self.slice_total_export`. Recompute `self.molecules_input_size`.
3. **Plumb a supply closure** into `ode_model` (line 812). The signature becomes either an additional positional arg in `args=...` or captured via the outer `run_model` closure. Source the closure from `get_charging_supply_function` (`kinetics.py:520-627`) — the same factory SteadyState uses.
4. **Pre-compute supply inputs before the solve**: `import_rates`, `export_rates`, `aa_supply`, `aa_supply_scaling`, etc. — mirror the setup at `polypeptide_elongation.py:1056-1095`. These feed `get_charging_supply_function`.
5. **Add supply terms to the AA balance** in the RHS:
   ```python
   v_synthesis, v_import, v_export = supply(counts_to_molar * amino_acids_remaining)
   v_supply_counts = (v_synthesis + v_import - v_export) / counts_to_molar  # μM → counts/s
   dx_dt[self.slice_amino_acids] = -(self.trnas_to_amino_acids @ charging_rate) + v_supply_counts
   dx_dt[self.slice_total_synthesis] = v_synthesis / counts_to_molar
   dx_dt[self.slice_total_import]    = v_import    / counts_to_molar
   dx_dt[self.slice_total_export]    = v_export    / counts_to_molar
   ```
   (Note the unit bridge — see Risk §8.)
6. **Post-solve, extract the accumulators** from `ode_result.y[-1]` and use them for `aa_supply`, `aa_synthesis`, `aa_exchange_rates` listeners. Today these listeners get empty arrays on the kinetic path.
7. **Compute `aa_count_diff`** = `total_synthesis + total_import − total_export − amino_acids_used` (matches SteadyState's `aa_supply − aa_used_trna` sign convention: positive = over-supplied → raise homeostatic target). This replaces the latent bug at line 775 (see §4).

### Unit bridge

The `supply` closure operates in μM (`MICROMOLAR_UNITS` magnitudes). The kinetic ODE operates in molecule counts. Two paths:

- **(A) Convert at every RK45 step**: inside `ode_model`, compute `aa_conc_uM = counts_to_molar * amino_acids_remaining * 1e6`, call `supply(aa_conc_uM)`, then convert flux back to counts/s. Costs one scalar mul + one scalar div per AA per step. RK45 with `rtol=1e-4` typically takes ~hundreds of steps per second of sim time, so this is cheap.
- **(B) Wrap supply at setup time** in a counts-domain closure that does the conversion once and is then called purely in counts. More state to thread but avoids the per-step bridge.

Recommendation: **(A) first**. Simpler, fewer indirection layers, easier to diff against SteadyState's reference behavior.

---

## 3. The ppGpp Wiring (Decoupled From the ODE)

ppGpp synthesis/degradation is NOT integrated — it's a single-shot delta computed by `ppgpp_metabolite_changes(...)` at `kinetics.py:30`. SteadyState calls it twice per tick:

- **Request side** (`_ppgpp_request` at `polypeptide_elongation.py:1481`): predict ppGpp turnover from current uncharged tRNA / RelA / SpoT / ppGpp concentrations, translate to bulk requests for the ppGpp-reaction stoichiometry (`request=True`, returns reactant-only deltas to avoid over-requesting).
- **Evolve side** (`_ppgpp_evolve` at line 1535): realize the synthesis/degradation events with the partitioned reactant counts as `limits` (`request=False`, returns net reactant+product deltas).

Both methods consume parameters already declared on `BasePolypeptideElongation` (lines 124-174: `KD_RelA`, `KI_SpoT`, `k_RelA`, `k_SpoT_syn`, `k_SpoT_deg`, `ppgpp_reaction_stoich`, etc.) — the parameter scaffold is inherited by the kinetic class for free. Only the wiring is missing.

Additionally, ppGpp affects elongation rate via `elong_rate_by_ppgpp(ppgpp_conc, basal_rate)` at `polypeptide_elongation.py:1000-1012`. This is applied BEFORE the ODE solve — it modifies `self.basal_elongation_rate` (or equivalent), which then becomes `target_codon_rate` on the kinetic side. Three things to wire:

1. **Pre-solve**: in the kinetic `request_state`, compute `ppgpp_conc` from bulk counts, then call `elong_rate_by_ppgpp(...)` if `self.ppgpp_regulation and not self.disable_ppgpp_elongation_inhibition`. Use the result to scale `target_codon_rate` passed to `solve_ivp` (line 925).
2. **Request side**: add a `_ppgpp_request(...)` call in `request` that appends bulk requests for the ppGpp reaction metabolites.
3. **Evolve side**: add a `_ppgpp_evolve(...)` call in `evolve_state` after the existing bulk deltas, with the partitioned reactant counts as `limits`. Emit `ppgpp_conc` and growth-limits listeners.

### Implementation options for porting the two methods

- **(A) Copy both methods** from `SteadyStatePolypeptideElongation` into `KineticTrnaChargingPolypeptideElongation`. Pros: zero impact on the SteadyState class, lowest blast radius. Cons: code duplication, parallel maintenance.
- **(B) Change inheritance**: `KineticTrnaChargingPolypeptideElongation(SteadyStatePolypeptideElongation)`. Inherit ppGpp + AA-supply machinery for free; override only the ODE/codon-tracking pieces. Pros: no duplication. Cons: requires auditing `SteadyState`'s init/setup so the kinetic class's overrides don't clash; some `SteadyState` init steps may be incompatible with kinetic state.
- **(C) Extract a `PpgppRegulationMixin`**: both classes inherit it. Pros: cleanest. Cons: most invasive — touches `SteadyState`.

Recommendation: **(A) first**, with (C) as a follow-up refactor after consensus parity gates pass. (B) is appealing but risks coupling the kinetic class to SteadyState's initialization order in ways that may not surface until generation 2+ in a sim.

---

## 4. The Latent `aa_count_diff` Bug

**Severity**: high — silently breaks metabolism's homeostatic feedback whenever `use_trna_charging=True`.

**Location**: `v2ecoli/processes/polypeptide/kinetic_charging.py:775`

```python
return net_charged, {}, update
```

The second tuple position is unpacked by the caller at line 634 as `aa_count_diff` and written to `update["polypeptide_elongation"]["aa_count_diff"]` at line 650 and to `update["listeners"]["growth_limits"]["aa_count_diff"]` at line 662. The schema (lines 314, 328, 354) declares this port as `array[float]`. Writing `{}` to an `array[float]` port either silently mismatches or errors at the partition merge.

**Consumer**: `v2ecoli/processes/metabolism.py:646` reads `states["polypeptide_elongation"]["aa_count_diff"]` to adjust homeostatic FBA targets when `use_trna_charging=True`.

**Reference for correct semantics**: `SteadyStatePolypeptideElongation` writes `aa_count_diff = aa_supply − aa_used_trna` (positive = over-supplied → raise homeostatic target). After §2's ODE merge lands, the kinetic class can compute this exactly: `total_synthesis + total_import − total_export − amino_acids_used`. So this bug-fix is **the tail of Phase 3**, not an independent phase — without the supply accumulators existing, there's no meaningful supply value to subtract.

If for some reason §2's ODE merge needs to be split out, an interim fix is `return net_charged, np.zeros(n_amino_acids, dtype=np.float64), update`. That gets metabolism a shape-correct (but content-empty) array, unbreaking the silent partition mismatch but providing no real feedback signal. Not a substitute for the real fix.

---

## 5. File-by-File Map

| File | Role | Consensus delta |
|---|---|---|
| `v2ecoli/processes/polypeptide/kinetic_charging.py` (1370 LOC) | `KineticTrnaChargingPolypeptideElongation` class | **Primary edit target.** Extend ODE state + RHS (P3, §2). Add `_ppgpp_request` / `_ppgpp_evolve` + pre-solve ppGpp wiring (P2, §3). |
| `v2ecoli/processes/polypeptide/kinetic_charging_kernel.py` (648 LOC) | Reconciliation kernel (codon-position binary search) | **No edits expected.** This is the reconcile-via-positions / reconcile-via-pools code, not an ODE. |
| `v2ecoli/processes/polypeptide/kinetics.py` (627 LOC) | Shared math: `dcdt` builder, `dcdt_jit` (numba), `ppgpp_metabolite_changes`, `get_charging_supply_function`, AA-transport functions | **No edits expected.** Already provides everything the kinetic class needs to reuse. The `supply` closure factory and the ppGpp delta function are designed to be callable by either class. |
| `v2ecoli/processes/polypeptide_elongation.py` (1811 LOC) | `BasePolypeptideElongation`, `TranslationSupplyPolypeptideElongation`, `SteadyStatePolypeptideElongation` | **Reference only.** Port methods FROM here (lines 1056-1095 for supply-arg setup; lines 1481+ for ppGpp request/evolve). No edits unless §3 Option B/C is chosen. |
| `v2ecoli/processes/metabolism.py` (1260 LOC) | Homeostatic FBA; consumes `aa_count_diff` (line 646) and `aa_exchange_rates` (line 673) | **No edits expected** — consumers already exist and operate gated on `use_trna_charging` / `mechanistic_aa_transport`. |
| `v2ecoli/composites/baseline.py` | Main composite; defines `ppgpp_regulation` feature; defaults `ppgpp_regulation=True` | **No edits expected** — already passes `ppgpp_regulation=True` through. |
| `v2ecoli/composites/kinetic_charging_baseline.py` (137 LOC) | Kinetic composite that swaps in the kinetic class via scoped `PARTITIONED_PROCESSES` swap | **Likely no edits.** After P2+P3 land, this composite IS the consensus model when run with default `ppgpp_regulation=True`. May add an alias `consensus_baseline` for discoverability in P4. |
| `tests/test_behavior_kinetic_charging.py` | Behavior test for kinetic class | **Extend** with ppGpp-on + supply-on assertions across P2/P3. |
| `tests/test_polypeptide_elongation_parity.py` | Parity vs upstream | **Extend** with consensus-mode parity gates in P4. |
| `tests/test_kinetic_charging_polypeptide_elongation_scaffold.py` | Scaffold-level tests for the kinetic class | **Add** `aa_count_diff` shape/value assertions + supply accumulator assertions in P3. |
| `tests/test_metabolism_flux_pin.py` | Metabolism flux pinning | **Verify** still passes after `aa_count_diff` becomes a non-zero array. |

---

## 6. Phase Plan

Two phases do most of the work. They are **independent and can run in parallel after Phase 1**.

| # | Phase | Scope | Tests | Status |
|---|---|---|---|---|
| 1 | Audit & Plan | This doc + HANDOFF.md. No code. | — | ✅ committed (e9c53dee) |
| 3a | ODE-merge scaffold | `include_aa_supply` flag + 3 accumulator slices in state vector. No RHS change. | `tests/test_kinetic_aa_supply_ode_scaffold.py` (7) | ✅ committed (774e0c12) |
| 3b-i | Supply plumbing | Unpack `amino_acid_synthesis`/`_import`/`_export`/`aa_supply_scaling`/`get_pathway_enzyme_counts_per_aa`/`import_constraint_threshold`. Override `inputs()` for `boundary` port + `outputs()` for supply listeners. Fix latent deep_merge nesting bug. | regression net | ✅ committed (db79adfe) |
| 3b-ii | **ODE merge: AA supply terms** | `_build_supply_function` helper. Plumb closure into `ode_model` via `args=`. Add `+v_syn +v_imp −v_exp` to AA balance. Write 3 accumulator dx_dt rows. Post-solve: extract accumulators → emit listeners + store on self. Fix `aa_count_diff` return. | `tests/test_kinetic_aa_supply_ode.py` (8) | ✅ ODE-merge proof passing; cache-backed listener test passing |
| 2 | ppGpp coupling on kinetic class | Port `_ppgpp_request` / `_ppgpp_evolve` (Option A). Wire pre-solve `elong_rate_by_ppgpp`. Emit `ppgpp_conc` listeners. | `tests/test_kinetic_ppgpp_coupling.py` | ⏳ pending |
| 4 | Composite + parity | Optional `consensus_baseline` alias. Parity tests. | `tests/test_consensus_composite.py` | ⏳ pending |
| 5 | Validation sweep | 5-gen × 3-seed × 4-media. ParCa rerun gate. | `scripts/consensus_validation_sweep.py` | ⏳ pending |

**Total active development**: ~4–6 sessions of code + tests, then a compute-bound validation phase. P2 and P3 can be split across parallel sessions if desired.

**Phase ordering note**: P3 (the ODE merge) is the larger and more invasive of the two parallel phases. If sequencing matters, do P3 first — getting AA supply into the kinetic ODE makes pool dynamics realistic, which is what ppGpp regulation actually needs to be tested against in behavior tests. P2's tests will be more meaningful after P3 lands.

---

## 7. Test Strategy

Each phase ships its own tests, gated to pass before moving on.

| Phase | New / changed tests | Acceptance |
|---|---|---|
| 2 | `tests/test_kinetic_ppgpp_coupling.py` (new) | (a) `kinetic_charging_baseline` with `ppgpp_regulation=True` emits non-zero `ppgpp_conc` listener (currently empty/missing). (b) Starvation scenario increases ppGpp 2–5× vs `minimal`. (c) Recovery: ppGpp falls back within 1–2 generations after restoring AAs. (d) `_ppgpp_request` numeric parity with SteadyState's `_ppgpp_request` on shared inputs (cross-class). (e) `target_codon_rate` is scaled by `elong_rate_by_ppgpp` when `ppgpp_regulation=True`. |
| 3 | `tests/test_kinetic_aa_supply_ode.py` (new) | (a) Kinetic ODE state vector has 9 slices (was 6); supply accumulators initialize to 0 and grow monotonically (synthesis, import) or grow (export). (b) `dx_dt[slice_amino_acids]` includes `+v_supply_counts` (assert by parametrized fixture: with supply > 0, AA pool drains less than supply=0 case). (c) Post-solve accumulators emit `aa_supply`, `aa_synthesis`, `aa_exchange_rates` listeners with correct shapes and units. (d) Unit bridge: convert input AA counts → μM → call supply → flux returned in counts/s matches a manual μM×N_A×V_cell calculation. |
| 3 (tail) | `tests/test_aa_count_diff_kinetic.py` (new) | (a) Pre-fix: reproduce bug — `evolve()` returns `{}` for second tuple position. (b) Post-fix: returns `ndarray[float64]` of shape `(n_amino_acids,)`. (c) Sign convention matches SteadyState: positive `aa_count_diff` = over-supplied = raise homeostatic target. (d) Metabolism receives a valid array when `use_trna_charging=True` and adjusts homeostatic targets in the expected direction. |
| 4 | `tests/test_consensus_composite.py` (new) | (a) `kinetic_charging_baseline(ppgpp_regulation=True)` instantiates clean. (b) 1-tick smoke. (c) Parity: `ppgpp_regulation=False` + supply-disabled mode ≡ pre-consensus kinetic baseline (no behavioral drift on the legacy path). (d) 5-tick smoke shows non-zero ppGpp dynamics + non-zero supply. |
| 5 | `scripts/consensus_validation_sweep.py` (new) | 5-gen × 3-seed × 4-media (`minimal`, `acetate`, `plus_amino_acids`, `no_glucose`). Acceptance per spec: growth ±5–10%, charging ≥85%, ppGpp 2–5× stringent rise, rare-codon 20–70% slowdown. |

Existing tests that must continue to pass throughout:
- `tests/test_behavior_kinetic_charging.py` — kinetic class regression net.
- `tests/test_polypeptide_elongation_parity.py` — SteadyState parity is not touched by these changes.
- `tests/test_metabolism_flux_pin.py` — metabolism path is not refactored.
- `scripts/parity_check.py --build-check` — composite-build gate from AGENTS.md.

---

## 8. Risks & Open Questions

| Risk / question | Mitigation / answer plan |
|---|---|
| **ODE merge stability**: extending the kinetic ODE's state vector from 6 to 9 slices and adding supply terms to the AA balance could destabilize RK45 (the `adjusted_amino_acid_saturation` sin²-roll-off at low pools was tuned for monotonic depletion; supply terms could overshoot zero). | Phase 3 starts with the supply terms behind an opt-in flag (`include_aa_supply`, default False) so legacy behavior is preserved. Tests parametrize both ways. Use `scipy.solve_ivp`'s `max_step` if needed. SteadyState ODE has been running with these terms for years on `main`; if it works there, the math should work here once units bridge. |
| **Unit bridge correctness**: `supply` closure expects μM; ODE state is counts; converting back and forth at every RK45 step is the most likely source of off-by-N_A or off-by-V_cell errors. | Phase 3 test (d) directly asserts numeric correctness against a manual hand-calculation. Use existing `MICROMOLAR_UNITS` constants from `polypeptide_elongation.py`. Pin `counts_to_molar` once per tick (cell volume doesn't change during the timestep). |
| **Option A duplicates ppGpp code** between SteadyState and Kinetic classes. | Accept as Phase 2 cost. Plan refactor to mixin (Option C) as a follow-up after consensus parity gates pass. |
| **`evolve_state` already has codon-aware pipeline**; bolting on `_ppgpp_evolve` mid-pipeline may break bulk-delta accounting. | Place `_ppgpp_evolve` AFTER the existing bulk deltas in `evolve` (line 737). ppGpp bulk deltas are independent of tRNA charging deltas. Add a unit test that conserved species (water, ATP, AMP) balance before vs after the ppGpp bolt-on. |
| **`aa_count_diff` sign convention**: spec wording is ambiguous. | Ground truth: read `SteadyStatePolypeptideElongation`'s `aa_count_diff` write site directly during Phase 3 tail; mirror that. Don't trust spec memos. |
| **Parity baseline**: Phase 3 changes behavior vs the current kinetic baseline (the bug fix flips a silently-broken path; the supply terms add real dynamics). What's the canonical "before"? | Define two parity gates: (1) `ppgpp_regulation=False` + `include_aa_supply=False` ≡ `trna_charging_final@5ffb76de` kinetic behavior (regression gate). (2) `ppgpp_regulation=True` + supply enabled is the consensus, which has no historical baseline — gate on the behavior tests in §7 instead of numeric parity. |
| **Phase 5 ParCa rerun is hours of compute**. | Schedule overnight or to coincide with another required ParCa rerun. Don't block Phase 4 sign-off on it. |

---

## 9. What This Audit Does NOT Cover

- **Codon-aware reconciliation logic** — already implemented in the kinetic class kernel; no changes needed for consensus.
- **K_M unit conversion at ODE startup** — already done in `kinetic_charging.py`. The ODE merge adds NEW unit-bridging at the supply boundary, but that's separate.
- **Transcription's ppGpp feedback on rRNA/tRNA operons** — `transcript_initiation.py` already has this from upstream; no edits expected. Verified indirectly by Phase 2 behavior tests (rRNA/tRNA synthesis should drop when ppGpp rises).
- **`mechanistic_aa_transport` flag mechanics** — falls out naturally once supply accumulators exist (Phase 3 emits `aa_exchange_rates`). The flag itself is a metabolism-side switch already in place.
- **Visualization / dashboards** — out of scope until validation reports.