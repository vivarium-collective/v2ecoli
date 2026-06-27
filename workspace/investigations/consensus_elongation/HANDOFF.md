# Consensus Elongation Model — Session Handoff

**Branch**: `consensus_elongation` (forked 2026-06-26 from `trna_charging_final` HEAD `5ffb76de`).
**Status**: Phase 1 complete. Phase 2 ready to start. No code changes committed yet.

## What's in this folder

- `audit.md` — file-by-file investigation, the latent `aa_count_diff` bug, the missing ppGpp hooks on the kinetic class, the proposed 6-phase plan, risks.
- `HANDOFF.md` — this file. Session boundaries + next-session prompt.

## Project context (1 paragraph)

The design spec (`v2ecoli_consensus_model.md`) calls for "merging two ODE systems." The merge is real but smaller than that framing suggests, and stratified into three coupling patterns: (1) **AA synthesis/import/export** lives INSIDE the SteadyState ODE RHS via a `supply(aa_conc)` closure and is genuinely absent from the kinetic ODE RHS — adding it is a real ODE-RHS extension (state vector grows from 6 to 9 slices, AA balance gains supply terms, accumulators emit listeners). (2) **ppGpp synthesis/degradation** is NOT integrated — it's a single-shot delta via `ppgpp_metabolite_changes` called from `_ppgpp_request` (predict) and `_ppgpp_evolve` (realize). Porting these two methods to the kinetic class + wiring the `elong_rate_by_ppgpp` pre-solve adjustment is decoupled outer-loop work. (3) **`aa_count_diff` to metabolism** is a latent bug — kinetic `evolve()` returns `{}` instead of an ndarray. Fixes naturally as the tail of (1) once accumulators exist (compute `aa_count_diff = total_synthesis + total_import - total_export - amino_acids_used`).

## Phase status

| # | Phase | Status |
|---|---|---|
| 1 | Audit & Plan | ✅ complete (this folder, rev 2) |
| 2 | ppGpp coupling | ⏳ ready to start (parallel with P3) |
| 3 | ODE merge: AA supply terms (+ aa_count_diff tail) | ⏳ ready to start (parallel with P2) |
| 4 | Composite + parity | ⏳ blocked on P2+P3 |
| 5 | Validation sweep | ⏳ blocked on P4 + ParCa rerun |

P2 and P3 are independent. Recommended order if sequential: **P3 first** — supply-realistic AA pools make P2's ppGpp behavior tests more meaningful.

## Next session prompt

```
Continue consensus elongation model build in v2ecoli.
Branch: consensus_elongation (see workspace/investigations/consensus_elongation/audit.md).

This session: tackle Phase 3 (the ODE merge) from audit.md §6.

Goal: extend the kinetic ODE in
v2ecoli/processes/polypeptide/kinetic_charging.py (ode_model at line 812)
to include AA synthesis/import/export terms in the AA balance RHS, mirroring
the SteadyState ODE at v2ecoli/processes/polypeptide/kinetics.py:330-355.

Specifics:
- Extend state vector from 6 to 9 slices: add total_synthesis,
  total_import, total_export accumulators.
- Pre-compute supply args before solve_ivp (mirror
  polypeptide_elongation.py:1056-1095).
- Plumb get_charging_supply_function closure (kinetics.py:520-627) into
  ode_model. Convert AA counts → μM at every RK45 step, convert flux back.
- Add `+v_synthesis + v_import - v_export` to dx_dt[slice_amino_acids].
- Post-solve: emit aa_supply, aa_synthesis, aa_exchange_rates listeners
  from accumulators.
- Tail: replace the `return net_charged, {}, update` at line 775 with
  `return net_charged, aa_count_diff, update` where aa_count_diff =
  total_synthesis + total_import - total_export - amino_acids_used
  (matches SteadyState sign convention).

Tests: tests/test_kinetic_aa_supply_ode.py + tests/test_aa_count_diff_kinetic.py.
Acceptance criteria in audit.md §7.
Gate Phase 3 behind an opt-in flag `include_aa_supply` (default False) so
the legacy path stays bit-identical for regression.
```

## Pointers (don't re-derive)

- Design spec: `v2ecoli_consensus_model.md` (repo root)
- Kinetic process: `v2ecoli/processes/polypeptide/kinetic_charging.py` (line 78 = class; line 457 = `evolve_state`; line 704 = inner `evolve`; line 775 = the bug)
- Steady-state reference: `v2ecoli/processes/polypeptide_elongation.py` (line 830 = `SteadyStatePolypeptideElongation`; line 1481+ = `_ppgpp_request`)
- Kinetic composite: `v2ecoli/composites/kinetic_charging_baseline.py` (already enables ppgpp_regulation=True via baseline delegation)
- Metabolism consumer: `v2ecoli/processes/metabolism.py:646` (aa_count_diff), `:673` (aa_exchange_rates)
- Env setup: `uv sync --extra dev --no-install-package vivarium-dashboard`
- Reference upstream: `/Users/arnabmutsuddy/projects/vEcoli_trna/vEcoli` (kinetic origin)
```
