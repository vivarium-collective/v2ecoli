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
| 3a | ODE-merge scaffold | ✅ committed (774e0c12) |
| 3b-i | Supply plumbing | ✅ committed (db79adfe) |
| 3b-ii | **ODE merge — AA supply in kinetic RHS** | ✅ landed; ODE merge gate + listener test passing |
| 2 | ppGpp coupling | 🟡 implementation landed; sim tests pending |
| 4 | Composite + parity | 🟡 implementation landed; sim tests pending |
| 5 | Validation sweep | 🟢 orchestrator + 16 unit tests landed; multi-hour sweep is user-driven |

Next: commit P3b-ii (after full sim-test pass), then start P2 (ppGpp coupling).

## Next session prompt

```
Continue consensus elongation model build in v2ecoli.
Branch: consensus_elongation (see workspace/investigations/consensus_elongation/audit.md).

This session: tackle Phase 2 (ppGpp coupling) from audit.md §6.

Goal: wire ppGpp regulation into KineticTrnaChargingPolypeptideElongation
so the kinetic_charging_baseline composite (which already passes
ppgpp_regulation=True) actually exercises ppGpp dynamics on the kinetic
path. Today the kinetic class has no _ppgpp_request / _ppgpp_evolve
hooks, so ppGpp is silently a no-op.

Approach (audit.md §3, Option A):
- Port _ppgpp_request from SteadyStatePolypeptideElongation:1481-1533
  into the kinetic class. Wire it into the existing request() method —
  append bulk requests for the ppGpp reaction metabolites alongside the
  existing AA/ATP/tRNA requests.
- Port _ppgpp_evolve from SteadyStatePolypeptideElongation:1535-...
  Add it to evolve() AFTER the existing bulk deltas — ppGpp deltas are
  independent of tRNA charging deltas.
- Pre-solve: if ppgpp_regulation and not disable_ppgpp_elongation_inhibition,
  call self.elong_rate_by_ppgpp(ppgpp_conc, basal_rate) and use the
  result to scale target_codon_rate passed to ode_model. (This is the
  ppGpp → elongation-rate inhibition.)
- Emit ppgpp_conc + rela_conc + spot_conc + rela_syn + spot_syn +
  spot_deg listeners to growth_limits.

Tests: tests/test_kinetic_ppgpp_coupling.py per audit.md §7 row P2:
- ppgpp_conc listener non-empty when ppgpp_regulation=True
- starvation 2-5x rise vs minimal
- numeric parity with SteadyState's _ppgpp_request on shared inputs
- target_codon_rate scaled when flag on

P3b-ii's deep_merge nesting fix is already in place — listeners flow
through the store correctly now.
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
