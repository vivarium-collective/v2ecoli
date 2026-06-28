# Consensus Elongation Model — Session Handoff

**Branch**: `consensus_elongation` (forked 2026-06-26 from `trna_charging_final` HEAD `5ffb76de`).
**Status (2026-06-27)**: Stage A smoke sweep ✅ complete. Model functionally runs 100 ticks. Biology partially-stressed (charging fraction 0.67 vs spec 0.85). See `stage_a_findings.md` for details.

## What's in this folder

- `audit.md` — original file-by-file investigation + 5-phase plan (Phases 1-5 all landed).
- `stage_a_findings.md` — **Stage A smoke sweep results, bug history, open issues, resume options**. Read this first if you're resuming.
- `HANDOFF.md` — this file: phase status table + next-session prompts.

## Phase status

| # | Phase | Status |
|---|---|---|
| 1 | Audit & Plan | ✅ complete (`e9c53dee`) |
| 3a | ODE-merge scaffold | ✅ committed (`774e0c12`) |
| 3b-i | Supply plumbing | ✅ committed (`db79adfe`) |
| 3b-ii | ODE merge — AA supply in kinetic RHS | ✅ committed (`bda11966`) |
| 2 | ppGpp coupling | ✅ committed (`b27b2569`) |
| 4 | Composite + parity | ✅ committed (`c20701e9`) |
| 5 | Validation sweep orchestrator | ✅ committed (`5fcedd5b`) |
| 5+ | BDF integrator switch | ✅ committed (`9929e5da`) |
| 5+ | Unit fix + recovery-loop guard | ✅ committed (`140298c6`) |
| 5+ | **Stage A smoke sweep — 100 ticks** | ✅ ran clean; see `stage_a_findings.md` |
| 5+ | Stage B single cell cycle | ⏳ ready to launch (~3h wall) |
| 5+ | Investigate biology gaps | ⏳ optional |

## Biology gaps from Stage A (not crashes — tuning issues)

1. **Charging fraction maxes at 0.67** (spec target ≥ 0.85). Climbs from 0.49 → 0.67 in 100s, then starts declining slightly.
2. **TRP[c] homeostatic target chronically negative** (metabolism clamps positive every tick — at least 24 warnings in 100 ticks).
3. **`cell_mass` not in `trajectory.json`** — likely a pint-Quantity serialization issue in `extract_trajectory`. Easy fix.
4. **ppGpp degradation rising sharply** (spot_deg 0.24 → 1.42 over 100s) while synthesis stays flat. Maybe normal recovery, maybe not — needs longer run to see if it settles.

None of these are crash conditions. The model is structurally sound.

## Next-session prompts

### Option A — Launch Stage B (single cell cycle)

```
Continue Stage B of consensus validation in v2ecoli.
Branch: consensus_elongation. Read
workspace/investigations/consensus_elongation/stage_a_findings.md first
for current state and bug history.

This session: launch the single-cell-cycle sweep, monitor, report on
whether the cell divides cleanly.

Command:
  .venv/bin/python -u scripts/consensus_validation_sweep.py \
      --stage cell-cycle \
      --output-dir out/consensus_validation_cellcycle \
      > logs/sweep_cellcycle.log 2>&1 &

Expected wall: ~3 hours (1500 ticks at 3.3s/tick). After completion,
read out/consensus_validation_cellcycle/minimal/seed-0/summary.txt and
trajectory.json to assess: cell mass growth, charging fraction
stability, ppGpp dynamics, whether one division event occurred.
```

### Option B — Investigate biology gaps before scaling up

```
Investigate the consensus model biology gaps from Stage A.
Branch: consensus_elongation. Read
workspace/investigations/consensus_elongation/stage_a_findings.md
§ "Open issues" first.

Pick ONE of:

(a) Per-AA charging fraction breakdown — modify extract_trajectory to
    keep per-AA fraction_trna_charged arrays (not just the mean).
    Re-run Stage A. Identify which AAs drag the mean down. Maps to
    open issue #1.

(b) TRP[c] deep dive — read aa_count_diff for TRP per tick, trace
    metabolism's reaction to the negative-target adjustment. Could
    reveal a calibration parameter that needs adjusting. Maps to
    open issue #2.

(c) Cell mass serialization fix — extract_trajectory needs to handle
    pint Quantity stored as dict/string in the SQLite state JSON.
    Quick win: ~5 LOC + a test. Maps to open issue #3.

After: re-run Stage A smoke to verify the fix doesn't regress.
```

### Option C — Document complete + freeze

```
The consensus elongation model build is functionally complete per
v2ecoli_consensus_model.md. Phases 1-5 all landed, model runs cleanly
through Stage A smoke (100 ticks), all design-spec features present
and verified through dedicated tests:

- Individual tRNA species (86 tracked): kinetic_charging.py state vector
- Codon-aware translation: ode_model RHS line 1514 (reading_rate)
- ppGpp regulation: _ppgpp_request + _ppgpp_evolve, elong_rate_by_ppgpp
- AA synthesis/import/export INSIDE the same ODE: ode_model lines
  1537-1553 (supply terms in dx_dt)
- Integrated ODE: single solve_ivp call per run_model, 9-slice state

Spec acceptance criteria (charging ≥ 0.85, etc.) not met but model
is structurally sound. Tuning to spec targets is deferred to future
work (calibration, parameter sweeps).

Close out the project:
1. Verify all tests still green
2. Update memory file project_consensus_elongation_model.md with final status
3. Open a PR (or keep branch local per current convention)
4. Write a final summary in workspace/investigations/consensus_elongation/FINAL.md
```

## Pointers (don't re-derive)

### Code
- **Kinetic process** (the consensus implementation lives here):
  `v2ecoli/processes/polypeptide/kinetic_charging.py`
  - Line 78 = class definition
  - Line 502 = `elongation_rate` (with ppGpp inhibition cap from P2)
  - Line 555 = `request` (with _ppgpp_request wiring from P2)
  - Line 759 = `evolve_state` (with _ppgpp_evolve wiring)
  - Line 782 = `evolve` (with aa_count_diff computation)
  - Line 833 = `_ppgpp_request` (ported from SteadyState in P2)
  - Line 887 = `_ppgpp_evolve` (ported from SteadyState in P2)
  - Line 1286 = `_build_supply_function` (builds supply closure for ODE RHS)
  - Line 1406 = `run_model` (orchestrates solve_ivp)
  - Line 1439 = `ode_model` (the MERGED ODE RHS — this is the consensus)
  - Line 1583 = `solve_ivp` call (BDF when supply on, RK45 when off)
  - Line 1741 = AA-availability recovery loop (with the chargings>=0 guard from 140298c6)

- **Composite alias**: `v2ecoli/composites/consensus_baseline.py`
- **Validation sweep**: `scripts/consensus_validation_sweep.py` (`--stage` flag)
- **Diagnostic runner**: `scripts/diagnose_consensus_tick.py` (one-tick-at-a-time, no exception swallowing)
- **Profiler**: `scripts/profile_consensus_tick.py` (A/B legacy vs consensus)

### Reference (SteadyState — where ppGpp + supply logic was ported from)
- `v2ecoli/processes/polypeptide_elongation.py:830` = `SteadyStatePolypeptideElongation`
- `v2ecoli/processes/polypeptide_elongation.py:1481+` = `_ppgpp_request` (verbatim source)
- `v2ecoli/processes/polypeptide_elongation.py:1535+` = `_ppgpp_evolve` (verbatim source)
- `v2ecoli/processes/polypeptide/kinetics.py:30+` = `ppgpp_metabolite_changes` (shared math)
- `v2ecoli/processes/polypeptide/kinetics.py:330-355` = SteadyState `dcdt` RHS (reference for the merge)
- `v2ecoli/processes/polypeptide/kinetics.py:520-627` = `get_charging_supply_function` (shared supply-closure factory)
- `v2ecoli/processes/parca/reconstruction/ecoli/dataclasses/process/metabolism.py:2001` = `amino_acid_synthesis` (the function the closure calls)

### Tests (50+ green across the consensus build)
- `tests/test_kinetic_aa_supply_ode_scaffold.py` — P3a scaffold (7 tests)
- `tests/test_kinetic_aa_supply_ode.py` — P3b-ii ODE merge (8 tests)
- `tests/test_kinetic_ppgpp_coupling.py` — P2 ppGpp (10 tests)
- `tests/test_consensus_composite.py` — P4 composite parity (9 tests)
- `tests/test_consensus_validation_sweep.py` — P5 orchestrator (24 tests)

### Cache + environment
- ParCa cache: `out/cache/` (last rebuilt 2026-06-26 17:21, see `logs/parca_full_rerun.log`)
- venv setup: `uv sync --extra dev --no-install-package vivarium-dashboard`
- Reference upstream (kinetic origin): `/Users/arnabmutsuddy/projects/vEcoli_trna/vEcoli`
