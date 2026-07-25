# Stage A Smoke Sweep — Findings

**Date:** 2026-06-27
**Branch:** `consensus_elongation` (HEAD: `140298c6` at time of Stage A v3)
**Scope:** 1 condition (`minimal`) × 1 seed (0) × 1 generation × 100 ticks
**Outcome:** ✅ Consensus model is **functional** for 100 simulated seconds. Biology partially-stressed (charging fraction 0.67 vs spec 0.85), not catastrophic.

This document captures everything a fresh session needs to resume the validation work.

---

## TL;DR

- The consensus elongation model (kinetic tRNA charging + ppGpp + AA synthesis/import/export merged into one ODE) **runs cleanly for 100 ticks**. First evidence the model can sustain itself beyond 1 tick.
- **Two bugs fixed during Stage A** (both commit `140298c6`):
  1. Unit conversion at the supply-call boundary: μM → mol/L was missing, causing `amino_acid_synthesis` to interpret AA concentrations as 1e6× too high.
  2. tRNA-recovery loop in `run_model` could drive `chargings` negative because it didn't check the source tRNA had any charging events left to undo.
- **Per-tick wall: 3.3s** (vs my pre-Stage-A estimate of ~42s). The unit fix dramatically reduced ODE stiffness; BDF takes much bigger adaptive steps when concentrations are physiologically realistic.
- **Biology gaps remaining**: charging fraction maxes at 0.67 (spec ≥ 0.85), starts declining after t=70, chronic TRP[c] homeostatic-target adjustment in metabolism. None of these are crashes — they're tuning gaps.

---

## How to reproduce Stage A

```bash
# From repo root with venv activated:
.venv/bin/python -u scripts/consensus_validation_sweep.py \
    --stage smoke \
    --output-dir out/consensus_validation_smoke
```

Output (typical, on this hardware):
- ~5.5 min wall total
- `out/consensus_validation_smoke/minimal/seed-0/`:
  - `run.db` — SQLite (10 history rows at chunk=10)
  - `trajectory.json` — per-emit listener values, pretty-printed
  - `summary.txt` — human-readable per-field stats + sanity-check verdict
- `out/consensus_validation_smoke/report.json` — acceptance check verdict

---

## Quantitative results

### Per-tick wall cost (BIG surprise — order-of-magnitude better than profile)

| Source | Per-tick wall | Notes |
|---|---|---|
| Legacy kinetic (RK45) | 2.54s | pre-consensus baseline |
| Consensus RK45 (pre-BDF) | 304s | profiled before BDF switch |
| Consensus BDF (profile, 1 tick) | 42s | profiled after BDF switch |
| Consensus BDF post-unit-fix (Stage A actual) | **3.3s** | unit fix unblocks BDF's big steps |

The unit-conversion bug had been forcing BDF into tiny steps because the wildly-incorrect AA concentrations made the supply function return extreme values, perpetually re-triggering Jacobian updates and step rejections. With concentrations in the correct order of magnitude, BDF runs near optimal stride.

### Per-emit trajectory (10 emits across t=10s → t=100s)

| t (s) | charged_frac | aa_supply (counts/s, mean) | rela_syn | spot_deg |
|---|---|---|---|---|
| 10 | 0.493 | +225,357 | 0.112 | 0.244 |
| 20 | 0.558 | +228,042 | 0.228 | 0.492 |
| 30 | 0.635 | +230,820 | 0.227 | 0.723 |
| 40 | 0.657 | +233,680 | 0.226 | 0.908 |
| 50 | 0.662 | +236,536 | 0.205 | 1.078 |
| 60 | 0.667 | +239,954 | 0.229 | 1.215 |
| 70 | **0.668** (peak) | +243,271 | 0.211 | 1.320 |
| 80 | 0.665 | +246,656 | 0.211 | 1.403 |
| 90 | 0.658 | +249,882 | (continuing) | 1.421 |
| 100 | ~0.65 | ~+253,000 | ~0.21 | ~1.42 |

### Acceptance check (against spec)

| Criterion | Target | Actual | Verdict |
|---|---|---|---|
| `trna_charging_fraction` mean | ≥ 0.85 | **0.6317** | ✗ below |
| `aa_supply` listener emits | non-zero | 10/10 ticks | ✓ |
| `rela_syn` listener emits | non-zero | 10/10 ticks | ✓ |
| ppGpp starvation rise | 2–5× on stress media | N/A (Stage A is single-condition) | — |
| Cell mass growth | positive | N/A (cell_mass not in trajectory dump — see open issue 3) | — |

---

## Bug history (Stage A diagnostic chain)

| # | Symptom | Root cause | File:line | Commit |
|---|---|---|---|---|
| 1 | RK45 took ~1.9M sub-steps per tick (~5 min/tick wall) | Merged supply terms made the ODE stiff; RK45 is explicit non-stiff | `kinetic_charging.py:1598-1601` | `9929e5da` (BDF switch) |
| 2 | Stage A v1: composite stopped at tick 10 with bare `AssertionError`; supply rates wildly wrong sign (aa_supply = −143,760 counts/tick); charging fraction stuck at 0.33 | `ode_model` passed μM magnitudes to `amino_acid_synthesis` which expects `METABOLITE_CONCENTRATION_UNITS = mol/L` (1e6× too high → enormous reverse fluxes) | `kinetic_charging.py:1538` (now `aa_conc_molar = (counts_to_uM_mag * amino_acids_remaining) * unit_conversion`) | `140298c6` (unit fix) |
| 3 | Stage A v2: composite stopped at tick 7 with bare `AssertionError` after unit fix | Pre-existing kernel bug: AA-availability recovery loop in `run_model` could drive `chargings` negative because it didn't mask tRNAs that already had `chargings == 0`. Exposed by the increased charging activity once unit fix landed. | `kinetic_charging.py:1741-1753` | `140298c6` (recovery-loop guard) |

The fixes are all in one source file (`v2ecoli/processes/polypeptide/kinetic_charging.py`) and are minimal. The diagnostic chain was:
- `scripts/profile_consensus_tick.py` → identified RK45 as the stiffness problem
- `scripts/consensus_validation_sweep.py --stage smoke` → caught the unit bug via the `aa_supply` listener sign check in `summary.txt`
- `scripts/diagnose_consensus_tick.py 15` → produced full traceback for the recovery-loop assert

---

## Open issues for next session

### 1. Charging fraction below spec (0.67 vs ≥ 0.85)

**Observation:** Charging fraction rises from 0.49 to a peak of 0.67 at t=70, then declines slightly. Doesn't reach the spec target of 0.85.

**Hypotheses (untested):**
- **Initial conditions**: Initial tRNA pools may be calibrated for the SteadyState path; the kinetic + consensus ODE needs longer equilibration. A 100-tick window may be too short to reach a true steady state.
- **Ribosome demand > supply**: If `target_codon_rate` is set too high relative to what the supply can sustain, the cell will chronically undercharge. Could be that `elongation_max` parameter in `charging_params` isn't being throttled enough by ppGpp.
- **Synthesis under-calibrated**: `amino_acid_synthesis_jit`'s reverse-reaction terms may dominate at low AA concentrations (well below the K_M for forward synthesis). Could be a parameter calibration issue from ParCa.

**How to investigate:**
- Run Stage B (1500-3500 ticks → ~83 min - 3.2h wall) and see if charging fraction settles to a stable value. If it stays at 0.65-0.70, that's the model's actual steady state.
- Add per-AA charging fraction to the trajectory dump (currently only the mean across AAs is emitted). One AA could be dragging the mean down.
- Compare with SteadyState model running the same composite (substitute the elongation process). If SteadyState reaches 0.85 in the same conditions, the kinetic path is what's mis-tuned.

### 2. TRP[c] homeostatic target chronically negative

**Observation:** Throughout the 100-tick run, metabolism prints `Warning: updated amino acid target for TRP[c] was negative - adjusted to be positive.` at least 24 times. Metabolism clamps it positive (benign), but it indicates the `aa_count_diff` feedback for TRP is repeatedly pushing the target negative.

**Hypotheses:**
- Tryptophan synthesis is intrinsically slow (TRP pathway has many steps; rate-limiting). The consensus model's elongation may be consuming TRP faster than synthesis can replace, especially in minimal media where TRP must be made de novo.
- `aa_count_diff = (synth + import - export) - amino_acids_used` could be heavily negative for TRP, telling metabolism "we used way more than we made" — metabolism then tries to lower the target, hitting zero.

**How to investigate:**
- Read TRP-specific values from the trajectory: per-AA `aa_supply` and `aa_used` for TRP. Need to enhance `extract_trajectory` to keep per-AA values for TRP rather than just means.
- Check upstream `CovertLab/vEcoli` for any TRP-specific handling.

### 3. `cell_mass` not in trajectory dump

**Observation:** `extract_trajectory` reads `listeners.mass.cell_mass`, but it's missing from the dumped JSON. Likely a serialization issue — `cell_mass` may be a pint Quantity stored as a dict/string in the JSON state column rather than a plain float.

**Fix:** in `extract_trajectory`, handle pint-Quantity dict serialization (e.g., `{"magnitude": 1.42, "units": "fg"}`) by extracting the magnitude. Quick win — single function edit.

### 4. Per-emit cadence

**Observation:** Chunk=10 in smoke stage gave us 10 trajectory points across 100 ticks. For Stage B with 1500-3500 ticks, default chunk=100 would give 15-35 emits — fine. But for diagnostic deep dives, smaller chunks would help. Already controllable via `--chunk` CLI flag.

---

## How to resume work in a new session

### Option A: Launch Stage B (single cell cycle)

```bash
.venv/bin/python -u scripts/consensus_validation_sweep.py \
    --stage cell-cycle \
    --output-dir out/consensus_validation_cellcycle
```

Wall: ~3 hours (1500 ticks default, can extend to 3500 with `--max-steps-per-gen 3500`). Goal: see one division event, capture full cell-cycle ppGpp + supply dynamics.

### Option B: Investigate biology gaps (faster turnaround)

1. Read `stage_a_findings.md` § "Open issues" above.
2. Re-run Stage A with extended per-AA trajectory extraction.
3. Compare against SteadyState model on the same composite.
4. Hypothesize-and-test rounds on the TRP issue.

### Option C: Stop and document the consensus model build as complete

The consensus model is functionally complete (P1-P5 all landed, model runs 100 ticks). Spec acceptance criteria (charging ≥ 0.85, etc.) aren't met but model is structurally sound. Could write up the final state and end the project here.

---

## All commits on `consensus_elongation` (as of Stage A completion)

```
140298c6 fix(consensus): correct μM → mol/L unit bridge + guard tRNA recovery loop
9929e5da perf(consensus): switch to BDF solver when AA-supply terms are active
5fcedd5b feat(consensus): P5 — validation sweep orchestrator + report generator
c20701e9 feat(consensus): P4 — consensus_baseline composite alias + parity gates
b27b2569 feat(consensus): P2 — ppGpp regulation on the kinetic-charging path
bda11966 feat(consensus): P3b-ii — AA supply terms inside the kinetic ODE RHS
942723be test(consensus): P3b-ii red gates — bind the ODE-merge claim
db79adfe feat(consensus): P3b-i plumbing — unpack AA-supply callables in kinetic initialize
774e0c12 feat(consensus): P3a scaffold — include_aa_supply flag and accumulator slices
e9c53dee docs(consensus): Phase 1 audit and handoff for consensus_elongation branch
```

---

## Key artifacts

- `v2ecoli/processes/polypeptide/kinetic_charging.py` — the kinetic class with all consensus changes
- `v2ecoli/composites/consensus_baseline.py` — composite alias with both flags forced on
- `scripts/consensus_validation_sweep.py` — staged sweep orchestrator
- `scripts/profile_consensus_tick.py` — A/B per-tick profiler
- `scripts/diagnose_consensus_tick.py` — one-tick-at-a-time diagnostic
- `tests/test_consensus_*.py` + `tests/test_kinetic_aa_supply_ode*.py` + `tests/test_kinetic_ppgpp_coupling.py` — 50+ green tests gating the build
- `workspace/investigations/consensus_elongation/audit.md` — the original phase plan
- `workspace/investigations/consensus_elongation/HANDOFF.md` — session-to-session resume prompts
- `workspace/investigations/consensus_elongation/stage_a_findings.md` — this file
- `out/consensus_validation_smoke/minimal/seed-0/` — Stage A outputs (trajectory + summary + run.db)
- `logs/sweep_smoke_v3.log` — Stage A v3 execution log
