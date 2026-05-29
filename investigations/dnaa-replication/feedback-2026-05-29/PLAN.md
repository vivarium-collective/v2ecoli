# dnaa-replication — Round 3.8 methodology pivot

Reviewer: **Rashmi** (2026-05-29 feedback widget, exported HTML
`investigation-dnaa-replication-2026-05-29-2.html`). Raw feedback:
`./feedback.yaml`.

## Rashmi's two annotations

> **`study-dnaa-00-parameter-foundation`** — "the goal was to get a
> simulation where the total doubling time was greater than C+D period.
> So succinate might be a better media to use for the baseline
> simulation."

> **`study-dnaa-00-parameter-foundation-embeds`** — "if the cell cycle
> time is longer than C+D, replication initiation should happen only
> once. So running the cell for one generation might not be enough.
> Need to run it for multiple generations to see if steady state has
> been achieved. By steady state, it means quantities (ori number, cell
> mass) per cell is periodic."

## The decision (Eran, 2026-05-29)

> "we decided to use the default succinate parameters instead of
> overriding the C and D periods."

This is a **methodology pivot, not a parameter tweak.** The investigation
abandons the minimal_glycerol + imposed-τ/C/D approach in favor of
ParCa's native succinate condition. Everything downstream of dnaa-00
inherits this.

## Why succinate is the right media

ParCa defaults under the shipped fixture (`models/parca/parca_state.pkl.gz`):

| quantity | value | source |
|---|---|---|
| τ (doubling time) | **82 min** | `sim_data.condition_to_doubling_time['succinate']` |
| C-period | **40 min** | `sim_data.process.replication.c_period` (ParCa default) |
| D-period | **20 min** | `sim_data.process.replication.d_period` (ParCa default) |
| C + D | **60 min** | derived |
| B-period (τ − C − D) | **22 min** | derived — non-overlapping cell cycle ✓ |
| `dry_mass_inc[minimal_succinate]` | 155 fg | `sim_data.expectedDryMassIncreaseDict` |
| `basal_elongation_rate` | 967 nt/s | derived from C-period + replichore length |

The `B > 0` condition means each cell cycle has a window where replication
is NOT in progress — exactly Rashmi's "replication initiation should
happen only once" regime. minimal_glycerol's τ=83-min emergent cycle
collapsed B to zero (replication was continuous), which is why
single-generation runs couldn't show periodic steady-state.

## What this pivot retires / preserves

**Retires:**
- The minimal_glycerol Stage-1 ParCa cache (`out/cache-stage1-glycerol`,
  `out/cache-stage1-heuristic-glycerol`) — kept on disk for historical
  comparison but no longer the baseline.
- The C-period / D-period override knobs in `LoadSimData` + `save_sim_input`
  / `build_condition_cache.py --c-period-min --d-period-min`. Plumbing
  stays (it's used elsewhere), but the dnaa investigation no longer
  exercises it.
- Most of Round 3's τ-gap framing. With C+D < τ, the τ-gap question
  ("why doesn't cell cycle = 150 min?") becomes irrelevant.
- The `decisions_needed[0]` (τ-gap) entry — supersede with a closure
  note pointing here.

**Preserves:**
- **The D-period wiring fix from Round 3.7** (commit `8b5d8cd`,
  `v2ecoli/steps/division.py` + `v2ecoli/composites/_helpers.py`). The
  bug it fixed — Division step ignoring the `divide` flag MarkDPeriod
  sets — was real and remains a fix regardless of which media we run
  on. Under the new succinate methodology, D=20 min still has to be
  enforced for the cycle to make sense.
- All parquet emitter migration work (Round 3.6).
- The dnaa-01..04 study yamls (their per-study verdicts re-validate
  on the new cache; their structure stands).

## Concrete actions

### 1. Build the succinate baseline cache
```
python scripts/build_condition_cache.py \
    --condition stage1-heuristic \
    --media-condition succinate \
    --fixed-media minimal_succinate \
    --cache out/cache-stage1-heuristic-succinate
```
No `--c-period-min` / `--d-period-min` overrides. The cache picks up
ParCa's defaults (C=40, D=20).

### 2. dnaa-00 baseline — multi-generation succinate run
Acceptance criteria (Rashmi):
- Cell cycle is non-overlapping (B-period > 0).
- Replication initiates **once per generation**.
- After ≥2 generations, per-cell observables are **periodic**:
  `n_oric` (ori count), `cell_mass` (mass), DnaA total — same value at
  the same point in each cycle.

Suggested run: `n_steps = 15000` (≈ 3 × τ = 246 min sim time). With
the D-period fix and Stage-1 expression, the first cell should:
- divide at t ≈ B + C + D ≈ 82 min into its own life
- daughter 0 divides at t ≈ 164 min from t=0
- granddaughter divides at t ≈ 246 min from t=0
- by generation 3, observables should be periodic (Rashmi's
  steady-state criterion).

Script: keep `scripts/run_dnaa00_tau_compare.py` for snapshot capture
or write a multi-gen variant if needed (the current script uses
single-cell run-to-end logic; multi-gen requires the multigen runner
machinery, which the existing dnaa-01..04 runners use via
`run_multigen_parquet` from `v2ecoli.library.parquet_run`).

### 3. dnaa-01..04 — re-validate against the new succinate cache
The Round 3.6 parquet-rerun confirmations were against the prior
minimal-glucose / minimal-glycerol caches. Re-execute each runner
with `--cache-dir out/cache-stage1-heuristic-succinate`. Acceptance:
the verdicts (dnaa-02 atp_fraction band failure intrinsic-only,
dnaa-02f variant B' band recovery, dnaa-03 four declarative tests
pass, dnaa-04 mechanism fires per cycle) survive.

### 4. Update study yamls + investigation YAML
- dnaa-00 `setup:` reflects the cache switch. `result:` gets a
  Round 3.8 addendum once the multi-gen run completes.
- investigation.yaml `executive.verdict` documents the pivot.
- `decisions_needed[0]` (the τ-gap) is marked superseded with a
  closure note pointing to this PLAN.md.
- `caveats[]` updated: the Round 3.7 "D-period unenforced" caveat
  stays (because the fix landed), but the Round 3.6 "C-period
  unenforced" caveat goes away entirely — under default succinate
  there's no imposition, no override, no plumbing to break.

### 5. Update PR #59 description
Note the Round 3.8 pivot in the PR body so a reviewer entering cold
sees the latest methodology immediately rather than having to read
back through the Round 3 / 3.5 / 3.6 / 3.7 history.

## Sequencing

1. Save feedback + plan. *(this commit)*
2. investigation.yaml + dnaa-00 study.yaml prose updates per §4.
3. Build the succinate cache per §1 (cheap, <1 min).
4. Multi-gen dnaa-00 run per §2 — verify periodic steady state.
5. Re-validate dnaa-01..04 per §3.
6. Final addendums + PR #59 description.

## What's out of scope

- The "remaining 36.8-min gap to declared τ=150 min" from Round 3.7
  becomes moot. The cell now divides at its ParCa-fit τ=82 min — no
  declaration to match.
- Stage-1 transcription/TE calibration on succinate (Haochen R2.G).
  Different concern from this methodology pivot; flag separately if
  the in-band DnaA claim fails on succinate.
