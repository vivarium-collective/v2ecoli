# PDMP-Phase-1 real W₂ run report — 2026-06-11

Factual run log for the Phase-0 reference ensemble and the Phase-1
multi-replicate Wasserstein-2 comparison that were previously claimed but
never committed. Generated on the Mac mini (darwin, 12 cores, 64 GB),
branch `v2ecoli-pdmp`, worktree `v2e-pdmp-ensemble`. **No study status was
advanced; this is a run log, not an acceptance.**

## A. Phase-0 reference ensemble (pdmp-00) — 3 conditions × N=32

| script | `scripts/run_phase0_trajectory_ensemble.py --n-seeds 32 --n-steps 600 --stride 5` |
|---|---|
| composite | `baseline` (v2ecoli kFBA Metabolism) |
| N | **32** seeds/condition (master seeds 0–31) — honest reduced ensemble; the canonical target is N=64/condition. **Not** N=64. |
| duration | 600 model-seconds, snapshot every 5 s (121 timepoints/seed) |

Endpoint stats at t=600 s (all 32/32 seeds succeeded each):

| condition (ParCa cache) | doubling | wall | cell_mass (fg) | dry_mass (fg) | ATP[c] |
|---|---|---|---|---|---|
| M9-glucose (`out/cache`) | 44 min | 295 s | 1461.1 ± 6.9 | 438.6 ± 2.1 | 7.70e6 |
| M9-acetate (`out/cache-acetate`) | 136 min | 259 s | 374.2 ± 4.8 | 112.3 ± 1.5 | 1.97e6 |
| M9-glucose+aa (`out/cache-with_aa`) | 25 min | 1796 s | 2958.2 ± 2.7 | 887.6 ± 0.8 | 1.56e7 |

Endpoint mass is ordered by doubling time as expected (25 < 44 < 136 min).
Glucose + acetate ran via the Ray seed fan-out; **with_aa ran sequentially**
because the Ray parallel path reliably hit a parquet-emitter `makedirs` race
on the fixed default run path (serializing the builds avoids it).

Bug fixed this run: mass listener values are pint-femtogram Quantities; the
prior script `float()`'d them directly, which raised and silently dropped
every cell_mass/dry_mass snapshot to `None`. Fixed with a `_to_float()`
unit-strip helper. Raw stores live under
`.pbg/runs/phase0-traj{,-acetate,-with_aa}/seed_*` (gitignored); the durable
committed artifacts are 16 regenerated pdmp-00 figures + the 3-condition
`pdmp-00-characterization/phase0_ensemble_provenance.json`.

## B. Phase-1 multi-replicate Wasserstein-2 (pdmp-01)

| field | value |
|---|---|
| script | `scripts/compare_pdmp_ensemble_vs_phase0.py --n-replicates 12 --duration 600 --sample-every 5 --flux-source consumption_matched` |
| PDMP composite | `millard_pdmp_baseline` (+ref-growth, consumption_matched) |
| N replicates | **12** (distinct master seeds 1000–1011), 12/12 usable |
| reference | the N=32 Phase-0 M9-glucose ensemble above |
| W₂ method | exact 1-D 2-Wasserstein via inverse-CDF integral (W₁/scipy reported for cross-ref); percentile bootstrap 95% CI |
| per-replicate wall | build ≈ 4.1 s, run ≈ 32.4 s; ensemble wall 437.7 s (sequential) |

### Real endpoint W₂ (t = 600 s)

| observable | W₂ (fg) | W₂ 95% CI | W₁ (fg) | σ Phase-0 | **W₂/σ** | PDMP mean | Phase-0 mean |
|---|---|---|---|---|---|---|---|
| cell_mass | **20.81** | [18.49, 23.16] | 19.85 | 6.91 | **3.01** | 1480.97 | 1461.12 |
| dry_mass  | **1.80**  | [1.31, 2.41]   | 1.51  | 2.08 | **0.86** | 437.68  | 438.61 |

Per-timepoint W₂ peaks: cell_mass 24.95 fg, dry_mass 1.80 fg.

### Gate verdict — **NOT met**

Mechanical gate (endpoint W₂ ≤ 2·σ_Phase-0 for **both** observables):
- dry_mass: W₂/σ = 0.86 → within ±σ ✓
- cell_mass: W₂/σ = 3.01 → ~3σ high ✗

Because the gate requires both, it is **NOT met**.

### Honest caveats

1. **Degenerate LQR on all 12 replicates.** Every replicate emitted the
   `consumption_matched` LQR Riccati `zero gain` / degenerate warning (the
   known issue). The driver therefore injects essentially no stochastic
   control, so the PDMP "ensemble" is near-deterministic: per-replicate
   σ ≈ 0.47 fg vs the Phase-0 6.9 fg (cell) / 2.08 fg (dry). The W₂ is
   dominated by a ~20 fg constant mean offset in cell_mass, not by
   distribution overlap. A non-degenerate gain is required before this W₂
   can be read as a genuine distribution-vs-distribution comparison.
2. **Much smaller gap than the prior figure implied.** The earlier
   single-replicate z-score figure cited "-18 to -600 σ"; the real
   multi-replicate W₂ shows cell_mass ≈ 3σ and dry_mass < 1σ. The gap is far
   smaller than that figure suggested — but still fails the ±σ gate.
3. **Reduced N, single PDMP condition.** The W₂ gate is M9-glucose only;
   PDMP N=12, Phase-0 N=32. The Phase-0 reference itself now spans 3
   conditions (§A), but the PDMP-vs-Phase-0 W₂ was run for M9-glucose. Not
   the canonical N=64.

## C. Scope / what was NOT done

- Phase-0 now covers all 3 canonical conditions at **N=32** each (not the
  canonical N=64). The PDMP-vs-Phase-0 W₂ (§B) was computed for M9-glucose
  only — acetate/with_aa PDMP-vs-Phase-0 W₂ was not run.
- The consumption_matched LQR is degenerate (zero gain); the W₂ gate cannot
  be read as a genuine distribution comparison until a non-degenerate gain
  lands.
- **No study was marked Accepted/complete.** Confidence/status on pdmp-00 and
  pdmp-01 are unchanged; only factual run logs, provenance, regenerated
  figures, and the real W₂ were added.

Raw artifacts (gitignored): `.pbg/runs/pdmp-ensemble-vs-phase0/` and
`.pbg/runs/phase0-traj{,-acetate,-with_aa}/`. Committed: 16 pdmp-00 figures,
the regenerated `reports/figures/pdmp-01/pdmp_vs_phase0.html`, the
3-condition provenance JSON, and this report.
