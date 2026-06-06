# Basal-condition population phenotype — v1↔v2 equivalence (Physiology + Composition) — report card

- **Model**: b01250e
- **Stimulus**: 2 seeds x 2 gens (smoke); vEcoli master; ParCa fast
- **Reference status**: populated
- **Generated**: 2026-06-05 21:52

## Overall: MISMATCH (3 ✓ · 4 ≈ · 2 ✗ · 0 –)

> **⚠ Stationarity:** `Doubling time` (gen 51%, ρ=-0.63) show generation-structured drift — the ensemble may not be at steady balanced growth across generations (burn-in insufficient or a generational instability). Diagnostic only; does not affect grades.

### Physiology

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Doubling time | 0.8393 h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +11.3% · p = 0.062 · d = +0.99 | ≈ drift |
| Cell mass | 1538 fg | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -9.9% · p = 0.001 · d = -1.12 | ≈ drift |
| Cell volume | 1.398 fL | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -9.9% · p = 0.001 · d = -1.12 | ≈ drift |
| Replication origins (oriC) | 2.251 origins | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -11.1% · p = 0.002 · d = -1.19 | ✗ mismatch |
| Replication initiation | 34.27 min | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +3.0% · p = 0.814 · d = +0.07 | ✓ within_tol |
| Replication completion | 32.93 min | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +30.3% · p = 0.015 · d = +0.98 | ✗ mismatch |

### Composition

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Protein / dry weight | 0.4366 g/gDW | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -5.8% · p = 0.050 · d = -1.09 | ≈ drift |
| Total RNA / dry weight | 0.1281 g/gDW | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -4.4% · p = 0.008 · d = -1.19 | ✓ within_tol |
| DNA / dry weight | 0.01833 g/gDW | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +3.3% · p = 0.003 · d = +0.84 | ✓ within_tol |

## Findings

- EQUIVALENCE reference: v2ecoli graded against a vEcoli v1 ensemble (not a self-pin). Welch t-test of v2 cell-level values vs v1 ref_values.
- Phase-1 scope: Physiology + Composition. Ribosomes/fluxes/omics omitted (vEcoli emits bulk positionally, not as bulk__id/bulk__count; needs the bulk-index adapter).
- v1 values read by scripts/pin_vecoli_equivalence_reference.py (self-contained cross-impl reader; shared analysis_runner is untouched).
- tolerance bands (within_pct/mismatch_pct) inherited from the self-pin template; revisit per-axis equivalence margins (delta) for the real run.

_Behavioral report card — see docs/report_cards/README.md for the index and how the cards compose._
