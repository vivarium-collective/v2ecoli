# Basal-condition population phenotype — v1↔v2 equivalence — report card

- **Model**: bd2123d2
- **Stimulus**: 
- **Reference status**: populated
- **Generated**: 2026-06-12 22:40

## Overall: MISMATCH (12 ✓ · 6 ≈ · 3 ✗ · 0 –)

> **⚠ Simulations (v2ecoli (v2)):** 24 of 256 generations hit the duration cap without dividing (232/256 divided).

### Physiology

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Doubling time | 0.8422 h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -2.2% · p = 0.014 · d = -0.26 | ✓ within_tol |
| Cell mass | 1507 fg | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -4.4% · p = 0.000 · d = -0.65 | ✓ within_tol |
| Cell volume | 1.37 fL | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -4.4% · p = 0.000 · d = -0.65 | ✓ within_tol |
| Replication origins (oriC) | 2.163 origins | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -6.3% · p = 0.000 · d = -0.81 | ≈ drift |
| Replication initiation | 38.28 min | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -7.2% · p = 0.030 · d = -0.23 | ≈ drift |
| Replication completion | 33.34 min | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +4.7% · p = 0.004 · d = +0.30 | ✓ within_tol |

### Composition

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Protein / dry weight | 0.4391 g/gDW | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +3.3% · p = 0.000 · d = +0.61 | ✓ within_tol |
| Total RNA / dry weight | 0.1284 g/gDW | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -1.5% · p = 0.000 · d = -0.38 | ✓ within_tol |
| DNA / dry weight | 0.01829 g/gDW | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +1.0% · p = 0.000 · d = +0.36 | ✓ within_tol |

### Ribosomes

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Total ribosomes | 2.016e+04 ribosomes | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -5.8% · p = 0.000 · d = -0.62 | ≈ drift |
| Active fraction | 0.8334 fraction | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.1% · p = 0.141 · d = +0.15 | ✓ within_tol |
| Elongation rate | 15.47 aa/s | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.2% · p = 0.800 · d = +0.03 | ✓ within_tol |
| Ribosome production (rRNA init) | 4.18 rRNA/s | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -8.7% · p = 0.000 · d = -0.42 | ≈ drift |

### Exchange fluxes

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Exchange-flux fingerprint | R² 0.9989 | R² ≥ 0.99 on matched fluxes; no appeared/disappeared ≥ 0.001 | R² = 0.9989 · 40 matched · 0 appeared · 0 lost · 6 sub-floor | ✓ within_tol |
| Glucose exchange | -5.171 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -4.4% · p = 0.000 · d = +0.41 | ✓ within_tol |
| O₂ exchange | -0.4524 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -40.4% · p = 0.000 · d = +0.89 | ✗ mismatch |
| Ammonium (N source) exchange | -7.063 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -2.4% · p = 0.020 · d = +0.24 | ✓ within_tol |
| CO₂ exchange | 1.705 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -20.3% · p = 0.000 · d = -0.87 | ✗ mismatch |
| Acetate exchange (overflow sentinel) | 2.223e-06 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -48.3% · p = 0.460 · d = -0.08 | ≈ drift |

### Gene expression

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Transcriptome (mRNA cistron counts) | R² 0.9461 | log-log R² ≥ 0.99 (4156 genes) | R² = 0.9461 | ✗ mismatch |
| Proteome (monomer counts) | R² 0.9613 | log-log R² ≥ 0.99 (4170 genes) | R² = 0.9613 | ≈ drift |

## Findings

- EQUIVALENCE reference: v2ecoli graded against a vEcoli v1 ensemble (not a self-pin). Welch t-test of v2 cell-level values vs v1 ref_values; r2 for omics; flux_scatter for the exchange fingerprint.
- v1<->v2 share cistron/monomer/flux ordering exactly — vectors align positionally (no ID remapping). Ribosome 30S/50S sliced from v1's positional bulk by index (s30/s50 from v1 sim_data).
- v1 values read by scripts/pin_vecoli_equivalence_reference.py (self-contained cross-impl reader; shared analysis_runner is untouched).
- tolerance bands inherited from the self-pin template; revisit per-axis equivalence margins (delta) — and consider TOST for the formal claim.

_Behavioral report card — see docs/report_cards/README.md for the index and how the cards compose._
