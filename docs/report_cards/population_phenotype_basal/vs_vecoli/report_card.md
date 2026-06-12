# Basal-condition population phenotype — v1↔v2 equivalence — report card

- **Model**: 52bbf43
- **Stimulus**: 8 seeds x 16 gens; vEcoli master; fresh ParCa; gen-lb 3
- **Reference status**: populated
- **Generated**: 2026-06-08 12:31

## Overall: MISMATCH (15 ✓ · 2 ≈ · 4 ✗ · 0 –)

> **⚠ Simulations (v2ecoli (v2)):** 11 of 128 generations hit the duration cap without dividing (117/128 divided).

### Physiology

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Doubling time | 0.8438 h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -4.0% · p = 0.005 · d = -0.39 | ✓ within_tol |
| Cell mass | 1562 fg | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.1% · p = 0.938 · d = +0.01 | ✓ within_tol |
| Cell volume | 1.42 fL | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.1% · p = 0.938 · d = +0.01 | ✓ within_tol |
| Replication origins (oriC) | 2.284 origins | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.0% · p = 0.981 · d = +0.00 | ✓ within_tol |
| Replication initiation | 36.76 min | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -5.4% · p = 0.287 · d = -0.15 | ≈ drift |
| Replication completion | 32.19 min | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -1.7% · p = 0.515 · d = -0.09 | ✓ within_tol |

### Composition

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Protein / dry weight | 0.4394 g/gDW | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +1.2% · p = 0.144 · d = +0.20 | ✓ within_tol |
| Total RNA / dry weight | 0.1293 g/gDW | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -0.4% · p = 0.623 · d = -0.07 | ✓ within_tol |
| DNA / dry weight | 0.0182 g/gDW | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.1% · p = 0.781 · d = +0.04 | ✓ within_tol |

### Ribosomes

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Total ribosomes | 2.088e+04 ribosomes | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -1.2% · p = 0.443 · d = -0.11 | ✓ within_tol |
| Active fraction | 0.8344 fraction | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.1% · p = 0.082 · d = +0.24 | ✓ within_tol |
| Elongation rate | 15.71 aa/s | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.7% · p = 0.599 · d = +0.07 | ✓ within_tol |
| Ribosome production (rRNA init) | 4.573 rRNA/s | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +2.5% · p = 0.455 · d = +0.10 | ✓ within_tol |

### Exchange fluxes

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Exchange-flux fingerprint | R² 1 | R² ≥ 0.99 on matched fluxes; no appeared/disappeared ≥ 0.001 | R² = 1.0000 · 37 matched · 0 appeared · 0 lost · 4 sub-floor | ✓ within_tol |
| Glucose exchange | -5.257 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -1.3% · p = 0.396 · d = +0.12 | ✓ within_tol |
| O₂ exchange | -0.4954 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -32.1% · p = 0.000 · d = +0.69 | ✗ mismatch |
| Ammonium (N source) exchange | -7.179 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.3% · p = 0.850 · d = -0.03 | ✓ within_tol |
| CO₂ exchange | 1.747 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -15.4% · p = 0.000 · d = -0.63 | ✗ mismatch |
| Acetate exchange (overflow sentinel) | 2.713e-06 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -76.6% · p = 0.289 · d = -0.14 | ≈ drift |

### Gene expression

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Transcriptome (mRNA cistron counts) | R² 0.9337 | log-log R² ≥ 0.99 (4131 genes) | R² = 0.9337 | ✗ mismatch |
| Proteome (monomer counts) | R² 0.9393 | log-log R² ≥ 0.99 (4165 genes) | R² = 0.9393 | ✗ mismatch |

## Findings

- EQUIVALENCE reference: v2ecoli graded against a vEcoli v1 ensemble (not a self-pin). Welch t-test of v2 cell-level values vs v1 ref_values; r2 for omics; flux_scatter for the exchange fingerprint.
- v1<->v2 share cistron/monomer/flux ordering exactly — vectors align positionally (no ID remapping). Ribosome 30S/50S sliced from v1's positional bulk by index (s30/s50 from v1 sim_data).
- v1 values read by scripts/pin_vecoli_equivalence_reference.py (self-contained cross-impl reader; shared analysis_runner is untouched).
- tolerance bands inherited from the self-pin template; revisit per-axis equivalence margins (delta) — and consider TOST for the formal claim.

_Behavioral report card — see docs/report_cards/README.md for the index and how the cards compose._
