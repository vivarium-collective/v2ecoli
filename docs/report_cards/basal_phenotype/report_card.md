# Basal-condition phenotype — report card

- **Model**: e6f8ea7
- **Stimulus**: 4 seeds x 8 gens, generation_lower_bound=3 -> 20 cells (17 divided); 29/32 generations divided
- **Reference status**: populated
- **Generated**: 2026-06-05 13:40

## Overall: PASS (21 ✓ · 0 ≈ · 0 ✗ · 0 –)

### Physiology

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Doubling time | 0.8393 h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.0% · p = 1.000 · d = +0.00 | ✓ within_tol |
| Cell mass | 1538 fg | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -0.0% · p = 1.000 · d = -0.00 | ✓ within_tol |
| Cell volume | 1.398 fL | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -0.0% · p = 1.000 · d = -0.00 | ✓ within_tol |
| Replication origins (oriC) | 2.251 origins | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.0% · p = 1.000 · d = +0.00 | ✓ within_tol |
| Replication initiation | 34.27 min | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.0% · p = 1.000 · d = +0.00 | ✓ within_tol |
| Replication completion | 32.93 min | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.0% · p = 1.000 · d = +0.00 | ✓ within_tol |

### Composition

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Protein / dry weight | 0.4366 g/gDW | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.0% · p = 1.000 · d = +0.00 | ✓ within_tol |
| Total RNA / dry weight | 0.1281 g/gDW | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.0% · p = 1.000 · d = +0.00 | ✓ within_tol |
| DNA / dry weight | 0.01833 g/gDW | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.0% · p = 1.000 · d = +0.00 | ✓ within_tol |

### Ribosomes

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Total ribosomes | 2.043e+04 ribosomes | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -0.0% · p = 1.000 · d = -0.00 | ✓ within_tol |
| Active fraction | 0.8328 fraction | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -0.0% · p = 1.000 · d = -0.00 | ✓ within_tol |
| Elongation rate | 15.4 aa/s | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.0% · p = 1.000 · d = +0.00 | ✓ within_tol |
| Ribosome production (rRNA init) | 4.356 rRNA/s | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = +0.0% · p = 1.000 · d = +0.00 | ✓ within_tol |

### Exchange fluxes

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Exchange-flux fingerprint | R² 1 | R² ≥ 0.99 on matched fluxes; 0 appeared/disappeared exchanges | R² = 1.0000 · 36 matched · 0 appeared · 0 lost | ✓ within_tol |
| Glucose exchange | -5.145 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -0.0% · p = 1.000 · d = +0.00 | ✓ within_tol |
| O₂ exchange | -0.481 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -0.0% · p = 1.000 · d = +0.00 | ✓ within_tol |
| Ammonium (N source) exchange | -7.017 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -0.0% · p = 1.000 · d = +0.00 | ✓ within_tol |
| CO₂ exchange | 1.708 mmol/gDCW/h | |Δ| < 5% (drift 5%–10%; mismatch >10% & p<0.05) | Δ = -0.0% · p = 1.000 · d = -0.00 | ✓ within_tol |
| Acetate exchange (overflow sentinel) | 0 mmol/gDCW/h | stays inactive (|mean| < 1e-06) | both ≈ 0 (ref +0, meas +0) | ✓ within_tol |

### Gene expression

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Transcriptome (mRNA cistron counts) | R² 1 | log-log R² ≥ 0.99 (3969 genes) | R² = 1.0000 | ✓ within_tol |
| Proteome (monomer counts) | R² 1 | log-log R² ≥ 0.99 (4161 genes) | R² = 1.0000 | ✓ within_tol |

## Findings

- Population-stat axes grade by Welch t-test on cell-level values (n=cells, not timepoints). Self-pin => p=1.0; this establishes the baseline. Caveat: a t-test flags DIFFERENCE; 'not significantly different' is not a formal equivalence proof (TOST would be), but it is a sound drift gate for a regression pin.
- 3 of 32 generations did not divide (ran to the 3600 s cap); their over-grown cells are still in the composition average (n=20). A refinement would restrict composition to divided cells; effect is small (fractions are per-dry-weight).

_Behavioral report card — see docs/report_cards/README.md for the index and how the cards compose._
