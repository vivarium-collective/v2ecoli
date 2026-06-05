# Basal-condition phenotype — report card

- **Model**: d139b9b
- **Stimulus**: `v2ecoli/configs/basal_phenotype_card.json`
- **Reference status**: populated
- **Generated**: 2026-06-04 21:20

## Overall: PASS (4 ✓ · 0 ≈ · 0 ✗ · 0 –)

### Growth

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Doubling time | 0.8393 h | Welch t-test p ≥ 0.05 (n=17 cells) | p = 1.000 · Δ = +0.0% · d = +0.00 | ✓ within_tol |

### Composition

| Axis | Value | Criterion | Summary | Verdict |
|---|---|---|---|---|
| Protein / dry weight | 0.4366 g/gDW | Welch t-test p ≥ 0.05 (n=20 cells) | p = 1.000 · Δ = +0.0% · d = +0.00 | ✓ within_tol |
| Total RNA / dry weight | 0.1281 g/gDW | Welch t-test p ≥ 0.05 (n=20 cells) | p = 1.000 · Δ = +0.0% · d = +0.00 | ✓ within_tol |
| DNA / dry weight | 0.01833 g/gDW | Welch t-test p ≥ 0.05 (n=20 cells) | p = 1.000 · Δ = +0.0% · d = +0.00 | ✓ within_tol |

## Findings

- Population-stat axes grade by Welch t-test on cell-level values (n=cells, not timepoints). Self-pin => p=1.0; this establishes the baseline. Caveat: a t-test flags DIFFERENCE; 'not significantly different' is not a formal equivalence proof (TOST would be), but it is a sound drift gate for a regression pin.
- 3 of 32 generations did not divide (ran to the 3600 s cap); their over-grown cells are still in the composition average (n=20). A refinement would restrict composition to divided cells; effect is small (fractions are per-dry-weight).

_Meta-tier card. A failure blocks merge; grades only move up. See `docs/meta_report_cards.md`._
