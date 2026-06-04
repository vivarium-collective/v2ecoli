# Basal-condition phenotype — report card

- **Model**: b162243
- **Stimulus**: `v2ecoli/configs/basal_phenotype_card.json`
- **Ensemble**: 20 cells after burn-in (generation_lower_bound=3)
- **Reference status**: populated
- **Generated**: 2026-06-04 15:20

## Overall: PASS ✅

| Axis | Measured (mean) | spread | Reference | Tol | Verdict |
|---|---|---|---|---|---|
| Growth — doubling time (s) | 3021 | ±312.6 (CV 0.1035) | 3021 | ±10% | ✅ pass |
| Composition — protein / dry weight | 0.4366 | ±0.02476 (CV 0.05671) | 0.4366 | ±5% | ✅ pass |
| Composition — RNA / dry weight | 0.1026 | ±0.003701 (CV 0.03607) | 0.1026 | ±5% | ✅ pass |
| Composition — DNA / dry weight | 0.01833 | ±0.0007343 (CV 0.04007) | 0.01833 | ±5% | ✅ pass |

## Findings

- RNA == rRna for v1 (analysis_runner._MASS_COLS pulls rRna only; ~80-85% of total RNA). Extend to total RNA (tRna+mRna) as a follow-up.
- 3 of 32 generations did not divide (ran to the 3600 s cap); their over-grown cells are still included in the composition average (n=20). A v1 refinement would restrict composition to divided cells; the effect is small (composition fractions are per-dry-weight and robust).

_Meta-tier card. A failure blocks merge; grades only move up. See `docs/meta_report_cards.md`._
