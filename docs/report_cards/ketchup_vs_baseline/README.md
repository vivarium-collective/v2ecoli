# KETCHUP kinetic models ↔ v2ecoli baseline — exchange-flux report cards

A behavioral report card (PR #134 grader) comparing the **central-carbon
exchange-flux phenotype** of two KETCHUP kinetic models
([k-ecoli74, k-ecoli307](https://github.com/vivarium-collective/pbg-ketchup))
against the v2ecoli whole-cell baseline, under glucose-minimal aerobic growth.

## Result

| per 100 glucose | v2ecoli baseline | k-ecoli74 | k-ecoli307 |
|---|---|---|---|
| O₂        | −9.4  | −135 | −136 |
| CO₂       | +33   | +144 | +146 |
| acetate   | 0     | +71  | +71  |
| NH₃       | −136  | −54  | −81  |

Both KETCHUP models **mismatch** the v2ecoli baseline
(k-ecoli74 R²=0.27, k-ecoli307 R²=0.37), and the grader flags **acetate as
"appeared"** — the KETCHUP core-metabolism models predict aerobic acetate
overflow and much heavier respiration per glucose, whereas v2ecoli's baseline
FBA is near-fermentative with no overflow. This is the dominant, qualitative
disagreement the report-card system surfaces.

See `index.html` for the side-by-side summary and `k-ecoli74.html` /
`k-ecoli307.html` for the full graded cards (with the signed flux-scatter plot).

## How it was built

- **Reference (v2ecoli baseline):** the pinned exchange-flux vector from
  `tests/fixtures/population_phenotype_basal_reference.json` (20-seed ensemble
  mean), restricted to the shared central-carbon exchanges and glucose-normalized.
- **Candidate (KETCHUP):** fitted exchange fluxes from `pbg-ketchup`
  (`KetchupEstimator` on k-ecoli74 / k-ecoli307), the model boundary reactions
  mapped to v2ecoli EcoCyc compound IDs and glucose-normalized. Cached in
  `ketchup_exchange.json`.
- **Grading:** the `flux_scatter` criterion (identity-line R² on signed,
  matched fluxes) from `v2ecoli/library/report_card.py` (PR #134), with
  cross-model thresholds (R²≥0.8 good, ≥0.4 moderate).

### Reproduce

```bash
# 1. regenerate KETCHUP exchange fluxes (needs pbg-ketchup + IPOPT):
#    see https://github.com/vivarium-collective/pbg-ketchup
#    -> writes docs/report_cards/ketchup_vs_baseline/ketchup_exchange.json
# 2. build the cards:
PYTHONPATH=. python scripts/ketchup_baseline_report_cards.py
```

## Caveats

- KETCHUP fluxes are at a bounded fit (status `maxIterations`); the two models
  agree closely with each other (shared K-FIT data lineage).
- The two models describe different conditions/objectives (13C-MFA-style core
  kinetics vs. whole-cell FBA), so the mismatch is expected and informative,
  not a defect — it quantifies where a core kinetic model and the whole-cell
  model diverge on byproduct partitioning.
- Comparison is over shared central-carbon exchanges only (KETCHUP is a
  core-carbon model; it does not exchange the amino acids/ions v2ecoli does).
