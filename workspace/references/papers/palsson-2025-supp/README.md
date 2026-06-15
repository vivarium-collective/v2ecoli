# Palsson 2025 (Beulig et al.) — supplementary data

Time-series CSVs cloned (with attribution) from
[`febedtu/hd_ecoli` `data/fermentation_data/`](https://github.com/febedtu/hd_ecoli/tree/main/data/fermentation_data)
on 2026-05-22 (MIT licence — see upstream `LICENSE`).

These are the **digitized fermentation trajectories** referenced from
`studies/mbp-05-palsson-benchmark/study.yaml` (`req-1` is now fully
resolved — no further digitization needed).

## Column format

Each per-observable CSV has rows indexed by `fed-batch time [h]` (negative
values = pre-fed-batch BATCH phase; t ≥ 0 = fed-batch phase) and columns
named by reactor ID (`DDB_PD_XXX_AMBR_RYY`). Map reactor ID → strain /
study via `experiment_mapping.csv`. `process_summary_interpol.csv` is
the time-aligned interpolated long-format combining all observables.

| File | Observable | Units |
|---|---|---|
| `reactor_OD_data.csv` | OD600 | — (apply paper's 0.34 g/L per OD for gDW) |
| `reactor_glucose_data.csv` | D-glucose concentration | mmol/L |
| `reactor_gluc_upt_data.csv` | Glucose uptake rate | — |
| `reactor_gluc_cons_data.csv` | Cumulative glucose consumed | cmol |
| `reactor_OTR_data.csv` | Oxygen transfer rate | mol/h |
| `reactor_CTR_data.csv` | CO2 transfer rate | mol/h |
| `reactor_feed_rate_data.csv` | Feed rate (the fed-batch profile) | L/h |
| `reactor_base_data.csv` | Base addition (pH control) | — |
| `reactor_stirrer_data.csv` | Impeller speed | RPM |
| `reactor_ethanol_data.csv`, `reactor_acetate_data.csv` (etc.) | Organic-acid byproducts | mmol/L |
| `reactor_melatonin_data.csv` | Melatonin (MEL strain product) | — |
| `reactor_tryptophan_data.csv` | Tryptophan (TRP strain product) | — |
| `experiment_mapping.csv` | precise_ID ↔ sample_id ↔ reactor_id ↔ strain ↔ time | — |
| `process_summary_interpol.csv` | Time-aligned interpolated long-format (all observables) | mixed |

## Notes for mbp-05

- **OD600 ↔ gDW conversion** in the paper: **0.34 g/L per OD600** (slightly
  different from the textbook 0.33 we listed in `references/claims.yaml`
  under `od-to-gdw-conversion` — refit per-paper if needed in the spec PR).
- **Batch-phase prefix** for mbp-05's batch-only scope: rows with
  `fed-batch time [h] < 0`. Length varies by reactor.
- **WT strain** for the in-scope comparison: filter `experiment_mapping.csv`
  for the WT BW25113 reactor IDs.
- **Acetate column**: not in this dump under that exact name — check
  `reactor_acetate_data.csv` (in the upstream tree) or derive from the
  organic-acid mass-balance via `reactor_CTR_data.csv` minus the named
  byproducts.

## Attribution

Upstream repository:
- Beulig F, ..., Palsson B. (2025). *Trade-off between resistance and persistence in high cell density cultures.* mSystems 10(7). https://doi.org/10.1128/msystems.00323-25
- Code/data repo: https://github.com/febedtu/hd_ecoli (MIT)
- Bib key: `@Beulig2025mSystems` in `references/papers.bib`
