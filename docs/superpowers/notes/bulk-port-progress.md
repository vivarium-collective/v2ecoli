# Bulk analyses port — progress log

Porting the remaining vEcoli DuckDB/`sim_data` analyses (`origin/ptools_viz`
in `~/code/vivarium-ecoli`) onto v2ecoli's native `Analysis` base.

## Summary counts

- Step 0 (compat shim centralization): **DONE**
- PORTED: 3
- SKIPPED: 0
- BLOCKED: 3
- Remaining: 19

## Infrastructure added
- `v2ecoli/workflow/analyses/_helpers.py` — native `read_stacked_columns`
  (aliases `global_time` → `time`), `num_cells`, `skip_n_gens`,
  `available_columns`, `bulk_field_ids`, `bulk_count_idx_expr` (parquet-order
  bulk indexing, fail-loud), `cumulative_time_history` (absolute time axis for
  multigeneration), `chart_to_html` (Altair view), and re-exports `named_idx` /
  `ndidx_to_duckdb_expr`. `tests/test_bulk_analyses.py` registration tests.

## Step 0 — centralize wholecell↔matplotlib-3.10 compat

`v2ecoli/workflow/analyses/_wholecell_compat.py` with idempotent `apply()`.
`mass_fraction_voronoi.py` now calls it instead of inline monkeypatching.
`pytest tests/test_view_analyses.py` green. Commit: (see git log)

## Target list & status

Legend: PORTED (sha) / SKIPPED (reason) / BLOCKED (reason) / TODO

### already done before this effort (skip)
- single/ptools_rna ✓ (pre-existing)
- single/ptools_rxns ✓ (pre-existing)
- single/ptools_proteins ✓ (pre-existing)
- single/mass_fraction_voronoi ✓ (pre-existing; refactored in Step 0)
- multiseed/centralCarbonMetabolismScatter ✓ (pre-existing)

### single
- single/blame — TODO
- single/mass_fraction_summary — TODO

### multigeneration
- multigeneration/new_gene_counts — TODO
- multigeneration/ptools_proteins — PORTED (multiscale module; abs-time wrapper) — name `ptools_proteins_multigeneration`
- multigeneration/ptools_rna — PORTED (multiscale module; abs-time wrapper) — name `ptools_rna_multigeneration`
- multigeneration/ptools_rxns — PORTED (multiscale module; abs-time wrapper) — name `ptools_rxns_multigeneration`
- multigeneration/replication — TODO
- multigeneration/ribosome_components — TODO
- multigeneration/ribosome_crowding — TODO
- multigeneration/ribosome_production — TODO
- multigeneration/ribosome_usage — TODO
- multigeneration/rna_decay_03_high — TODO

### multiseed
- multiseed/ecocyc_table — TODO
- multiseed/protein_counts_validation — TODO
- multiseed/ptools_proteins — BLOCKED (cross-seed list aggregation; see below)
- multiseed/ptools_rna — BLOCKED (cross-seed list aggregation; see below)
- multiseed/ptools_rxns — BLOCKED (cross-seed list aggregation; see below)

  **BLOCKED reason (multiseed ptools ×3):** at multiseed scale multiple seeds
  share each `(generation, time)`, so vEcoli's `read_outputs` sums the list
  columns element-wise across seeds (its `time` is absolute and list columns are
  ndarrays). v2ecoli's pandas `groupby("time").sum()` over the per-row
  `bulk__id` (string lists) and `bulk__count` / flux (python lists) columns
  *concatenates* instead of element-wise adding, so the single-scale read path
  cannot be reused. A faithful port needs a dedicated cross-seed read
  (`first(bulk__id)` + element-wise ndarray sum of count/flux columns). Deferred;
  revisit after the distinct analyses.
- multiseed/ribosome_spacing — TODO
- multiseed/subgenerational_expression_table — TODO

### multivariant
- multivariant/average_monomer_counts — TODO
- multivariant/cell_mass — TODO
- multivariant/doubling_time_hist — TODO
- multivariant/doubling_time_line — TODO
- multivariant/dummy — TODO
- multivariant/new_gene_translation_efficiency_heatmaps — TODO

### skipped scales/dirs (different signature or unsupported scale)
- antibiotics_colony/* — SKIPPED (different signature / colony scale)
- causality_network/* — SKIPPED (network builder, not a plot() analysis)
- colony/* — SKIPPED (colony scale, not in ANALYSIS_SCALES)
- multiexperiment/* — SKIPPED (scale not in ANALYSIS_SCALES)
