# Bulk analyses port — progress log

Porting the remaining vEcoli DuckDB/`sim_data` analyses (`origin/ptools_viz`
in `~/code/vivarium-ecoli`) onto v2ecoli's native `Analysis` base.

## Summary counts

- Step 0 (compat shim centralization): **DONE**
- PORTED: 18
- SKIPPED: 0
- BLOCKED: 7
- Remaining: 0

**All 25 in-scope targets processed; 3 previously-BLOCKED multiseed ptools now PORTED.**

## Infrastructure added
- `v2ecoli/workflow/analyses/_helpers.py` — native `read_stacked_columns`
  (aliases `global_time` → `time`), `num_cells`, `skip_n_gens`,
  `available_columns`, `bulk_field_ids`, `bulk_count_idx_expr` (parquet-order
  bulk indexing, fail-loud), `cumulative_time_history` (absolute time axis for
  multigeneration), `collapse_cross_seed` (element-wise list-column aggregation
  for multiseed, fail-loud on shape mismatch), `chart_to_html` (Altair view),
  and re-exports `named_idx` / `ndidx_to_duckdb_expr`.
  `tests/test_bulk_analyses.py` registration tests + unit tests for
  `collapse_cross_seed`.
- Each ptools base class (`PtoolsRna`, `PtoolsRxns`, `PtoolsProteins`) now
  exposes `_do_read_outputs()` so the `_MultiseedMixin` can override it without
  duplicating `analyze()` logic.

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
- single/blame — BLOCKED (upstream plot() is `raise NotImplementedError`; requires --log_updates data + experiment topology, not present in DuckDB output)
- single/mass_fraction_summary — PORTED (name `mass_fraction_summary_view`; bare name taken by record-based AnalysisStep)

### multigeneration
- multigeneration/new_gene_counts — PORTED (orderings from sim_data; informational view when no new genes present)
- multigeneration/ptools_proteins — PORTED (multiscale module; abs-time wrapper) — name `ptools_proteins_multigeneration`
- multigeneration/ptools_rna — PORTED (multiscale module; abs-time wrapper) — name `ptools_rna_multigeneration`
- multigeneration/ptools_rxns — PORTED (multiscale module; abs-time wrapper) — name `ptools_rxns_multigeneration`
- multigeneration/replication — PORTED (abs-time axis; critical-mass panels absent — columns not emitted by v2ecoli)
- multigeneration/ribosome_components — PORTED (bulk→parquet-order; active_ribosome shim; monomer order from sim_data)
- multigeneration/ribosome_crowding — BLOCKED (requires listeners__ribosome_data__target_prob_translation_per_transcript and __actual_prob_translation_per_transcript; both absent in v2ecoli parquet)
- multigeneration/ribosome_production — PORTED (bulk→parquet-order; active_ribosome shim; agent_id filter dropped)
- multigeneration/ribosome_usage — PORTED (did_initialize absent → activation panels skipped; decimals cast to float)
- multigeneration/rna_decay_03_high — BLOCKED (requires listeners__rna_degradation_listener__count_RNA_degraded_per_cistron; entire rna_degradation_listener group absent in v2ecoli)

### multiseed
- multiseed/ecocyc_table — BLOCKED (needs validation_data.protein.schmidt2015Data for the minimal-media branch (validation_data is None) and active_ribosome shim; very large multi-TSV; deferred)
- multiseed/protein_counts_validation — BLOCKED (entire analysis compares to validation_data.protein.wisniewski2014Data/schmidt2015Data; validation_data is None in the Analysis framework)
- multiseed/ptools_proteins — PORTED (`ptools_proteins_multiseed`; `_MultiseedMixin` + `collapse_cross_seed`)
- multiseed/ptools_rna — PORTED (`ptools_rna_multiseed`; `_MultiseedMixin` + `collapse_cross_seed`)
- multiseed/ptools_rxns — PORTED (`ptools_rxns_multiseed`; `_MultiseedMixin` + `collapse_cross_seed`)

  **Implementation:** each base class (`PtoolsRna/Rxns/Proteins`) gained a
  `_do_read_outputs()` method that delegates to the module-level `read_outputs`.
  `_MultiseedMixin` overrides `_do_read_outputs()` to fetch raw per-seed rows
  (bypassing `groupby.sum()`) and apply `collapse_cross_seed()`:
  `bulk__id` first-preserved (length-assert), list/array columns element-wise
  numpy-summed (shape-assert, fail-loud on mismatch), scalar columns plain-summed.
  Smoke-tested on the 2-seed parquet (`out/compare_harness/v2_sim/parquet/
  two_generations/`): element-wise match `s0 + s1 == collapsed` verified
  exactly; `active_ribosome` scalar sum verified (12801 + 12802 = 25603);
  array length preserved at 16321 (NOT 32642 concatenation artifact).
  vEcoli uses SUM (not mean) across seeds — v2ecoli matches this exactly.
- multiseed/ribosome_spacing — BLOCKED (requires listeners__ribosome_data__ribosome_init_event_per_monomer; absent in v2ecoli parquet)
- multiseed/subgenerational_expression_table — PORTED (field_metadata orderings substituted from sim_data; ignore_first_n_gens defaults to 0)

### multivariant
- multivariant/average_monomer_counts — PORTED (sim_data orderings; needs >=2 variants — informational note with single-variant reference data)
- multivariant/cell_mass — PORTED
- multivariant/doubling_time_hist — PORTED (skip_n_gens defaults to 0)
- multivariant/doubling_time_line — PORTED (death-point overlay omitted; no success_sql)
- multivariant/dummy — PORTED (column-drift canary re-baselined to v2ecoli's 176-col schema)
- multivariant/new_gene_translation_efficiency_heatmaps — BLOCKED (requires a new-gene translation-efficiency variant grid (exp_trl_eff) AND multiple absent columns: did_initialize, target/actual_prob_translation_per_transcript, ribosome_init_event_per_monomer, max_p, tu/mRNA_is_overcrowded)

### skipped scales/dirs (different signature or unsupported scale)
- antibiotics_colony/* — SKIPPED (different signature / colony scale)
- causality_network/* — SKIPPED (network builder, not a plot() analysis)
- colony/* — SKIPPED (colony scale, not in ANALYSIS_SCALES)
- multiexperiment/* — SKIPPED (scale not in ANALYSIS_SCALES)
