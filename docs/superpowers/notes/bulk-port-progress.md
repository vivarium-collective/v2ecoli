# Bulk analyses port — progress log

Porting the remaining vEcoli DuckDB/`sim_data` analyses (`origin/ptools_viz`
in `~/code/vivarium-ecoli`) onto v2ecoli's native `Analysis` base.

## Summary counts

- Step 0 (compat shim centralization): **DONE**
- PORTED: 0
- SKIPPED: 0
- BLOCKED: 0
- Remaining: 25

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
- multigeneration/ptools_proteins — TODO
- multigeneration/ptools_rna — TODO
- multigeneration/ptools_rxns — TODO
- multigeneration/replication — TODO
- multigeneration/ribosome_components — TODO
- multigeneration/ribosome_crowding — TODO
- multigeneration/ribosome_production — TODO
- multigeneration/ribosome_usage — TODO
- multigeneration/rna_decay_03_high — TODO

### multiseed
- multiseed/ecocyc_table — TODO
- multiseed/protein_counts_validation — TODO
- multiseed/ptools_proteins — TODO
- multiseed/ptools_rna — TODO
- multiseed/ptools_rxns — TODO
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
