# v2ecoli Native Analyses — Status & Roadmap

**Date:** 2026-06-09
**PRs:** framework + 5 ports in `main` (orig. PR #144, closed/superseded); bulk ports in **PR #152**.

This document records what the native-analyses effort delivered, what was deliberately left out, and the roadmap — in particular for the original goal of **offering ptools as a visualization in the v2ecoli dashboard**.

---

## TL;DR

vEcoli's DuckDB/`sim_data` analyses now run **natively inside v2ecoli** as process-bigraph `Analysis` steps — no vEcoli `Analysis`/`plot()` wrapper, no sms-api/HPC. **24 analyses are ported and tested.** The **ptools omics data pipeline works end-to-end** (v2ecoli sim → EcoCyc-frame-ID × timepoint TSVs in the exact Pathway-Tools format). The **ptools _visualization_ is not built yet** (the ports emit data, not a rendered map); that's the main roadmap item.

---

## What works

### The `Analysis` abstraction (in `main`)
- `Analysis(V2Step)` — sibling of the record-based `AnalysisStep`; declares DuckDB `conn` + scale-scoped `history_sql` + ParCa `sim_data` input ports; emits `{view: html, data: map}`. Shared `ANALYSIS_REGISTRY`.
- Runner (`workflow/analysis_runner.py`) provisions one DuckDB connection + the paired `sim_data` per run, builds a per-scale `history_sql`, writes `data → analysis.json` and `view → <sweep>/viz/*.html`.
- Dashboard already surfaces `<study>/viz/*.html`; a server-side picker edit to list `Analysis` classes is **pending** (see roadmap).

### 24 ported analyses (`Analysis`, DuckDB/`sim_data`)
| Scale | Analyses |
|---|---|
| single (5) | `ptools_rna`, `ptools_rxns`, `ptools_proteins`, `mass_fraction_voronoi`, `mass_fraction_summary_view` |
| multigeneration (8) | `ptools_{rna,rxns,proteins}_multigeneration`, `new_gene_counts`, `replication`, `ribosome_components`, `ribosome_production`, `ribosome_usage` |
| multiseed (6) | `ptools_{rna,rxns,proteins}_multiseed`, `central_carbon_metabolism_scatter`, `protein_counts_validation`, `subgenerational_expression_table` |
| multivariant (5) | `average_monomer_counts`, `cell_mass`, `doubling_time_hist`, `doubling_time_line`, `dummy` |

(Plus the 5 pre-existing record-based `AnalysisStep`s: `mass_fraction_summary`, `daughter_mass_symmetry`, `mass_growth_across_generations`, `doubling_time_distribution`, `metric_across_variants`.)

Output types: the **ptools** modules emit **data** (frame-ID × timepoint TSV); the **plot/table** analyses emit a **rendered HTML `view`** (matplotlib→SVG or Altair/Vega-Lite).

### Supporting infrastructure
- **`_shims.py`** — bridges v2ecoli's parquet schema to vEcoli's: `bulk__id`/`bulk__count` → reindexed bulk matrix; `active_ribosome`/`oriC`/`active_RNAP` derived columns.
- **`_helpers.collapse_cross_seed`** — multiseed analyses sum list columns element-wise across seeds (v2ecoli's pandas `groupby.sum` would concatenate).
- **`_wholecell_compat.py`** — one shim for the vendored `wholecell` plotting utils' matplotlib-3.10 incompatibilities (Voronoi etc.).
- **Validation data** — Schmidt 2015 / Wisniewski 2014 copied from vEcoli into `v2ecoli/validation/ecoli/flat/`; minimal loader `v2ecoli/library/validation_data.py`; `resolve_validation_data()` wired into the runner.

### Verified correctness (highlights)
- `ptools_rna` output matches the sms-api oracle format (`EG#####` frame IDs; EG10002 anchored at 1.0).
- Reaction analyses align FBA fluxes **1:1** with `base_reaction_ids` and **assert** on mismatch (no silent truncation — caught a real ~2687-reaction mislabel in review).
- Cross-seed sums verified exact against a real 2-seed sweep (e.g. active_ribosome 12801+12802=25603; array length stays 16321, no concatenation artifact).
- `protein_counts_validation`: log10 Pearson r = **0.74** (Schmidt) / **0.61** (Wisniewski) vs proteomics — a real, sensible validation.
- **61 tests** green across the analysis suite.

### Critical conventions (don't trip on these)
- **sim_data ↔ parquet pairing:** an analysis must use the `sim_data` the sweep ran with. The compare_harness sweep pairs with `out/workflow/simData.cPickle` (2820 `base_reaction_ids`), **not** `out/kb/simData.cPickle` (2821). `resolve_sim_data` no longer matches `kb/`.
- **Fail loud:** never truncate/pad/concatenate where you mean to align or sum; raise `ValueError` on shape/length mismatch.
- Run via `.venv/bin/python` / `.venv/bin/pytest` (bare `python` lacks `unum`).

---

## Not done (deliberate)

- **Emit-trim analyses — descoped (maintainer: not needed):** `ribosome_crowding`, `rna_decay_03_high`, `ribosome_spacing` (+ `new_gene_translation_efficiency_heatmaps`, which also needs an `exp_trl_eff` variant grid). Their listeners (`rna_degradation_listener.count_RNA_degraded_per_cistron`, `ribosome_data.{target,actual}_prob_translation_per_transcript`, `ribosome_init_event_per_monomer`) **are computed** by the processes but **not emitted** — v2ecoli's feature-based emit (`composites/baseline.py:183`, `feat['listeners']`) was deliberately trimmed for clutter (176 cols vs vEcoli's 231). Unblocking = re-declare those listeners in the emit features + re-run a sweep.
- **`ecocyc_table`** (multiseed) — deferred for size (large multi-TSV); its data blockers (validation_data + active_ribosome shim) are now resolved, so it's portable when wanted.
- **`blame`** (single) — permanent skip: vEcoli's upstream `plot()` is `raise NotImplementedError`.

---

## Roadmap

### A. ptools as a dashboard visualization (the original goal)
The ptools data pipeline is done; the *visualization* is not. In increasing depth:
1. **Tier 1 — native render.** Give the ptools ports a `view` (Plotly/Altair heatmap of the frame-ID × timepoint matrix), so they render in the dashboard's Visualizations tab instead of returning raw TSV.
2. **Tier 2 — BioCyc annotation.** Resolve `EG#####` frame IDs to readable reaction/compound/pathway names (vEcoli's `biocyc_service` queries `websvc.biocyc.org`; or use the reconstruction flat files offline). Lets the viz group/label by pathway.
3. **Tier 3 — Pathway Tools Omics Viewer.** Feed the TSVs to the licensed `sms-ptools` Pathway Tools server (ports 1555/5008) and embed the painted EcoCyc cellular-overview map — the signature "ptools visualization." Coupled to the licensed image; biggest lift.

### B. Dashboard surfacing
- Redo + commit the `vivarium-dashboard` `server.py` picker edit (`_list_visualization_classes` lists `Analysis` classes). The earlier edit was lost when that repo's WIP branch moved; it's inert until v2ecoli is importable in the serving venv.

### C. Other follow-ups
- **`validation_data`:** the current minimal loader covers Schmidt/Wisniewski protein counts only. A fuller `ValidationDataEcoli` port (or adding a `validation/` tier to the **ecoli-sources** package, which is reconstruction-only today) would generalize it.
- **Emit completeness:** if the descoped analyses are ever wanted, re-enable their listeners in the emit features (a clutter trade-off) and re-run.
- **`ecocyc_table`:** port when its size is worth the effort.
- **DRY:** the ptools modules share `build_query`/`read_outputs`; further consolidation into `_helpers.py` is possible.

---

## Provenance
Spec/plan/parity notes under `docs/superpowers/`. The bulk port was built largely by a headless agent on the persistent mini, then unblocked + consolidated locally. See memory `project_v2ecoli_native_analyses` for cross-session context.
