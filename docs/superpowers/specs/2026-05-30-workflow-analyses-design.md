# v2ecoli Workflow Analyses — Design

**Date:** 2026-05-30
**Status:** Approved (design); implementation pending
**Scope:** Port a first round of vEcoli's per-scale analyses into v2ecoli as
`AnalysisStep`s computed from emitted observables, and wire `analysis_options`
into a post-sweep analysis phase (in `run_workflow`) plus a standalone
`v2ecoli-analyze` CLI.

## Background

PR #95 added the workflow framework with an `AnalysisStep` base + a five-scale
registry (`single / multidaughter / multigeneration / multiseed / multivariant`),
but only one analysis (`MassFractionSummary`, single scale) is implemented and
`analysis_options` is parsed yet never executed (it only emits a warning).
vEcoli has ~21 core scale-based analyses; v2ecoli has ported ~1.

This spec ports **one meaningful native analysis per scale** and **wires the
post-sweep analysis phase** so `analysis_options` actually runs.

## Decisions (locked during brainstorming)

1. **Fidelity — native from emitted observables.** Each analysis is reimplemented
   as an `AnalysisStep` computing from data the sweep already emits (parquet
   listeners + per-cell summaries). No `sim_data`/validation-dataset dependency
   and no Altair/DuckDB-SQL parity. Faithful vEcoli ports are explicitly deferred.
2. **Runner — post-sweep phase in `run_workflow` + standalone CLI.** After all
   branches complete, `run_workflow` runs the configured analyses and writes
   `analysis.json`; the same runner is exposed as `v2ecoli-analyze <sweep_dir>`
   to re-run on a finished sweep without re-simulating.
3. **Breadth — one analysis per scale + full wiring.** Lands the runner + CLI +
   `analysis_options` wiring and one native analysis at each scale, proving the
   pipeline end-to-end across all five scales. More analyses then become drop-in.

## Architecture

```
run_workflow → branches complete
  → write <out_dir>/summary.json   (per-cell: variant/seed/generation/agent_id,
                                    division_time, divided, newborn/final mass)
  → run_analyses(sweep_dir, analysis_options):
       build per-cell records (summary.json + parquet-derived fraction means)
       for each scale in analysis_options:
         group cells for that scale
         for each configured AnalysisStep: run it over each group
  → write <out_dir>/analysis.json
v2ecoli-analyze <sweep_dir> [--config cfg.json]   # same runner, standalone
```

**Two data sources** (both already produced or trivially added):

- The sweep's emitted **parquet** → per-cell timeseries (single-scale analyses
  that need the full trajectory, e.g. mass fractions).
- A **`summary.json`** written by `run_workflow` from the per-cell generations
  summary it already computes (division time, duration, divided flag,
  newborn/final dry mass) → per-cell summary records for cross-scale analyses.
  This fills the gap that parquet alone lacks division metadata.

## Components

### 1. `AnalysisStep` interface + registry — `v2ecoli/workflow/analysis.py`

- Keep `analyze(rows) -> dict`. `rows` is defined **per scale**:
  - **single**: a list of one cell's emitted timeseries records (current
    `MassFractionSummary` already consumes this).
  - **cross-scale**: a list of **per-cell summary records** (one dict per cell in
    the group), each shaped:
    ```python
    {"variant": int, "lineage_seed": int, "generation": int, "agent_id": str,
     "divided": bool, "division_time": float,
     "newborn_dry_mass": float, "final_dry_mass": float,
     "protein_fraction_mean": float, "rRna_fraction_mean": float,
     "dna_fraction_mean": float}
    ```
- Add `ANALYSIS_REGISTRY: dict[str, type[AnalysisStep]]` keyed by analysis name
  (e.g. `"mass_fraction_summary"`, `"doubling_time_distribution"`). Populated by
  the implemented Steps; `analysis_options` names resolve through it.

### 2. Grouping per scale — in the runner

Cells are keyed by `(variant, lineage_seed, generation, agent_id)`:

| scale | group key | cells in a group |
|---|---|---|
| single | each cell individually | 1 cell's timeseries |
| multidaughter | `(variant, lineage_seed, generation, parent_agent_id)` | sister cells (share parent) |
| multigeneration | `(variant, lineage_seed)` | all generations of a lineage |
| multiseed | `(variant,)` | all cells across seeds |
| multivariant | `()` (all) | all cells |

`parent_agent_id` = the cell's `agent_id` minus its last phylogeny char
(`"00" → "0"`); sisters share it.

### 3. The five native analyses — `v2ecoli/workflow/analysis.py`

- **single — `MassFractionSummary`** (exists, unchanged): mean
  protein/rRNA/DNA mass fractions over a cell's timeseries.
- **multidaughter — `DaughterMassSymmetry`** (`scale="multidaughter"`): from
  sister per-cell summaries, birth-mass asymmetry
  `|m1 - m0| / (m1 + m0)` using `newborn_dry_mass`. Returns
  `{"n_sisters": int, "mass_asymmetry": float}`; if `< 2` sisters →
  `{"n_sisters": n, "skipped": "needs ≥2 daughters (single_daughters=false)"}`.
  *Dormant for single-lineage sweeps (only one daughter is carried); activates
  when binary-tree lineages land.*
- **multigeneration — `MassGrowthAcrossGenerations`** (`scale="multigeneration"`):
  from a lineage's per-cell summaries (one per generation), returns
  `{"n_generations": int, "per_generation": [{"generation", "newborn_dry_mass",
  "final_dry_mass", "division_time", "fold_change"}...], "mean_division_time": float}`.
- **multiseed — `DoublingTimeDistribution`** (`scale="multiseed"`): over a
  variant's cells, division-time stats of divided cells:
  `{"n_cells": int, "n_divided": int, "doubling_time_mean": float,
  "doubling_time_std": float, "final_dry_mass_mean": float}`.
- **multivariant — `MetricAcrossVariants`** (`scale="multivariant"`): over all
  cells grouped by variant,
  `{"per_variant": {variant: {"mean_division_time": float,
  "mean_final_dry_mass": float, "n_cells": int}}}`.

All cross-scale Steps consume per-cell summary records; only the single-scale one
consumes timeseries.

### 4. Analysis runner — `v2ecoli/workflow/analysis_runner.py` (new)

- `build_cell_records(sweep_dir) -> dict[cellkey, dict]`: merge two sources into
  the per-cell summary record shape above —
  - from `summary.json`: `divided`, `division_time` (the generation's `duration`);
  - from the parquet timeseries: `newborn_dry_mass` (first `dry_mass`),
    `final_dry_mass` (last `dry_mass`), and the fraction means
    (`protein/rRna/dna`, mean of `<comp>_mass / dry_mass` over the trajectory).

  Also retains each cell's raw timeseries rows for single-scale Steps. (Note:
  `summary.json`'s own `dry_mass` field is the post-division daughter reading, so
  newborn/final mass are taken from parquet, not from it.)
- `group_for_scale(scale, records) -> dict[groupkey, list[record]]`: the grouping
  table above.
- `run_analyses(sweep_dir, analysis_options) -> dict`: for each `scale` in
  `analysis_options`, for each configured analysis name, resolve via
  `ANALYSIS_REGISTRY`, run it over each group (timeseries for single; summary
  records for cross-scale), collect results keyed
  `scale → analysis_name → group_key → result`. Writes `<sweep_dir>/analysis.json`.
- `main()`: `v2ecoli-analyze <sweep_dir> [--config cfg.json]` — reads
  `analysis_options` from `--config` (or a sibling config) and runs the analyses.

### 5. `run_workflow` integration — `v2ecoli/workflow/run.py`

- After branches complete, write `<out_dir>/summary.json` from the per-branch
  generations summaries already in the result.
- If `config["analysis_options"]` is non-empty, call `run_analyses(out_dir,
  analysis_options)` and include the path in the result. Remove the existing
  "analysis_options is set but not wired" warning.

### 6. Console script + configs

- `pyproject.toml`: `v2ecoli-analyze = "v2ecoli.workflow.analysis_runner:main"`.
- `v2ecoli/configs/two_generations.json`: extend `analysis_options` with
  `multigeneration` (`mass_growth_across_generations`) and `multiseed`
  (`doubling_time_distribution`) entries alongside the existing `single`.

## Error handling

- The runner isolates each analysis: a Step raising is caught, recorded in
  `analysis.json` as `{"error": "<type>: <msg>"}`, and the remaining analyses
  continue (`AnalysisStep.invoke` itself still surfaces errors loudly when a
  Step is run directly).
- Unknown analysis name in `analysis_options` → `warnings.warn` + skip.
- Empty or too-small groups → the Step returns a `skipped` result (e.g.
  `DaughterMassSymmetry` with `< 2` sisters).
- Missing `summary.json` (standalone CLI on a sweep that didn't write one) →
  clear error telling the user to re-run the sweep or that division metadata is
  unavailable.

## Testing

- **Unit (per Step):** each `analyze()` over synthetic per-cell summaries /
  timeseries → correct numbers (asymmetry, doubling-time stats, per-generation
  fold change, per-variant means); skip paths.
- **Registry:** `analysis_options` names resolve to the right Step classes.
- **Runner grouping:** `group_for_scale` produces the expected groups per scale
  from synthetic cell records (no IO).
- **Cache-gated end-to-end:** run a tiny sweep with `analysis_options` spanning
  single + multigeneration + multiseed → `analysis.json` has a result block for
  each configured scale/analysis.
- **CLI:** `v2ecoli-analyze <sweep_dir>` on a generated sweep dir reproduces the
  same `analysis.json`.

## File layout

```
v2ecoli/workflow/analysis.py          extend: ANALYSIS_REGISTRY + 4 new Steps
v2ecoli/workflow/analysis_runner.py   new: build_cell_records, group_for_scale,
                                           run_analyses, main (CLI)
v2ecoli/workflow/run.py               write summary.json; call run_analyses
pyproject.toml                        v2ecoli-analyze console script
v2ecoli/configs/two_generations.json  multigeneration + multiseed analysis_options
tests/test_workflow_analysis.py       extend: 4 new Steps + registry
tests/test_analysis_runner.py         grouping, run_analyses, CLI, end-to-end
```

## Out of scope (deferred / non-goals)

- Faithful vEcoli analysis ports (DuckDB SQL, `sim_data` molecule maps,
  validation datasets such as Schmidt 2015, Altair charts).
- Rendering analysis results into the sweep HTML report.
- Reading analyses from the xarray/zarr path (parquet is the sweep default;
  the runner reads parquet + summary.json).
- `multidaughter` actually producing data — needs binary-tree lineages
  (`single_daughters=false`), still deferred; the Step lands but stays dormant.
- More than one analysis per scale (drop-in once the framework is proven).
