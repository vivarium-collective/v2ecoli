# vEcoli ↔ v2ecoli Comparison Harness — Design

**Date:** 2026-06-03
**Status:** Approved (design); pending implementation plan
**Repo:** `v2ecoli`
**Branch:** `feat/vecoli-comparison-harness`

## Purpose

Produce a rigorous, reviewable comparison between the original **vEcoli**
(monolithic, wcEcoli-derived) and **v2ecoli** (process-bigraph reimplementation),
arranged as a single self-contained two-column HTML report (vEcoli left,
v2ecoli right). The report has three sections:

0. **Config & schema diff** — what a vEcoli config maps to in v2ecoli.
1. **ParCa / `sim_data` comparison** — run both ParCas (full mode), diff the
   parameter-calculator output in detail.
2. **2-generation simulation comparison** — feed the *same* config into both,
   run 2 generations × 2 init sims, compare dynamics in detail.

A core goal: if the two configs are not equivalent, surface exactly how they
differ and make v2ecoli **accept vEcoli configs correctly** via a translation
adapter, so the equivalence is an explicit, reviewed mapping rather than a
silent guess.

## Decisions (locked during brainstorming)

| Decision | Choice |
|---|---|
| ParCa section | **Reuse & extend** `scripts/parca_compare.py` (per-step) **plus** a final-`sim_data` field-by-field diff |
| Config compatibility | **Adapter / translation layer** in the harness; v2ecoli core untouched |
| Match bar | **Statistical agreement + per-metric tolerances** (sim_data held tight; dynamics use tolerances + KS) |
| Execution | **Orchestrate + cache** — harness runs both pipelines if outputs missing/stale, caches, reuses on rerun |
| Home & entry | `v2ecoli/scripts/compare_harness.py` |
| Sim observables | Mass & growth, molecule counts, listener fields, division & lineage |
| ParCa mode | **Full mode** (real comparison); `--fast-plumbing` flag for wiring iteration only, banner-labeled NOT VALID |

## Entry point

```
.venv/bin/python scripts/compare_harness.py \
    --config <path/to/vEcoli-config.json> \
    -o out/compare/report.html \
    [--fast-plumbing] [--seed 0] [--force-rerun {parca,sim,all}]
```

The **vEcoli** JSON config is the single source of truth. The harness derives
the v2ecoli config from it via the adapter.

## Pipeline (5 stages)

### Stage 1 — Config ingestion + adapter

- Load the vEcoli config honoring its `inherit_from` chain (vEcoli's
  `load_config_with_inheritance`).
- `translate_vecoli_config(vecoli_cfg) -> v2_cfg` maps vEcoli keys to v2ecoli
  keys. Known gaps to bridge (from initial inspection):
  - vEcoli-only: `emitter`, `emitter_arg`, `parca_options`, `fail_at_max_duration`,
    `suffix_time`, `sim_data_path`.
  - v2ecoli-only: `cache_dir`, `out_dir`, `time_step`, `max_duration_per_gen`,
    `variants`, `lineage_seed`, `different_seeds_per_variant`, `skip_baseline`.
  - shared: `experiment_id`, `generations`, `n_init_sims`, `single_daughters`,
    `analysis_options` (sub-shape differs).
- Emit a **schema-diff table** into the report: only-in-vEcoli, only-in-v2ecoli,
  shared-but-different. This table is the reviewed record of the mapping.
- The adapter lives in the harness (e.g. `scripts/_compare/config_adapter.py`),
  not in `v2ecoli/workflow/config.py`. v2ecoli core stays untouched.

### Stage 2 — ParCa orchestration (full mode, cached)

- **vEcoli:** `runscripts/parca.py … --save-intermediates` → `kb/sim_data.cPickle`
  + per-step intermediates (`sim_data_<step>.cPickle`, `cell_specs_<step>.cPickle`).
- **v2ecoli:** its ParCa run → per-step checkpoints (`checkpoint_step_N.pkl` +
  `runtimes.json`), as consumed today by `parca_compare.py`.
- Cache under `out/compare_harness/<key>/` where `<key>` = SHA of
  (resolved config + engine git commit + mode). Stale ⇒ rerun.

### Stage 3 — ParCa / `sim_data` comparison

- Reuse `scripts/parca_compare.py` for the **per-step** diff (runtime, port
  manifest, scalar/distribution/cell_specs deltas, KS p-values, overlaid
  histograms).
- **Extend** with a **final-`sim_data` field-by-field diff** walking the two
  `sim_data` objects (not just step checkpoints).
- `sim_data` is held to **tight** tolerances — v2ecoli ParCa is meant to
  reproduce `fitSimData_1`, so diffs here are findings.

### Stage 4 — Sim orchestration + comparison (cached)

- Run 2 generations × 2 init sims with a fixed lineage seed on **both** engines,
  each fed the appropriate config (vEcoli native; v2ecoli translated).
- Cache keyed by (config hash + sim_data hash + seed).
- Compare four observable families with **per-metric tolerances + KS tests** and
  overlaid trajectory plots:
  - **Mass & growth** — cell/dry mass trajectories, growth rate, doubling
    time per generation.
  - **Molecule counts** — bulk molecule counts (key metabolites, proteins,
    RNAs): distributions and trajectories.
  - **Listener fields** — ribosome/RNAP activity, `fba_results` fluxes,
    transcription/translation rates.
  - **Division & lineage** — division firing times, daughter mass split,
    generation-boundary alignment between the two engines.

### Stage 5 — Report renderer

- One self-contained HTML, sticky left nav, two columns (vEcoli | v2ecoli).
- Per-metric **verdict badges**: within-tol / drift / mismatch.
- Sections: (0) Config & schema diff, (1) ParCa/`sim_data`, (2) 2-gen dynamics.

## Modules

Organized as a small `scripts/_compare/` package imported by
`scripts/compare_harness.py`:

- `config_adapter` — `translate_vecoli_config()` + schema diff.
- `orchestrator` — run/cache vEcoli & v2ecoli ParCa and sim; cache-key logic.
- `parca_section` — wraps/extends `parca_compare.py`; final-sim_data diff.
- `sim_section` — observable extraction + stats (tolerances, KS) + plots.
- `report` — HTML assembly (two-column layout, nav, badges).

Each module has one clear purpose and a small interface; the orchestrator is the
only stateful piece (filesystem + subprocess).

## Data flow

```
vEcoli config JSON
   └─[config_adapter]→ {vEcoli cfg, v2 cfg, schema diff}
        └─[orchestrator]→ {vEcoli sim_data + intermediates, v2 checkpoints}
             └─[parca_section]→ ParCa/sim_data diff
        └─[orchestrator]→ {vEcoli parquet, v2 emitted output}
             └─[sim_section]→ dynamics diff (4 families)
   └─[report]→ report.html
```

## Error handling

- Missing checkpoint/observable on one side ⇒ "not compared" with the reason
  listed; partial pipelines don't break the report (parca_compare already does
  this for ParCa).
- A failed run is captured (stderr tail) and rendered in the report, not fatal —
  one broken section doesn't kill the others.
- All tolerances in one editable `TOLERANCES` dict.
- `--fast-plumbing` uses ParCa `--mode fast` for wiring iteration only and
  stamps a prominent **"NOT SCIENTIFICALLY VALID"** banner in the report.

## Testing

- Unit-test `config_adapter`: vEcoli→v2 mapping and the schema-diff computation.
- Unit-test the stats/tolerance logic with synthetic arrays (within-tol, drift,
  mismatch cases; KS behavior).
- Smoke-test the renderer with a tiny fake result dict (asserts HTML structure,
  badge classes, both columns present).
- Real end-to-end ParCa/sim is too heavy for CI — gate behind a manual pytest
  marker; run by hand.

## Risks / to verify during implementation (not baked-in assumptions)

1. **Exact v2ecoli ParCa run command** — confirm against `scripts/parca_run.py` /
   `v2ecoli/cli/parca.py` and that it emits the checkpoints `parca_compare.py`
   expects.
2. **Driving vEcoli sim outside Nextflow** — confirm `ecoli/experiments/
   ecoli_master_sim.py` can run a 2-gen lineage from a plain config. If it
   realistically needs Nextflow for lineage, the orchestrator shells out to
   vEcoli's workflow runner instead.
3. **Emitter parity for reading dynamics** — vEcoli emits parquet (DuckDB);
   v2ecoli emits parquet/xarray/zarr. The `sim_section` reader must normalize
   both to a common in-memory shape before diffing.

## Out of scope (YAGNI)

- Variants / interventions beyond the single baseline 2-gen lineage.
- Cloud/Sherlock execution paths.
- Modifying v2ecoli's core config loader (adapter-only this round).
- More than 2 generations / 2 seeds (parameterizable later, not now).
