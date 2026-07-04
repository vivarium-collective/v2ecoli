# Unified post-simulation analysis flush — design

**Status:** approved design (2026-06-29), pending implementation plan.

## Goal

After a simulation run finishes, do **one** post-simulation pass — the *analysis
flush* — that extracts the run's emitted data once and dispatches it to every
registered post-sim step (Analyses, Visualizations, ReportCards), then routes
each output to the place in the **report** where the dashboard renders it. This
generalizes today's analysis-only, ad-hoc `run_analyses` into a single,
kind-aware orchestrator, and absorbs the interim per-study report-card runner.

## Background / current state

- `v2ecoli/workflow/run.py:run_workflow` already runs the sim, writes
  `summary.json` to `out_dir`, then calls
  `v2ecoli/workflow/analysis_runner.run_analyses(out_dir, analysis_options)`
  (run.py ~line 133). This is the de-facto post-run hook — but it runs **only**
  Analyses.
- `run_analyses` already does the extraction: reads the sweep's emitted parquet
  → `build_cell_records` → provisions a shared DuckDB `conn` + `sim_data`
  (`_analysis_ctx`) → runs `AnalysisStep`s, writing `view`→`out_dir/viz/<name>.html`,
  `data` tsv→`out_dir/ptools/`, and an `analysis.json`. It branches on
  `issubclass(step_cls, Analysis)` to provision the DuckDB context.
- `Analysis(V2Step)` / `AnalysisStep(V2Step)` register in `ANALYSIS_REGISTRY`
  via `__init_subclass__`; both expose output ports `{"view":"string","data":"map"}`.
- `ReportCardStep(V2Step)` (shipped, `v2ecoli/workflow/report_cards/`) registers
  in `REPORT_CARD_REGISTRY`, same `{view,data}` ports, input `{"study":"any"}`.
- There is **no** `Visualization` step type and **no** unified registry.
- **The placement gap:** the dashboard study report reads everything from the
  *owning study's* `viz/`:
  - visualizations / analysis views → `studies/<slug>/viz/*.html`
    (`single_study_report.py` inlines `studies/<slug>/viz/*.html` via iframe).
  - report cards → `studies/<slug>/viz/report_card/*.{html,verdict.json}`.
  But `run_analyses` writes to the **run's** `out_dir/viz/`, which is not the
  study's `viz/` — so run analyses can land where the study report never looks.
  Closing this is the core requirement.

## Architecture (chosen approach)

A single `run_flush(run_ctx)` orchestrator with four layers:

```
run finishes (run_workflow) → run_flush(run_ctx):
  1. EXTRACTION (lazy, once)   emitters/parquet ──► shared context bag
                               {records, conn, history_sql, sim_data, run_meta, study}
                               conn/sim_data built ONLY if some step declares them.
  2. DISPATCH                  for step in POST_SIM_REGISTRY (kind-tagged):
                                 feed step the SUBSET of the bag its inputs() declares
                                 → collect (view, data)
  3. PLACEMENT (by kind)       route each output to the OWNING STUDY's report dir:
                                 report_card  → studies/<slug>/viz/report_card/<name>.{html,verdict.json}
                                 visualization→ studies/<slug>/viz/<name>.html
                                 analysis     → studies/<slug>/viz/<name>.html (view)
                                                + analysis.json / ptools/ (data), as today
  4. MANIFEST                  one analysis.json-style summary of what ran + where it landed.
```

### 1. Unified registry — `POST_SIM_REGISTRY`

One registry keyed by step `name`, each entry tagged with a `kind` ∈
`{"analysis","visualization","report_card"}`. Realized by a shared
`PostSimStep` mixin (or a `register_post_sim(kind)` hook) that
`Analysis`/`AnalysisStep`/`ReportCardStep`/`Visualization` all funnel into via
their existing `__init_subclass__`. The existing `ANALYSIS_REGISTRY` and
`REPORT_CARD_REGISTRY` keep working (back-compat: they mirror, or become views
over the unified registry). The flush iterates `POST_SIM_REGISTRY` only.

`Visualization` is a **kind**, not a heavy new base — a thin `Visualization`
class mirroring `Analysis` (same `{view,data}` ports) tagged `kind="visualization"`.
Existing Analyses that emit a `view` continue to work unchanged; the kind tag
just lets placement route a pure-visualization output distinctly when desired.

### 2. Extraction — `RunExtract`

A reusable object built from the run's `out_dir` that lazily exposes
`records()`, `conn()`/`history_sql()`, `sim_data()`, `run_meta()`, and the
**owning `StudyContext`** (resolved from run provenance — the run's study slug).
Generalizes the existing `build_cell_records` + `_analysis_ctx`; the heavy
DuckDB/sim_data provisioning happens once and only when a dispatched step's
`inputs()` declares those keys.

### 3. Dispatch — shared context bag filtered by `inputs()`

The flush assembles one context dict with every available key, then for each
registered step passes only the subset its `inputs()` ports name. No per-kind
branching for input provisioning — a report card declaring `{"study":"any"}`
gets `study`; an analysis declaring `{conn,history_sql,sim_data,…}` gets those.
Each step is run via its `update()` (the process-bigraph surface) or `build`/
`analyze`, yielding `(view, data)`.

### 4. Placement — per-kind sinks into the study report dir

A `PLACEMENT` mapping kind → sink writes each output to the owning study's
canonical report location (paths above), so the report renderer surfaces it.
The owning study comes from the run provenance via `RunExtract`. For a run with
no owning study (ad-hoc), placement falls back to `out_dir/viz/` (today's
behavior) so nothing regresses. This layer is what guarantees "analyses,
visualizations and report cards all end up in the necessary place of the report."

### 5. Trigger & the absorbed runner

- `run_workflow` replaces its `run_analyses(out_dir, …)` call with
  `run_flush(RunExtract(out_dir, …))`. A standalone CLI re-flushes an existing
  run dir (parity with `analysis_runner.main`).
- `scripts/study_report_cards.py` becomes a thin wrapper that invokes the flush
  filtered to `kind="report_card"` (the no-new-run path); the per-run path is
  the flush inside `run_workflow`.

## Data flow

```
run_workflow → out_dir (+summary.json)
  RunExtract(out_dir) ──► context bag (lazy conn/sim_data; records; run_meta; study)
  for step in POST_SIM_REGISTRY:
     out = step(update / inputs()-subset of bag) → {view, data}
     PLACEMENT[step.kind](study, name, out) → studies/<slug>/viz[/report_card]/...
  write flush manifest (what ran, verdicts, output paths)
dashboard report (unchanged) renders studies/<slug>/viz/*.html + viz/report_card/*
```

## Error handling

- Per-step: a step that raises (build/extract/placement) logs a skip and the
  flush continues — one step never aborts the whole flush (the invariant the
  report-card runner already honors).
- Extraction failures (no parquet, unreadable sim_data) degrade gracefully:
  steps that need the missing key are skipped; run-free report cards still run.
- Placement: non-finite floats sanitized before JSON write (`allow_nan=False`);
  writes are atomic; a per-kind sink only touches its own subtree.

## Testing

- **Registry:** `Analysis`, `Visualization`, `ReportCardStep` subclasses each
  land in `POST_SIM_REGISTRY` with the right `kind`; back-compat views
  (`ANALYSIS_REGISTRY`/`REPORT_CARD_REGISTRY`) still resolve.
- **Dispatch:** a fake step declaring `{"study":"any"}` receives only `study`;
  one declaring DuckDB keys receives the extraction subset; an exception in one
  step does not abort the others.
- **Placement:** a `report_card` output lands at
  `studies/<slug>/viz/report_card/<name>.{html,verdict.json}`; a `visualization`
  view at `studies/<slug>/viz/<name>.html`; an `analysis` data payload at the
  manifest/ptools location; ad-hoc (no study) falls back to `out_dir/viz/`.
- **End-to-end:** `run_flush` over a fixture run dir (small parquet + a study)
  produces the expected study `viz/` artifacts and a manifest; re-running is
  deterministic for run-free outputs.
- **Trigger parity:** `run_workflow` invoking the flush reproduces today's
  analysis outputs (no regression) plus the new study-dir placement.

## Scope

**In (first plan):** the `run_flush` orchestrator (extraction + dispatch +
placement + manifest); `POST_SIM_REGISTRY` + `kind` tagging with back-compat
views; a thin `Visualization` kind; wiring Analyses + ReportCards through the
flush; `run_workflow` trigger swap; the report-card runner wrapper.

**Out (later):** investigation-level (cross-study) report aggregation; porting
the full vEcoli analysis library; migrating every existing Analysis's bespoke
sink semantics beyond parity; the dashboard-side rendering (already reads the
study `viz/` locations — no change needed).
