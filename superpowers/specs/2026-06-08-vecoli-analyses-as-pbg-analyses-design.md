# vEcoli Analyses as Native Process-Bigraph Analyses — Design

**Date:** 2026-06-08
**Status:** Approved (design); implementation pending
**Scope:** Evolve v2ecoli's `AnalysisStep` into a unified **`Analysis`** abstraction
that ports vEcoli's DuckDB/`sim_data`-based analyses *faithfully and natively* as
process-bigraph Steps that are "Visualizations-like" (each emits a rendered `view`
HTML plus optional `data`). This spec lands the base, the runner refactor, the
dashboard surfacing, and a **proving set** of 5 analyses. A follow-up spec bulk-ports
the remaining ~43.

## Background

PR #95 added the `AnalysisStep` base + five-scale registry; the
[2026-05-30 workflow-analyses spec](2026-05-30-workflow-analyses-design.md) then
ported one native analysis per scale computed from emitted observables. That spec
made a deliberate **decision #1**: *native from emitted observables — no `sim_data`,
no DuckDB-SQL parity; faithful vEcoli ports explicitly deferred* (its "Out of scope"
section lists "Faithful vEcoli analysis ports (DuckDB SQL, `sim_data` molecule maps,
validation datasets …)" and "Rendering analysis results").

**This spec supersedes that decision #1.** We now want the faithful ports: the
analyses that read parquet via **DuckDB SQL** and the ParCa **`sim_data`** object,
reimplemented as native process-bigraph Steps — *without* importing vEcoli's
`Analysis`/`plot()` machinery and *without* a wrapper. The `analysis.py` module
docstring already anticipates this: *"Porting the full vEcoli analysis library onto
this base is a follow-up spec."* This is that spec.

The full target set on `vivarium-ecoli@origin/ptools_viz`: **48 analyses** —
single:7, multidaughter:1, multigeneration:11, multiseed:9, multivariant:7,
multiexperiment:1.

## Decisions (locked during brainstorming)

1. **Data access — DuckDB handle + `sim_data` as input ports.** An `Analysis`
   receives a live DuckDB connection (`conn`), the scale-scoped `history_sql`
   (plus `config_sql`/`success_sql`), and the run's `sim_data` as declared inputs.
   `analyze()` runs SQL over the partitioned parquet exactly as vEcoli's `plot()`
   does, so query logic ports near-verbatim. (Reverses 2026-05-30 decision #1.)
2. **Analysis unifies with Visualization.** One concept, one registry, one
   dashboard tab. Each `Analysis` produces a rendered `view` (HTML) **and** an
   optional `data` product (map/TSV). A pure visualization is just an `Analysis`
   with no `data` output. (Reverses 2026-05-30's "no rendering" non-goal.)
3. **Rendering is HTML, mixed engines per analysis.** `view` is an HTML string.
   Plot-heavy ports keep **matplotlib → embedded SVG** in an HTML wrapper
   (near-verbatim); ptools/simple analyses may use **Plotly**. No forced rewrite to
   a single charting engine.
4. **Scope — base + proving set now; bulk port next.** This spec lands the
   `Analysis` base, the runner refactor, dashboard surfacing, and a **5-analysis
   proving set** that exercises every axis. The remaining ~43 are a recipe-driven,
   parallel-agent-friendly follow-up spec.

## Architecture

```
run_workflow → branches complete → parquet sweep output (partitioned by
                                    experiment_id/variant/lineage_seed/generation/agent_id)
  → run_analyses(sweep_dir, analysis_options):
       conn = duckdb.connect over the sweep's partitioned parquet
       sim_data = resolve+load the run's ParCa sim_data
       for each scale in analysis_options:
         history_sql = scale-scoped SELECT (the WHERE/partition that scale aggregates)
         for each configured Analysis (resolved via ANALYSIS_REGISTRY by name+scale):
           result = analysis.update({conn, history_sql, config_sql, success_sql,
                                     sim_data, validation_data, variant_metadata, params})
           write result["data"] → analysis.json / <module>.tsv
           write result["view"] → <study>/viz/<analysis>.html
  → dashboard surfaces <study>/viz/*.html in the Visualizations tab (existing
    embed_visualizations mechanism); Analysis classes also listed in the viz picker.
v2ecoli-analyze <sweep_dir> [--config cfg.json]   # same runner, standalone
```

**Provisioning layer = the runner.** `conn`/`sim_data` are live, non-serializable
Python objects, so they are *not* wired through serialized bigraph state. The runner
injects them into the plain state dict passed to `Analysis.update()` — exactly how
the current `AnalysisStep.update(state)` is already invoked directly. `inputs()`/
`outputs()` declarations exist for typing/discoverability; port types for the live
handles are permissive (`any`).

## Components

### 1. The `Analysis` base — `v2ecoli/workflow/analysis.py`

Evolve `AnalysisStep` → **`Analysis(V2Step)`** *in place* (keep an `AnalysisStep =
Analysis` alias for one release so existing imports/`MassFractionSummary` keep
working). Retains the auto-registry (`ANALYSIS_REGISTRY`, `ANALYSIS_SCALES`,
`__init_subclass__` registration by `name`). New unified shape:

```python
class Analysis(V2Step):
    scale = "single"                          # one of ANALYSIS_SCALES
    config_schema = {}                         # was vEcoli's `params` (e.g. n_tp, time_unit)

    def inputs(self):
        return {
            "conn": "any", "history_sql": "string",
            "config_sql": "string", "success_sql": "string",
            "sim_data": "any", "validation_data": "any",
            "variant_metadata": "any",
        }
    def outputs(self):
        return {"view": "string", "data": "map"}   # either may be empty

    def analyze(self, *, conn, history_sql, sim_data, **ctx) -> dict:
        """Return {"view": <html>, "data": <map>} (either key optional)."""
        raise NotImplementedError

    def update(self, state, interval=None):
        out = self.analyze(**{k: state.get(k) for k in self.inputs()})
        return {"view": out.get("view", ""), "data": out.get("data", {})}
```

**Port mapping from vEcoli `plot()`** (`params, conn, history_sql, config_sql,
success_sql, sim_data_paths, validation_data_paths, outdir, variant_metadata,
variant_names`):

| vEcoli `plot()` arg | `Analysis` source |
|---|---|
| `conn`, `history_sql`, `config_sql`, `success_sql` | input ports (runner-provided) |
| `sim_data_paths` → `sim_data` | `sim_data` port (runner loads via v2ecoli's loader) |
| `validation_data_paths` → `validation_data` | `validation_data` port (optional) |
| `variant_metadata`, `variant_names` | `variant_metadata` port |
| `params` | `config_schema` / step config |
| `outdir` (file writes) | **returned** `view`/`data`; runner owns the filesystem |

Porting an analysis = paste the `plot()` body into `analyze()`, swap each
`outdir`-file-write for a `view`/`data` return entry, and read `params[...]` from
config. A `render` helper on the base wraps a matplotlib figure → embedded SVG HTML
(`fig_to_html(fig)`), so plot ports stay near-verbatim.

### 2. The runner — `v2ecoli/workflow/analysis_runner.py` (refactor)

Replace the record-building/grouping data path (`build_cell_records`,
`group_for_scale`) with **SQL provisioning**:

- `open_connection(sweep_dir) -> duckdb.Connection`: connect over the sweep's
  partitioned parquet (the layout `simulations_index.py` already documents:
  `parquet/<experiment_id>/…/variant=…/lineage_seed=…/generation=…/agent_id=…`).
- `scale_history_sql(scale, conn, partition) -> str`: build the per-scale SELECT —
  the WHERE/partition each scale aggregates over. This **replaces** the old
  per-scale grouping table; same semantics, expressed in SQL:

  | scale | partition the `history_sql` selects |
  |---|---|
  | single | one `(variant, lineage_seed, generation, agent_id)` cell |
  | multidaughter | sisters: one `(variant, lineage_seed, generation, parent_agent_id)` |
  | multigeneration | one `(variant, lineage_seed)` lineage, all generations |
  | multiseed | one `(variant,)`, all seeds |
  | multivariant | all cells |

- `resolve_sim_data(sweep_dir) -> sim_data`: locate + load the run's ParCa
  `sim_data` via v2ecoli's loader (see Risk §). Loaded once per run, reused across
  analyses.
- `run_analyses(sweep_dir, analysis_options) -> dict`: for each `scale` →
  configured analysis name → resolve via `ANALYSIS_REGISTRY` (asserting
  `step_cls.scale == scale`), iterate the scale's partitions, inject
  `{conn, history_sql, …, sim_data}` into `update()`, collect `data` into
  `analysis.json` and write each `view` to the sweep's `viz/<analysis>[_<partition>].html`.
  When the sweep belongs to a dashboard study/investigation, this `viz/` dir is the
  one the Visualizations tab reads (§3) — same location, no copy.
- `main()`: `v2ecoli-analyze <sweep_dir> [--config cfg.json]` unchanged in spirit.

### 3. Dashboard surfacing — `vivarium-dashboard/vivarium_dashboard/server.py`

- **Views (no new plumbing):** the runner writes `view` HTML into `<study>/viz/*.html`,
  which the dashboard **already** surfaces in the Visualizations tab via
  `embed_visualizations` (server.py ~690‑701).
- **Class picker (small extension):** extend `_list_visualization_classes()` to also
  enumerate `ANALYSIS_REGISTRY` entries (kind `"analysis"`), so ported Analyses appear
  in the same picker as `TimeSeriesPlot` etc. The existing discovery recognizes
  `issubclass(cls, Visualization)`; we add an `ANALYSIS_REGISTRY` source rather than
  forcing `Analysis` to subclass `Visualization` (which would collide with `V2Step`
  and the live-handle port model).

### 4. The proving set (5 analyses) — `v2ecoli/workflow/analyses/`

Chosen to exercise **every axis**: both render paths, sim_data-heavy and -light,
data-output and view-output, single and cross-cell scales.

| analysis | scale | exercises |
|---|---|---|
| `ptools_rna` | single | **sim_data-heavy** (rna_data, complexation monomers, rna_maturation_stoich, molecule_groups); data TSV output; the original integration goal |
| `ptools_rxns` | single | sim_data reaction maps; data TSV |
| `ptools_proteins` | single | sim_data monomer maps; data TSV |
| `mass_fraction_summary` | single | already native; re-expressed on the duckdb path as the migration reference; matplotlib→SVG `view` |
| one multiseed plot analysis (e.g. `centralCarbonMetabolismScatter`) | multiseed | **cross-cell** SQL aggregation + matplotlib `view`; proves the scale-scoped `history_sql` |

Each ported from `vivarium-ecoli@origin/ptools_viz:ecoli/analysis/<scale>/<name>.py`.

### 5. Console script + configs

- `pyproject.toml`: keep `v2ecoli-analyze = "v2ecoli.workflow.analysis_runner:main"`.
- A proving-set config (`v2ecoli/configs/*.json`) with `analysis_options` naming the
  5 analyses across `single` + `multiseed`.

## Error handling

- The runner isolates each analysis/partition: a Step raising is caught, recorded in
  `analysis.json` as `{"error": "<type>: <msg>"}`, remaining analyses continue.
  (`Analysis.invoke` still surfaces errors loudly when run directly.)
- Unknown analysis name or scale mismatch in `analysis_options` → `warnings.warn` + skip.
- Missing `sim_data` for a sim_data-dependent analysis → clear error naming the
  analysis and the expected sim_data path.
- Empty partition / no rows from `history_sql` → the Step returns a `skipped` view+data.

## Testing

- **Unit (per Analysis):** `analyze()` against a tiny fixture parquet + a fixture
  `sim_data` → expected TSV values (ptools row/column shape, normalization) and a
  non-empty `view`; skip paths.
- **Port fidelity:** ptools TSV output compared against the vEcoli reference TSVs
  checked into sms-api (`tests/fixtures/analysis_data/ptools_rna.txt`,
  `ptools_rxns.txt`) for the same input — the cross-implementation oracle.
- **Runner:** `scale_history_sql` produces the expected partition SELECT per scale;
  `run_analyses` over a generated mini-sweep yields an `analysis.json` block + a
  `viz/*.html` per configured analysis.
- **Registry:** `analysis_options` names resolve to the right classes; scale assert.
- **Dashboard:** the viz picker lists the ported Analyses; a produced `viz/*.html`
  appears as an `embed_visualizations` entry.

## File layout

```
v2ecoli/workflow/analysis.py            evolve AnalysisStep→Analysis (alias kept);
                                        {view,data} outputs; duckdb/sim_data ports
v2ecoli/workflow/analysis_runner.py     refactor: open_connection, scale_history_sql,
                                        resolve_sim_data, run_analyses (SQL provisioning)
v2ecoli/workflow/analyses/              new: ptools_{rna,rxns,proteins}, +mass_fraction,
                                        +1 multiseed plot analysis
v2ecoli/configs/<proving>.json          analysis_options for the proving set
vivarium-dashboard/.../server.py        _list_visualization_classes += ANALYSIS_REGISTRY
pyproject.toml                          (unchanged) v2ecoli-analyze console script
tests/test_workflow_analysis.py         Analysis base + ports + render helper
tests/test_analysis_runner.py           scale_history_sql, run_analyses, fidelity oracle
```

## Known risk — gates the proving set

The ports lean on the ParCa **`sim_data`** object (`transcription.rna_data`,
`complexation.get_monomers`, `rna_maturation_stoich_matrix`, `molecule_groups`) and
**exact parquet column names** (`listeners__rna_counts__full_mRNA_counts`, `bulk`,
`listeners__unique_molecule_counts__active_ribosome`). v2ecoli is vEcoli-derived but
may have attribute/column drift. **First plan step = a parity check** diffing
v2ecoli's `sim_data` + parquet schema against what the 5 proving analyses require.
The proving set deliberately front-loads the most `sim_data`-dependent analysis
(`ptools_rna`) so any drift surfaces here, not in the bulk-port spec. Drift outcomes:
small (add a column alias / attribute shim in the runner) vs. large (a missing
listener → flag for a separate emitter change).

## Out of scope (deferred / non-goals)

- The remaining ~43 analyses (recipe-driven follow-up spec, parallel-agent-friendly).
- `multiexperiment` scale (not in v2ecoli's `ANALYSIS_SCALES`; add when needed).
- The live **Pathway Tools Omics Viewer** painted-map embed (the licensed `sms-ptools`
  container) — Tier 3; these native Analyses render the ptools TSVs themselves.
- BioCyc *web-service* annotation as a hard dependency — frame-ID→name resolution uses
  the reconstruction flat files offline; live BioCyc enrichment is optional/later.
- Importing vEcoli's `Analysis`/`plot()` runner or any wrapper around it (explicit
  non-goal — these are native reimplementations).
```
