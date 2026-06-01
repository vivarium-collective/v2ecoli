# v2ecoli

**vEcoli, reimagined as a composable [process-bigraph](https://github.com/vivarium-collective/process-bigraph)
model and a research workspace.** v2ecoli takes the Covert lab's whole-cell
*E. coli* simulation ([vEcoli](https://github.com/CovertLab/vEcoli)) and rebuilds
it on the process-bigraph engine and the
[bigraph-schema](https://github.com/vivarium-collective/bigraph-schema) type
system. The model is no longer a monolithic simulation tied to `vivarium-core` —
it is a set of typed, independently-wireable processes you **compose** into
whatever architecture a question needs.

Two things follow from that:

- **Composition is a first-class operation.** Every process declares typed
  input/output ports, so building a model is wiring, not patching. A whole cell
  is one `build_composite("baseline")` call; a colony embeds many cells through a
  single bridge process; a kinetic-metabolism variant is a *different wiring of
  the same parts*. Swapping a subsystem is a one-line change, not a fork.
- **The repository is a pbg research workspace.** Alongside the model code live
  **investigations** (a research question) and **studies** (the simulations that
  answer it) — browsable and runnable in the
  [vivarium-dashboard](https://github.com/vivarium-collective/vivarium-dashboard).
  The current tree carries the `colonies` and `v2ecoli-pdmp` investigations
  across seven studies (`workspace/`). New science is added as a study, not a
  patch to a monolith.

The biology stays faithful to upstream. v2ecoli reproduces vEcoli's cell-cycle
trajectories from birth to division — across dry mass, mass composition
(protein / RNA / DNA / water), bulk-molecule counts, and replication dynamics —
not just a single endpoint. The
[composite comparison](https://vivarium-collective.github.io/v2ecoli/composite_comparison.html)
and [vEcoli-vs-v2ecoli](https://vivarium-collective.github.io/v2ecoli/v1_v2_comparison.html)
reports lay the two engines side by side metric-by-metric. Through the full cell
cycle the dry-mass trajectories track to within a fraction of a percent
(707.2 fg vs 705.3 fg at division), with division timing of ~42 min in both.

Under the hood, the biology is **17 biological process modules** plus 8
listener/deriver steps (`v2ecoli/processes/`, `v2ecoli/steps/derivers/`); the
partitioned baseline composite schedules these — after splitting some processes
into requester/evolver halves and adding infrastructure steps — into ~45 steps
per simulation tick. The Parameter Calculator (ParCa) is decomposed into nine
process-bigraph Steps and ships pre-computed, so a fresh clone simulates
end-to-end without the ~70 min knowledge-base rebuild.

**New here?** Read in this order: [What v2ecoli is](#what-v2ecoli-is) →
[Install](#install) → [Quick start](#quick-start) →
[The process-bigraph framework](#the-process-bigraph-framework) →
[Architectures](#architectures). Then dive into
[running pipelines](#running-pipelines-multiseed--multigen--multivariant) and
[the reports](#reports--interactive-viewers).

---

## Contents

- [What v2ecoli is](#what-v2ecoli-is)
- [Install](#install)
- [Quick start](#quick-start)
- [The process-bigraph framework](#the-process-bigraph-framework)
- [Architectures](#architectures)
- [Running pipelines: multiseed / multigen / multivariant](#running-pipelines-multiseed--multigen--multivariant)
- [Emitters: how output is written (Parquet & Xarray)](#emitters-how-output-is-written-parquet--xarray)
- [Reports & interactive viewers](#reports--interactive-viewers)
- [ParCa](#parca)
- [What changed since vEcoli](#what-changed-since-vecoli)
- [Performance & validation](#performance--validation)
- [Known limitations](#known-limitations)
- [Repository layout](#repository-layout)
- [Dependencies & ecosystem](#dependencies--ecosystem)

---

## What v2ecoli is

vEcoli is a whole-cell *E. coli* model: a mechanistic simulation that grows a
single cell from birth to division by integrating transcription, translation,
metabolism, replication, and regulation. v2ecoli re-implements that biology on
the [process-bigraph](https://github.com/vivarium-collective/process-bigraph)
engine and [bigraph-schema](https://github.com/vivarium-collective/bigraph-schema)
type system, and wraps it in a research workspace.

What you get over upstream vEcoli:

- **No `vivarium-core`.** The simulation engine is process-bigraph; the model is
  a plain process-bigraph state document.
- **Composition over configuration.** Architectures are *generated* by
  `@composite_generator`-decorated functions and reached by name
  (`build_composite("baseline" | "colony" | "millard_pdmp_baseline")`). A new
  architecture is a new wiring of existing parts, not a new config flag inside a
  monolith.
- **Explicit, typed ports.** Every process declares its `inputs`/`outputs`
  schema with units (`pint.Quantity`), and state round-trips through
  bigraph-schema JSON (no pickle in the save path).
- **A research workspace.** The repo is a pbg workspace (`workspace.yaml`):
  biology sits next to **investigations** (a shared research question) and
  **studies** (the runs that answer it), all browsable and runnable in the
  vivarium-dashboard. The `colonies` and `v2ecoli-pdmp` investigations live in
  `workspace/`. Manage them with the `pbg-investigation` / `pbg-study` skills.
- **A decomposed ParCa.** The monolithic `fitSimData_1()` is broken into nine
  inspectable Steps, and the fitted `sim_data` is shipped pre-computed.
- **Workflow pipelines.** Multiseed / multigeneration / multivariant sweeps are
  driven by a single config-file CLI (`v2ecoli-workflow`).
- **Self-describing HTML reports** published to GitHub Pages.

For the full diff against upstream, see
[What changed since vEcoli](#what-changed-since-vecoli).

---

## Install

Requires [`uv`](https://docs.astral.sh/uv/) and a C compiler (Xcode CLI tools on
macOS, `build-essential` on Linux).

```bash
git clone https://github.com/vivarium-collective/v2ecoli.git
cd v2ecoli
uv sync
```

`uv sync` provisions Python 3.12, installs every dependency (vEcoli,
process-bigraph, bigraph-schema, the pbg workspace stack), and compiles the
vendored Cython extensions.

> Run everything through the project venv: `.venv/bin/python …` (or activate it).
> A bare `python` on your `PATH` will be missing `unum` and other deps.

---

## Quick start

```bash
# Single cell to division (~42 simulated min), writes + opens an HTML report
.venv/bin/python reports/workflow_report.py

# A multiseed × multigeneration sweep driven by a config file
v2ecoli-workflow --config v2ecoli/configs/two_generations.json

# A 3-cell colony with N adder-grow-divide surrogate cells
.venv/bin/python reports/colony_report.py --n-adder 5 --duration 45
```

The `reports/*.py` scripts each write a self-contained HTML report under `out/`
and open it. `v2ecoli-workflow` writes partitioned Parquet plus a `summary.json`
under `out/workflow/` (see [Running pipelines](#running-pipelines-multiseed--multigen--multivariant)).

Programmatic use:

```python
import v2ecoli
composite = v2ecoli.build_composite("baseline", seed=0, cache_dir="out/cache")
composite.update({}, 60.0)            # advance 60 simulated seconds
```

---

## The process-bigraph framework

v2ecoli is built on **process-bigraph** (the engine) and **bigraph-schema** (the
type system). Four concepts are enough to read the codebase:

| Concept | What it is | Where it lives |
|---|---|---|
| **Process** | A unit of computation with typed `inputs`/`outputs` schemas and an `update(state, interval) -> update` method. Updates are merged into the shared store each tick. | `v2ecoli/processes/*.py` |
| **Step** | A process that runs *to convergence within a tick* rather than stepping through time (e.g. listeners, allocators, the ParCa fit). | `v2ecoli/steps/*.py` |
| **Store** | A named, schema-typed state container addressed by path (`bulk`, `listeners.mass.dry_mass`, `unique.ribosome`). | declared in the composite document |
| **Composite** | Processes + stores + **wires** (edges from a process port to a store path), assembled into one runnable model. | `v2ecoli/composites/*.py` |

A minimal sketch of a process:

```python
class MyProcess(Step):
    def inputs(self):  return {"bulk": "bulk_array", "timestep": "float[s]"}
    def outputs(self): return {"bulk": "bulk_array"}
    def update(self, state, interval):
        # read state["bulk"], compute, return only what you write
        return {"bulk": delta}
```

**Types & units.** Project-specific types live in `v2ecoli/types/` (e.g.
`Quantity`, `CSRMatrix`, `BulkNumpyUpdate`, `ListenerStore`). Dimensioned
quantities at ports are `pint.Quantity`; the only place `Unum` survives is the
upstream-interop bridge at `v2ecoli/library/unit_bridge.py`.

**Composite generators.** Each architecture is a function decorated with
`@composite_generator` (from `pbg_superpowers`). The decorator registers the
generator under a name, with its parameters, default visualizations, and default
emitter:

```python
@composite_generator(
    name="baseline",
    description="Partitioned whole-cell E. coli model — upstream-parity",
    parameters={"seed": {"type": "integer", "default": 0}, ...},
    emitters=[{"address": "local:ParquetEmitter", "paths": ["global_time", "bulk", "listeners"]}],
)
def baseline(core=None, *, seed=0, cache_dir="out/cache", config_overrides=None):
    return { ... }   # a process-bigraph state document (a dict), not a Composite
```

Callers reach any registered architecture by name:
`v2ecoli.build_composite("baseline", seed=42)` looks up the generator, calls it,
and wraps the returned document in a `Composite`.

**Auto-discovery.** When v2ecoli is installed alongside bigraph-schema and
process-bigraph, its biological processes register themselves on
`allocate_core()` — no manual `core.register_link()` calls:

```python
from process_bigraph import allocate_core
core = allocate_core()   # core.link_registry now has v2ecoli.processes.*
```

**The `pbg_v2ecoli/` package** at the repo root is the *workspace* package the
[vivarium-dashboard](https://github.com/vivarium-collective/vivarium-dashboard)
uses. Its `build_core()` pre-registers the v2ecoli types **plus** the `EcoliWCM`
bridge before composites are built (so the dashboard's subprocess runner can pass
a fully-populated `core`). The model package (`v2ecoli/`) and the workspace
package (`pbg_v2ecoli/`) are distinct: edit biology in `v2ecoli/`, workspace
wiring in `pbg_v2ecoli/`. See `workspace.yaml` for the workspace config.

> Deeper framework questions: invoke the `pbg-expert` skill, or read
> [AGENTS.md](AGENTS.md).

---

## Architectures

Four composite generators are registered (`v2ecoli/composites/`). All share the
same biological processes; they differ in how cells are scheduled and embedded.

| Architecture | `build_composite("…")` | What it is |
|---|---|---|
| **baseline** | `baseline` | Partitioned requester/allocator/evolver scheduling — the vEcoli-parity reference. |
| **colony** | `colony` | Many whole cells embedded in a 2D pymunk physics environment via the `EcoliWCM` bridge (multi-agent). |
| **parca** | `parca` | The nine-Step ParCa parameter-calculation pipeline (builds `sim_data`). |
| **millard_pdmp_baseline** | `millard_pdmp_baseline` | Experimental variant replacing tFBA metabolism with a Millard-2017 kinetic ODE + LQR controller (piecewise-deterministic Markov process). |

Each whole cell in the colony is wrapped by **`EcoliWCM`** (`v2ecoli/bridge.py`),
a process-bigraph `Process` holding an internal `Composite` and a port bridge:

| External port | Internal store |
|---|---|
| `local` (in) | `boundary.external` |
| `mass` (out) | `listeners.mass.dry_mass` |
| `length` (out) | from `volume` via capsule geometry |
| `volume` (out) | `listeners.mass.volume` |

At division (~702 fg dry mass) the bridge swaps the mother for two daughters with
fresh `EcoliWCM`s and phylogeny-mutated colors.

To add an architecture, see [AGENTS.md → Adding a new composite
architecture](AGENTS.md).

---

## Running pipelines: multiseed / multigen / multivariant

All three sweep types are driven by **one CLI and a JSON config**:

```bash
v2ecoli-workflow --config <config.json> [--out <dir>] [--build-only] [--max-sim-time <s>]
```

Configs support inheritance (`"inherit_from": ["default.json"]`). Three example
configs ship in `v2ecoli/configs/`: `default.json` (1 seed, 1 generation),
`two_generations.json`, and `two_generations_xarray.json`.

The three pipelines are not separate commands — they are three knobs on the same
grid. The sweep expands to `variants × seeds × generations` independent lineages:

| Pipeline | Knob | What it sweeps | Conceptually |
|---|---|---|---|
| **multiseed** | `n_init_sims` | The same model across N random seeds | Stochastic replicates of one cell — variance across seeds |
| **multigeneration** | `generations` | A single lineage across N divisions (one daughter carried forward) | Mass growth & division timing down one lineage |
| **multivariant** | `variants` | A parameter grid (product / zip / linspace) | Sensitivity / robustness across parameter values |

### Config grammar

```jsonc
{
  "experiment_id": "kcat_sweep",
  "generations": 2,            // multigen depth (one daughter carried forward)
  "n_init_sims": 2,            // multiseed: seeds per variant
  "lineage_seed": 0,           // base seed; replicate s uses lineage_seed + s
  "single_daughters": true,    // single-lineage walk
  "max_duration_per_gen": 3600.0,
  "out_dir": "out/workflow",

  "variants": {                // multivariant: omit/empty for baseline-only
    "kcat_scale": {
      "target": "ecoli-metabolism.kcat",
      "linspace": {"start": 0.5, "stop": 2.0, "num": 5}
    }
  },

  "analysis_options": {        // post-sweep aggregations, run by v2ecoli-analyze
    "multiseed":       {"doubling_time_distribution": {}},
    "multigeneration": {"mass_growth_across_generations": {}},
    "multivariant":    {"metric_across_variants": {}}
  }
}
```

Variant blocks accept `value: [...]` (explicit), `linspace: {start, stop, num}`,
or any numpy generator, and combine multiple parameters with `"op": "prod"`
(Cartesian, default), `"zip"`, or `"add"`. The config above expands to
1 baseline + 5 variants × 2 seeds = 12 lineages, each run for 2 generations.

### Output & analysis

The sweep writes **hive-partitioned Parquet** plus metadata under `--out`:

```
out/workflow/
  parquet/experiment_id=…/variant=…/lineage_seed=…/generation=…/agent_id=…/*.pq
  sweep.pbg        # the full process-bigraph sweep document
  summary.json     # per-branch division metadata (duration, dry_mass, divided)
```

Aggregate across the grid with the companion analysis CLI:

```bash
v2ecoli-analyze out/workflow            # runs the config's analysis_options
```

These mirror vEcoli's own workflow grammar (`lineage_seed` + `n_init_sims`,
`single_daughters`, variant `target`/`value`/`op`), so configs translate
directly.

---

## Emitters: how output is written (Parquet & Xarray)

An **emitter** is a process-bigraph Step attached to a composite that records
selected store paths each tick. v2ecoli ships two production emitters plus an
in-memory default.

| Emitter | Backend | Best for | Read back with |
|---|---|---|---|
| **ParquetEmitter** *(default)* | Hive-partitioned Parquet on disk | Large multiseed/multigen/multivariant sweeps; vEcoli-compatible analysis | DuckDB / Polars |
| **XArrayEmitter** | Zarr store → xarray `DataTree` | numpy/scipy analysis & plotting; lazy slicing of big runs | `xarray.open_datatree` |
| `RAMEmitter` | In-memory | Tests, quick interactive runs | direct `.query()` |

**ParquetEmitter** re-exports the shared
[`pbg-emitters`](https://github.com/vivarium-collective/pbg-emitters)
implementation (`v2ecoli/library/parquet_emitter.py`). Its on-disk hive layout
matches vEcoli's exactly — partitioned by `experiment_id / variant /
lineage_seed / generation / agent_id` — so each lineage is its own subtree and
DuckDB can query subsets without scanning everything:

```python
import duckdb
duckdb.sql("""
  SELECT * FROM read_parquet('out/workflow/parquet/**/*.pq', hive_partitioning=true)
  WHERE generation = 2
""")
```

**XArrayEmitter** (`v2ecoli/library/xarray_emitter/`, vendored from vEcoli
PR #414 and re-rooted onto `process_bigraph.emitter.Emitter`) buffers ticks and
writes a Zarr store with one group per generation/agent, preserving units and
encodings as array metadata:

```python
import xarray as xr
tree = xr.open_datatree("out/run.zarr", engine="zarr")
tree["generation_1"]["agent_id_0"]["dry_mass"]   # an xarray DataArray
```

Choosing an emitter:

- **Config-driven** — the workflow `emitter` field accepts `"parquet"`
  (default) or `"xarray"`; see `v2ecoli/configs/two_generations_xarray.json`.
- **Presets** — `parquet_vecoli(...)` / `xarray_vecoli(...)` in
  `v2ecoli/library/emitter_presets.py` build vEcoli-compatible configs.
- **Override context managers** — `with parquet_emitter(experiment_id=…) as e:`
  wraps a build and auto-flushes on exit (`v2ecoli/composites/_helpers.py`).

> The vivarium-dashboard's Simulations-DB tab currently reads SQLite, not
> Parquet/Zarr — those are for offline DuckDB / xarray analysis.

---

## Reports & interactive viewers

Every report is generated by a script and (for the published set) committed under
`docs/`, served at
**[vivarium-collective.github.io/v2ecoli](https://vivarium-collective.github.io/v2ecoli/)**.
They fall into four groups.

### Interactive model viewers — *explore the model in your browser*

| Report | Published | Generate locally |
|---|---|---|
| **[Baseline composite](https://vivarium-collective.github.io/v2ecoli/bigraph_baseline.html)** — processes, stores, wiring, port schemas (with units), and a formal equation for each process. Pan/zoom, expand store chips, toggle nodes. | `bigraph_baseline.html` | `scripts/viz_baseline_interactive.py` |
| **[ParCa pipeline](https://vivarium-collective.github.io/v2ecoli/bigraph_parca.html)** — the nine-Step parameter calculator, same interactive viewer. | `bigraph_parca.html` | `scripts/viz_parca_interactive.py` |
| **[ParCa network](https://vivarium-collective.github.io/v2ecoli/parca_network.html)** / **[Baseline network](https://vivarium-collective.github.io/v2ecoli/network_baseline.html)** — Cytoscape topology; click a process for ports, schema, docstring, math. | `parca_network.html`, `network_baseline.html` | `scripts/parca_network.py` |

### Simulation result reports — *what the cell actually did*

| Report | Published | Generate locally |
|---|---|---|
| **[Cell lifecycle](https://vivarium-collective.github.io/v2ecoli/workflow_report.html)** — one cell, mother → division → both daughters. | `workflow_report.html` | `reports/workflow_report.py` |
| **[Multigeneration lineage](https://vivarium-collective.github.io/v2ecoli/multigeneration_report.html)** — N-generation single lineage, mass trajectories & fold-change. | `multigeneration_report.html` | `reports/multigeneration_report.py --generations 3` |
| **[Colony](https://vivarium-collective.github.io/v2ecoli/colony_report.html)** — mixed colony with pymunk physics, growth & division, synced animations. | `colony_report.html` | `reports/colony_report.py --n-adder 9` |

### Comparison & benchmark — *v2ecoli vs vEcoli*

| Report | Published | Generate locally |
|---|---|---|
| **[v1 vs v2](https://vivarium-collective.github.io/v2ecoli/v1_v2_comparison.html)** — vEcoli 1.0 vs v2ecoli baseline: wall/sim time, dry mass, growth. | `v1_v2_comparison.html` | `reports/v1_v2_report.py` |
| **[Composite comparison](https://vivarium-collective.github.io/v2ecoli/composite_comparison.html)** — any set of engines side-by-side (load/wall/sim, composition, sparklines). | `composite_comparison.html` | `reports/composite_comparison.py --engines baseline millard_pdmp_baseline` |
| Benchmark — v2ecoli vs the vEcoli composite (local-only). | — | `reports/benchmark_report.py` |

### Model structure & ParCa — *the math and the parameters*

| Report | Published | Generate locally |
|---|---|---|
| **[Mathematical structure](https://vivarium-collective.github.io/v2ecoli/math_structure.html)** — every process's governing equations grouped by subsystem, the per-tick execution flow, and the partition→allocate→evolve contract. | `math_structure.html` | `reports/math_structure_report.py` |
| **[ParCa workflow](https://vivarium-collective.github.io/v2ecoli/parca_workflow_report.html)** — the nine-Step run with per-step runtimes, port manifests, and raw-data stats. | `parca_workflow_report.html` | (ParCa pipeline) |

> **Provenance banners.** PR-evidence reports (`scripts/pr_session_report.py`,
> `scripts/sweep_report.py`) embed a self-describing header — ISO timestamp, git
> SHA/branch, dirty-tree badge, last commit, host/OS/Python — so an HTML file
> stays meaningful months later. Attach these to PRs that change biology; see
> [AGENTS.md → Reports](AGENTS.md).

---

## ParCa

ParCa (the Parameter Calculator) fits ~130 EcoCyc-derived knowledge-base TSVs
into a `SimulationDataEcoli` blob the runtime reads from. Upstream this is the
single monolithic `fitSimData_1()`; v2ecoli decomposes it into **nine
process-bigraph Steps** with explicit ports. Stage 5 (`fit_condition`) is the
~70 min cost; everything else is seconds.

- **Pre-computed `sim_data`** ships at `models/parca/parca_state.pkl.gz` (18 MB)
  — a fresh clone simulates without re-running ParCa.
- **Re-run from scratch:** `v2ecoli-parca --mode fast` (~70 min).
- **Resume from the cached step-5 checkpoint:** `bash scripts/parca_rerun_from_step5.sh`.
- **Refresh BioCyc flat files:** `python scripts/parca_update_biocyc.py`.
- **Rebuild the runtime cache** (fast; reuses the committed ParCa fixture):
  `python scripts/build_cache.py`. The cache is fingerprinted, so a stale cache
  raises `StaleCacheError` with a one-line rebuild command rather than a deep
  `AttributeError`.

Full path: `docs/generate_full_parca.md`.

---

## What changed since vEcoli

**Engine & architecture**

- **No `vivarium-core`.** The model is a process-bigraph state document run by
  the process-bigraph engine.
- **Partitioned scheduling** (`v2ecoli/steps/partition.py`, `allocator.py`):
  contended processes split into a *requester* (declares demand) and an
  *evolver* (acts on the allocation), coordinated by an allocator — the
  vEcoli-parity execution order, made explicit in the composite.
- **`EcoliWCM` bridge** (`v2ecoli/bridge.py`) wraps a whole cell as a single
  process so many cells compose into a colony.
- **ParCa decomposed** into nine inspectable Steps and shipped pre-computed.
- **Typed, serializable state** — bigraph-schema types with units; save states
  round-trip through JSON (no pickle outside the ParCa cache).

**Biology — process inventory (ground truth)**

| Group | Count | Location |
|---|---|---|
| Biological process modules | 17 | `v2ecoli/processes/*.py` |
| Listener / deriver steps | 8 | `v2ecoli/steps/derivers/*.py` |
| ParCa pipeline Steps | 9 | `v2ecoli/processes/parca/steps/*.py` |

The 17 modules implement the same biology vEcoli spreads across more process
classes; after requester/evolver partitioning and infrastructure steps, the
running baseline composite schedules ~45 steps per tick. One notable
restructuring: polypeptide elongation's old strategy-pattern variants
(base / translation-supply / steady-state) are now three sibling
`PartitionedProcess` subclasses chosen by wiring rather than a config flag
(`v2ecoli/processes/polypeptide_elongation.py`).

**Active extensions beyond a straight port**

- **PDMP metabolism** — a Millard-2017 kinetic-ODE + LQR variant
  (`millard_pdmp_baseline`), opt-in.
- **DnaA replication-initiation** — a mechanistic-vs-heuristic investigation
  (DnaA-box catalog at `v2ecoli/data/dnaa_box_catalog.py`), on an investigation
  branch, not yet on `main`.

**Parity.** At 60 s, v2ecoli's dry mass is 384.6 fg vs vEcoli's 384.5 fg
(0.0 % drift), and time-to-division matches at ~42 min — see the
[v1 vs v2](https://vivarium-collective.github.io/v2ecoli/v1_v2_comparison.html)
and [composite comparison](https://vivarium-collective.github.io/v2ecoli/composite_comparison.html)
reports.

---

## Performance & validation

A single cell runs from birth to division (~42 simulated min) at faster than
real time. Measured for the baseline composite at a 60 s checkpoint (seed 0):

| Metric (baseline, 60 s sim) | v2ecoli |
|---|---|
| Build + cache load | ~5.6 s |
| Run (60 s simulated) | ~7.0 s |
| Realtime factor | ~8.5× |
| Dry mass at 60 s | ~384 fg |
| `vivarium-core` dependency | none |

**Validated against vEcoli observable-by-observable** — not by a single
endpoint. The published
[vEcoli-vs-v2ecoli](https://vivarium-collective.github.io/v2ecoli/v1_v2_comparison.html)
and [composite comparison](https://vivarium-collective.github.io/v2ecoli/composite_comparison.html)
reports put the engines side by side across dry mass, mass composition,
bulk-molecule counts, and replication dynamics; through the full cell cycle the
dry-mass trajectories agree to within a fraction of a percent (707.2 fg vs
705.3 fg at division). Regenerate with `reports/v1_v2_report.py` /
`reports/composite_comparison.py`.

---

## Known limitations

- **Colony throughput** — the `EcoliWCM` bridge runs each cell's internal
  composite synchronously (~0.7 s/tick), so a colony with one whole-cell runs at
  ~2.6× realtime.
- **Daughter state** — daughter `EcoliWCM` processes start from a fresh composite
  rather than inheriting the mother's internal state at division.
- **Cell-length transient** — at the WCM's starting volume the capsule-geometry
  volume→length map gives a shorter cell than expected, so length dips before
  climbing.
- **Division mechanism** — the `Division` step fires via exception handling (it
  attempts a structural modification that crashes; the bridge catches it and
  applies the handoff). Clean structural division is on the roadmap.

---

## Repository layout

```
v2ecoli/
  composites/      baseline · colony · parca · millard_pdmp_baseline
  processes/       17 biological processes + parca/ (9-Step pipeline)
  steps/           infrastructure steps + derivers/ (8 listeners)
  types/           bigraph-schema types (units, bulk/unique arrays, listeners)
  workflow/        run.py (v2ecoli-workflow) · lineage · variants · analysis
  library/         emitters (parquet, xarray), unit bridge, cache versioning
  visualizations/  Visualization Steps backing each report
  configs/         default · two_generations · two_generations_xarray
  bridge.py        EcoliWCM whole-cell wrapper
  core.py          build_core() + cache loading
cli/               v2ecoli-parca · v2ecoli-colony entry points
pbg_v2ecoli/       workspace package (dashboard build_core + EcoliWCM link)
reports/           CLI orchestrators (one per published report)
scripts/           viz_*, parca_*, build_cache, pr_session_report, sweep_report
models/            pre-computed sim_data + serialized .pbg documents
docs/              published GitHub Pages reports
tests/             unit + integration + ParCa-alignment + behavior gates
workspace.yaml     pbg workspace config
```

---

## Dependencies & ecosystem

- [process-bigraph](https://github.com/vivarium-collective/process-bigraph) — simulation engine (Composite / Process / Step)
- [bigraph-schema](https://github.com/vivarium-collective/bigraph-schema) — type system + auto-discovery
- [pbg-superpowers](https://github.com/vivarium-collective/pbg-superpowers) — `@composite_generator`, `Visualization`
- [pbg-emitters](https://github.com/vivarium-collective/pbg-emitters) — `ParquetEmitter`
- [vivarium-dashboard](https://github.com/vivarium-collective/vivarium-dashboard) — interactive workspace UI (reads `workspace.yaml` + `pbg_v2ecoli/`)
- [vEcoli](https://github.com/CovertLab/vEcoli) — ParCa reference data & biology
- [multi-cell](https://github.com/vivarium-collective/pymunk-process) — 2D colony physics

---

## Contributing

Humans: read [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). AI assistants and anyone editing
process code, composite wiring, or the type system: read [AGENTS.md](AGENTS.md)
first — it documents the schema round-trip / port-contract / units / conservation
checks every process change must pass, plus the parity gate and PR conventions.
