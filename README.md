# v2ecoli

> **Explore in your browser — no install:**
> &nbsp;🖥️ **[Interactive dashboard](https://vivarium-collective.github.io/v2ecoli/dashboard/)** (browse the whole workspace)
> &nbsp;·&nbsp; 🧬 **[3D *E. coli* cell](https://pub-eb913fbbdc584bd7add047c823570b13.r2.dev/viewer/index.html?models=https://pub-eb913fbbdc584bd7add047c823570b13.r2.dev/ecoli-3d/viz/3d/models.json)** (interactive whole-cell structural model — switch between a newborn cell and a pre-division cell with two segregated chromosomes; opens directly; **"View in VR"** on a Meta Quest)
> &nbsp;·&nbsp; 📊 **[Baseline Showcase report](https://vivarium-collective.github.io/v2ecoli/investigations/v2ecoli-baseline-showcase.html)** ⭐ (best starting point)
> &nbsp;·&nbsp; 🗂️ **[Report gallery](https://vivarium-collective.github.io/v2ecoli/)**
> — see **[Explore v2ecoli](#explore-v2ecoli)** for what each one is.

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
  the same parts* ([Architectures](#architectures)). Swapping a subsystem is a
  one-line change, not a fork.
- **The repository is a pbg research workspace.** Alongside the model code live
  **investigations** (a research question) and **studies** (the simulations that
  answer it) — browsable and runnable in the
  [vivarium-workbench](https://github.com/vivarium-collective/vivarium-workbench).
  Investigations live under `workspace/investigations/` (baseline showcase,
  PDMP, colonies, and more), each publishing a self-contained
  [investigation report](#explore-v2ecoli).
  New science is added as a study, not a patch to a monolith.

The biology stays faithful to upstream: v2ecoli reproduces vEcoli's cell-cycle
trajectories from birth to division — dry mass, mass composition, bulk-molecule
counts, and replication dynamics, not just a single endpoint — tracking dry mass
to within a fraction of a percent through the full cell cycle, with ~42 min
division timing in both ([Performance & validation](#performance--validation)).
The ParCa is shipped **pre-computed**, so a fresh clone simulates end-to-end
without the ~70 min knowledge-base rebuild.

**New here?** Browse the published [dashboard and reports](#explore-v2ecoli)
first, then read [What v2ecoli is](#what-v2ecoli-is) → [Install](#install) →
[Quick start](#quick-start) →
[the framework](#the-process-bigraph-framework) →
[Architectures](#architectures) →
[pipelines](#running-pipelines-multiseed--multigen--multivariant).

> 🤖 **Using an AI coding assistant (Claude Code / Cursor / …)?** Hand it
> **[docs/first-run-agent-guide.md](docs/first-run-agent-guide.md)** — a gated
> runbook that takes you from a clean clone to a running vivarium-workbench with
> the `baseline` cell open, then on to authoring and contributing.

---

## Explore v2ecoli

Everything is published to GitHub Pages and runs in the browser — no install.
There are **three** kinds of output, each from a different pipeline.

### 🖥️ Interactive dashboard — *browse the whole workspace*

**[→ vivarium-collective.github.io/v2ecoli/dashboard/](https://vivarium-collective.github.io/v2ecoli/dashboard/)**

A read-only snapshot of the v2ecoli workspace in the
[vivarium-workbench](https://github.com/vivarium-collective/vivarium-workbench):
investigations & studies, the process/type **registry**, navigable **composite**
wiring graphs (`baseline`, `parca`), and the **sources** bundle. Auto-rebuilt
from `main` on every push. For the full interactive version (authoring, running
studies), clone and run `vivarium-workbench serve` locally.

### 📊 Investigation reports — *one self-contained report per research question*

Each investigation under `workspace/investigations/` auto-publishes a single,
self-contained HTML report — overview → studies → figures → reviewer decisions —
rebuilt from `main` on every push. **Start with the
[Baseline Showcase ⭐](https://vivarium-collective.github.io/v2ecoli/investigations/v2ecoli-baseline-showcase.html)**,
then browse the rest from the
[report gallery](https://vivarium-collective.github.io/v2ecoli/) or interactively
in the [dashboard](https://vivarium-collective.github.io/v2ecoli/dashboard/). Full
list with research questions: **[docs/reports.md](docs/reports.md#2-investigation-reports-auto-generated)**.

### 🧬 3D whole-cell structural model — *molecular-scale, in your browser*

An interactive **[3D structural model of the *E. coli* cell](https://pub-eb913fbbdc584bd7add047c823570b13.r2.dev/viewer/index.html?models=https://pub-eb913fbbdc584bd7add047c823570b13.r2.dev/ecoli-3d/viz/3d/models.json)** —
hundreds of molecular species (ribosomes, RNA polymerase, metabolic enzymes,
the supercoiled chromosome, flagella) packed at true abundance from a v2ecoli
cell state. Switch between a **newborn cell** and a **pre-division cell** with
two segregated chromosomes, toggle/isolate species by functional category, and
**"View in VR"** on a Meta Quest. Built by the
**[3d-ecoli](https://github.com/vivarium-collective/3d-ecoli)** workspace — which
imports v2ecoli (for the molecular state) +
**[pbg-parsimony](https://github.com/vivarium-collective/pbg-parsimony)** (the
packing engine) — from the molecular counts of a simulated cell.

### 🔬 Model viewers & technical reports — *standalone HTML*

Generated on demand by `reports/*.py` / `scripts/*.py` (committed under `docs/`):
interactive [model-wiring viewers](https://vivarium-collective.github.io/v2ecoli/baseline-viewer/),
[simulation-result reports](https://vivarium-collective.github.io/v2ecoli/workflow_report.html),
[vEcoli comparisons](https://vivarium-collective.github.io/v2ecoli/v1_v2_comparison.html), and the
[mathematical structure](https://vivarium-collective.github.io/v2ecoli/math_structure.html).
Browse them from the **[report gallery](https://vivarium-collective.github.io/v2ecoli/)** —
full list + how to regenerate each in **[docs/reports.md](docs/reports.md#3-standalone-html-reports)**.

---

## Contents

- [Explore v2ecoli](#explore-v2ecoli) — dashboard, investigation reports, HTML reports
- [First-run agent guide](docs/first-run-agent-guide.md) — hand it to your AI assistant to get running end-to-end
- [What v2ecoli is](#what-v2ecoli-is)
- [Install](#install)
- [Quick start](#quick-start)
- [The process-bigraph framework](#the-process-bigraph-framework)
- [Architectures](#architectures)
- [Running pipelines: multiseed / multigen / multivariant](#running-pipelines-multiseed--multigen--multivariant)
- [Emitters: how output is written (Parquet & Xarray)](#emitters-how-output-is-written-parquet--xarray)
- [ParCa](#parca)
- [What changed since vEcoli](#what-changed-since-vecoli)
- [Performance & validation](#performance--validation)
- [v2ecoli ↔ vEcoli comparison harness](#v2ecoli--vecoli-comparison-harness)
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
  **studies** (the runs that answer it) under `workspace/`, all browsable and
  runnable in the vivarium-workbench. Manage them with the `viva-investigation` /
  `viva-study` skills (from the [viva-superpowers](https://github.com/vivarium-collective/viva-superpowers)
  Claude Code plugin).
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

> 🤖 **Want an AI assistant to drive the whole setup** (clone → workbench →
> first sim)? Hand it [docs/first-run-agent-guide.md](docs/first-run-agent-guide.md).

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

**Types & units.** Project-specific types live in `v2ecoli/types/` (e.g.
`Quantity`, `CSRMatrix`, `BulkNumpyUpdate`, `ListenerStore`). Dimensioned
quantities at ports are `pint.Quantity`; the only place `Unum` survives is the
upstream-interop bridge at `v2ecoli/library/unit_bridge.py`.

**The `pbg_v2ecoli/` package** at the repo root is the *workspace* package the
[vivarium-workbench](https://github.com/vivarium-collective/vivarium-workbench)
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

> The vivarium-workbench's Simulations-DB tab currently reads SQLite, not
> Parquet/Zarr — those are for offline DuckDB / xarray analysis.

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

Beyond the headline changes in [What v2ecoli is](#what-v2ecoli-is) (no
`vivarium-core`, composition over configuration, typed/serializable state, a
decomposed pre-computed ParCa), the architecture-level specifics are:

- **Partitioned scheduling** (`v2ecoli/steps/partition.py`, `allocator.py`):
  contended processes split into a *requester* (declares demand) and an
  *evolver* (acts on the allocation), coordinated by an allocator — the
  vEcoli-parity execution order, made explicit in the composite.
- **`EcoliWCM` bridge** (`v2ecoli/bridge.py`) wraps a whole cell as a single
  process so many cells compose into a colony ([Architectures](#architectures)).

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

**Parity.** v2ecoli matches vEcoli observable-by-observable through the full
cell cycle — see [Performance & validation](#performance--validation).

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

## v2ecoli ↔ vEcoli comparison harness

A single, reproducible entry point that validates v2ecoli as a **faithful port**
of vEcoli by running BOTH engines on GovCloud and grading them with one canonical
report card. The whole flow is `scripts/comparison_harness.sh`
(`register` → `launch` → `report`, or `all`).

**What it compares — matched timepoints, not snapshots.** v2ecoli and vEcoli share
the same ParCa, processes, and initial state, so they should track each other. The
harness lines both engines up on a shared simulation-seconds axis and grades
**matched generation-1 timepoints**. It does *not* compare end-of-run snapshots —
those catch the two cells at different cell-cycle phases (one may have just
divided) and produce ±100% artifacts that are pure phase, not divergence. At
matched basal timepoints the masses track to ~1%.

**Grading — the canonical report card.** `scripts/comparison_report_card.py` feeds
per-seed gen-1 means into `v2ecoli.library.report_card.grade_card`, which emits a
machine-readable `verdict.json` (schema `report_card_verdict/v1`, **vEcoli =
reference model**). Each of the 7 axes (cell / dry / protein / RNA mass +
growth_rate + active_RNAP + active_ribosome) gets a Welch **t-test** plus a
relative-Δ band: `within_tol` (≤5%), `drift` (≤10%), or `mismatch` (>10%);
absent observables are reported `ungraded`, not faked.

### Reproduce it

Prereqs (one-time per session):

```bash
aws sso login --profile stanford-sso
# SSM tunnel to sms-api on localhost:8080 (run in its own terminal — flaky under a harness)
nohup bash ~/code/sms-cdk/scripts/ptools-proxy.sh -s smsvpctest >/tmp/ptools.log 2>&1 &
```

Then the whole chain (5 conditions × 2 engines, defaults 4 seeds × 2 generations):

```bash
# register v2ecoli at the current commit (vEcoli is fixed sim id 47), launch, report:
bash scripts/comparison_harness.sh all                       # register→launch→report
# …or step by step:
bash scripts/comparison_harness.sh register                  # → v2ecoli simulator_id
bash scripts/comparison_harness.sh launch --v2-sim <ID> --seeds 4 --gens 2
#   …wait for the runs to finish on S3…
bash scripts/comparison_harness.sh report --only all
```

`launch` POSTs each run to sms-api — v2ecoli via **Ray** (`composite=v2ecoli`,
`condition=<c>`), vEcoli via **Nextflow** (`simulation_config_filename=cond_<c>.json`,
clearing the stale `nf-cond-<c>` K8s job/configmap first) — and writes the
per-condition experiment ids it created to `out/full_compare/experiments.json`.
`report` exports env AWS creds (aiobotocore's SSO refresh is unreliable; s3fs +
duckdb need real env creds) and reads that file, so launch → report is one
deterministic chain with no hand-editing. Runs land at
`s3://smsvpctest-shared-sharedbucket60d199d6-abfvwv0day91/vecoli-output/<exp>/`
(us-gov-west-1; v2ecoli zarr, vEcoli parquet under `cond_<c>/history`).

### Output artifacts (`out/full_compare/`)

- `standardized_comparison_report.html` — the multi-section report (overview →
  ParCa/initial-state → report card → per-condition matched-trajectory overlays);
  also copied to `~/Downloads/`.
- `report_card.html` — the standalone rendered card.
- `verdict.json` — the machine-readable `report_card_verdict/v1` verdict.
- `experiments.json` — the launched experiment ids the report read from.

### Known caveats

- **`total_rna_init` is excluded** — it is a unit mismatch between the engines,
  not a real divergence.
- **`active_RNAP` / `active_ribosome`** require `include_vectors=True` in
  `scripts/run_comparison_ensemble.py` (their scalar counts share a leaf name with
  the unique-molecule coordinate vectors, so the old `include_vectors=False`
  dropped them); emitters that don't export them show `ungraded`.
- **vEcoli runs via Nextflow, not the in-process native composite** — the
  `build_composite_native` path is wrapped in `run_comparison_ensemble.py` but is
  effectively dead for production comparison; the launched vEcoli reference is the
  Nextflow workflow (`simulator_id=47`, `vEcoli@62924758`).
- Until the per-condition-initial-state v2ecoli runs (`sim59-v2fix-*`) land, only
  **basal** is a fully valid v2↔vE comparison; the other conditions are launched
  but their v2ecoli initial state was basal in the older `sim48-*` runs.

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
- [vivarium-workbench](https://github.com/vivarium-collective/vivarium-workbench) — interactive workspace UI (reads `workspace.yaml` + `pbg_v2ecoli/`)
- [vEcoli](https://github.com/CovertLab/vEcoli) — ParCa reference data & biology
- [multi-cell](https://github.com/vivarium-collective/pymunk-process) — 2D colony physics
- [3d-ecoli](https://github.com/vivarium-collective/3d-ecoli) — 3D structural model (imports v2ecoli + pbg-parsimony)

---

## Contributing

Humans: read [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). AI assistants and anyone editing
process code, composite wiring, or the type system: read [AGENTS.md](AGENTS.md)
first — it documents the schema round-trip / port-contract / units / conservation
checks every process change must pass, plus the parity gate and PR conventions.
