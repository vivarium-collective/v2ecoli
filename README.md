# v2ecoli

**Pure process-bigraph E. coli whole-cell model**

v2ecoli is the next-generation whole-cell *E. coli* simulation built entirely on 
[process-bigraph](https://github.com/vivarium-collective/process-bigraph) — 
no vivarium-core dependency. It ports all 55 biological processes from 
[vEcoli](https://github.com/CovertLab/vEcoli) with identical biological output 
and comparable performance.

The Parameter Calculator (ParCa) that turns the raw knowledge base into
a fitted `sim_data` also lives in this repo, as the nine-Step composite
at [`v2ecoli/processes/parca/`](v2ecoli/processes/parca/).  vEcoli is
used only by the side-by-side comparison report — the runtime and the
flat-file knowledge base are both vendored, so a fresh clone can run
the full build → simulate → divide pipeline end-to-end without any
external vEcoli checkout.

## Reports

**[All reports](https://vivarium-collective.github.io/v2ecoli/)** —
cell lifecycle, colony, architecture comparisons, ParCa pipeline, and
composition-graph visualizations.

## ParCa

The Parameter Calculator (ParCa) is the build step that turns ~130
TSVs of raw knowledge-base data (genes, RNAs, proteins, reactions,
concentrations, DNA sites) into a fitted `SimulationDataEcoli` that
the online model reads from.  In the Covert-lab vEcoli the ParCa is a
single monolithic `fitSimData_1()` call; v2ecoli decomposes it into
nine process-bigraph `Step`s with explicit, named ports, so each
stage's inputs and outputs are wired through the composite's store
rather than threaded through a deeply-mutated `sim_data` blob.

### Pipeline

| Step | What it does |
|---|---|
| 1. initialize | Bootstraps `SimulationDataEcoli` from the flat files; scatters tables across the 20 subsystem objects (Mass, Constants, Transcription, Translation, Metabolism, …). |
| 2. input_adjustments | Hand-curated corrections to expression, degradation rates, translation efficiencies, and small-molecule targets. |
| 3. basal_specs | Builds the reference cell spec (M9 + glucose, no TF perturbations) — the anchor every condition perturbs away from. |
| 4. tf_condition_specs | Per-TF activation replay of step 3; produces one spec per (condition, TF) pair.  ~50 s in fast mode. |
| 5. fit_condition | Solves for bulk distributions + translation supply rates that make the specs self-consistent.  **~70 min** — the dominant cost, and the reason the step-5 checkpoint is memoized. |
| 6. promoter_binding | CVXPY fits of TF-promoter recruitment strengths and binding probabilities. |
| 7. adjust_promoters | Ligand concentrations + RNAP recruitment adjusted to match step 6. |
| 8. set_conditions | Materializes per-nutrient dicts the online simulation indexes by `condition`. |
| 9. final_adjustments | ppGpp kinetics + amino-acid supply constants + mechanistic kcat/KM corrections. |

The per-step port manifests ship in
[`v2ecoli/processes/parca/steps/`](v2ecoli/processes/parca/steps/); the
wire table is [`STORE_PATH`](v2ecoli/processes/parca/composite.py).

### Using the fitted `sim_data`

A gzipped pickle of the final state ships at
[`models/parca/parca_state.pkl.gz`](models/parca/README.md) (18 MB)
so downstream code can skip the ~70-min build:

```python
from v2ecoli.processes.parca.data_loader import load_parca_state
state = load_parca_state()

transcription = state['process']['transcription']
sim_data_root = state['sim_data_root']   # full SimulationDataEcoli
```

The loader transparently aliases legacy `v2parca.*` / `vparca.*`
pickle module paths, so older artifacts deserialize cleanly.

The `workflow.py` pipeline's `step_parca` hydrates this fixture by
default (~2 s) and dills its `sim_data_root` into
`out/workflow/simData.cPickle` for downstream `save_cache()`.  Pass
`--parca-rerun` to re-execute the full 9-Step composite instead.

### Re-running from scratch

```bash
# One-time: build the three vendored Cython extensions for the
# current Python.
bash scripts/parca_cython_build.sh

# Full pipeline (~70 min in fast mode)
python scripts/parca_run.py --mode fast --cpus 4

# Or resume from a cached step-5 checkpoint (~15 s)
bash scripts/parca_rerun_from_step5.sh
```

### Reconstruction data

The flat-file knowledge base is vendored at
[`v2ecoli/processes/parca/reconstruction/ecoli/flat/`](v2ecoli/processes/parca/reconstruction/ecoli/flat/)
(133 files, ~12.9 MB).  Ten of these come from EcoCyc and can be
refreshed from the BioCyc webservice:

```bash
python scripts/parca_update_biocyc.py
```

The remaining ~120 are Covert-lab-curated overlays (diffs, modifier
files, manually-added entries, compartment definitions, …) that live
only in this repo's history.  The comparison report surfaces both
sets with per-file GitHub links and EcoCyc/Curated source badges.

### Comparison report

[`scripts/parca_compare.py`](scripts/parca_compare.py) generates a
self-contained HTML report with per-step runtime, port manifests, and
scalar/distribution/cell_specs diffs against the original vEcoli
`fitSimData_1` (the only place v2ecoli pulls from the Covert-lab
vEcoli tree).  The rendered output is published at
[vivarium-collective.github.io/v2ecoli/parca_comparison_report.html](https://vivarium-collective.github.io/v2ecoli/parca_comparison_report.html).

### Tests

| File | What it covers |
|---|---|
| [`tests/test_parca_ports_and_wiring.py`](tests/test_parca_ports_and_wiring.py) | Static checks: every declared port is in `STORE_PATH`; tick sequencing is valid; composite builds with a mocked raw_data.  Fast. |
| [`tests/test_parca_fixture_roundtrip.py`](tests/test_parca_fixture_roundtrip.py) | Loads `models/parca/parca_state.pkl.gz`, asserts 24 top-level keys + 9 subsystem classes. |
| [`tests/test_parca_alignment_vs_vecoli.py`](tests/test_parca_alignment_vs_vecoli.py) | Scalar-drift gate vs vEcoli's step-1-4 intermediates.  Skips cleanly when reference pickles aren't present. |
| [`tests/test_workflow_parca_integration.py`](tests/test_workflow_parca_integration.py) | `workflow.step_parca` fast-path hydration (< 15 s); path sanity (no `../vEcoli/` references); KB stats via merged `KnowledgeBaseEcoli`. |

### Composition diagrams

Interactive per-architecture network views: click a process to inspect its
inputs/outputs, port schemas, config, docstring, and mathematical model
(with role-specific math for requester/evolver halves of partitioned
processes). Bipartite layout by default (stores on the left, processes on
the right); switch to dagre/fcose/grid/etc. from the dropdown.

- **[Baseline (partitioned)](https://vivarium-collective.github.io/v2ecoli/network_baseline.html)** 
- **[Departitioned](https://vivarium-collective.github.io/v2ecoli/network_departitioned.html)** 
- **[Reconciled](https://vivarium-collective.github.io/v2ecoli/network_reconciled.html)** 

These are also embedded as iframes inside the Architecture Comparison
report and generated by `python reports/compare_report.py` or, individually, by
`python reports/network_report.py --model baseline|departitioned|reconciled`.

## Architecture

```
multi-cell colony (pymunk 2D physics)
  └─ cells/
       ├─ surrogate_1 (AdderGrowDivide)   ← grey
       ├─ surrogate_2 (AdderGrowDivide)   ← grey
       ├─ ecoli_0 (EcoliWCM bridge)       ← colored
       │    └─ internal Composite
       │         ├─ 11 PartitionedProcesses (requester/allocator/evolver)
       │         ├─ 4 non-partitioned Steps (metabolism, chromosome structure, ...)
       │         ├─ 9 listener Steps
       │         └─ infrastructure (global clock, unique update, emitter)
       └─ ...
```

Each whole-cell E. coli is wrapped by `EcoliWCM`, a process-bigraph `Process` 
that holds an internal `Composite` with a bridge mapping:

| External port | Internal store |
|---------------|----------------|
| `local` (input) | `boundary.external` |
| `mass` (output) | `listeners.mass.dry_mass` |
| `length` (output) | computed from volume via capsule geometry |
| `volume` (output) | `listeners.mass.volume` |

When the internal model reaches division (~702 fg dry mass), the bridge 
removes the mother cell and adds two daughter cells with fresh `EcoliWCM` 
processes and phylogeny-mutated colors.

## Performance

| Metric | vEcoli (composite) | v2ecoli |
|--------|-------------------|---------|
| 60s simulation | 6.5s wall | 7.3s wall |
| Speed | 9.2x realtime | 8.2x realtime |
| Division time | ~42 min | ~42 min |
| dry_mass at 60s | 384.5 fg | 384.6 fg |
| Mass difference | — | 0.0% |
| vivarium-core | required | **none** |

## Quick Start

```bash
# Single cell simulation (runs to division)
python3 reports/workflow_report.py

# Colony simulation (1 wc-ecoli + N surrogates)
python3 reports/colony_report.py --duration 45 --n-adder 5

# Partitioned vs departitioned architecture comparison (42 min)
python3 reports/compare_report.py

# Benchmark comparison
python3 reports/benchmark_report.py

# Three-way comparison (vEcoli 1.0, vEcoli 2.0, v2ecoli)
python3 reports/v1_v2_report.py --duration 2500
```

## Dependencies

- [process-bigraph](https://github.com/vivarium-collective/process-bigraph) — composable simulation framework
- [bigraph-schema](https://github.com/vivarium-collective/bigraph-schema) — type system
- [vEcoli](https://github.com/CovertLab/vEcoli) — ParCa parameter calculator (for config generation)
- [multi-cell](https://github.com/vivarium-collective/pymunk-process) — 2D colony physics (pymunk)

## Project Structure

```
v2ecoli/
├── v2ecoli/
│   ├── composite.py        # make_composite() entry point
│   ├── generate.py         # build_document(), execution layers
│   ├── bridge.py           # EcoliWCM: whole-cell model as Process with bridge
│   ├── colony.py           # make_colony() for multi-cell simulations
│   ├── processes/          # 15 biological processes (from vEcoli)
│   ├── steps/              # infrastructure + 9 listeners
│   ├── types/              # custom bigraph-schema types
│   └── library/            # shared utilities
├── reports/
│   ├── workflow_report.py        # single cell lifecycle report
│   ├── multigeneration_report.py # N-generation single lineage report
│   ├── colony_report.py          # colony simulation report
│   ├── compare_report.py         # partitioned vs departitioned comparison
│   ├── network_report.py         # per-architecture Cytoscape diagram
│   ├── benchmark_report.py       # v2ecoli vs vEcoli benchmark
│   └── v1_v2_report.py           # three-way engine comparison
├── models/
│   └── partitioned.pbg     # serialized model document
└── docs/                   # GitHub Pages reports
```
