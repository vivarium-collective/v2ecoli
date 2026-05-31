# v2ecoli

Pure [process-bigraph](https://github.com/vivarium-collective/process-bigraph)
port of the Covert lab's whole-cell *E. coli* model
([vEcoli](https://github.com/CovertLab/vEcoli)). 55 biological processes, no
vivarium-core dependency, same biology to within 0.0% dry-mass drift at 60 s.

The Parameter Calculator (ParCa) — the ~70 min knowledge-base build that
fits `sim_data` — is decomposed into nine process-bigraph Steps and ships
pre-computed at `models/parca/parca_state.pkl.gz`, so a fresh clone can
simulate end-to-end without rebuilding.

## 🔬 Interactive model viewer

Explore the model's bigraph structure in your browser — processes, stores,
wiring, input/output port schemas (with units), and a formal mathematical
description for each process:

- **[▶ Baseline composite](https://vivarium-collective.github.io/v2ecoli/bigraph_baseline.html)** — the whole-cell *E. coli* model
- **[▶ ParCa pipeline](https://vivarium-collective.github.io/v2ecoli/bigraph_parca.html)** — the nine-Step parameter calculator

Pan/zoom, expand store chips, toggle nodes on/off, and click a process to see
its inputs/outputs and governing equations. Regenerate locally with
`python scripts/viz_baseline_interactive.py` (or `viz_parca_interactive.py`).

## Install

Requires [`uv`](https://docs.astral.sh/uv/) and a C compiler (Xcode CLI
tools on macOS, `build-essential` on Linux).

```bash
git clone https://github.com/vivarium-collective/v2ecoli.git
cd v2ecoli
uv sync
```

`uv sync` provisions Python 3.12, installs every dependency (vEcoli,
process-bigraph, bigraph-schema, multi-cell), and compiles the three
vendored Cython extensions.

## Quick start

```bash
# Single cell to division (~42 simulated min)
python reports/workflow_report.py

# Three-cell colony with N adder-grow-divide surrogates
python reports/colony_report.py --duration 45 --n-adder 5

# vEcoli ↔ v2ecoli benchmark
python reports/benchmark_report.py
```

Each script writes a self-contained HTML report under `out/` and opens it.

## Published reports

[vivarium-collective.github.io/v2ecoli](https://vivarium-collective.github.io/v2ecoli/)
— cell lifecycle, colony, ParCa pipeline,
multi-generation lineage, and the baseline composition graph.
Regenerated from the corresponding `reports/*.py` scripts.

## Performance

| | vEcoli | v2ecoli |
|---|---|---|
| 60 s simulated wall | 6.5 s | 7.3 s |
| Realtime factor | 9.2× | 8.2× |
| Dry mass at 60 s | 384.5 fg | 384.6 fg |
| Time to division | ~42 min | ~42 min |
| vivarium-core | required | none |

## Known limitations

- **Colony throughput**: the `EcoliWCM` bridge runs each cell's internal
  composite synchronously (~0.7 s per tick), so a colony sim with one
  whole-cell ecoli runs at ~2.6× realtime.
- **Daughter state**: daughter `EcoliWCM` processes start from a fresh
  composite — they don't inherit the mother's internal state at division.
- **Cell length transient**: at the WCM's starting volume, the
  capsule-geometry volume→length map gives a shorter cell than expected,
  so length dips before climbing.
- **Division mechanism**: the `Division` step fires via exception
  handling (it attempts a structural modification that crashes; the
  bridge catches and applies the handoff). Clean structural division is
  on the roadmap.

## Architecture

Three architectures share the same 55 processes and 9 listeners; they
differ in how they're scheduled.

| Architecture | Composite generator | What it is |
|---|---|---|
| baseline | `v2ecoli.composites.baseline` | Partitioned (requester/allocator/evolver) — vEcoli-parity |

Each whole cell is wrapped by `EcoliWCM`, a process-bigraph `Process`
that holds an internal `Composite` and a bridge:

| External port | Internal store |
|---|---|
| `local` (in) | `boundary.external` |
| `mass` (out) | `listeners.mass.dry_mass` |
| `length` (out) | from `volume` via capsule geometry |
| `volume` (out) | `listeners.mass.volume` |

At division (~702 fg dry mass) the bridge swaps the mother for two
daughters with fresh `EcoliWCM`s and phylogeny-mutated colors.

## ParCa

ParCa is the build step that fits ~130 knowledge-base TSVs into a
`SimulationDataEcoli` blob the runtime reads from. The original is one
monolithic `fitSimData_1()`; v2ecoli decomposes it into nine
process-bigraph Steps with explicit ports, wired through the composite's
store. Stage 5 (`fit_condition`) is the ~70 min cost; everything else is
seconds.

- Pre-computed `sim_data`: `models/parca/parca_state.pkl.gz` (18 MB).
- Re-run from scratch: `v2ecoli-parca --mode fast` (~70 min).
- Resume from cached step-5 checkpoint: `bash scripts/parca_rerun_from_step5.sh`.
- Refresh BioCyc-sourced flat files: `python scripts/parca_update_biocyc.py`.

## Auto-discovery

When v2ecoli is pip-installed alongside `bigraph-schema` and
`process-bigraph`, all biological processes register themselves on
`allocate_core()` — no manual `core.register_link()` calls:

```python
from process_bigraph import allocate_core
core = allocate_core()
# core.link_registry now contains v2ecoli.processes.* by both bare and
# fully-qualified names.
```

## Dependencies

- [process-bigraph](https://github.com/vivarium-collective/process-bigraph) — simulation framework
- [bigraph-schema](https://github.com/vivarium-collective/bigraph-schema) — type system + discovery
- [vEcoli](https://github.com/CovertLab/vEcoli) — ParCa reference data
- [multi-cell](https://github.com/vivarium-collective/pymunk-process) — 2D colony physics

## Layout

```
v2ecoli/
  composites/    baseline · colony · millard_pdmp_baseline
  processes/     15 biological processes + parca/ (9-step pipeline)
  steps/         infrastructure + 9 listeners
  visualizations/ Visualization Steps backing each report
  bridge.py      EcoliWCM wrapper
  colony.py      make_colony() multi-cell builder
reports/         CLI orchestrators (one per published report)
models/          pre-computed sim_data + serialized .pbg documents
tests/           unit + integration + ParCa alignment
```
