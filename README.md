# v2ecoli

**Pure process-bigraph E. coli whole-cell model**

v2ecoli is the next-generation whole-cell *E. coli* simulation built entirely on 
[process-bigraph](https://github.com/vivarium-collective/process-bigraph) — 
no vivarium-core dependency. It ports all 55 biological processes from 
[vEcoli](https://github.com/CovertLab/vEcoli) with identical biological output 
and comparable performance.

## Reports

- **[Workflow Report](https://vivarium-collective.github.io/v2ecoli/workflow_report.html)** — 
  Full cell lifecycle: single cell to division (~42 min), daughter simulations, 
  chromosome replication, growth metrics
- **[Colony Report](https://vivarium-collective.github.io/v2ecoli/colony_report.html)** — 
  Mixed colony: 1 whole-cell E. coli + surrogate cells with 2D pymunk physics, 
  growth, and division
- **[Architecture Comparison](https://vivarium-collective.github.io/v2ecoli/comparison_report.html)** — 
  Partitioned vs departitioned: 42-min side-by-side comparison of mass trajectories, 
  bulk counts, growth rates, and timing (55 vs 41 steps)

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
python3 workflow.py

# Colony simulation (1 wc-ecoli + N surrogates)
python3 colony_report.py --duration 45 --n-adder 5

# Partitioned vs departitioned architecture comparison (42 min)
python3 compare_report.py

# Benchmark comparison
python3 benchmark.py

# Three-way comparison (vEcoli 1.0, vEcoli 2.0, v2ecoli)
python3 compare_v1_v2.py --duration 2500
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
├── workflow.py             # single cell lifecycle report
├── colony_report.py        # colony simulation report
├── compare_report.py       # partitioned vs departitioned comparison
├── benchmark.py            # v2ecoli vs vEcoli benchmark
├── compare_v1_v2.py        # three-way engine comparison
├── models/
│   └── partitioned.pbg     # serialized model document
└── docs/                   # GitHub Pages reports
```
