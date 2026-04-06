# v2ecoli

Whole-cell *E. coli* model built natively on [process-bigraph](https://github.com/vivarium-collective/process-bigraph).

[![Benchmark](https://github.com/vivarium-collective/v2ecoli/actions/workflows/benchmark.yml/badge.svg)](https://github.com/vivarium-collective/v2ecoli/actions/workflows/benchmark.yml)

**[Benchmark Report](https://htmlpreview.github.io/?https://github.com/vivarium-collective/v2ecoli/blob/main/doc/benchmark_report.html)** | **[GitHub Pages](https://vivarium-collective.github.io/v2ecoli/benchmark/benchmark_report.html)**

## Overview

v2ecoli migrates the [vEcoli](https://github.com/CovertLab/vEcoli) whole-cell *E. coli* model to run on process-bigraph's `Composite.run()`. All 55 biological steps execute through the standard process-bigraph pipeline with custom types handling bulk molecule arrays, unique molecule updates, and listener stores.

### Key Results

| Metric | Value |
|--------|-------|
| Worst mass % error vs v1 | **0.62%** |
| Mass component R² | **> 0.99** |
| Simulation speed | **~6x realtime** |
| Biological processes | **55 steps + 1 process** |
| Division | **State splitting + daughter viability** |
| Bulk molecule conservation | **Exact (binomial split)** |

## Quick Start

```bash
# Install
git clone https://github.com/vivarium-collective/v2ecoli.git
cd v2ecoli
uv sync

# Run simulation (requires cached simData)
uv run python -c "
from v2ecoli.composite import make_composite
ecoli = make_composite(cache_dir='out/cache')
ecoli.run(60.0)
print(f'global_time: {ecoli.state[\"global_time\"]}')
"

# Run benchmark report
uv run python benchmark.py
open out/benchmark/benchmark_report.html
```

## Architecture

```
v2ecoli/
├── reconstruction/          # ParCa pipeline + 133 TSV raw data files
│   └── ecoli/              # knowledge_base_raw, fit_sim_data_1, simulation_data
├── v2ecoli/
│   ├── library/            # Schema helpers, units, polymerize, FBA, fitting
│   ├── types/              # Custom bigraph-schema types
│   │   ├── bulk_numpy.py   # BulkNumpyUpdate (index-add on structured arrays)
│   │   ├── unique_numpy.py # UniqueNumpyUpdate (accumulate/flush pattern)
│   │   └── stores.py       # InPlaceDict, SetStore, ListenerStore
│   ├── steps/              # process-bigraph Steps
│   │   ├── partition.py    # PartitionedProcess, Requester, Evolver
│   │   ├── allocator.py    # Priority-based molecule allocation
│   │   ├── division.py     # Cell division detection
│   │   └── listeners/      # 9 listener steps (mass, RNA, protein, etc.)
│   ├── processes/          # 15 biological processes
│   │   ├── metabolism.py, chromosome_replication.py, ...
│   │   ├── transcript_initiation.py, polypeptide_elongation.py, ...
│   │   └── tf_binding.py, equilibrium.py, complexation.py, ...
│   ├── composite.py        # make_composite() → Composite.run()
│   ├── generate.py         # build_document() from simData
│   └── cache.py            # JSON state caching with numpy support
├── benchmark.py            # Benchmark suite with HTML report
└── test_types.py           # Type system unit tests
```

## How It Works

### Simulation Pipeline

```
Raw Data (133 TSVs) → ParCa → simData → LoadSimData → Process Configs
                                              ↓
                                    build_document()
                                              ↓
                               Composite (55 steps + GlobalClock)
                                              ↓
                                    Composite.run(duration)
```

### Custom Types

The simulation uses custom bigraph-schema types with dispatch methods:

- **`BulkNumpyUpdate`**: Bulk molecule structured arrays. `apply` adds `(index, value)` tuples to the `count` field.
- **`UniqueNumpyUpdate`**: Unique molecule arrays. `apply` accumulates set/add/delete operations and flushes on signal.
- **`InPlaceDict`**: Dicts that mutate in place (preserving object identity across apply cycles).
- **`SetStore`**: Full replacement on apply (for request/allocate stores).
- **`ListenerStore`**: In-place merge at top level, set at leaves.

### Sequential Execution

The Composite uses `sequential_steps=True` mode which runs one step at a time by priority (highest first), ensuring the correct execution order for the partitioned process workflow (requesters → allocators → evolvers).

## Benchmark

The benchmark suite (`benchmark.py`) measures:

1. **Document Build** — time to load cached simData and instantiate all processes
2. **Simulation** — wall time, molecules changed, mass growth trajectory
3. **v1 Comparison** — per-timestep Pearson correlation of all 16,321 molecule counts
4. **Long Simulation** — 8+ minute run showing growth dynamics
5. **Step Diagnostics** — per-step analysis of config, ports, wiring

Run locally:
```bash
uv run python benchmark.py
```

## Prerequisites

- Python 3.12.9
- [uv](https://docs.astral.sh/uv/)
- vEcoli (for v1 comparison and simData generation): `pip install vEcoli[dev]`
- `simData.cPickle` — generated by ParCa or cached in `out/cache/`
- graphviz (for bigraph visualization): `brew install graphviz` / `apt install graphviz`
