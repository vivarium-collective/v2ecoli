# v2ecoli

Whole-cell *E. coli* model built natively on [process-bigraph](https://github.com/vivarium-collective/process-bigraph).

[![Benchmark](https://github.com/vivarium-collective/v2ecoli/actions/workflows/benchmark.yml/badge.svg)](https://github.com/vivarium-collective/v2ecoli/actions/workflows/benchmark.yml)

**[Benchmark Report](https://htmlpreview.github.io/?https://github.com/vivarium-collective/v2ecoli/blob/main/doc/benchmark_report.html)**

## Overview

v2ecoli runs the [vEcoli](https://github.com/CovertLab/vEcoli) whole-cell *E. coli* model entirely through process-bigraph's `Composite.run()`. All 55 biological steps execute through the native process-bigraph pipeline with custom bigraph-schema types for bulk molecules, unique molecules, and listener stores.

### Key Results

| Metric | Value |
|--------|-------|
| Worst mass % error vs v1 | **0.62%** |
| Mass component R² | **> 0.99** (all except small molecules) |
| Simulation speed | **~6x realtime** (60s sim in ~10s wall) |
| Execution mode | **Native flow** (`sequential_steps=False`) |
| Execution layers | **31** (layer-based flow tokens) |
| Biological processes | **55 steps + 1 process** |
| Division | **Verified at t=1800** (2 chromosomes, daughter viability) |

### v1 Comparison (60s)

| Component | Mean % Error | Max % Error | R² |
|-----------|-------------|-------------|-----|
| Dry Mass | 0.02% | 0.22% | 0.99 |
| Protein | 0.02% | 0.03% | 1.00 |
| RNA | 0.03% | 0.06% | 1.00 |
| DNA | 0.03% | 0.03% | 1.00 |
| Small Molecules | 0.02% | 0.62% | 0.78 |

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
├── v2ecoli/
│   ├── generate.py          # build_document(): simData → process edges → document
│   ├── composite.py         # make_composite(), save_state(), run_and_cache()
│   ├── cache.py             # JSON state caching with numpy support
│   ├── processes/           # 15 biological processes (PartitionedProcess subclasses)
│   ├── steps/
│   │   ├── partition.py     # PartitionedProcess, ExplicitRequester/Evolver
│   │   ├── allocator.py     # Priority-based molecule allocation (flat per-process stores)
│   │   ├── division.py      # Cell division with _add/_remove structural updates
│   │   └── listeners/       # 9 listener steps (mass, RNA, protein, etc.)
│   ├── types/               # Custom bigraph-schema types
│   │   ├── bulk_numpy.py    # BulkNumpyUpdate (index-add on structured arrays)
│   │   ├── unique_numpy.py  # UniqueNumpyUpdate (accumulate/flush pattern)
│   │   └── stores.py        # InPlaceDict, SetStore, ListenerStore
│   ├── library/             # Schema helpers, units, polymerize, FBA, fitting
│   └── reconstruction/      # ParCa pipeline + 133 TSV raw data files
├── benchmark.py             # Benchmark suite with HTML report
├── diagnose.py              # Per-step v1/v2 diagnostic comparison
└── doc/
    └── benchmark_report.html
```

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

### Execution Model

Steps are organized into **31 execution layers** with layer-based flow tokens enforcing inter-layer ordering. Within each partition layer:

1. **Requesters** compute molecule requests → write to flat `request_{proc_name}` stores
2. **Allocator** partitions available molecules by priority → writes to flat `allocate_{proc_name}` stores
3. **Evolvers** run with allocated counts → write bulk/unique/listener updates

Input/output topology separation ensures requesters only declare what they read (bulk, unique, listeners) and what they write (request store). This eliminates false dependency cycles that previously required `sequential_steps=True`.

### Custom Types

| Type | Description |
|------|-------------|
| `BulkNumpyUpdate` | Structured arrays with `(index, value)` tuple application |
| `UniqueNumpyUpdate` | Accumulate set/add/delete operations, flush on signal |
| `InPlaceDict` | Deep merge on apply |
| `SetStore` | Full replacement on apply (allocation stores) |
| `ListenerStore` | Top-level merge, leaf-level set |

### Division

Tested on pre-division state (t=1800s, 2 chromosomes, 578 fg dry mass):

1. **Detection**: dry mass >= threshold with >= 2 chromosomes
2. **State splitting** (`divide_cell()`): bulk binomial, chromosomes alternating, attached molecules follow domain
3. **Daughter construction**: `build_document_from_configs()` with divided state + cached configs
4. **Structural update**: `{'agents': {'_remove': [mother], '_add': [(d1, state), (d2, state)]}}`
5. **Viability**: daughter builds and runs 1s successfully

Generate pre-division checkpoint:
```python
from benchmark import generate_predivision_state
generate_predivision_state()  # ~4 min, saves to out/predivision.dill
```

## Benchmark

The benchmark suite (`benchmark.py`) generates an interactive HTML report:

1. **Document Build** — cache generation + document build timing
2. **Simulation (60s)** — mass trajectories, growth rate
3. **v1 Comparison** — per-category mass accuracy, R² scores
4. **Division** — bulk conservation, unique molecule partitioning, daughter viability (pre-division state)
5. **Long Simulation (8 min)** — extended growth dynamics
6. **Step Diagnostics** — per-step config, ports, wiring analysis
7. **Network Visualization** — interactive pan-zoom process-bigraph graph
8. **Timing Summary** — wall time breakdown

## Dependencies

- Python 3.12.9
- [uv](https://docs.astral.sh/uv/)
- process-bigraph with `skip_initial_steps` support (PR [#111](https://github.com/vivarium-collective/process-bigraph/pull/111))
- No bigraph-schema changes (unmodified PyPI version)
- vEcoli (for v1 comparison): `pip install vEcoli[dev]`
- graphviz: `brew install graphviz` / `apt install graphviz`
