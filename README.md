# v2ecoli

Whole-cell *E. coli* model built natively on [process-bigraph](https://github.com/vivarium-collective/process-bigraph).

[![Benchmark](https://github.com/vivarium-collective/v2ecoli/actions/workflows/benchmark.yml/badge.svg)](https://github.com/vivarium-collective/v2ecoli/actions/workflows/benchmark.yml)

**[Benchmark Report](https://htmlpreview.github.io/?https://github.com/vivarium-collective/v2ecoli/blob/main/doc/benchmark_report.html)**

## Overview

v2ecoli migrates the [vEcoli](https://github.com/CovertLab/vEcoli) whole-cell *E. coli* model to run entirely through process-bigraph's `Composite.run()`. All 55 biological steps execute through the standard process-bigraph pipeline with custom bigraph-schema types for bulk molecules, unique molecules, and listener stores.

### Key Results

| Metric | Value |
|--------|-------|
| Worst mass % error vs v1 | **0.62%** |
| Mass component R² | **> 0.99** |
| Dry mass error at t=1 | **0.003 fg** |
| Simulation speed | **~6x realtime** (60s sim in ~10s wall) |
| Biological processes | **55 steps + 1 process** |
| Division | **State splitting + daughter viability verified** |
| Bulk conservation | **Exact** (binomial split) |

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
├── reconstruction/          # ParCa pipeline + 133 TSV raw data files
│   └── ecoli/              # knowledge_base_raw, fit_sim_data_1, simulation_data
├── v2ecoli/
│   ├── library/            # Schema helpers, units, polymerize, FBA, fitting
│   │   └── division.py     # State splitting functions (bulk binomial, domain-based)
│   ├── types/              # Custom bigraph-schema types
│   │   ├── bulk_numpy.py   # BulkNumpyUpdate (index-add on structured arrays)
│   │   ├── unique_numpy.py # UniqueNumpyUpdate (accumulate/flush pattern)
│   │   └── stores.py       # InPlaceDict, SetStore, ListenerStore
│   ├── steps/              # process-bigraph Steps
│   │   ├── partition.py    # PartitionedProcess, Requester, Evolver, _protect_state
│   │   ├── allocator.py    # Priority-based molecule allocation
│   │   ├── division.py     # Cell division with _add/_remove structural updates
│   │   └── listeners/      # 9 listener steps (mass, RNA, protein, etc.)
│   ├── processes/          # 15 biological processes
│   ├── composite.py        # make_composite(), save_state(), run_and_cache()
│   ├── generate.py         # build_document() from simData
│   └── cache.py            # JSON state caching with numpy support
├── benchmark.py            # Benchmark suite with HTML report
├── diagnose.py             # Per-step v1/v2 diagnostic comparison
├── compare_v1_v2.py        # Per-timestep mass comparison
└── doc/
    └── benchmark_report.html  # Latest benchmark report (self-contained)
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

### Partition System

Biological processes that share molecules go through a request → allocate → evolve cycle:

1. **Requesters** compute what each process needs (bulk molecule requests)
2. **Allocator** partitions available molecules by priority
3. **Evolvers** run with allocated counts, producing state updates

Requesters write to a shared `request` store keyed by process name. The Allocator reads all requests and writes allocations. Evolvers read their allocation and apply bulk count replacement before running `evolve_state`.

### Custom Types

- **`BulkNumpyUpdate`**: Bulk molecule structured arrays. `apply` adds `(index, value)` tuples to the `count` field.
- **`UniqueNumpyUpdate`**: Unique molecule arrays. `apply` accumulates set/add/delete operations and flushes on signal.
- **`InPlaceDict`**: Dicts that mutate in place (deep merge on apply).
- **`SetStore`**: Full replacement on apply (for allocation stores).
- **`ListenerStore`**: In-place merge at top level, set at leaves.

### Division

The Division step uses process-bigraph's `_add`/`_remove` structural updates:

1. **Detection**: dry mass ≥ threshold with ≥ 2 chromosomes
2. **State splitting** (`divide_cell()`):
   - Bulk: binomial p=0.5 on each molecule count
   - Chromosomes: alternating (even→D1, odd→D2) with domain tree tracking
   - Chromosome-attached molecules: follow their domain
   - RNAs: full transcripts binomial, partial follow RNAP
   - Ribosomes: follow mRNA, degraded-mRNA ribosomes binomial
3. **Daughter construction**: `build_document_from_configs()` with divided state + cached configs
4. **Structural update**: `{'agents': {'_remove': [mother], '_add': [(d1, state), (d2, state)]}}`

Pre-division caching:
```python
from v2ecoli.composite import run_and_cache
composite = run_and_cache(intervals=[500, 1000, 1500, 1800, 2000])
```

### Store Routing

The `_protect_state` function copies bulk/unique arrays before passing to processes, preventing v1-style in-place mutations from corrupting the shared state. Request sub-dicts are pre-created for each partitioned process so `core.apply` can route requester updates through the shared request store.

## Benchmark

The benchmark suite (`benchmark.py`) generates an HTML report with:

1. **Document Build** — cache generation + document build timing
2. **Simulation (60s)** — mass trajectories, growth rate, volume
3. **v1 Comparison** — per-category mass accuracy (dry mass, protein, RNA, DNA, small molecules), R² scores
4. **Division** — bulk conservation, unique molecule partitioning, daughter viability
5. **Long Simulation (8 min)** — extended growth dynamics
6. **Step Diagnostics** — per-step config, ports, wiring analysis
7. **Process-Bigraph Network Visualization** — full process network graph
8. **Timing Summary** — wall time breakdown

Run locally:
```bash
uv run python benchmark.py
open out/benchmark/benchmark_report.html
```

## Dependencies

- Python 3.12.9
- [uv](https://docs.astral.sh/uv/)
- process-bigraph PR [#111](https://github.com/vivarium-collective/process-bigraph/pull/111): `skip_initial_steps` config option
- No bigraph-schema changes needed (works with unmodified PyPI version)
- vEcoli (for v1 comparison): `pip install vEcoli[dev]`
- graphviz: `brew install graphviz` / `apt install graphviz`
