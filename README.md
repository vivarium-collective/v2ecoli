# v2ecoli

Whole-cell *E. coli* model built natively on [process-bigraph](https://github.com/vivarium-collective/process-bigraph).

[![Tests](https://github.com/vivarium-collective/v2ecoli/actions/workflows/test.yml/badge.svg)](https://github.com/vivarium-collective/v2ecoli/actions/workflows/test.yml)
[![Benchmark](https://github.com/vivarium-collective/v2ecoli/actions/workflows/benchmark.yml/badge.svg)](https://github.com/vivarium-collective/v2ecoli/actions/workflows/benchmark.yml)

## Goal

Re-implement the [vEcoli](https://github.com/CovertLab/vEcoli) whole-cell *E. coli* model as a native [process-bigraph](https://github.com/vivarium-collective/process-bigraph) composite. The simulation runs entirely through `Composite.run()` with custom [bigraph-schema](https://github.com/vivarium-collective/bigraph-schema) types — no Vivarium engine, no custom scheduler, no monkey-patching. This validates process-bigraph as a general-purpose framework for large-scale biological simulations while enabling architectural experiments (departitioning, caching, parallelism) that are difficult in the original codebase.

## Reports

| Report | Description |
|--------|-------------|
| **[Workflow Report](https://vivarium-collective.github.io/v2ecoli/workflow_report.html)** | Full pipeline: ParCa → simulation → v1 comparison → division → multicell |
| **[Architecture Comparison](https://vivarium-collective.github.io/v2ecoli/comparison_report.html)** | Partitioned vs departitioned: mass trajectories, bigraph viz, timing |

## Quick Start

```bash
git clone https://github.com/vivarium-collective/v2ecoli.git
cd v2ecoli
uv sync

# Run simulation (requires cached simData in out/cache/)
uv run python -c "
from v2ecoli.composite import make_composite
ecoli = make_composite(cache_dir='out/cache')
ecoli.run(60.0)
print(ecoli.state['agents']['0']['listeners']['mass']['dry_mass'])
"

# Run workflow report
uv run python workflow.py
open out/workflow/workflow_report.html

# Run architecture comparison
uv run python compare_architectures.py
open out/comparison_report.html
```

## Models

Two simulation architectures are maintained side by side:

| Architecture | Steps | Allocator | Description | Model |
|-------------|-------|-----------|-------------|-------|
| **Departitioned** | 41 | No | Each process is a single Step operating on live state | [`models/departitioned.pbg`](models/departitioned.pbg) |
| **Partitioned** | 55 | Yes (×3) | Requester → Allocator → Evolver pattern for 11 processes | [`models/partitioned.pbg`](models/partitioned.pbg) |

The **partitioned** model matches v1 vEcoli's architecture and produces near-identical results (< 0.05% mass error over 60s). The **departitioned** model merges each Requester+Evolver into a single Step, removing the allocator. Over short runs (60s) they agree within 0.2%, but over a full cell cycle the departitioned model grows ~30% slower because processes compete for shared molecules without fair allocation.

## Testing

```bash
pytest tests/test_types.py -v          # Type system (fast, no cache)
pytest tests/test_integration.py -v    # 10s simulation (needs cache)
pytest tests/ -v                       # All tests
pytest tests/ -m slow -v               # Full comparisons only
```

| Test | Validates | Speed |
|------|-----------|-------|
| `test_types.py` | BulkNumpyUpdate, UniqueNumpyUpdate, type registration | ~1s |
| `test_integration.py` | Composite loading, 10s sim, mass growth | ~15s |
| `test_partitioned.py` | Partitioned model loads and grows | ~15s |
| `test_rnap_count.py` | ppGpp regulation, RNAP count correctness | ~18s |
| `test_pbg_files.py` | .pbg model file structure and freshness | ~5s |
| `test_departitioned.py` | Departitioned model vs v1 vEcoli (60s) | ~2 min |
| `test_architecture_comparison.py` | Partitioned vs departitioned agreement (60s) | ~4 min |

## Architecture

```
v2ecoli/
├── v2ecoli/
│   ├── generate.py              # build_document(): simData → document dict
│   ├── composite.py             # make_composite(), save/load state
│   ├── processes/               # 15 biological processes (standalone Steps)
│   ├── partitioned/             # Frozen partitioned architecture for comparison
│   │   ├── processes.py         # Logic + Requester + Evolver classes
│   │   └── generate_partitioned.py
│   ├── steps/
│   │   ├── partition.py         # RequesterBase, EvolverBase, _protect_state
│   │   ├── allocator.py         # Priority-based molecule allocation
│   │   ├── division.py          # Cell division with structural updates
│   │   └── listeners/           # 9 listener steps (mass, RNA, protein, etc.)
│   ├── types/                   # Custom bigraph-schema types
│   └── reconstruction/          # ParCa pipeline + raw data TSVs
├── tests/                       # pytest test suite
├── models/                      # .pbg model metadata files
├── workflow.py                  # Full pipeline with HTML report
├── compare_architectures.py     # Architecture comparison report
└── compare_baseline.py          # Quick baseline comparison
```

### Simulation Pipeline

```
Raw Data (TSVs) → ParCa → simData → LoadSimData → Process Configs
                                          ↓
                                build_document()
                                          ↓
                           Composite (41 steps + GlobalClock)
                                          ↓
                                Composite.run(duration)
```

### Custom Types

| Type | Description |
|------|-------------|
| `BulkNumpyUpdate` | Structured arrays with `(index, value)` tuple application |
| `UniqueNumpyUpdate` | Accumulate set/add/delete operations, flush on signal |
| `InPlaceDict` | Deep merge on apply |
| `ListenerStore` | Top-level merge, leaf-level set |

## Dependencies

- Python 3.12.9
- [uv](https://docs.astral.sh/uv/)
- [process-bigraph](https://github.com/vivarium-collective/process-bigraph) with view/project caching ([PR #112](https://github.com/vivarium-collective/process-bigraph/pull/112))
- [bigraph-schema](https://github.com/vivarium-collective/bigraph-schema) (PyPI version)
- [vEcoli](https://github.com/CovertLab/vEcoli) (for v1 comparison)
- graphviz: `brew install graphviz` / `apt install graphviz`
