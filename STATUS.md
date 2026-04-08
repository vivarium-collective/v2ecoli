# v2ecoli Status

**Date**: 2026-04-07
**Repo**: https://github.com/vivarium-collective/v2ecoli
**Branch**: `serializable-configs` (active development), `main` (stable)

## Current State

### Two Architectures

| Architecture | Steps | Allocator | Division | Branch |
|-------------|-------|-----------|----------|--------|
| **Partitioned** | 55 | Yes (×3) | t=2644s | `main` |
| **Departitioned** | 41 | No | t=2656s | `serializable-configs` |

Both produce equivalent results (bulk correlation 1.000000 over 60s).

### Accuracy (60s, partitioned model vs v1)

| Component | Mean % Error | Max % Error | R² |
|-----------|-------------|-------------|-----|
| Dry Mass | 0.00% | 0.01% | 1.0000 |
| Protein | 0.00% | 0.01% | 0.9999 |
| RNA | 0.01% | 0.04% | 0.9993 |
| DNA | 0.00% | 0.00% | 1.0000 |
| Small Molecules | 0.01% | 0.01% | 0.9988 |

### Config Serialization Progress

28/30 bound methods and class instances eliminated from configs via function registry:

| Category | Count | Status |
|----------|-------|--------|
| Bound methods → function registry | 22 | ✅ Done |
| Precomputed at config time | 3 | ✅ Done |
| Class instances → extracted | 2 | ✅ Done |
| Kept as dill (unit complexity) | 2 | ⏳ exchange_constraints, external_state |
| Unum values | ~20 | ⏳ Need per-process unit refactoring |

### Performance

| Metric | Value |
|--------|-------|
| 60s simulation | ~8s wall (8x realtime) |
| Workflow to division | ~8 min (with cached v1) |
| view/project caching | 14% speedup (process-bigraph PR #112) |

### Testing

| Test | What | Speed |
|------|------|-------|
| `test_types.py` | Type system | ~1s |
| `test_integration.py` | 10s simulation | ~15s |
| `test_partitioned.py` | Partitioned model | ~15s |
| `test_rnap_count.py` | ppGpp regulation correctness | ~18s |
| `test_pbg_files.py` | .pbg model files | ~5s |
| `test_departitioned.py` | vs v1 (60s) | ~2 min |
| `test_architecture_comparison.py` | Partitioned vs departitioned | ~4 min |

### Reports

- **Workflow Report**: full cell cycle to division with v1 comparison
- **Architecture Comparison**: partitioned vs departitioned metrics

## Recent Work (2026-04-07)

### Departitioning
- Merged all 11 Requester+Allocator+Evolver pairs into standalone Steps
- Removed allocator entirely from departitioned model
- 25% speedup (9.42s → 7.05s for 60s sim)

### Config Serialization
- Created function registry (`v2ecoli/library/function_registry.py`) with 15 registered functions
- Created config resolver (`v2ecoli/library/config_resolver.py`)
- Migrated ODE solvers (equilibrium, two-component system) with dill-encoded rate functions
- Extracted metabolism dataclass into individual config attributes
- **Fixed ppGpp unit conversion bug**: `.asNumber()` without target units returned mol/L (5e-5) instead of umol/L (50), making the Hill function return ~0 instead of ~0.81. This doubled active RNAP count (787 vs 717) and caused 80% higher growth rate. Fixed by using `.asNumber(units.umol / units.L)`.

### Infrastructure
- pytest test suite with fast/slow markers
- GitHub Actions CI (`test.yml` + `benchmark.yml`)
- `.pbg` model files (process-bigraph serialized documents)
- Workflow speedups: cached v1, skip EcoCyc API, short daughter sims

## Known Issues

1. **exchange_constraints** and **external_state** — still dill-serialized (complex unit interactions with FBA)
2. **~20 Unum values** in configs — need per-process arithmetic refactoring
3. **PolypeptideInitiationStep** — `KeyError: 'ribosome_data'` on first timestep (non-critical)
4. **Departitioned growth rate** — ~30% slower than partitioned over full cell cycle (no allocator means unfair molecule competition)

## Architecture

```
v2ecoli/
├── v2ecoli/
│   ├── generate.py              # build_document(): simData → document
│   ├── composite.py             # make_composite(), save/load
│   ├── pbg.py                   # .pbg file serialization
│   ├── processes/               # 15 biological processes (standalone Steps)
│   ├── partitioned/             # Frozen partitioned architecture
│   ├── steps/
│   │   ├── partition.py         # RequesterBase, EvolverBase
│   │   ├── allocator.py         # Priority-based molecule allocation
│   │   ├── division.py          # Cell division
│   │   └── listeners/           # 9 listener steps
│   ├── types/                   # BulkNumpyUpdate, UniqueNumpyUpdate, etc.
│   ├── library/
│   │   ├── function_registry.py # Serializable function references
│   │   ├── config_resolver.py   # Resolve function refs in configs
│   │   └── sim_data.py          # Config generation from simData
│   └── reconstruction/          # ParCa pipeline + raw data
├── tests/                       # pytest suite (7 test files)
├── models/                      # .pbg model documents
├── workflow.py                  # Pipeline with HTML report
└── compare_architectures.py     # Architecture comparison report
```
