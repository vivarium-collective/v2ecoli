# v2ecoli Migration Status

**Date**: 2026-04-05
**Repo**: https://github.com/vivarium-collective/v2ecoli

## What Works

- **All 55 biological steps** run through `Composite.run()` — no custom simulation engine
- **1.0000 per-timestep correlation** with vEcoli (v1) across all 16,321 molecules
- **6x realtime** simulation speed (60s sim in ~10s wall)
- **Division detected** at t=1857s (~31 min), dry mass threshold 702 fg
- **Custom bigraph-schema types**: BulkNumpyUpdate, UniqueNumpyUpdate, InPlaceDict, SetStore, ListenerStore
- **Sequential step execution** via `sequential_steps` mode in Composite
- **ParCa pipeline** included — can generate simData from raw TSV data
- **JSON state caching** — initial_state.json (10MB) for fast reloading
- **Benchmark suite** with per-timestep v1 comparison, mass plots, step diagnostics
- **CI workflow** — GitHub Actions runs benchmarks on push, deploys report to Pages

## Architecture

```
v2ecoli/
├── reconstruction/          # ParCa + 133 TSV raw data files
├── v2ecoli/
│   ├── library/            # 20 modules (schema, sim_data, fitting, etc.)
│   ├── types/              # Custom types (bulk_numpy, unique_numpy, stores)
│   ├── steps/              # Steps (partition, allocator, listeners, division)
│   ├── processes/          # 15 biological processes
│   ├── composite.py        # make_composite() → Composite
│   ├── generate.py         # build_document() from simData
│   ├── cache.py            # JSON caching with numpy support
│   └── migrate.py          # Migration audit utility
├── benchmark.py            # Benchmark suite with HTML report
├── .github/workflows/      # CI: benchmark on push, deploy to Pages
└── test_types.py           # Type system unit tests
```

## Known Issues

### 1. Process Migration (In Progress)
Processes still use v1 patterns (`ports_schema`, `defaults`, `next_update`) wrapped
in v2 shells via `_translate_schema`. The `V2Step` base class auto-generates
`inputs()`/`outputs()` from `ports_schema()` at runtime. A full migration to native
v2 patterns requires `core.fill()` improvements in bigraph-schema for callable and
array config values.

### 2. Division Not Fully Implemented
Division is detected (dry mass threshold) but daughter cell generation is not yet
implemented. The state checkpoint system (`save_state`/`load_state`) supports
resuming from pre-division state.

### 3. Compiled Dependencies
Five library modules wrap wholecell compiled extensions:
polymerize (Cython), fba (GLPK), mc_complexation (Cython), units (unum), unit_struct_array.

### 4. Unit System
Uses unum (via wholecell) for units. Migration to pint planned.

### 5. bigraph-schema Modifications
In-place dict mutation in `apply(schema: dict, ...)` and `sequential_steps` mode
in Composite were applied to local copies. These should be upstreamed.

## Benchmark Results (60s comparison)

| Metric | Value |
|--------|-------|
| Per-timestep correlation | 1.000000 |
| Delta correlation | 1.0000 |
| v1 runtime (60s) | ~6.4s |
| v2 runtime (60s) | ~8.5s |
| Molecules changed (v1) | 2,195 |
| Molecules changed (v2) | 2,145 |
| Long sim (500s) | 59s wall, 3,206 changed |

## Next Steps

1. **Complete process migration** — native `inputs()`/`outputs()`, `config_schema` with defaults
2. **Upstream bigraph-schema changes** — in-place apply, sequential_steps
3. **Implement full division** — daughter cell generation, state splitting
4. **Replace unum with pint** — unified unit system
5. **Vendor compiled extensions** — remove wholecell runtime dependency
6. **Modularize ParCa** — process-bigraph workflow for parameter fitting
