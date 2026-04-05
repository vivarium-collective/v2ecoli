# v2ecoli Migration Status

**Date**: 2026-04-04
**Repo**: https://github.com/vivarium-collective/v2ecoli

## What Works

- **101 Python files, 35,714 lines** — self-contained E. coli whole-cell model
- **All 30+ biological processes** ported from vEcoli as process-bigraph Steps
- **Simulation runs via `Composite.run()`** — no custom engine
- **55 Steps + 1 Process (GlobalClock)** execute through process-bigraph's pipeline
- **3828/16321 bulk molecules change** in 10s simulated time
- **1.51s runtime** (2.3x faster than v1's 3.48s)
- **ParCa pipeline** (raw data → simData) included in the repo
- **133 TSV raw data files** in reconstruction/ecoli/flat/
- **Custom types**: BulkNumpyUpdate, UniqueNumpyUpdate with proper apply dispatches
- **HTML comparison report** with mass fractions, bigraph viz, scatter plots, JSON viewer
- **RAMEmitter** from process-bigraph collects per-timestep data

## Known Issues

### 1. Step Execution Order (Critical)
**Impact**: 3828 vs 1595 bulk changes (v1). Correlation ~0.03 instead of 1.0.

The Composite's dependency-driven step cascade doesn't match v1's strict sequential
execution. Steps share `bulk`, `unique`, `listeners` stores, creating a dense
dependency graph where evolvers can run before their allocators produce partitioned
counts.

**Root cause**: `determine_steps()` in process-bigraph uses data dependencies (overlapping
input/output paths) to order steps. With many steps sharing stores, this creates cycles
that get broken by priority — but the cycle detection groups steps differently than our
flow ordering expects.

**Potential fixes**:
- Enhance `determine_steps` in process-bigraph to respect priority ordering within
  triggered batches (not just cycle-breaking)
- Use separate stores per partition layer (request_1, allocate_1, etc.) to reduce
  spurious dependencies
- Add a "sequential mode" to Composite that runs all triggered steps strictly by
  priority, one at a time

### 2. Silent Step Failures
Steps that crash (KeyError, AssertionError) are caught by `_SafeInvokeMixin.invoke()`
and return `{}`. This prevents cascade crashes but means some steps don't contribute
their updates. Proper fix: handle missing data gracefully in each step.

### 3. Compiled Dependencies
Five library modules still wrap wholecell compiled extensions:
- `polymerize.py` → Cython polymerization
- `fba.py` → GLPK flux balance analysis
- `mc_complexation.py` → Cython complexation
- `units.py` → unum units
- `unit_struct_array.py` → UnitStructArray

These need to be vendored or reimplemented for full independence.

### 4. vivarium.library.units
Three files import `vivarium.library.units` for pint-based mM units:
- `steps/media_update.py`
- `steps/exchange_data.py`
- `processes/metabolism.py`

### 5. bigraph-schema Modifications
In-place dict mutation in `apply(schema: dict, ...)` and null-handling in
`apply(schema: Node, ...)` were applied to the local bigraph-schema copy.
These should be upstreamed or the types need to handle this differently.

## Architecture

```
v2ecoli/
├── reconstruction/          # ParCa + 133 TSV raw data files
│   └── ecoli/              # knowledge_base_raw, fit_sim_data_1, simulation_data
├── v2ecoli/
│   ├── library/            # 20 modules (schema, sim_data, fitting, etc.)
│   ├── types/              # 10 custom bigraph-schema types
│   ├── steps/              # 16 step files + listeners/ + base.py + partition.py
│   │   ├── base.py         # V2Step with auto schema translation
│   │   ├── partition.py    # PartitionedProcess, Requester, Evolver
│   │   ├── allocator.py    # Allocator
│   │   ├── unique_update.py
│   │   ├── global_clock.py # Process (drives time)
│   │   └── listeners/      # 9 listener steps
│   ├── processes/          # 15 biological processes
│   ├── composite.py        # make_composite() → Composite
│   └── generate.py         # build_document() from simData
├── report.py               # HTML comparison report generator
└── test_types.py           # Type system unit tests
```

## Next Steps

1. **Fix step ordering in process-bigraph** — the critical path to matching v1
2. **Upstream bigraph-schema changes** (in-place apply, null handling)
3. **Add proper error handling** to biological processes for missing data
4. **Vendor compiled extensions** (polymerize, fba, mc_complexation)
5. **Replace vivarium.library.units** with direct pint usage
6. **Run ParCa from v2ecoli's own code** (currently uses cached simData)
7. **Add more analysis plots** (growth rate, gene expression, etc.)
