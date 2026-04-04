# v2ecoli

Whole-cell *E. coli* model built natively on [process-bigraph](https://github.com/vivarium-collective/process-bigraph).

Wraps the biological logic from [vEcoli](https://github.com/CovertLab/vEcoli) in v2-native `BigraphStep` and `BigraphProcess` classes with typed schemas, running on the standard `Composite` engine.

## Quick Start

```bash
uv run python -c "
from v2ecoli import generate_document, load_composite
generate_document()
ecoli = load_composite()
ecoli.run(10.0)
print(f'global_time: {ecoli.state[\"global_time\"]}')
"
```

## Architecture

```
v2ecoli/
  types/        # Custom bigraph-schema types (BulkNumpyUpdate, UniqueNumpyUpdate, ...)
  steps/        # v2-native BigraphStep wrappers for v1 steps
  processes/    # v2-native BigraphProcess wrappers for v1 processes
  __init__.py   # Public API: generate_document, load_composite
```

### Types

- **BulkNumpyUpdate** — Structured numpy array with `count` field. Apply = index-add.
- **UniqueNumpyUpdate** — Structured numpy array with `_entryState` field. Apply = accumulate set/add/delete, flush on signal.
- Plus: CSRMatrix, UnitsArray, Unum, Quantity, Method (from genEcoli)

### Steps and Processes

Each biological step/process is a `BigraphStep` or `BigraphProcess` subclass that:
1. Declares typed `inputs()` and `outputs()` schemas
2. Wraps the v1 `next_update()` / `calculate_request()` / `evolve_state()` logic
3. Returns deltas that the Composite applies through the type system

## Prerequisites

- Python 3.12.9
- [uv](https://docs.astral.sh/uv/)
- Local editable installs of vEcoli, process-bigraph, bigraph-schema
- `simData.cPickle` at `<vEcoli>/out/kb/simData.cPickle`
