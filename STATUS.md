# v2ecoli Status

**Date**: 2026-04-05
**Repo**: https://github.com/vivarium-collective/v2ecoli

## Current State

All 55 biological steps run through process-bigraph's `Composite.run()` with **0.62% worst-case mass error** vs v1 (vEcoli). No custom simulation engine — pure process-bigraph.

### Accuracy (60s comparison with v1)

| Component | Mean % Error | Max % Error | R² |
|-----------|-------------|-------------|-----|
| Dry Mass | 0.02% | 0.22% | 0.99 |
| Protein | 0.02% | 0.03% | 1.00 |
| RNA | 0.03% | 0.06% | 1.00 |
| DNA | 0.03% | 0.03% | 1.00 |
| Small Molecules | 0.02% | 0.62% | 0.78 |

### What Works

- **55 biological steps** through `Composite.run()`
- **Native flow execution** via `sequential_steps=False` with layer-based flow tokens
- **Explicit Requester/Evolver Steps** with input/output topology separation for all 11 partitioned processes
- **31 execution layers** with requesters/evolvers/listeners grouped for parallel execution
- **Flat per-process stores**: `request_{proc}` and `allocate_{proc}` replace shared nested dicts
- **Partition system**: Requesters → Allocators → Evolvers with per-process store routing
- **Store routing fix**: `_protect_state` copies bulk/unique
- **Custom types**: BulkNumpyUpdate, UniqueNumpyUpdate, InPlaceDict, SetStore, ListenerStore
- **Division**: `_add`/`_remove` structural updates with state splitting + daughter viability
- **State splitting**: bulk (binomial), unique (domain-based), RNA (RNAP-following), ribosomes (mRNA-following)
- **Benchmark suite**: per-category mass accuracy, division test, network visualization, TOC navigation
- **ParCa pipeline**: raw TSV data → simData → initial state → document
- **JSON caching**: initial_state.json (10MB) + sim_data_cache.dill (190MB)
- **Checkpoint system**: `save_state`/`load_state`/`run_and_cache` for resumption

### Division

- Division step detects condition (dry mass >= threshold + 2 chromosomes)
- `divide_cell()` splits all state stores with conservation guarantees
- `build_document_from_configs()` constructs complete daughter cell states
- Returns `_add`/`_remove` structural update for the Composite
- Daughter viability verified: builds + runs 1s successfully
- Pre-division caching via `run_and_cache()` for repeated testing

### Dependencies

- `process-bigraph` PR #111: `skip_initial_steps` config (only change needed)
- No bigraph-schema changes (works with unmodified PyPI version)
- No modifications to the Composite execution engine

## Branches

- **`main`**: Production — correlation 1.0000, 0.62% worst error, PartitionedProcess architecture
- **`explicit-steps-parallel`**: Future — explicit Requester/Evolver Steps, layer-parallel execution, 2.12% worst error

## Known Issues

1. **`sequential_steps` removed** — input/output topology separation eliminates dependency cycles. Layer-based flow tokens enforce inter-layer ordering. `sequential_steps=False` produces bit-identical results to the sequential baseline.

2. **Small molecule R² = 0.78** — small molecules have the lowest correlation due to metabolism's sensitivity to upstream state differences. All other components > 0.99.

3. **Division not tested at actual division time** — the benchmark tests division on the initial state (t=0). At actual division (~1857s), the cell has 2+ chromosomes with proper domain trees. Mass isn't growing enough to reach the division threshold in v2 (same as v1 behavior for short runs).

4. **Compiled dependencies** — polymerize (Cython), fba (GLPK), mc_complexation (Cython) from wholecell.

5. **unum units** — uses unum via wholecell. Migration to pint planned.

## Next Steps

1. **Run to actual division** — cache pre-division state, test full division cycle
4. **Upstream process-bigraph PR** — get `skip_initial_steps` merged
5. **CI workflow** — get GitHub Actions passing with PyPI dependencies
6. **Replace unum with pint** — unified unit system
