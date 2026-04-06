# v2ecoli Status

**Date**: 2026-04-05
**Repo**: https://github.com/vivarium-collective/v2ecoli

## Current State

All 55 biological steps run through process-bigraph's `Composite.run()` with **0.62% worst-case mass error** vs v1 (vEcoli). Native flow execution (`sequential_steps=False`) with layer-based flow tokens — no custom simulation engine.

### Accuracy (60s comparison with v1)

| Component | Mean % Error | Max % Error | R² |
|-----------|-------------|-------------|-----|
| Dry Mass | 0.02% | 0.22% | 0.99 |
| Protein | 0.02% | 0.03% | 1.00 |
| RNA | 0.03% | 0.06% | 1.00 |
| DNA | 0.03% | 0.03% | 1.00 |
| Small Molecules | 0.02% | 0.62% | 0.78 |

### Architecture

- **Native flow execution**: `sequential_steps=False` with 31 execution layers
- **Explicit Requester/Evolver Steps**: input/output topology separation for all 11 partitioned processes
- **Flat per-process stores**: `request_{proc}` and `allocate_{proc}` (no shared nested dicts)
- **Layer-based flow tokens**: requesters/evolvers/listeners grouped by execution layer
- **Custom types**: BulkNumpyUpdate, UniqueNumpyUpdate, InPlaceDict, SetStore, ListenerStore
- **Division**: `_add`/`_remove` structural updates, tested on pre-division state (t=1800, 2 chromosomes)
- **ParCa pipeline**: raw TSV data → simData → initial state → document
- **Benchmark**: interactive process-bigraph network visualization, per-category mass accuracy, division test

### Dependencies

- `process-bigraph`: `skip_initial_steps` config (PR #111)
- No bigraph-schema changes (unmodified PyPI version)
- No modifications to the Composite execution engine

## Known Issues

1. **Within-layer parallelism blocked** — layer token W/W conflicts serialize steps in the same layer. Flat stores are ready; needs process-bigraph barrier token support.
2. **Small molecule R² = 0.78** — metabolism sensitivity to upstream state. All other components > 0.99.
3. **Compiled dependencies** — polymerize (Cython), fba (GLPK), mc_complexation (Cython) from wholecell.

## Next Steps

1. **Upstream process-bigraph PR** — get `skip_initial_steps` merged
2. **Barrier tokens** — process-bigraph support for W/W sync points (enables within-layer parallelism)
3. **CI workflow** — GitHub Actions with PyPI dependencies
4. **Replace unum with pint** — unified unit system
