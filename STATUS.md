# v2ecoli Status

**Date**: 2026-04-06
**Repo**: https://github.com/vivarium-collective/v2ecoli

## Current State

All 55 biological steps run through process-bigraph's `Composite.run()` with **0.04% worst-case mass error** vs v1 (vEcoli). Native flow execution (`sequential_steps=False`) with layer-based flow tokens — no custom simulation engine.

### Accuracy (60s comparison with v1)

| Component | Mean % Error | Max % Error | R² |
|-----------|-------------|-------------|-----|
| Dry Mass | 0.00% | 0.01% | 1.0000 |
| Protein | 0.00% | 0.01% | 0.9999 |
| RNA | 0.01% | 0.04% | 0.9993 |
| DNA | 0.00% | 0.00% | 1.0000 |
| Small Molecules | 0.01% | 0.01% | 0.9988 |

### Architecture

- **No PartitionedProcess**: all 15 processes use plain Logic classes with per-process Steps
- **Native flow execution**: `sequential_steps=False` with 31 execution layers
- **Per-process Requester/Evolver Steps**: explicit `inputs()`/`outputs()`, `initialize()` pattern
- **Flat per-process stores**: `request_{proc}` and `allocate_{proc}` (no shared nested dicts)
- **Layer-based flow tokens**: requesters/evolvers/listeners grouped by execution layer
- **Custom types**: BulkNumpyUpdate, UniqueNumpyUpdate, InPlaceDict, SetStore, ListenerStore
- **Division**: verified at t=2687s (2 chromosomes, dry mass 702 fg, daughter viability confirmed)
- **Workflow testing**: 8-step cached pipeline (EcoCyc API → ParCa → simulation → division)

### Dependencies

- `process-bigraph`: `skip_initial_steps` config (PR #111)
- No bigraph-schema changes (unmodified PyPI version)
- No modifications to the Composite execution engine

## Known Issues

1. **Within-layer parallelism blocked** — layer token W/W conflicts serialize steps in the same layer. Flat stores are ready; needs process-bigraph barrier token support.
2. **Compiled dependencies** — polymerize (Cython), fba (GLPK), mc_complexation (Cython) from wholecell.

## Next Steps

1. **Upstream process-bigraph PR** — get `skip_initial_steps` merged
2. **Barrier tokens** — process-bigraph support for W/W sync points (enables within-layer parallelism)
3. **CI workflow** — GitHub Actions with PyPI dependencies
4. **Replace unum with pint** — unified unit system
