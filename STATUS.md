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

### Full Cell Cycle

- **Division at t=2645s** (dry mass 702 fg, 2 chromosomes)
- **Chromosome replication**: initiates, forks progress bidirectionally, terminate ~t=1350s
- **Re-initiation**: second round begins ~t=1950s (4 active forks), matching v1 behavior
- **Daughter viability**: confirmed (builds + runs 1s)
- **Lifecycle v1/v2 comparison**: mass, chromosomes, forks, RNAP tracked over full cycle

### Architecture

- **No PartitionedProcess**: all 15 processes use plain Logic classes with per-process Steps
- **Native flow execution**: `sequential_steps=False` with 31 execution layers
- **Per-process Requester/Evolver Steps**: explicit `inputs()`/`outputs()`, `initialize()` pattern
- **Flat per-process stores**: `request_{proc}` and `allocate_{proc}` (no shared nested dicts)
- **Layer-based flow tokens**: requesters/evolvers/listeners grouped by execution layer
- **Custom types**: BulkNumpyUpdate, UniqueNumpyUpdate, InPlaceDict, SetStore, ListenerStore
- **Error logging**: `_SafeInvokeMixin` logs warnings instead of silently swallowing errors

### Workflow Testing

9-step cached pipeline with comprehensive HTML report:

0. **EcoCyc API** — fetch 10 BioCyc data files
1. **Raw Data** — catalog 133 TSV files (4,641,652 bp genome, 4747 genes)
2. **ParCa** — parameter calculator (27 process configs, 16,321 bulk molecules)
3. **Load Model** — build composite from cache
4. **Short Simulation** — 60s with mass/growth diagnostics
5. **v1 Comparison** — per-category accuracy (0.04% worst error)
6. **Long Simulation** — run to division with chromosome snapshots every 50s
6b. **Lifecycle v1/v2** — full cell cycle comparison (mass, chromosomes, forks, RNAP)
7. **Division** — conservation, unique molecule splits, daughter viability

All steps cache metadata + state. Cached run completes in ~5s.

### Dependencies

- `process-bigraph`: `skip_initial_steps` config (PR #111)
- No bigraph-schema changes (unmodified PyPI version)
- No modifications to the Composite execution engine

## Recent Fixes

- **Chromosome re-initiation** (2026-04-06): `np.in1d` → `np.isin` in chromosome_replication.py. The deprecated NumPy function silently crashed, preventing second-round replication. Now matches v1 (4 forks from ~t=1950).
- **Error visibility**: `_SafeInvokeMixin` now logs warnings instead of silently swallowing exceptions.

## Known Issues

1. **Within-layer parallelism blocked** — layer token W/W conflicts serialize steps in the same layer. Flat stores are ready; needs process-bigraph barrier token support.
2. **Compiled dependencies** — polymerize (Cython), fba (GLPK), mc_complexation (Cython) from wholecell.
3. **Unum dependency** — units still use unum via wholecell.utils. Pint compatibility layer ready but full migration requires cache regeneration under pint.
4. **PolypeptideInitiationRequester** — minor `KeyError: 'ribosome_data'` (surfaced by error logging, non-critical).

## Next Steps

1. **Fix ribosome_data KeyError** in PolypeptideInitiationRequester
2. **Upstream process-bigraph PR** — get `skip_initial_steps` merged
3. **Barrier tokens** — process-bigraph support for W/W sync points (enables within-layer parallelism)
4. **Replace unum with pint** — unified unit system
5. **CI workflow** — GitHub Actions with PyPI dependencies
