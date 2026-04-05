# v2ecoli Migration Status

**Date**: 2026-04-05
**Repo**: https://github.com/vivarium-collective/v2ecoli

## What Works

- **All 55 biological steps** run through `Composite.run()` — no custom simulation engine
- **Layer-parallel execution** matching v1's model (35 batches, requesters in parallel)
- **1.67% worst-case mass error** vs v1 across dry mass, protein, RNA, DNA, small molecules
- **7.5x realtime** simulation speed (60s sim in ~8s wall)
- **Explicit Requester/Evolver Steps** for each biological process — no PartitionedProcess abstraction
- **Division step** with `_add`/`_remove` structural updates for daughter cell generation
- **State splitting functions** (divide_bulk, divide_by_domain, etc.) ported from v1
- **Benchmark suite** with per-category mass accuracy metrics
- **No sequential_steps needed** — pure dependency-based execution via input/output topology separation
- **No bigraph-schema changes needed** — works with unmodified PyPI version

## Architecture

- Each biological process has `XxxRequester(Step)` and `XxxEvolver(Step)` classes
- Standalone processes (tf-unbinding, metabolism, etc.) are plain `Step` subclasses
- Shared Logic instances pass cached values from Requester to Evolver
- Request store: InPlaceDict with per-process sub-dicts
- Allocation stores: flat `allocate_{proc_name}` with SetStore type
- Input/output topology separation controls the dependency graph
- Requesters: read bulk/unique (inputs), write request only (outputs) → parallel within layer
- Evolvers: read allocate (inputs), write bulk/unique (outputs)
- Flow layer tokens enforce inter-layer ordering

## Benchmark Results (60s comparison)

| Component | v1 Final (fg) | v2 Final (fg) | Max % Error | R² |
|-----------|---------------|---------------|-------------|-----|
| Dry Mass | ~380 | ~380 | 1.67% | TBD |
| Protein | ~183 | ~183 | ~1% | TBD |
| RNA | ~48 | ~48 | ~1.7% | TBD |
| DNA | ~7 | ~7 | ~1% | TBD |
| Small Molecules | ~142 | ~142 | ~0.5% | TBD |

## Dependencies

- `process-bigraph` PR #111: `skip_initial_steps` config option (only change needed)
- No bigraph-schema changes needed

## Next Steps

1. **Fix remaining 1.67% error** — evolver parallel execution, listener data routing
2. **Implement full division** — daughter cell generation with `_add`/`_remove`
3. **Replace unum with pint** — unified unit system
4. **Get CI workflow passing** — after process-bigraph PR merged
