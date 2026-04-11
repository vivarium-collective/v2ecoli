# v2ecoli Status

## What Works

### Single Cell Lifecycle
- Full 55-step partitioned whole-cell model runs to division at ~42 min
- Dry mass grows from 380 fg → 702 fg (division threshold)
- Chromosome replication initiates at ~23 min (2 chromosomes)
- Division produces 2 viable daughter cells that continue growing
- Mass output matches vEcoli within 0.0% at 60s
- Performance: 8.2x realtime (v2ecoli) vs 9.2x (vEcoli composite)

### Colony Simulation
- Whole-cell E. coli embedded in 2D pymunk physics colony via EcoliWCM bridge
- Ecoli cell grows visibly (length computed from capsule geometry)
- Division at ~40 min: mother replaced by 2 daughter cells with fresh WCM processes
- Surrogate cells (AdderGrowDivide) grow and divide alongside
- Phylogeny coloring: grey surrogates, colored ecoli lineage with mutated daughter colors
- GIF animation of colony dynamics

### Vivarium Cleanup
- Zero vivarium-core imports
- `ports_schema()` removed (1098 lines deleted)
- `defaults` dicts merged into `config_schema`
- All process definitions ported from vEcoli composite branch
- Custom resolve dispatches for bigraph-schema type compatibility

## Known Limitations

### Colony
- EcoliWCM bridge runs internal composite synchronously (~0.7s per tick)
  making colony sim ~2.6x realtime with 1 ecoli cell
- Daughter EcoliWCM processes start fresh (don't inherit mother's internal state)
- Cell length initially decreases then increases (volume-to-length mapping
  at WCM's starting volume gives shorter cell than expected)

### Type System
- 40+ plum resolve dispatches needed for cross-type resolution
  (InPlaceDict vs Float, BulkNumpyUpdate vs Array, etc.)
- Config defaults loaded from pickle (config_defaults.pickle, port_defaults.pickle)
  rather than inline in source files

### Division
- WCM division fires via exception handling (Division step tries structural
  modification that crashes — bridge catches and handles)
- No clean division state handoff from mother to daughter WCM composites

## Architecture Decisions

### Partitioned Architecture
The model uses the Requester → Allocator → Evolver pattern:
- 11 PartitionedProcesses split into requester + evolver steps
- 3 allocator layers distribute bulk molecules by priority
- 11 UniqueUpdate steps flush accumulated unique molecule changes

### EcoliWCM Bridge
Each whole-cell model runs as an internal `Composite` inside an `EcoliWCM`
Process node. The bridge:
- Maps external `local` concentrations to internal `boundary.external`
- Reads `listeners.mass.dry_mass/volume` and computes length via capsule geometry
- Returns mass/length/volume deltas to the colony's pymunk physics
- Detects division and produces structural `_add`/`_remove` updates

### No vivarium-core
All process-bigraph patterns used directly:
- `EcoliStep`/`EcoliProcess` adapters bridge vEcoli's `(parameters=)` signature
- `_build_parameters()` reads from `config_schema` + extracted pickle defaults
- `port_defaults()` replaces `ports_schema()` for initial state seeding
- Type resolve dispatches handle schema conflicts in Composite realization
