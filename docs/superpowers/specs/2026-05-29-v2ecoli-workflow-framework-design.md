# v2ecoli Workflow Framework — Design

**Date:** 2026-05-29
**Status:** Approved (design); implementation pending
**Scope:** Bring vEcoli's multiseed / multigeneration / variant workflow framework into
v2ecoli as **pure process-bigraph**, replacing NextFlow with a single inspectable
meta-composite. Also port vEcoli's JSON composite-config files so the same simulations
can be inspected and run.

## Background

vEcoli orchestrates simulations with NextFlow: a DAG of
`runParca → createVariants → simGen0 → sim (per generation) → analysis`. The conceptual
hierarchy is **variants → seeds (`n_init_sims`) → generations → daughters**, with analysis
aggregation at five scales (`single`, `multidaughter`, `multigeneration`, `multiseed`,
`multivariant`). Configs are JSON with `inherit_from` inheritance chains and a `variants`
block using a `value` / `linspace` / `nested` grammar combined via `op: prod|zip|add`.
Variants are applied as Python functions that mutate the ParCa `sim_data` pickle before
simulation.

v2ecoli already provides, in pure process-bigraph:

- **Multiseed**: per-process RNG derivation from a master `seed` (`baseline._derive_process_seed`).
- **Multigeneration**: a working lineage walk (today via the manual rebuild loop in
  `reports/multigeneration_report.py`).
- **In-state division**: `v2ecoli/steps/division.py:Division` detects division and spawns
  daughters in-state, returning `agents: {_remove: [mother], _add: [(d1, …), (d2, …)]}`
  with phylogeny ids (`"0"` → `"00"`/`"01"`). The `agents` map therefore grows as cells
  divide within a single running composite.
- **Emitters** whose metadata keys (`experiment_id, variant, lineage_seed, generation,
  agent_id`) already match vEcoli's parquet partitioning (`library/emitter_presets.py:parquet_vecoli`).

**Missing**: the variant/sweep framework, a config-file system, and any orchestration layer
tying seeds × variants × generations together.

## Decisions (locked during brainstorming)

1. **Orchestration model — meta-composite (all bigraph).** The entire
   `variants × seeds × generations` sweep is expressed as one process-bigraph Composite,
   serializable/inspectable as a `.pbg` document. Not a NextFlow port; not a plain-Python driver
   over many composites.
2. **Config format — port vEcoli JSON as-is.** Reuse vEcoli's exact schema (`inherit_from`,
   `variants`, `n_init_sims`, `generations`, `single_daughters`, `lineage_seed`,
   `different_seeds_per_variant`, `analysis_options`). Existing vEcoli configs run unchanged.
3. **Variant application — declarative process-config override.** A variant is a set of
   `path → value` overrides on process `config_schema` values, written into each branch's
   process edges and visible in the saved `.pbg`. Variants requiring `sim_data` recomputation
   (e.g. gene knockouts that rescale matrices) are **out of scope**.
4. **Analysis — analyses are process-bigraph Steps that run on simulation results.** This spec
   establishes the framework, the five-scale registry, the reserved post-sweep wiring slot, and
   **one example** Step. Porting the full analysis library is a follow-up spec.

## Architecture

```
configs/*.json  ──load+inherit──▶  expanded grid  ──build──▶  MetaComposite (.pbg)
 (vEcoli schema)                  (variants×seeds)            │  run loop (driver)
                                                              ▼
                                          branches[variant_i][seed_j] = LineageComposite
                                              agents: {gen0 → gen0/d → …}  (Division spawns)
                                              GenerationController (prune + stop)
                                              emitter (parquet, partitioned by metadata)
                                                              │
                                                              ▼  (reserved)
                                          AnalysisStep(s) read results @ 5 scales
```

- **Branch = embedded sub-composite.** Each `(variant_i, seed_j)` is its own
  `LineageComposite` (today's `baseline` composite) embedded under `state.branches[...]`, the
  same way `bridge.py:EcoliWCM` embeds a whole cell. This isolates each branch's `agents` map
  (division-spawned daughters stay within their branch) and yields a clean `.pbg` to inspect.
  *(Considered and rejected: one flat `agents` map keyed `variant/seed/phylogeny` — simpler
  topology but mixes branches and complicates per-branch analysis slicing.)*
- **Static grid, builder-time fan-out.** The grid is fully determined by the config, so all
  branches are constructed in `build_meta_composite`; no runtime "spawn seeds/variants"
  controller is needed. The only new runtime control logic is per-lineage
  (`GenerationController`).

## Components

### 1. Config loader & variant expansion — `v2ecoli/workflow/config.py`, `v2ecoli/workflow/variants.py`

Direct port of vEcoli's:
- `load_config_with_inheritance` + `_merge_configs` (the `inherit_from` chain and
  `LIST_KEYS_TO_MERGE` semantics).
- `parse_variants` (the `value` / `linspace` / `nested` grammar combined via
  `op: prod|zip|add`).

Output: a flat list of branch specs crossed with the seed range:

```python
[
  {"variant_index": 0, "variant_name": "baseline", "overrides": {}, "metadata": {}},
  {"variant_index": 1, "variant_name": "kcat",
   "overrides": {"ecoli-metabolism.kcat": 2}, "metadata": {"kcat": 2}},
  ...
]
```

Seeds: `seed ∈ [lineage_seed, lineage_seed + n_init_sims)`, honoring
`different_seeds_per_variant` (non-overlapping per-variant seed ranges). Existing vEcoli
configs (`default.json`, `two_generations.json`, …) are copied into `v2ecoli/configs/` and run
unchanged.

### 2. Variant application — declarative config override

`build_meta_composite` threads each branch's `overrides` into the cell builder
(`baseline(core, seed, cache_dir, config_overrides=...)`), patching the per-process config dict
**before** the `Step` is instantiated (Steps read config at init). Override path is
`"<process-name>.<config-key>"` (dotted for nested keys). Applied overrides are visible in the
saved `.pbg`.

### 3. Generation control — `v2ecoli/steps/generation_controller.py` (the one genuinely new Step)

`Division` already spawns both daughters in-state. `GenerationController` runs **after**
`Division` in each lineage's `flow_order` and adds what vEcoli's NextFlow channels encoded:

- **`single_daughters`**: if true, prune the `"…1"` daughter (`agents: {_remove: ["…1"]}`),
  keeping a linear lineage; if false, keep both (binary tree).
- **Generation depth** derived from phylogeny-id length; **stop** a lineage once it reaches
  `generations` (the agent is no longer stepped/divided).
- A **branch is complete** when no live cell remains below the generation cap.

The exact generation-indexing convention (off-by-one of `generations`) will be matched to
vEcoli's `generate_lineage` during implementation.

### 4. Meta-composite builder — `v2ecoli/workflow/meta_composite.py`

`build_meta_composite(config) -> document` constructs the top-level Composite: one embedded
`LineageComposite` per `(variant_index, seed)` under `state.branches[...]`, each built from the
variant-overridden bundle with the correct per-process seed and emitter metadata. Returns a
document saveable via the existing `pbg.save_pbg`.

### 5. Emission (already supported)

Each cell carries an emitter via `parquet_vecoli()`; the builder fills
`experiment_id / variant / lineage_seed / agent_id` (and `generation`, tracked as the lineage
advances) from the branch spec. Emitted parquet is directly comparable to vEcoli output.

### 6. Analyses as Steps — `v2ecoli/workflow/analysis.py` (framework + one example)

- An `AnalysisStep` base and a **scale registry** for the five vEcoli scales
  (`single / multidaughter / multigeneration / multiseed / multivariant`). Each scale declares
  which slice of results it reads (single cell → across daughters → across generations →
  across seeds → across variants).
- A reserved **post-sweep analysis phase** in the meta-composite where these Steps wire in,
  reading the partitioned parquet (and/or in-state listener data).
- **MVP ships the base + registry + one example** (a `mass_fraction_summary`-style single-scale
  Step) to prove the wiring. The full analysis-Step library is a follow-up spec.

### 7. Driver / CLI — `v2ecoli/workflow/run.py` → `v2ecoli-workflow`

```
v2ecoli-workflow --config configs/two_generations.json [--out out/<exp>] [--build-only]
```

Loads + inherits config → expands grid → `build_meta_composite` → runs `composite.run(dt)` in a
loop until all branches complete (or a global safety cap) → saves `sweep.pbg` → prints a
per-branch summary. `--build-only` writes the `.pbg` without running (for inspection in
loom/explorer).

## File layout (new)

```
v2ecoli/configs/                      default.json + ported example configs
v2ecoli/workflow/__init__.py
v2ecoli/workflow/config.py            inheritance + merge (port)
v2ecoli/workflow/variants.py          variant grammar → branch grid (port)
v2ecoli/workflow/meta_composite.py    build_meta_composite()
v2ecoli/workflow/run.py               driver loop + CLI entry
v2ecoli/workflow/analysis.py          AnalysisStep base + scale registry + 1 example
v2ecoli/steps/generation_controller.py
tests/test_workflow_config.py         inheritance + variant-grid expansion
tests/test_meta_composite_build.py    grid → branches, overrides land in edges
tests/test_generation_controller.py   single_daughters prune + gen-cap stop
tests/test_workflow_smoke.py          tiny config (1 variant, 1 seed, 1 gen) runs + emits
```

## Testing

- **Config**: inheritance chains merge with correct precedence; `LIST_KEYS_TO_MERGE` behavior;
  variant grammar (`value`/`linspace`/`nested` × `op`) expands to the expected grid;
  `different_seeds_per_variant` produces non-overlapping seed ranges.
- **Build**: grid → expected branch set; declarative overrides land in the right process edge
  config; `.pbg` round-trips.
- **Generation control**: `single_daughters` prunes the `"…1"` daughter; lineage stops at the
  generation cap; branch-complete detection.
- **Smoke**: a minimal config (1 variant, 1 seed, 1 generation) builds, runs to completion, and
  emits partitioned parquet with correct metadata.

## Out of scope (deferred / non-goals)

- Parallel / distributed execution (NextFlow, HPC/SLURM, cloud, HyperQueue). The meta-composite
  runs sequentially in one process — speed traded for inspectability.
- ParCa-recomputing variants (gene knockouts that rescale matrices); cache-bundle-transform
  variants.
- Resume / caching of partial sweeps.
- The full analysis-Step library (this spec ships the framework + one example only).
