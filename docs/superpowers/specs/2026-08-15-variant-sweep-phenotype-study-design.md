# Variant-sweep phenotype study — design

**Date:** 2026-08-15
**Status:** approved (brainstorming)
**Scope:** v2ecoli generic capability only. No downstream-model or perturbation-specific
content lives here — this repo stays application-agnostic.

## Problem

The whole-config WCM node (`VivariumEcoliProcess` in
`v2ecoli/library/vivarium_ecoli_engine.py`) loads a full upstream vEcoli config
natively via `EcoliSim.from_cli(--config ...)` and runs it faithfully as one
process-bigraph node. But it **never applies the config's `variants` block** — it
always runs the undosed base `sim_data`. `run_vivarium_ecoli_pbg_multigen` already
carries a `variant: int` param, but only records it as provenance metadata; it does
not change what runs.

A config's `variants` block declares a parameter grid (via `op: prod|zip|add`,
`value`/`linspace`/`arange`) that upstream expands into N perturbed `sim_data`
pickles in its workflow layer (`runscripts/create_variants.py`). Each grid point is
applied by a variant module's `apply_variant(sim_data, params)`. A single
`EcoliSim.from_cli` does none of this. So today no sweep over a config-declared
perturbation is possible through the node.

## Goal

Make the whole-config node **variant-aware**, expose it as a first-class workbench
composite, and provide a generic study template that sweeps a config-declared
variant index and compares a configurable observable across the sweep axis. All
of it perturbation-agnostic: the node delegates entirely to whatever
`ecoli.variants.*` module the *fork config* names, so this repo carries no
knowledge of any specific perturbation.

## Design

### A. Apply variants in the whole-config path (`vivarium_ecoli_engine.py`)

In the whole-config build block, after `EcoliSim.from_cli(--config)` has loaded
`sim_data`:

1. If `sim.config` declares a non-empty `variants` mapping, take its single
   variant name + config.
2. Import the fork's own grid expander and apply it, fork-bound (via
   `$V2E_VECOLI_DIR`, same import discipline as the serializer fix):
   - `parse_variants(variant_config)` → ordered list of param dicts.
   - `variant_mod = importlib.import_module(f"ecoli.variants.{name}")`.
   - `sim_data = variant_mod.apply_variant(sim_data, param_dicts[variant])`.
3. Record `{variant_name, variant_index, resolved_params}` in the node's
   provenance metadata.

Index `variant` comes from the existing param. Out-of-range → clear error listing
the grid size. `variants` absent/empty → unchanged base behaviour (back-compatible).

This is pure delegation to the fork. No perturbation names, timelines, or biology
appear in this repo.

### B. Register the whole-config node as a composite generator

Add `v2ecoli/composites/vecoli_whole_config.py` exposing a composite generator
`vecoli_whole_config` referenceable by dotted path
`v2ecoli.composites.vecoli_whole_config.vecoli_whole_config`, with params:
`{from_vecoli_config, variant, seed, n_generations}`. It wraps the existing
`build_vivarium_ecoli_composite` / `run_vivarium_ecoli_pbg_multigen` path so a
`study.yaml` can name it directly. Today the node is only reachable from the
comparison-harness script.

### C. Generic phenotype extractor + report

A small extractor that, given a set of runs (one per sweep index) and a list of
**configurable observable store paths**, collects each observable's time-series per
run and renders a sweep-axis comparison panel (index/parameter on x, observable on
y). Reuses the existing comparison-report renderer rather than adding a new one.
Observable paths are inputs — the extractor knows nothing about what they mean.

### D. Public generic study template

A workbench study (`workspace/studies/variant-sweep-phenotype-demo/`) that
demonstrates the full pattern end-to-end on a **neutral** built-in variant that
needs no new fork content (the `condition` variant). It:

- references composite `v2ecoli.composites.vecoli_whole_config.vecoli_whole_config`,
- declares a small sweep over the neutral variant's index,
- extracts a generic observable panel (e.g. growth rate / mass),
- serves as the copy-me template for any downstream instantiation.

## Data flow

```
fork config
  → EcoliSim.from_cli loads base sim_data
  → parse_variants(config.variants) → [params_0 … params_{N-1}]
  → apply_variant(sim_data, params[i])          # once per sweep index i
  → vivarium Engine runs the (spatial or single-cell) composite
  → emit configured observable store paths
  → extractor collects per-index series
  → sweep-axis comparison report
```

## Boundaries / interfaces

- **`vivarium_ecoli_engine` variant hook** — in: `(sim.config, variant_index)`;
  out: perturbed `sim_data` + provenance; depends on: fork's
  `parse_variants` + `ecoli.variants.*`. Testable by asserting the perturbed
  `sim_data` differs from base at the variant's documented attributes.
- **`vecoli_whole_config` composite** — in: `{from_vecoli_config, variant, seed,
  n_generations}`; out: a runnable composite; depends on: the variant hook.
- **phenotype extractor** — in: `(runs, observable_paths)`; out: sweep panel;
  depends on: emitted zarr/parquet only. No coupling to the node internals.

## Testing

- Unit: variant hook applies index i and leaves base unchanged for empty/absent
  `variants`; out-of-range index errors clearly.
- Unit: extractor produces one series per run for each requested path; missing
  path → explicit skip, not a crash.
- Integration: the neutral-variant demo study builds + simulates a 2–3 point
  sweep and produces a non-empty sweep panel.

## YAGNI

- The node/runner support the full declared grid, but the demo study instantiates
  only a small sweep.
- No new fork content: the template uses the existing `condition` variant.
- No perturbation-specific observable logic — observable paths are pure inputs.

## Downstream

The capability (A–C) and template (D) flow to downstream private repos via the
existing deterministic overlay sync. Any perturbation-specific study is authored
*there*, by copying template D and naming that repo's fork config + variant +
observable paths. This repo never names a specific perturbation.
