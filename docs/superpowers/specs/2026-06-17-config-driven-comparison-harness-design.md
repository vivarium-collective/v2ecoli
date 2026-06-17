# Config-driven vEcoli↔v2ecoli comparison harness — design

**Date:** 2026-06-17
**Status:** Approved design, pre-implementation
**Worktree/branch:** `v2e-compare-harness` / `feat/comparison-harness-config` (off `origin/main`)

## Problem

We want to compare the behavior of a **vEcoli fork** (a separate local checkout
that carries a few *new* Vivarium-1.0 processes plus a new config) against the
process-bigraph port (v2ecoli), automatically. Today this requires manual,
ad-hoc steps. The goal is a single robust command that:

1. Reads the fork's vEcoli config.
2. Runs that config natively in the fork (vEcoli side).
3. Auto-translates the fork's new Vivarium-1.0 processes into process-bigraph
   (`wrap_vivarium_process`) and injects them into the v2ecoli composite.
4. Runs the equivalent v2ecoli composite.
5. Emits an extended self-contained HTML comparison report.

**Comparison is behavioral, not bit-exact.** The v2ecoli result does not have to
reproduce the fork's trajectory bit-for-bit; the report characterizes
qualitative/quantitative similarity and divergence.

## What already exists (reused, not rebuilt)

- `scripts/compare_harness.py` — two-column vEcoli↔v2ecoli harness driven by one
  vEcoli config; runs both engines, diffs ParCa, diffs sim dynamics, renders HTML.
- `scripts/_compare/` — `config_adapter.py` (resolve + translate vEcoli config),
  `orchestrator.py` (run/cache each engine in isolated subprocesses),
  `parca_section.py`, `sim_section.py`, `stats.py` (tolerance/verdict),
  `report.py` (self-contained two-column HTML renderer with verdict badges).
- `reports/composite_comparison.py` — richer N-engine report (behavioral overlays,
  trajectory sparklines, per-reaction flux divergence). Source of visuals to port.
- `v2ecoli/library/vivarium_bridge.py` — `wrap_vivarium_process(v1_cls, ...)` and
  `translate_ports()`: a tested, documented converter from Vivarium-1.0 processes
  (`ports_schema()` + `next_update()`) to process-bigraph `Process`/`Step`.
- `vEcoli` config mechanism — `add_processes`, `exclude_processes`,
  `swap_processes`, `process_configs`, per-process `topology`; classes resolved by
  name via `process_registry`.
- **Key enabler:** v2ecoli's store-path layout is structurally identical to
  vEcoli's topology paths (e.g. `("bulk",)`, `("unique","promoter")`), so a
  process's topology is reused **verbatim** — no path translation needed.

## Goal / non-goals

**Goal:** one deterministic command — fork repo path + config → both engines run →
extended HTML report, with the new processes auto-translated and injected.

**Non-goals (v1 scope boundaries):**
- **Partitioned processes deferred.** New processes must be simple
  (`next_update` / `inputs`-`outputs`), not partitioned
  (`calculate_request`/`evolve_state`, Requester+Evolver). Detect partitioned new
  processes and **fail fast** with a clear message + extension-point note.
- **sim_data-derived configs deferred** for *new* processes. New-process configs
  are an explicit dict or `"default"`; `process_configs: "sim_data"` on a new
  process fails fast (baseline processes still use sim_data as today).
- **ParCa unchanged.** New processes ride on existing sim_data; the harness does
  not re-fit ParCa for them.
- No bit-parity requirement.

## Entry point

Extend `scripts/compare_harness.py`:

```
python scripts/compare_harness.py \
  --vecoli-repo /path/to/vEcoli-fork \
  --config configs/<fork-config>.json \
  -o out/compare/report.html \
  [--duration 2520 --interval 50] \
  [--mode full|fast] \
  [--tol-rel 0.25] \
  [--force]
```

- `--vecoli-repo` (NEW): path to the fork checkout. Defaults to `~/code/vEcoli`.
  The orchestrator runs the fork's config from here and the injector imports the
  fork's new process classes from here.
- `--tol-rel` (NEW): relative-trajectory tolerance for divergence badges (no
  bit-parity → looser, configurable).
- `--force` (NEW): bypass run cache.

## Architecture

Three new/changed pieces; everything else reused.

### 1. Config carry-through — extend `_compare/config_adapter.py`

`translate_vecoli_config` currently **drops** vEcoli-only keys. Change it to
**preserve** the process-set keys into the v2 config:
`add_processes`, `exclude_processes`, `swap_processes`, `process_configs`, and
per-process `topology`. Topology is carried verbatim (paths are shared between
the models). Keep recording genuinely-dropped keys in `_dropped_vecoli_keys` for
report transparency.

### 2. Process injection + translation — NEW `_compare/inject.py` (the novel core)

Chosen approach: **inject into the prebuilt baseline** (lowest risk — topology
paths are shared; reuses the existing baseline builder and converter).

Inputs: resolved fork config + `--vecoli-repo` path.

Steps:
1. Prepend the fork repo to `sys.path`; import its `ecoli.processes` so the
   fork's `process_registry` is populated.
2. For each name in `add_processes` and each replacement class in
   `swap_processes`: resolve the class via the fork's `process_registry`.
3. **Classify** each class:
   - process-bigraph-native (`inputs()`/`outputs()` present) → use as-is;
   - Vivarium-1.0 (`ports_schema()` + `next_update()`) → `wrap_vivarium_process()`;
   - partitioned (`calculate_request`/`evolve_state` present) → **fail fast**
     (out of scope v1) with a clear message and an extension-point pointer.
4. Register wrapped classes into v2ecoli core via `register_link`.
5. Return an **injection spec**: `{name → {address, config, topology, interval}}`,
   plus metadata (source kind, translation status, port list) for the report.

Units: if a new v1 process uses Unum quantities, convert at the boundary with
`v2ecoli/library/unit_bridge.py` (existing); Unum must not leak into v2ecoli.

### 3. Composite assembly hook — v2ecoli build path

Build the baseline composite, then apply the injection spec:
- add each new process node with its topology (verbatim) and process_config
  (explicit dict or `"default"`);
- honor `exclude_processes` / `swap_processes` against the baseline process set;
- fail fast if a topology path references a store that does not exist in the v2
  state tree (clear message naming the port and path).
Run via the existing v2ecoli workflow runner.

### vEcoli side

The orchestrator runs the **fork's** config natively
(`python -m runscripts.workflow --config <cfg>` from `--vecoli-repo`). No
translation needed — the fork already registers and wires its processes. Reuse
existing per-engine subprocess isolation and run caching.

## Report — extend `_compare/report.py`

Keep: two-column vEcoli|v2ecoli layout, verdict badges, ParCa diff, sticky nav.

Add:
- **"New processes" panel** — one row per injected process: name, source kind
  (Vivarium-1.0 / PBG-native), translation status, declared ports + topology, and
  whether it produced non-empty updates in *both* engines (a basic "did it run"
  sanity gate).
- **Behavioral overlays** + **trajectory sparklines** (ported from
  `composite_comparison.py`) for the observables the new processes touch, plus the
  standard mass/protein/RNA/DNA series.
- **Looser, configurable tolerances** — badges (`within_tol` / `drift` /
  `mismatch`) computed from *relative trajectory distance* (controlled by
  `--tol-rel`), not exact equality, reflecting the no-bit-parity stance.

## Robustness / automation

- Single deterministic command; each engine in an isolated subprocess (existing).
- Idempotent run caching keyed on (config, repo ref, duration, mode); `--force`
  to rerun.
- **Fail-fast** with actionable messages on: fork import failure; unknown process
  name; partitioned new process; sim_data config on a new process; missing
  topology store path; engine run failure (surface the subprocess log path).
- **Smoke test** — a tiny example Vivarium-1.0 process (Counter/lysis-style)
  added via a minimal example config, exercising config carry-through →
  translation → injection → run → report, so the harness is CI-testable **without
  the real fork**. The real fork is pointed at later via `--vecoli-repo`.

## Open items deferred to the plan

- Exact injection-spec data shape and where the assembly hook lives in the v2
  build path (baseline builder vs. a thin post-build injector).
- Precise relative-trajectory distance metric and badge thresholds.
- Whether the "New processes" panel reuses `composite_comparison.py`'s sparkline
  helpers directly or via a shared extracted module.
