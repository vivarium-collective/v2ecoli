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

**Run model (decided):** **2-generation lineage** on both engines — vEcoli via
its Nextflow workflow (as today), v2ecoli via the meta-composite/`LineageProcess`
path (as today). The injected processes are added to the inner baseline composite
that the lineage rebuilds each generation, so they participate in every
generation. (A single-generation duration-runner path was considered and rejected
in favor of multigen fidelity.)

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
- No bit-parity requirement — comparison is statistical/behavioral equivalence.

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

Chosen approach: **inject into the baseline composite the lineage rebuilds each
generation** via a new `injected_processes` parameter on `baseline()`
(`v2ecoli/composites/baseline.py`). Topology paths are shared between the models,
so a process's vEcoli topology is reused verbatim. This is the single integration
point that reaches every generation (the lineage calls `baseline()` directly —
`v2ecoli/workflow/lineage.py:_build_generation`).

`inject.py` (harness-side, runs in the v2 sim subprocess) provides two functions:

`resolve_injections(fork_repo, resolved_config)` → `list[InjectionSpec]`. Inputs:
the fork repo path + resolved fork config.
1. Prepend the fork repo to `sys.path`; import its `ecoli.processes` so the
   fork's `process_registry` is populated.
2. For each name in `add_processes` and each replacement class in
   `swap_processes`: resolve the class via the fork's `process_registry`.
3. **Classify** each class: PBG-native (`inputs()`/`outputs()`) → use as-is;
   Vivarium-1.0 (`ports_schema()` + `next_update()`) → mark for
   `wrap_vivarium_process()`; partitioned (`calculate_request`/`evolve_state`) →
   **fail fast** (out of scope v1) with a clear message + extension-point pointer.
4. Resolve each process's topology (`config["topology"][name]`, falling back to
   the class's `TOPOLOGY`) and config (explicit dict / `"default"`; `"sim_data"`
   on a new process → fail fast).
5. Return `InjectionSpec` dicts: `{name, module, qualname, kind, as_step, config,
   topology, interval}` — plus report metadata (source kind, ports, translation
   status). This list is JSON-serializable and is written into the v2 config under
   `injected_processes` so it reaches the sim subprocess.

`apply_injected_processes(cell_state, flow_order, core, specs)` → mutates the
baseline doc's per-cell `cell_state` + `flow_order`. For each spec: import
`module.qualname`, `wrap_vivarium_process()` if Vivarium-1.0, `core.register_link`
the resulting class, then `cell_state[name] = make_edge(instance, topology,
edge_type=...)` and append `name` to `flow_order`. Fail fast if a topology path
references a store absent from the cell state tree (message names port + path).

Units: if a new v1 process uses Unum quantities, convert at the boundary with
`v2ecoli/library/unit_bridge.py` (existing); Unum must not leak into v2ecoli.

### 3. `baseline()` + `LineageProcess` threading — v2ecoli build path

- Add an `injected_processes: list | None = None` parameter to `baseline()`
  (decorator `parameters` + signature). After the standard build loop, call
  `apply_injected_processes(cell_state, flow_order, core, injected_processes)`.
- Thread it through: `LineageProcess.config_schema` gains `injected_processes`;
  `_build_generation()` passes `injected_processes=self.config.get(
  "injected_processes")` to `baseline()`; `meta_composite._lineage_node` copies
  `config["injected_processes"]` into the LineageProcess node config.
- These are no-ops when `injected_processes` is empty/absent (baseline unchanged).

### vEcoli side

The orchestrator runs the **fork's** config natively
(`python -m runscripts.workflow --config <cfg>` from `--vecoli-repo`). No
translation needed — the fork already registers and wires its processes. Reuse
existing per-engine subprocess isolation and run caching.

## Report — extend `_compare/report.py`

The report has four required parts (per user requirement: show the loaded config,
the converted processes, rich behavior, and statistical-equivalence via report
cards). Keep the existing two-column vEcoli|v2ecoli layout, verdict badges, sticky
nav as the shell.

1. **Loaded-config panel** — the resolved fork config that drove the run:
   experiment_id / generations / seeds, the `add_processes` / `swap_processes` /
   per-process `topology` / `process_configs` that were applied, and the
   `_dropped_vecoli_keys` (vEcoli-only keys not consumed by v2ecoli). Rendered
   from the resolved config dict.

2. **Converted-processes panel** — one row per injected process: name, source
   class (`module.qualname`), kind (Vivarium-1.0 / PBG-native), translation status
   (wrapped / native), the translated ports + topology, and whether it produced
   non-empty updates in *both* engines (basic "did it run" gate). Built from the
   `InjectionSpec` metadata + a per-engine run probe.

3. **Behavior detail** — behavioral overlays + trajectory sparklines ported from
   `reports/composite_comparison.py` (`_overlay_section`, `_trajectory_section`,
   `_sparkline`) for the standard observables (mass/protein/RNA/DNA/volume) and
   any observable the new processes touch.

4. **Statistical-equivalence report card** — reuse the existing card library
   (`v2ecoli/library/report_card.py` + `card_criteria.py`): pair the two engines'
   per-cell observables, call `grade_axis()` (Welch *t*/p, Cohen's d, relative Δ,
   R² fingerprints) per axis, `grade_card()` to roll up groups, `verdict_json()`
   to emit a `report_card_verdict/v1` JSON, and embed `render_html()` output as a
   report section. The emitted `report_card_verdict.json` is also written to disk
   so the existing `report_card_axis` study evaluator can gate on it.

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

- The exact observable→axis→group mapping for the report card (which observables
  become which `card_criteria` axes), and the criterion thresholds per axis.
- Whether behavior-detail SVG helpers are imported from
  `reports/composite_comparison.py` directly or lifted into a shared module
  (`scripts/_compare/charts.py`) to avoid a `reports/` ← `scripts/` import.
