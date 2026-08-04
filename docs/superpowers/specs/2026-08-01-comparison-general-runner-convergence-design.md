# Comparison Harness → General Workbench Convergence

**Date:** 2026-08-01
**Status:** design (approved directionally; pending written-spec review)
**Repo/worktree:** `v2ecoli` @ `origin/main`, worktree `v2ecoli--compare-generalize` (branch `compare-generalize`)

## Purpose

Retire the bespoke `v2e-compare` runner and run the whole-cell model comparison
through the **general `vivarium-workbench` capabilities**. The leverage: because
both models are Composites made through the process-bigraph template, the generic
runner already knows how to run them — a comparison needs no special engine
machinery, only a way to **pair** two Composite runs and **render** the diff.
Outcome: a comparison is "just an investigation you run," with one maintained
runner instead of a parallel bespoke CLI.

## Non-goals

- Adding a comparison primitive to the `vivarium-workbench` core (a separate
  repo). The reference is expressed with existing general capabilities, in the
  v2ecoli repo.
- Changing the science: candidate/reference fits, matched-initial-state, gating,
  and the card set are unchanged — only the *driver* changes.
- Retiring `v2e-compare` before its capabilities are fully covered (it stays
  working, in parallel, until Phase 3).

## Current state

- `v2e-compare` (`scripts/compare_cli.py` → `scripts/_compare/runner.py` →
  `scripts/run_comparison_ensemble.py`) is **standalone** — it does NOT call
  `vivarium-workbench`; its `run_study`/`run_investigation` are its own functions.
- The candidate is the `ecoli_baseline` Composite (a `@composite_generator` in
  `v2ecoli/composites/`). The reference (genuine vEcoli) runs via
  `run_comparison_ensemble.py`'s `--composite vecoli --vecoli-source vivarium-process`
  ("genuine vEcoli as a single composite, its own Engine inside") — but it is NOT
  a registered workspace Composite the general runner can invoke.
- The general runner (`vivarium-workbench run {study|investigation|composite|process}`,
  `run-remote`, `prepare-investigation`, `--render-only`) runs registered
  Composites/studies uniformly, but has no candidate-vs-reference concept.

## Design

### 1. Everything runnable is a registered, template-made Composite
Candidate = `ecoli_baseline` (exists). Reference = a **new registered `vecoli`
Composite** wrapping the vivarium-process vEcoli. Once registered, the generic
runner runs either uniformly (`run composite` / `run study`).

### 2. Register vEcoli as a Composite (`v2ecoli/composites/vecoli.py`)
A `@composite_generator` that wraps the genuine vivarium-process vEcoli (the same
path `run_comparison_ensemble.py` builds today), exposing params:
- `reference_repo` — the fork path (local) / git ref (sms-api build). **The fork
  is an explicit, declared composite param** (§5), threaded into the
  vivarium-process loader locally and the sms-api build ref remotely, and recorded
  in run provenance.
- `condition`, `seed`, and an optional `fork_config` (a vEcoli config JSON driving
  a process swap).
Verify it builds and runs via `vivarium-workbench run composite vecoli --steps N`
before pairing it in a comparison.

### 3. A comparison is an investigation of paired Composite studies
Per config, the investigation declares two runs: a **candidate study**
(`ecoli_baseline`) and a **reference study** (`vecoli`), with matched inputs
(`condition`, `seed`, and matched-initial-state as a param — §Notes). Run via the
general runner: `vivarium-workbench run study <config>` (candidate) +
`run study <config> --variant reference` (reference), or `run investigation
<slug>` / `prepare-investigation --investigation <slug>` for the whole set.

### 4. Comparison rendering becomes a workbench Analysis
The report cards, cross-config matrix, and gating move into a comparison
**Analysis** that reads the two study runs from the workbench run store (runs.db +
stored zarr) and emits the same outputs — **reusing the existing
`scripts/_compare` theme / report_cards / verdict / overview code** as the
analysis body (a thin adapter from the run store to the cards' inputs), not a
rewrite. `--render-only` regenerates all cards from stored runs.

### 5. Fork as an explicit composite param
`reference.repo` is a first-class param of the `vecoli` Composite: a repo/commit
locally (into the vivarium-process loader) or a git ref for the sms-api build.
Retargeting a fork = one declared param, run through the same generic call, and it
lands in the run's provenance (reproducible) rather than an ambient env var.

### 6. Retire `v2e-compare` (Phase 3)
Its orchestration → the generic runner; its render/gating → the Analysis. The
`comparison:` block survives as the declarative spec the investigation
materializes into paired studies.

## Phasing

- **Phase 1 — viable slice.** Register the `vecoli` Composite; run one config's
  candidate + reference through `vivarium-workbench run study`/`run composite`; a
  **minimal comparison Analysis** rendering ONE card (e.g. `summary` or
  `statistical`) from the two runs. Proves the generic path end-to-end. (This
  spec's implementation plan covers Phase 1.)
- **Phase 2.** Matched-initial-state as a param; the full card set + cross-config
  matrix as the Analysis; `run investigation` / `prepare-investigation` over all
  configs.
- **Phase 3.** Retire `v2e-compare`; wire `run-remote` for GovCloud; update the
  reuse-guide PDF with real, runnable general examples (closing the loop with the
  guide that currently describes this as the target).

## Risks / notes

- **Workbench study runner + vivarium-process vEcoli:** the `vecoli` Composite
  must run under the workbench's study runner (registration, env, emitter).
  Verify via `run composite vecoli` before pairing — this is the Phase 1 gate.
- **Matched initial state:** the candidate loads the reference's `simData`; expressed
  as a study param/input (a path), not CLI logic — same mechanism, declarative.
- **Run store adapter:** the comparison Analysis reads the workbench run store
  (runs.db + zarr), a different source than v2e-compare's out-dir. A small adapter
  maps stored runs → the existing cards' expected inputs.
- **Parallel safety:** keep `v2e-compare` working through Phases 1–2; only retire
  it in Phase 3 once the general path covers its capabilities.
- **ParCa matched-initial-state:** the candidate starting from the reference's
  ParCa `simData` is unchanged; the cache-correctness work (PR A) is orthogonal
  and compatible.
