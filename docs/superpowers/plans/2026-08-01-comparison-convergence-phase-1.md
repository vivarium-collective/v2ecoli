# Comparison Convergence — Phase 1 (viable slice) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** Prove the comparison can run through the *general* `vivarium-workbench` runner, end-to-end, for one config — by registering vEcoli as a Composite and rendering one comparison card as a workbench Analysis.

**Architecture:** vEcoli (already runnable via `vivarium_ecoli_engine.run_vivarium_ecoli_pbg_multigen`) is wrapped as a registered `@composite_generator` so `vivarium-workbench run composite vecoli` runs it. A minimal comparison Analysis reads a candidate run + a reference run and renders the `summary` card, reusing `scripts/_compare`.

**Tech Stack:** Python 3.12, process-bigraph composites, vivarium-workbench runner + Analysis framework, pytest.

## Global Constraints
- Worktree `~/code/v2ecoli--compare-generalize`, branch `compare-generalize` (off `origin/main` @ `c53017b7`). Commit here only, by explicit path (never `git add -A`).
- **Parallel-safe:** do NOT modify or remove `v2e-compare` / `scripts/_compare/runner.py` in Phase 1 — it must keep working. Reuse `scripts/_compare` modules read-only where possible.
- Reference spec: `docs/superpowers/specs/2026-08-01-comparison-general-runner-convergence-design.md`.
- Test runner: `PYTHONPATH=~/code/v2ecoli--compare-generalize ~/code/v2ecoli/.venv/bin/python -m pytest`.
- Heavy engine runs (actually running vEcoli/v2ecoli) are gated behind an env flag (e.g. `COMPARE_CONVERGE_E2E=1`) so CI/unit runs stay fast.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1 — Register the `vecoli` Composite
**Files:** Create `v2ecoli/composites/vecoli.py`; register it (follow the `v2ecoli/composites/__init__.py` + `_helpers.py` registration path used by `ecoli_baseline`). Test: `tests/test_vecoli_composite.py`.

**Interfaces:**
- Produces: a `@composite_generator`-decorated builder `vecoli(...)` taking params `reference_repo: str` (fork path / git ref — explicit, per spec §5), `condition: str = "basal"`, `seed: int = 0`, `fork_config: str | None = None`. It wraps the genuine vivarium-process vEcoli (the same engine `scripts/run_comparison_ensemble.py` drives via `v2ecoli/library/vivarium_ecoli_engine.py`). Registered so it is discoverable in the composite registry as `vecoli` (or `v2ecoli.composites.vecoli.vecoli`).

- [ ] **Step 1: Read the two patterns.** Read `v2ecoli/composites/ecoli_baseline.py` (the `@composite_generator` shape, params, return) and `v2ecoli/library/vivarium_ecoli_engine.py` (`run_vivarium_ecoli_pbg_multigen` + how it builds/loads the fork — `reference_repo`/`V2E_VECOLI_DIR`, condition, seed). Note how vEcoli is built as a single pbg composite ("its own Engine inside").
- [ ] **Step 2: Write the failing test.** `tests/test_vecoli_composite.py`: import the generator, assert it registers (appears in the composite registry / `composite_lookup`) and that building it with `reference_repo=<path>, condition="basal", seed=0` returns a composite/document without running a sim (build-only, or a `--dry-run`-style construction). Assert the `reference_repo` param is surfaced (declared) so the fork is an explicit input. Keep it hermetic (no heavy engine run).
- [ ] **Step 3: Run it — confirm it fails** (module/registration absent).
- [ ] **Step 4: Implement `v2ecoli/composites/vecoli.py`** as the minimal wrapper: a `@composite_generator` that constructs the vivarium-process vEcoli composite via `vivarium_ecoli_engine`, threading `reference_repo`/`condition`/`seed`/`fork_config`. Register it alongside the other `ecoli_*` composites. Do not duplicate engine logic — call the existing engine builder.
- [ ] **Step 5: Run the test — confirm it passes.** Then `PYTHONPATH=$PWD .venv/bin/python -c "from pbg_superpowers... import composite_lookup"` (or the project's registry API) to print that `vecoli` is discoverable — paste output.
- [ ] **Step 6: Commit** (`v2ecoli/composites/vecoli.py`, `__init__.py`, test).

## Task 2 — Phase-1 gate: run `vecoli` through the general runner
**Files:** Test/verification only: `tests/test_vecoli_composite_runs.py` (gated).

**Interfaces:** Consumes Task 1's registered `vecoli`.

- [ ] **Step 1: Dry-run via the workbench.** `vivarium-workbench run composite vecoli --steps 2 --dry-run` (or the nearest available dry construction) — assert it resolves + builds the composite through the *general runner* (no bespoke script). Capture output.
- [ ] **Step 2: Gated real run.** Behind `COMPARE_CONVERGE_E2E=1`: `vivarium-workbench run composite vecoli --steps <small> --param reference_repo=$V2E_VECOLI_DIR --param condition=basal` — assert it runs vEcoli end-to-end via the general runner and produces a run in the workbench run store. This is the **Phase-1 gate**: it proves the generic runner runs the reference engine. Document the exact command + where the run lands.
- [ ] **Step 3: Write a smoke test** that asserts the composite resolves via the workbench registry (ungated) and, under `COMPARE_CONVERGE_E2E=1`, that a short run produces a run record. Commit.

## Task 3 — Run-store adapter
**Files:** Create `scripts/_compare/run_store_adapter.py`. Test: `tests/compare/test_run_store_adapter.py`.

**Interfaces:**
- Produces: `load_run_observables(run_ref) -> dict` mapping a workbench study/run reference to the observables shape the existing cards consume (the same `observables`/`plot_trajs`/zarr shape `scripts/_compare/report_cards` expects). A thin translator from the workbench run store (runs.db + stored zarr) to the cards' inputs — NOT new science.

- [ ] **Step 1: Identify the two input shapes.** Read `scripts/_compare/report_cards/__init__.py` `CARD_INPUTS` (what a card needs) and how the workbench stores a run's outputs (runs.db + zarr; check `vivarium-workbench runs`/`status` + the stored zarr path). Note the mapping.
- [ ] **Step 2: Write the failing test** with a small fixture run store (a tiny zarr + run record, or a monkeypatched reader) asserting `load_run_observables` returns the expected observables dict for a known input.
- [ ] **Step 3: Implement** the adapter (reuse `scripts/_compare/vecoli_parquet_reader.py` / the local zarr reader where possible).
- [ ] **Step 4: Run the test — passes.** Commit.

## Task 4 — Minimal comparison Analysis (renders the `summary` card)
**Files:** Create `v2ecoli/workflow/analyses/comparison_summary.py` (follow the `v2ecoli/workflow/analyses/*.py` pattern, e.g. `growth_overlay.py`). Test: `tests/test_comparison_summary_analysis.py`.

**Interfaces:**
- Consumes: Task 3's `load_run_observables`; the existing `scripts/_compare/report_cards/summary.py::build_summary_html` + verdict logic.
- Produces: an Analysis `comparison_summary(candidate_run, reference_run, config) -> {card_html, verdict}` that loads both runs via the adapter, computes the per-observable verdict (reuse the existing `statistical`/verdict path), and renders the `summary` card HTML.

- [ ] **Step 1: Read the Analysis pattern** (`v2ecoli/workflow/analysis.py` + one `analyses/*.py`) — signature, registration, output contract.
- [ ] **Step 2: Write the failing test** with two fixture runs (candidate + reference observables) → assert the Analysis produces a `summary` card HTML containing the per-observable |Δ| + a verdict, reusing `build_summary_html`. Hermetic (fixtures, no engine run).
- [ ] **Step 3: Implement** `comparison_summary.py`: adapter → observables for both runs → verdict (reuse existing verdict computation) → `build_summary_html`. Register it as an Analysis.
- [ ] **Step 4: Run the test — passes.** Then render smoke: given the two fixtures, write the card HTML to scratch and confirm it's substantial + contains both models. Commit.

## Task 5 — End-to-end Phase-1 proof (gated)
**Files:** `tests/test_convergence_phase1_e2e.py` (gated `COMPARE_CONVERGE_E2E=1`).

- [ ] **Step 1:** Behind the gate: run `vecoli` (reference) + `ecoli_baseline` (candidate) for one config via `vivarium-workbench run study`/`run composite`, then run the `comparison_summary` Analysis over the two runs → assert a `summary` card + verdict is produced **entirely through general capabilities** (no `v2e-compare`). Document the command sequence.
- [ ] **Step 2:** Full non-sim suite green (`pytest tests/ -m "not sim" -q`); classify any failure vs `origin/main` base (the ~11 pre-existing failures are out of scope). Commit any fixups.

---

## Self-Review Notes
- Coverage: spec §2 (register vecoli)→T1; §1/§3 run via general runner→T2/T5; §4 Analysis + run-store→T3/T4; §5 fork param→T1 (`reference_repo` explicit). Phases 2–3 (matched-initial-state param, full card set + matrix, retire v2e-compare, run-remote, guide update) are OUT of Phase 1.
- Parallel-safe: no edits to `v2e-compare`; `scripts/_compare` reused read-only (adapter + summary builder).
- Heavy runs gated behind `COMPARE_CONVERGE_E2E=1`; unit tests hermetic.
- The Phase-1 GATE is Task 2 Step 2 — the general runner actually running vEcoli. If that can't work (registration/env), stop and re-scope before Tasks 3–5.
