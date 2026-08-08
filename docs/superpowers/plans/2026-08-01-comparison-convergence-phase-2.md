# Comparison Convergence — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox syntax.
> **BLOCKED-ON:** the Phase-1 GATE (mini e2e: vEcoli runs via the general runner + the run-store layout matches `run_store_adapter`). Do NOT start Task 2+ until the gate report confirms the adapter's run-store assumption; if it MISMATCHES, fix `scripts/_compare/run_store_adapter.py` first (localized to `_resolve_zarr_store`/`_lookup_emitter_path`).

**Goal:** Run the *whole* comparison — all configs, the full card set, and the cross-config matrix — through the general `vivarium-workbench` runner, from the `comparison:` spec, with matched initial state.

**Architecture:** Extends Phase 1 (registered `vecoli` composite + run-store adapter + `comparison_summary` Analysis). Phase 2 adds matched-initial-state as a composite/study param, the full card set as Analyses, an investigation-level cross-config matrix Analysis, and drives it all via `run investigation` / `prepare-investigation`.

**Tech Stack:** process-bigraph composites, vivarium-workbench runner + Analysis framework, `scripts/_compare` (reused), pytest.

## Global Constraints
- Worktree `~/code/v2ecoli--compare-generalize`, branch `compare-generalize`. Commit by explicit path (never `git add -A`).
- **Parallel-safe:** `v2e-compare` keeps working through Phase 2 (retired only in Phase 3). Reuse `scripts/_compare` read-only.
- Heavy engine runs gated behind `COMPARE_CONVERGE_E2E=1`; unit tests hermetic.
- Spec: `docs/superpowers/specs/2026-08-01-comparison-general-runner-convergence-design.md`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1 — Matched initial state as a param
**Files:** `v2ecoli/composites/ecoli_baseline.py` (or a thin wrapper) + a study-param path; test `tests/test_matched_initial_state_param.py`.
**Interfaces:** the candidate accepts a `match_simdata: str | None` param (a path to the reference's `simData.cPickle`); when set, it loads the candidate's initial state from that simData (the same matched-initial-state mechanism `run_comparison_ensemble.py` uses today via `--match-vecoli-simdata`), so t=0 is identical. Declarative param, not CLI logic.
- [ ] Read how `run_comparison_ensemble.py` applies matched-initial-state (`--match-vecoli-simdata`/`--match-initial-state` → `save_sim_input(match_...)`); express the same as a composite/study param.
- [ ] TDD: with `match_simdata=<fixture simData>`, the candidate's initial state derives from it (assert a known bulk count matches the reference's), hermetic against a small fixture.
- [ ] Implement; commit.

## Task 2 — Full comparison Analysis (all cards)
**Files:** `v2ecoli/workflow/analyses/comparison_cards.py` (or extend `comparison_summary.py`); test `tests/test_comparison_cards_analysis.py`.
**Interfaces:** `comparison_cards(candidate_run, reference_run, config, cards=[...])` renders the full per-config card set (summary, parca, statistical, standard, trajectory, distribution, metabolism, composition) reusing the existing `scripts/_compare/report_cards/*` builders + verdict logic, from the two runs via the adapter. Returns `{cards: {name: html}, verdict}`.
- [ ] Read each `report_cards/*.py`'s input contract; note which read paired `observables`/`plot_trajs` vs raw `v2_dir`/`ve_dir` (the adapter returns per-run; pair as needed — some cards multi-seed, single-run pairing may only support a subset in Phase 2; document which cards are wired now vs deferred).
- [ ] TDD: given two fixture runs, the Analysis renders each wired card's HTML + a coherent verdict; status via glyph+label; `|Δ|` real (not `--`).
- [ ] Implement (reuse builders; no new science); commit.

## Task 3 — Cross-config matrix Analysis (investigation-level)
**Files:** `v2ecoli/workflow/analyses/comparison_matrix.py`; test `tests/test_comparison_matrix_analysis.py`.
**Interfaces:** `comparison_matrix(config_verdicts: dict[str, verdict])` → the configs × observables matrix HTML, reusing `reports/_summary`/the existing overview builder + `scripts/_compare/theme`. Consumes the per-config verdicts produced by Task 2 across configs.
- [ ] Reuse the existing matrix/overview renderer (`reports/_summary` or `scripts/_compare/overview` if present); do not rebuild.
- [ ] TDD: given a multi-config verdict set, produces the matrix with the right cells/verdicts.
- [ ] Implement; commit.

## Task 4 — Drive the comparison investigation via the general runner
**Files:** materializer that turns the `comparison:` block into paired candidate+reference studies the general runner runs; hook into `run investigation` / `prepare-investigation`. Test `tests/test_comparison_investigation_materialize.py`.
**Interfaces:** from a `comparison:` block (candidate, reference, configs), materialize per-config paired studies (candidate `ecoli_baseline` + reference `vecoli`, matched-initial-state param), so `vivarium-workbench run investigation <slug>` / `prepare-investigation --investigation <slug>` runs both engines per config and the post-sim Analyses (Task 2/3) render the cards + matrix.
- [ ] Read how the workbench discovers an investigation's studies + registers post-sim analyses (Phase-1 Task 4 registration + `prepare-investigation`).
- [ ] TDD: given a small `comparison:` block, materialization yields the expected paired studies + wires the comparison Analyses; hermetic.
- [ ] Implement; commit.

## Task 5 — Full-config e2e (gated)
**Files:** `tests/test_convergence_phase2_e2e.py` (gated `COMPARE_CONVERGE_E2E=1`).
- [ ] Behind the gate (on the mini): `vivarium-workbench prepare-investigation --investigation <slug>` runs all configs' candidate+reference via the general runner and renders the full card set + matrix — entirely through general capabilities, no `v2e-compare`. Compare the verdicts against a `v2e-compare` run of the same configs (they should agree within tolerance) as a cross-check.
- [ ] Full non-sim suite green; classify failures vs base. Commit.

---

## Phase 3 (outline, separate plan later)
Retire `v2e-compare` (orchestration → general runner; render → Analyses); wire `run-remote` for GovCloud; update the reuse-guide PDF with real, runnable general examples.

## Self-Review Notes
- Coverage: spec §Notes matched-initial-state→T1; §4 full Analysis→T2; matrix→T3; §3 run investigation→T4; e2e→T5. Retire v2e-compare + run-remote + guide = Phase 3.
- **Gate dependency:** T2–T5 assume the run-store adapter reads real runs correctly — validated by the Phase-1 gate. Fix the adapter first if the gate mismatches.
- Some cards are multi-seed (statistical) — single candidate-run vs single reference-run may only fully wire a subset in Phase 2; document which, defer multi-seed pairing if needed.
