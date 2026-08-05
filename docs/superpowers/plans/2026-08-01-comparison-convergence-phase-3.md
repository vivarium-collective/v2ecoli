# Comparison Convergence — Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development.
> **BLOCKED-ON:** the Phase-2 gated e2e must pass first — `run investigation <slug>` must run vEcoli + candidate through the general runner end-to-end and render the full card set + a resolved cross-config matrix. Retiring `v2e-compare` (Task 1) is unsafe until the general path demonstrably covers its capabilities. The matrix cross-study data-flow (the Phase-2 placeholder) must also be resolved as part of the e2e.

**Goal:** Finish the convergence — retire `v2e-compare`, add GovCloud via `run-remote`, and update the reuse-guide PDF so its general-capability examples are literally runnable.

**Architecture:** Phase 1+2 delivered the general path (vecoli composite + adapter + Analyses + materializer). Phase 3 removes the bespoke runner, wires remote execution, and closes the docs loop.

## Global Constraints
- Worktree `~/code/v2ecoli--compare-generalize`, branch `compare-generalize`. Commit by explicit path.
- Do NOT remove `v2e-compare` until Task 1's parity check confirms the general path covers each capability it provided (run all configs, all cards, the matrix, matched-init, verdicts) — with the Phase-2 e2e as evidence.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 0 (gate) — resolve the cross-config matrix cross-study data-flow
**Files:** `v2ecoli/workflow/comparison_materialize.py` + `comparison_matrix.py`. Prereq for the e2e.
- Determine how the workbench passes one study/analysis's output (a per-config `comparison_cards` verdict) into an investigation-level analysis (`comparison_matrix`). Options: a workbench post-investigation analysis that reads each study's run-store verdict; or the materializer aggregates per-config verdicts from the run store after the per-config analyses complete. Replace the `<run>::comparison_cards` placeholder token with the real resolution.
- TDD against a fixture set of per-config runs; then confirm at the e2e.

## Task 1 — Retire `v2e-compare`
**Files:** `scripts/compare_cli.py`, `scripts/_compare/runner.py`, docs. Test: parity.
- **Parity check first:** run the same comparison via BOTH `v2e-compare run <slug>` and the general path (`run investigation`/`prepare-investigation`) on the mini; assert the verdicts + cards agree. Only after parity:
- Make `v2e-compare` a thin deprecation shim that delegates to the general runner (or remove it and update all references — `pyproject.toml` `[project.scripts]`, docs, AGENTS.md). Keep the `comparison:` block as the declarative spec.
- Remove `scripts/_compare/runner.py`'s bespoke orchestration once nothing calls it; keep the reused pieces (theme, report_cards, verdict, adapter, overview) that the Analyses depend on.
- TDD: the general path produces the same investigation outputs; no dangling references to the removed entry point.

## Task 2 — GovCloud via `run-remote`
**Files:** wiring so `vivarium-workbench run-remote` runs the vecoli/candidate composites on sms-api; test (gated).
- Confirm the `vecoli` + candidate composites build + run under sms-api (the container is built from the git ref; `reference_repo` becomes the git ref). Wire the comparison spec so a remote run emits to S3 and lands for local `--render-only`.
- Gated e2e on GovCloud (behind an env flag + tunnel up).

## Task 3 — Update the reuse-guide PDF
**Files:** the guide source (`~/AI-Generated/…` HTML) → re-render.
- Replace the "general examples as target" framing with **verified runnable** examples now that `run investigation`/`run-remote` drive the comparison. Keep the two-audience structure. Re-render to `~/AI-Generated/whole-cell-model-comparison-reuse-guide.pdf`.

---

## Self-Review Notes
- Coverage: spec §6 retire v2e-compare→T1; §Phase-3 run-remote→T2; guide→T3; the matrix data-flow gap→T0.
- Ordering: T0 (matrix) + Phase-2 e2e are prerequisites; then T1 (parity → retire); T2/T3 independent-ish after.
- Retiring `v2e-compare` is irreversible-ish for users — the parity check + keeping a deprecation shim de-risks it.
