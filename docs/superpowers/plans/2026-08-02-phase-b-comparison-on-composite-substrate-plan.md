# Phase B — comparison on the composite substrate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** Materialize the `comparison:` block into native `investigation.yaml` + per-config `study.yaml` files (one study per config: baseline=candidate, variant=reference) that the investigation-as-composite substrate runs — ParCa first, configs next, cross-config matrix last.

**Architecture:** Re-model `comparison_materialize` from paired studies to native single-study-per-config + a file writer; a `parca_prep` composite for the ParCa prerequisite (Gap 1); `comparison_matrix` reads per-config verdicts from disk (Gap 2). Both gaps resolve without changing the substrate's `run_study` contract.

**Tech Stack:** v2ecoli `workflow/comparison_materialize.py`, `composites/`, `workflow/analyses/comparison_matrix.py`, `workflow/parca_study.py`; the substrate (`vivarium_workbench.lib.investigation_execution`) via PYTHONPATH; pytest.

## Global Constraints
- Worktree `~/code/v2ecoli--compare-generalize`, branch `compare-generalize`. Commit by explicit path (never `git add -A`).
- **Test env:** run against the substrate branch + the v2ecoli venv:
  `PYTHONPATH=/Users/eranagmon/code/vivarium-workbench--inv-composite /Users/eranagmon/code/v2ecoli--compare-generalize/.venv/bin/python -m pytest <file> -v`
- **Do NOT change the substrate's `run_study` contract.** Gap 1's only allowed substrate touch is the prep-harvest fallback, and only if a marker-only prep run is rejected — escalate as BLOCKED first.
- Keep `v2e-compare` working (parallel-safe; reuse `scripts/_compare` read-only).
- Heavy engine runs gated; unit tests hermetic. Real paired e2e is a DEFERRED follow-up (mini) — not in this plan.
- Design: `docs/superpowers/specs/2026-08-02-phase-b-comparison-on-composite-substrate-design.md`.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Task 1 — `parca_prep` composite (Gap 1: ParCa as an ordinary study)

**Files:** Create `v2ecoli/composites/parca_prep.py`; Test `tests/test_parca_prep_composite.py`.

**Interfaces:**
- Produces: `@composite_generator(name="parca_prep") parca_prep(core=None, *, candidate_cache_dir, reference_cache_dir="", reference_repo="", build_if_stale=True) -> dict` — calls `resolve_or_build_parca(engine="candidate", cache_dir=candidate_cache_dir)` and, when `reference_cache_dir`, `(engine="reference", cache_dir=reference_cache_dir, reference_repo=reference_repo)`; when a check returns `stale` and `build_if_stale`, re-calls with `build=True`. Emits a minimal state carrying each engine's resolved status/path so a `run_study` harvest records a `runs.db` row.
- Consumes: `v2ecoli.workflow.parca_study.resolve_or_build_parca` (already returns `{status, path, reason}`).

- [ ] **Step 1: Write the failing test** — with a fixture candidate cache dir that `verify_cache_version` accepts (mock `resolve_or_build_parca` to return `{"status":"reused","path":...}`), assert `parca_prep(...)` returns a state dict reflecting `reused` for candidate (no build call); with a stale return then a build return, assert `build=True` was invoked once. Hermetic (mock `resolve_or_build_parca`).
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** the generator (thin wrapper; no ParCa science).
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** (`composites/parca_prep.py`, test).

## Task 2 — Native single-study-per-config materializer

**Files:** Modify `v2ecoli/workflow/comparison_materialize.py`; Test `tests/test_native_comparison_materialize.py`.

**Interfaces:**
- Produces: a new `to_native_study_specs(materialized) -> {study_name: spec}` (or refactor `to_study_specs`) that emits, per config, ONE study spec: `{name: <config>, baseline: [{name: <config>, composite: ecoli_baseline, params: {..., match_simdata}}], variants: [{name: reference, composite: vecoli, params: {...}}], comparative_visualizations: [...], analyses: [{name: comparison_cards, params: {candidate_run: <config>, reference_run: reference, cards: [...]}}], pipeline_gate: {prerequisites: [{study: parca, relation: leads-to}]}}` — plus the `parca` study spec (composite=`parca_prep`) and the investigation-level `comparison_matrix` analysis entry (members' config slugs in params).
- Consumes: existing `MaterializedInvestigation`/`ComparisonPair`/`ParcaPrerequisite` (reuse the resolved cache dirs, `match_simdata`, fork configs).

- [ ] **Step 1: Write the failing test** — given a 1-config `comparison:` block, assert `to_native_study_specs` yields (a) a `parca` study using `parca_prep`, (b) one `<config>` study with baseline=candidate + variant=reference + comparative_visualizations (runs referencing sim_names `<config>` and `reference`) + `analyses:[comparison_cards]` (candidate_run=`<config>`, reference_run=`reference`) + prereq `parca`, (c) an investigation-level `comparison_matrix` entry naming the config slugs. Hermetic.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** Keep the old paired `to_study_specs` intact (parallel-safe) or clearly supersede; add the native emitter.
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit.**

## Task 3 — Write native files to the workspace

**Files:** Modify `comparison_materialize.py` (add `write_native_investigation(materialized, workspace, invest_slug)`); Test `tests/test_write_native_investigation.py`.

**Interfaces:**
- Produces: `write_native_investigation(materialized, workspace, invest_slug) -> {investigation_path, study_paths}` — `yaml.safe_dump` each native study spec to `<workspace>/studies/<name>/study.yaml` and the `investigation.yaml` (members = `[parca, <config>...]`, `analyses: [comparison_matrix]`, keep the `comparison:` block) to `<workspace>/investigations/<invest_slug>/investigation.yaml`. Reuse `pipeline_gate` `{study, relation}` edge shape.
- Uses atomic writes if available; creates dirs.

- [ ] **Step 1: Write the failing test** — write to a tmp workspace, then reload each `study.yaml`/`investigation.yaml` and assert the round-tripped shape (members, prereqs, analyses, comparative_visualizations). Hermetic.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit.**

## Task 4 — `comparison_matrix` reads per-config verdicts from disk (Gap 2)

**Files:** Modify `v2ecoli/workflow/analyses/comparison_matrix.py`; Test `tests/test_comparison_matrix_disk_verdicts.py`.

**Interfaces:**
- Produces: `comparison_matrix` accepts `config_studies: list[str]` + a `workspace`/`report` root in its config; when `config_verdicts` isn't directly supplied, it LOADS each config study's persisted `comparison_cards` verdict from disk (the study's analysis output / conclusion artifact — locate the exact path by reading how `run_study_analyses`/`comparison_cards` persists its verdict) keyed by slug, then renders the matrix. Replaces the `<candidate_run>::comparison_cards` placeholder token.
- Backward-compatible: an explicit `config_verdicts` dict still works (existing tests).

- [ ] **Step 1: Write the failing test** — create fixture per-config verdict files on disk under a tmp workspace (mirroring how `comparison_cards` writes its verdict); call `comparison_matrix(config_studies=[...], workspace=...)`; assert it assembles `config_verdicts` from disk and renders the matrix cells. Hermetic.
- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement** the disk-loading path (find the real verdict artifact path first — read `comparison_cards`/`run_study_analyses` persistence).
- [ ] **Step 4: Run → pass** (plus the existing explicit-`config_verdicts` matrix test stays green).
- [ ] **Step 5: Commit.**

## Task 5 — Substrate integration (stubbed worker): order + matrix

**Files:** Test `tests/test_phase_b_substrate_integration.py` (gated? no — hermetic with stubbed worker).

**Interfaces:** drive the written investigation through the substrate: `from vivarium_workbench.lib.investigation_execution import run_investigation_composite` with the worker stubbed (`run_study_fn`), on a workspace written by Task 3.

- [ ] **Step 1: Write the test** — materialize a 1-config comparison → write to a tmp workspace (Task 3) → `run_investigation_composite(ws, invest_slug, run_study_fn=<recorder>)`; assert (a) `parca` runs BEFORE `<config>`; (b) the `comparison_matrix` analysis step runs AFTER the config study; (c) the recorder saw a `run_study` for `parca` and `<config>`. Stub the worker so no real sim runs; needs `process_bigraph` (v2ecoli venv + PYTHONPATH substrate).
- [ ] **Step 2: Run → iterate** until the ordering + analysis invocation assert. If the substrate rejects the `parca_prep` marker-only run in a real (non-stubbed) harvest, that surfaces only in the deferred real e2e — the stubbed path asserts ordering/wiring. Note any prep-harvest concern for the deferred e2e.
- [ ] **Step 3: Full non-sim suite** for the touched modules; classify failures vs base. Commit.

---

## Deferred (separate, mini) — real paired e2e
Not in this plan. Requires `$V2E_VECOLI_DIR` + a vEcoli ParCa rebuilt against current HEAD (Jul-cache-vs-HEAD skew = the `monomer_counts` shape mismatch). Smallest real proof: `basal seeds:1 gens:1 cards:[parca]` (t=0) paired run on the mini. Track as a follow-up.

## Self-Review Notes
- Coverage: design re-model→T2+T3; Gap 1→T1; Gap 2→T4; substrate order/matrix→T5; deferred e2e noted.
- Both gaps resolve without changing the substrate `run_study` contract (T1 composite-study; T4 disk-load) — the only possible substrate touch is the T5 prep-harvest fallback, gated behind BLOCKED-first.
- Type consistency: `to_native_study_specs`, `write_native_investigation(materialized, workspace, invest_slug)`, `parca_prep(...)`, `comparison_matrix(config_studies=..., workspace=...)` used consistently.
- Parallel-safe: `v2e-compare` + the old paired `to_study_specs` untouched/preserved.
