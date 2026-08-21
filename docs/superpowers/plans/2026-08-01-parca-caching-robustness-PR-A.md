# ParCa Caching + Robustness (PR A) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox syntax.

**Goal:** Restore the ParCa cache's correctness guarantees and fail-loud robustness — the safe, calibration-neutral subset of Fable's review (`PARCA_REVIEW.md`). No change to fit output.

**Source of truth:** `PARCA_REVIEW.md` (in this worktree). Each task cites its finding IDs (A1…); implementers read that finding's section for full detail + rationale.

**Architecture:** All changes are to the caching/bundling/provenance apparatus (`v2ecoli/library/cache_version.py`, `v2ecoli/core.py`, `scripts/build_cache.py`, `v2ecoli/cli/parca.py`, `step_09`, `.github/workflows/ci.yml`, docs) — NOT the fit math (`processes/parca/steps` worker logic, `sim_data.py` fields).

**Tech Stack:** Python 3.12, pytest, hashlib/SHA256 fingerprints, dill/pickle bundles.

## Global Constraints
- Worktree `~/code/v2ecoli--parca-review`, branch `parca-review` (off `origin/main` @ `c53017b7`). Commit here only.
- **Calibration-neutral is the pass criterion:** every task must leave `sha256(sim_data_cache.dill)` and `sha256(parca_state.pkl)` UNCHANGED. `inputs_hash` / `cache_version.json` MAY (and for A1/A7/A8/A9 MUST) change — that's the key, not the fit.
- Bump `SCHEMA_VERSION` in `cache_version.py` exactly once across this PR (any task that changes the CacheVersion format sets it; coordinate so it lands at one final value).
- Test runner: `PYTHONPATH=~/code/v2ecoli--parca-review ~/code/v2ecoli/.venv/bin/python -m pytest`.
- Commit BY EXPLICIT PATH (never `git add -A`); running pytest may mutate tracked artifacts.
- Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Preserve deliberate escape hatches (`V2ECOLI_SKIP_CACHE_VERIFY`, `--allow-partial-fit`).

---

## Task 1 — Fingerprint repair + guard tests (A1 + A4)
**Files:** `v2ecoli/library/cache_version.py`, `.github/workflows/ci.yml`, `tests/` (new guards). Read PARCA_REVIEW.md A1, A4.
- Repoint the 5 dead `INPUT_FILES` to the real `ecoli_*` composites (`ecoli_baseline.py`, `ecoli_population.py`, `ecoli_time_varying_env.py`, `ecoli_colony.py`, `ecoli_millard.py`); add `v2ecoli/composites/_helpers.py` if it's a shared builder on the fit/composite path. Make a **missing** input `raise` (not encode `"MISSING"`). Bump `SCHEMA_VERSION`.
- Make CI `hashFiles` agree with `INPUT_FILES` (derive from it, or fix the one divergent name `millard_pdmp_baseline.py`↔`baseline_millard.py`).
- **TDD:** `test_input_files_all_exist` (every INPUT_FILES path exists under repo root) and `test_ci_key_matches_input_files` (parse `ci.yml` hashFiles set == INPUT_FILES set) — both must FAIL on current `main`, PASS after. Also a test that a missing input raises.
- **Verify:** editing an `ecoli_*` composite now moves `compute_cache_version().inputs_hash`.

## Task 2 — Verify cache in production (A2)
**Files:** `v2ecoli/core.py` (`load_cache_bundle`/`_load_cache_bundle_cached`), `AGENTS.md`. Read A2.
- Call `verify_cache_version(cache_dir)` at the top of the production load path, with a `V2ECOLI_SKIP_CACHE_VERIFY=1` escape hatch. Correct the false claim at `AGENTS.md:287`.
- **TDD:** loading a cache whose `cache_version.json` mismatches the current fingerprint raises `StaleCacheError` (and the env escape hatch bypasses it). A fresh matching cache loads clean.

## Task 3 — Cache-key completeness: untracked inputs (A7 + A8 + A9)
**Files:** `cache_version.py` (extend `CacheVersion`), `v2ecoli/core.py` (`save_sim_input`), `step_05_fit_condition.py`, `pyproject.toml`. Read A7, A8, A9.
- Add a `build_params` block to `CacheVersion` (`condition`, `fixed_media`, `seed`, patch/condition-manifest id) and fold into `inputs_hash` (A7).
- Promote `V2PARCA_N_SEEDS` to `FitConditionStep.config_schema` (env fallback) and echo the resolved value into the composite state (A8).
- Add a runtime-`context` block (versions of `python, scipy, numpy, numba, dill, cvxpy, ecos, stochastic-arrow`) to `CacheVersion` + `inputs_hash` — copy the existing pattern from `v2ecoli/comparison/vecoli_parca.py:167-176`. Add a scipy **floor** in `pyproject.toml` (A9).
- Coordinate the single `SCHEMA_VERSION` bump with Task 1.
- **TDD:** changing `V2PARCA_N_SEEDS`, the condition/media/seed, or a pinned dependency version changes `inputs_hash`; identical inputs produce identical hash. `sim_data_cache.dill` bytes unchanged by these key additions.

## Task 4 — Fail loud on partial / incomplete fits (A3 + A6)
**Files:** `step_09_final_adjustments.py`, `v2ecoli/core.py:197-216`, `cache_version.py`. Read A3, A6.
- A3: record `{label: ok|error}` for the three mechanistic fits into composite state + provenance; refuse to write `parca_state.pkl` on failure unless `--allow-partial-fit`.
- A6: record `sorted(configs)` into `CacheVersion`; `verify_cache_version` asserts the expected set (or at least hard-fails on a required subset).
- **TDD:** a simulated mechanistic-fit exception aborts the write (no `parca_state.pkl`) unless `--allow-partial-fit`; a bundle missing a required config fails verification.

## Task 5 — Atomic bundle write + messaging (A5 + A21)
**Files:** `v2ecoli/core.py` (`_write_sim_input_bundle`), `v2ecoli/cache.py`. Read A5, A21.
- Write the bundle into `<dir>.tmp/` then `os.replace` the directory (or, minimum: `os.remove(cache_version.json)` as the FIRST action of a rebuild so an interrupted rebuild can't leave a valid marker over truncated data).
- Improve the `_rebuild_message`/stale messaging noted in A21.
- **TDD:** an interrupted rebuild (simulated by writing a partial bundle then invoking verify) is detected as stale, not passed as valid.

## Task 6 — Provenance sidecar + README (A11, sidecar only)
**Files:** `v2ecoli/cli/parca.py` (write a provenance sidecar next to `parca_state.pkl`), `models/parca/README.md`. Read A11.
- Emit a provenance sidecar recording `--mode`, `--cpus`, `--no-operons`, `--bundle-manifest-path`, resolved `V2PARCA_N_SEEDS`, producing git SHA, and dependency versions. Correct `models/parca/README.md` to describe the current full-mode fixture (51 conditions, ~2.5 min) — NOT the stale fast-mode/71-min claim.
- **NOT in this PR:** regenerating the committed `parca_state.pkl.gz` fixture (that's a validated, separate step in a follow-up).
- **TDD:** running the CLI writes a sidecar containing the mode + N_SEEDS + git SHA + dep versions.

## Task 7 — Docs sweep + cheap robustness (A19 + A18/A23/A26/A27/A28)
**Files:** the eight docs with stale "4–8 hours" (A19); `cli/parca.py`, `step_02`, `build_cache.py`, `data_loader.py`, `_scipy_compat.py`, `tests/fixtures/cache`, `AGENTS.md`/`CONTRIBUTING.md` per A18/A23/A26/A27/A28. Read A19, A18, A23, A26, A27, A28.
- Sweep the stale ParCa runtime claims to the measured ~2.5 min (full mode). Apply the cheap robustness nits (each is 1–2 lines; read each finding). Skip any that turn out to touch the fit — flag instead.
- **TDD/verify:** a doc-lint or grep asserting no "4-8 hours"/"4–8 hours"/"70 min" ParCa-runtime claims remain; the cheap-fix findings each get a minimal assertion where testable.

---

## Task 8 — Calibration-neutrality validation (the pass gate)
**Not TDD — the safety proof.** Read PARCA_REVIEW.md "Verification plan" (A1/A2/A4 subsection).
- Build ParCa once on `main` and once on this branch under a fixed env (`V2PARCA_N_SEEDS` unset, `PYTHONHASHSEED=0`, `OPENBLAS_NUM_THREADS=1`): `v2ecoli-parca --mode full -o out/parca_X`.
- Assert `sha256(parca_state.pkl)` is **identical** across main vs branch, and `sha256(sim_data_cache.dill)` identical after `build_cache.py`. Assert `inputs_hash` DID move and `SCHEMA_VERSION` bumped.
- Run `pytest -m "sim and not slow"` on the branch (behavior gate) + the full non-sim suite; classify any failure vs `main` base.

---

## Self-Review Notes
- Coverage: A1,A4→T1; A2→T2; A7,A8,A9→T3; A3,A6→T4; A5,A21→T5; A11(sidecar)→T6; A19,A18,A23,A26,A27,A28→T7; validation→T8. Deferred to later PRs: perf (A10,A12,A14,A16,A17,A20,A22 — note A14 dead-90MB-write is safe and may fold in if trivial), fixture regeneration (A11 tail), all Tier B (B1–B6, calibration-risky).
- Single `SCHEMA_VERSION` bump coordinated across T1/T3.
- Every task's pass criterion includes "sim_data/parca_state bytes unchanged"; T8 proves it end-to-end.
