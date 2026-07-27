# Hardening summary — `multiscale-bioprocess`

**Date:** 2026-07-27 · **Branch:** `harden/multiscale-bioprocess` (worktree, based on origin/main `cd75eae9`)

## The load-bearing gap I picked

The **keystone study mbp-03 (BiRD reactor ↔ cell coupling)** carried canonical
axes `simulation_status: ran` / `evaluation_status: in_progress`, but on disk it
had **no supporting evidence**: `runs: []`, `tests: []`, an **"ungraded"**
report card, a `gate_status_summary` and `runtime` still saying *"BLOCKED on the
upstream fork"*, and `conclusion_verdicts: PENDING`. Worse, its behavior tests
could not even be collected — `pbg_bioreactordesign` is **not installed in the
shared v2ecoli venv**. So the headline coupling claim was **never demonstrated
green anywhere on disk.**

Simultaneously, the same study *and* the investigation executive cited the
O2-saturation temperature-sign bug (`pbg-bioreactordesign#2`) as an **open,
unfixed upstream divergence** biasing mbp-03/04 dissolved-O2 — but that bug was
**already fixed upstream** (PR #3, `a06709b`, 2026-06-15, with a regression
test). The docs overclaimed an open bug **and** overclaimed "ran."

This is the gap with the most leverage on the headline claim, so I closed it.

## Mode (per skill step 2)

**Unbacked claim / deferred scaffold → RUN it, measure** + **overclaimed
verdict → reconcile to evidence**. (Not the O2 bug fix the driving prompt
floated as a candidate — that was already merged upstream; the redundant fix
worktree I opened was removed.)

## What I did

1. **Confirmed the O2 fix and its effect.** `saturation_concentration('O2', T)`
   now correctly *decreases* with temperature: **8.17 mg/L @298.15 K →
   6.55 mg/L @310.15 K** (buggy version reported ~10.2). Verified standalone
   against `pbg-bioreactordesign` origin/main (`a360f92`, includes PR #3).

2. **Ran the keystone coupling to green.** Executed all 7 behavior tests in
   `tests/test_mbp_03_bird_reactor_coupling.py` → **6 passed, 1 skipped**. The
   skip is a *documented FBA-model limitation* (M9-minimal-glucose yields zero
   CO2 environmental efflux, so the CO2 leg can't be exercised), **not** a
   wiring failure. Measured through the full coupled composite:
   - `cstar_o2 = 6.58 mg/L` at 310.15 K (corrected calibration flows through).
   - DO drops below saturation under cell O2 demand (2nd-half mean 6.24 mg/L).
   - no-cells DO → Henry saturation; higher kLa raises steady-state DO;
     one generation completes without divergence; O2 mass balance closes ±5%.

3. **Reconciled the docs to the data:**
   - mbp-03 `study.yaml`: O2-bug `open_question` → **`status: resolved`** (cites
     PR #3 + verified numbers); `gate_status_summary` → **UNBLOCKED/PASSING**;
     `runtime` → **ran** (with reproduce command + env-shadow note);
     `conclusion_verdicts` → **SUPPORTED**; visualization note de-blocked.
   - mbp-03 report card (`tests.verdict.json` + `tests.html`): ungraded → **pass
     (6 passed, 1 skipped)**.
   - investigation `executive`: O2-bug moved out of `evidence_against` (it's
     fixed) into `evidence_for`; added mbp-03-green evidence; the CO2-leg
     deferral now stated honestly under `evidence_against`.
   - investigation `decisions_needed`: the stale *"mbp-03 blocked on upstream
     fork"* → **resolved**; added a new open decision **`add-bioreactor-dep-to-env`**.

## Verdict

The keystone coupling is **built and demonstrated green**; the O2-sign
correctness gap is **closed and verified**. The report card, the verdict, and
the data now tell the same story.

## Environment note (important for the driving session)

- `pbg_bioreactordesign` is **NOT in the shared v2ecoli venv**
  (`~/code/v2ecoli/.venv`). I ran mbp-03 by shadowing `pbg-bioreactordesign`
  **origin/main** on `PYTHONPATH` (worktree at `~/code/pbg-bioreactordesign-origin`,
  a git worktree off its origin/main — no commits needed there; the fix is
  already merged upstream).
- The shared venv also ships a `bigraph_schema` **older** than this worktree's
  origin/main code needs (no `bigraph_schema.contract` module → `import
  v2ecoli` fails). I shadowed `bigraph_schema 1.4.3` in a **git-ignored
  `.deps/`** (added to `$GIT_COMMON_DIR/info/exclude`; copied from a sibling
  hardening worktree). **The shared venv was not mutated.**
- Reproduce:
  ```
  PYTHONPATH=~/code/v2e-hbioprocess/.deps:~/code/v2e-hbioprocess:~/code/pbg-bioreactordesign-origin \
    ~/code/v2ecoli/.venv/bin/python -m pytest tests/test_mbp_03_bird_reactor_coupling.py
  ```

## Residual gaps → recommended followups

1. **`add-bioreactor-dep-to-env` (filed as a `decisions_needed`).** Add
   `pbg-bioreactordesign` (pinned ≥ PR #3) and bump `bigraph_schema` (≥1.4.3) in
   the canonical v2ecoli environment so the reactor-coupled studies are
   reproducible **without** a PYTHONPATH shadow. This is the top blocker to
   mbp-03/mbp-04 being CI-reproducible.
2. **CO2 leg unexercised.** Run mbp-03 under an overflow/anaerobic condition
   (nonzero CO2 efflux) to exercise `cells-raise-dissolved-co2-above-saturation`
   and un-skip that test.
3. **mbp-04 coupled multigen** (`phase: Simulate`, `evaluation_status:
   not_evaluated`) now inherits the corrected O2 calibration — its coupled runs
   should be executed + evaluated next; not attempted here (larger than one
   bounded headless slice).
4. **Static chart render** for `do-vs-saturation-coupled` (the behavior tests
   confirm the shape; the SVG is the remaining render task).

## Not changed (deliberately)

- No code was modified (the correctness fix is already upstream; the coupling
  code is correct as-is — proven by the green run).
- The 3.3-order Beulig scale gap (mbp-05/06) is a deliberately-framed gap, left
  untouched.
