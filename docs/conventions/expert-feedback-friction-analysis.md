# Expert-feedback friction analysis (dnaa-replication, 2026-06-22)

This doc mines `workspace/investigations/dnaa-replication/feedback*/*.yaml` — **49
unique annotations across 22 files** — from the v2ecoli dnaa-replication
investigation (experts: **Rashmi**, **Haochen**) to find the recurring friction
the framework should eliminate. The goal is simple: experts should spend their
review budget on **science**, not on hygiene, completeness, and plotting
conventions.

The corpus is reproducible. Feedback is stored under the investigation's
`feedback*/` directories and read back via
`pbg_superpowers.feedback_import.load_investigation_feedback`, which returns the
deduplicated annotation set used for the counts below.

---

## Top friction patterns

| # | Pattern | Verbatim signal | Count | Root cause |
|---|---------|-----------------|-------|------------|
| 1 | Stale charts/findings linger across runs | "why do you still have the old visualizations… as per my previous feedback"; "this section is stale" (×3) | ~9 | pipeline + skill |
| 2 | Same plot conventions re-requested | "x axis in minutes"; "demarcate lineages"; "label axes with units"; "remove gridlines"; "labels overlapping" — re-asked 05-30, 06-05, 06-10, 06-12 | ~7 | pipeline (no house style) |
| 3 | "Just run it" | "I have already approved the plan. Please start the study"; "run the actual simulation and record results" | ~5 | skill / attitude |
| 4 | Tests not run | "run the tests as well"; "Tests haven't run yet" | ~4 | pipeline (no auto-eval) |
| 5 | Steady-state not respected | "initial gens not steady state — focus later gens / use generation average / gradual filling" | ~5 | pipeline default |
| 6 | Un-provided mechanisms used to force a result | "Do not use outside references… ChIP-seq… Do not use"; "Do not use mechanisms other than those explicitly provided" | 3 (sharpest) | attitude |
| 7 | Run-config provenance missing | "indicate exactly which parameters were used — v, kd's, hydrolysis rate" | 2 | pipeline |
| 8 | One-at-a-time V tuning | repeated "try 1.2e"; "still overshooting"; "try other v + multiseed" | ~6 | skill (sweep upfront) |

---

## The meta-pattern

Across these 49 annotations, the expert was effectively doing the **agent's QA**:
run it, run the tests, remove the stale charts, add the units, focus on
steady-state, show the parameters. **Almost none of this is science.** It is the
completeness-and-hygiene checklist the agent should have closed before handing
the work back.

The fix is structural, not exhortative: the **pipeline and skills should enforce
hygiene, completeness, and conventions automatically**, so the expert only ever
sees a clean, complete, convention-compliant artifact and can spend their
attention on the biology.

---

## Computational-pipeline improvements

Six pipeline changes, each tied to the pattern it eliminates and where it lives.

1. **Run-tagged chart store + auto-prune of superseded-run charts** — kills #1.
   Lives in pbg-superpowers / vivarium-dashboard chart discovery: tag every
   chart with its originating run id and drop charts whose run has been
   superseded by the canonical run.

2. **Auto-evaluate behavior tests on canonical-run completion → `runs[].outcomes`**
   — kills #4. Lives in a spine post-run hook: when the canonical run finishes,
   evaluate the study's behavior tests and write the outcomes back, so "tests
   haven't run yet" can't happen.

3. **Steady-state-by-default test windows + generation-average** — kills #5. The
   `gen_steady_state` window already exists; make it the **default** evaluation
   window (later generations, generation-averaged) rather than opt-in.

4. **A house plotting-style module** — kills #2. Time axis in minutes,
   lineage-boundary demarcation lines, axis labels carrying units, no stray
   gridlines, steady-state-focused occupancy. Lives in `scripts/pbg_plot_style.py`
   (being built now) and should graduate into pbg-superpowers as the shared
   house style.

5. **Auto-captured per-run parameter provenance, surfaced in the report** —
   kills #7. The `enforced_params` machinery exists but isn't auto-populated or
   displayed; capture v, kd's, hydrolysis rate, etc. per run and render them.

6. **Calibration-sweep helper** — kills #8. Runs a parameter grid × multiseed in
   one pass and reports the in-band choice, instead of the one-at-a-time
   "try 1.2e / still overshooting" loop.

---

## Agent skill / attitude improvements

Four skill/attitude changes. **These are the higher-leverage half** — they
address the behaviors that generated the most pointed feedback.

1. **Bias to execute** — kills #3. Once a plan is approved: run the sim → record
   outcomes → run the tests → report, all before handing back. Approval is the
   signal to act, not to re-confirm.

2. **Provided-mechanisms-only honesty guardrail** — kills #6, the sharpest
   pattern. Never invent mechanisms, parameters, or outside literature to force a
   target number; report honest open questions instead. The `cap=32` /
   ChIP-seq / sink episode is the cautionary tale here.

3. **Freshness discipline on re-run** — kills #1 / #9. Replace or delete
   superseded charts and findings, but **preserve valuable rich views** — don't
   regress to the boring minimum while pruning.

4. **Self-serve the standard asks** — kills #2 / #5 / #7. Units on axes, time in
   minutes, lineage demarcation lines, steady-state framing, and run-config
   provenance should all be produced by default, never waited on.

---

## Roadmap / status

This session already shipped two of these as precedent:

- the **canonical DnaA-observable single-source module**
  (`docs/conventions/dnaa-observable-definitions.md`), and
- the **gate-derived dashboard badges**.

In progress alongside this doc:

- the **house plotting style** (pipeline #4), and
- the **pbg skill guardrails** (skills — all four).
