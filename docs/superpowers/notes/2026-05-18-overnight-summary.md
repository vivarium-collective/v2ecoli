# Overnight session summary — 2026-05-18

For the morning read. This is what landed while you were asleep.

## TL;DR — Biology decision made

**dnaa-02f resolves with variant E + clamp as the canonical mechanism.**
RIDA-v0 (fork-active 100× hydrolysis multiplier) plus the ATP-fraction
clamp produces a stable, in-band atp_fraction WITHOUT mutating the
equilibrium reaction set and WITHOUT inventing any species.

Multi-seed (0/1/2), 1800s each:

  Variant                                      atp_fraction   CV
  ──────────────────────────────────────────────────────────────
  B  (current dnaa-02 monkey-patch + clamp)    0.216-0.270    ~16%
  B' (recalibrated equilibrium reverse rate)   FAIL (binary)  —
  E  (RIDA-v0 + clamp @ literature 100×)       0.496          < 1%

Critical realisation: **dnaa-02's current PASS came from the clamp, not
the biology.** Intrinsic hydrolysis (Sekimizu 0.046/min) is too slow to
produce ANY meaningful DnaA-ADP accumulation against the equilibrium.
The clamp did all the work; the monkey-patch was just to make the
clamp's transfers stick. Variant E shows that adding literature-rate
RIDA gives the clamp enough background ATP→ADP flux to stick against
an INTACT equilibrium — that's real biology.

## What's now in the worktree

### Runs.db populated for 4 studies via pbg_runner

  Study                        Runs in db   Latest sim
  ─────────────────────────────────────────────────────────
  dnaa-01-expression-dynamics  12           pbg-runner-smoke
  dnaa-02-atp-hydrolysis        ~33          variant-b-control × 3 seeds
  dnaa-02f-equilibrium-cleanup  ~42          variant-e-clamp-lit × 3 seeds
  dnaa-03-box-binding           ~15          dnaa03-baseline-seed{0,1,2}-dnaA500

All four show on the Runs tab (it now reads runs.db, not study.yaml).
All four have rendered Plotly viz HTML at studies/<slug>/viz/ that the
Visualizations tab auto-discovers.

### dnaa-02f study yaml updated

  status:           planning → complete-variant-e-wins
  phase:            Plan → Evaluate
  gate_status:      open → passed
  conclusion_verdicts:
    regression_compatibility:  PASS
    biological_validation:     PASS
    explanatory_gain:          POSITIVE
  follow_up_studies:
    retire-dnaa-02-monkey-patch        planned
    variant-e-downstream-isolation     planned
    hda-clamp-loading-dnaa05           planned-dnaa-05

### Dashboard improvements committed on PR #40

  - Runs tab now reads studies/<slug>/runs.db (was: study.yaml.runs only)
  - Visualizations tab auto-discovers studies/<slug>/viz/*.html iframes
  - /studies/<slug>/viz/<file>.html URL pattern now resolves to static
    (was: intercepted by study-detail page route, 404'd)
  - Two latent bugs in lib/investigations.py + lib/simulations_index.py
    fixed (nested observable resolution + mixed-type ts sort)

### pbg_superpowers/runner.py committed on PR #27

The `pbg_runner` context manager that closes the runner ↔ runs.db gap.
Every CLI-launched run via pbg_runner now auto-populates runs_meta +
history + simulations in the per-study db. Goodbye one-shot backfill
script (still in tree as scripts/backfill_runs_db.py for legacy data).

### Friction log appended (9 new entries)

`docs/superpowers/notes/2026-05-17-dnaa-investigation-friction-log.md`

  Session-3 / #20  Bash cwd resets confound multi-step file ops
  Session-3 / #21  Two Python interpreters in same workspace
  Session-3 / #22  Background-task harness silently ate output
  Session-3 / #23  Cherry-picking commit across upstream branches
  Session-3 / #24  out/<study>/ + runs.db + viz/ not gitignored
  Session-3 / Pattern  Decision-by-empirical-sweep before commit
  Session-3 / Biology  Variant E wins dnaa-02f
  Session-3 / dnaa-03 status  3 of 4 tests pass

## PRs (all unmerged, per your standing instruction)

  vivarium-dashboard #40   feat/viz-pipeline-fixes      3 commits, ready to review
  pbg-superpowers    #27   feat/lint-viz-addresses      2 commits, ready to review
  v2ecoli            #59   feat/dnaa-mock-investigation 5+ commits, integration branch (NOT a merge target)

PR #59 stays open as the investigation integration branch per your
direction. Companion PRs land on main first; this branch absorbs them
via uv.lock bump.

## Open items / next sessions

In rough priority:

1. **Retire dnaa-02 monkey-patch.** Add `dnaa_02_canonical` recipe
   using variant-E mechanism; migrate dnaa-03's recipe to chain off
   it. (Risk: may shift dnaa-03's test results. Worth a manual
   review pass.) — attempted as a NEW recipe addition this session,
   left to manual migration.

2. **dnaa-03 multi-seed reproducibility.** Seeds 0/1/2 produce
   different test verdicts (chrom-monotonic ✓ at seed 0 but ✗ at
   seeds 1/2). High stochastic variance; either longer sim windows
   or refined kd parameters.

3. **dnaa-04 implementation.** Still planning-phase; no composite
   recipe, no Steps, no runner authored. Needs biology decisions on
   SeqA-v0 fixed-timer refractory + initiation trigger driven by
   oriC occupancy.

4. **Free-DnaA-rises-after-titration test (dnaa-03).** Documented as
   needing multi-generation simulation; current 1800s window is
   sub-doubling-time.

5. **Polish:** add `out/`, `**/runs.db*`, `**/viz/`, `.pbg/server/`,
   `reports/` to .gitignore so the worktree stops accumulating
   untracked artefacts.

## What I did NOT do (intentionally)

- Did not merge any PR — your standing rule.
- Did not delete branches.
- Did not migrate dnaa-03 to use a new recipe — that's a chain-affecting
  change worth manual review.
- Did not author dnaa-04 from scratch — biology decisions needed.
- Did not touch dnaa-01 since it was already converted in session 2.
