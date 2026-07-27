# sm-04-through-division — Extension Plan D summary

**Branch:** `harden/surrogate-through-division` (off `origin/main`)
**Goal:** extend the `surrogate-modeling` investigation to roll its emulators
**to and through a real cell division**, and test whether the emulators that win
within a generation still hold across the cell-cycle discontinuity — the single
load-bearing boundary the investigation named as its top next direction but never
ran (sm-00 halts every trajectory at division).

## Verdict — informative NEGATIVE (not tuned away)

The emulators do **not** survive division. Following one daughter lineage across
a real division event, every emulator — trivial and neural — emulates coarse
growth within a generation but **none represents the cell_mass halving** at the
boundary.

6-fold leave-one-lineage-out CV, cell_mass, teacher-forced one-step nRMSE
(trained on within-generation transitions only, then applied across the boundary):

| model | within-gen 1-step | division-tick 1-step | ratio | wg→full rollout |
|---|---|---|---|---|
| persistence | 0.0105 | 3.58 | 342× | 1.79 → 1.75 |
| mean-delta | 0.0013 | 3.59 | 2738× | 0.16 → 0.81 |
| **linear** (the deliverable) | 0.0006 | 3.59 | **6204×** | 0.10 → 0.81 |
| neural (one-step) | 0.0003 | 3.59 | 13100× | 0.18 → 0.85 |
| neural (rollout-loss) | 0.0038 | 3.43 | 892× | 0.39 → 0.81 |

At division the true cell_mass halves cleanly (median daughter/mother **0.498**,
number_of_oric **4 → 2**) at tick 2526, consistent across all 6 seeds. Fed the
true pre-division mother state, every emulator predicts smooth continued growth
(linear rollout 2337 → 2341 fg while truth 2433 → 1221 fg) — the one-step error
at the boundary is ~3.5 nRMSE, thousands of times each learned model's
within-generation error. The best within-generation models fail *worst* at the
boundary (largest ratio), precisely because they are most confident about
smoothness. Autoregressive full-lineage rollout error jumps ~8× crossing division.

**Verified separately** that ADDING the halving examples to training does not
help the boundary (3.48 vs 3.59) and corrupts the within-generation fit
(0.0006 → 0.0128): the discontinuity is not learnable from the observable
(mass + chromosome) view — the same shape of result the investigation already
found for the high-dimensional panel. This **extends, not overturns**, the
scoping conclusion: the cheap linear emulator is valid only within a generation
and is simply undefined across division.

## What was built

- **`workspace/studies/sm-04-through-division/study.yaml`** — schema-v4 study
  with `investigation: surrogate-modeling` back-ref, primary behavior test
  `emulators-hold-across-division` (status **failed**, 6204× ≫ 1), honest
  fail verdict, `conclusion_verdicts`, and `decisions_needed` /
  `followup_proposals` seeding the latent encode–decode direction
  (`sm-05-latent-encode-decode`). Registered in `investigation.yaml`
  (members, at_a_glance, acceptance_criteria).
- **Code** (`workspace/investigations/surrogate-modeling/code/`):
  - `sample_through_division.py` — steps a lineage **past** division by reusing
    v2ecoli's own multigeneration machinery (`workflow/lineage.select_carry_daughter`
    + `apply_carry_state`); a single `Composite` cannot be `run()` through the
    Division step's `_remove`/`_add`, so the surviving daughter's state is carried
    into a rebuilt composite. Tags each transition within-gen vs across-division.
  - `eval_through_division.py` — 6-fold CV reusing `baselines.py` +
    `rollout_train.py`; teacher-forced one-step (within-gen vs boundary) +
    autoregressive rollout; `--stride` temporal subsample (dt=8 s) to keep the
    batch-size-1 rollout-loss NN tractable while preserving the halving as one
    clean transition.
  - `make_through_division_figure.py` — rollout-vs-halving trace + error bars.
- **Data:** `run_through_division/{transitions.npz, layout.json, meta.npz,
  metrics_through_division.json}` — 15,876 dt=1 transitions × 8 observables,
  6 lineages each crossing one division.
- **Figure:** `reports/figures/sm-04-through-division/through_division.html`.

## Environment note (important for reproduction)

This worktree has no `.venv`; it uses `~/code/v2ecoli/.venv/bin/python`. That
shared venv did **not** ship three things this branch's code needs:
`pbg-torch`, `plotly`, and a newer `bigraph_schema` that has `.contract`. They
are vendored into **`.deps/`** (gitignored) and wired in via
`PYTHONPATH=<worktree>:<worktree>/.deps`. `out/cache` is symlinked to the
canonical ParCa cache. Every command in this work used:

```bash
PYTHONPATH=~/code/v2e-hsurrogate:~/code/v2e-hsurrogate/.deps \
  ~/code/v2ecoli/.venv/bin/python <script>
```

To recreate `.deps`: `uv pip install --python ~/code/v2ecoli/.venv/bin/python
--target .deps "git+https://github.com/vivarium-collective/pbg-torch.git@cfee4f06" plotly`
plus a `bigraph_schema` (≥1.4.3, with `contract.py`).

## Commits (this branch)

1. sm-04 scaffold (study.yaml + code + investigation registration)
2. sm-04 data (6-lineage through-division transition dataset)
3. sm-04 results (metrics + figure + findings; README folded in)
