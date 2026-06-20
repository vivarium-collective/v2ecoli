# Mini run brief — colonies-03-wcm-rss-leak

Autonomous brief (`mct v2ecoli <this file>`). Branch `feat/colonies-parallel-multigen`.
Use an isolated worktree + `.venv/bin/python` (as the colonies-02 run did);
do NOT disturb the main clone's branch. Verify via git commits, not the log.

## Goal
Localize and fix the inner-EcoliWCM RSS leak (~7.7 MB/sim-s, emitter- and
transport-independent) that colonies-02 found OOMs the colony before HPC
deployment. Bounded per-cell RSS is the gate for colonies-04.

## 1. Localize (build phase)
```
.venv/bin/python studies/colonies-03-wcm-rss-leak/sims/profile_leak.py --ticks 2500 --snap-every 200
```
Reads a single EcoliWCM (no colony/pymunk). Commit `profiling/leak_profile.csv`
+ `profiling/top_sites.txt`. Identify the site(s) whose cumulative tracemalloc
size tracks the RSS climb. Known unbounded suspect: `bridge.py:243`
`self.chromosome_history.append(...)` (every tick, never trimmed) — confirm or
refute it is the dominant term; if not, name what is (inner-composite per-tick
retention, unique-molecule arrays, etc.). Need >=~80% attribution → clears
`leak-localized`.

## 2. Fix (build phase)
Apply the smallest fix the profile justifies (e.g. cap/disable
`chromosome_history` when unused; release inner-composite per-tick state). If
it changes recorded outputs, gate it behind a flag. Commit the fix.

## 3. Verify
```
# bounded RSS on a single cell after the fix:
.venv/bin/python studies/colonies-03-wcm-rss-leak/sims/profile_leak.py --ticks 2500 --snap-every 200
# full colony, must complete >=3 generations without OOM:
.venv/bin/python studies/colonies-02-parallel-multigen-perf/sims/run.py --sim-name seq-1cell-4div
```
Check the 3 behavior_tests: leak-localized, per-cell-rss-bounded-after-fix,
biology-unchanged-by-fix (first division STILL tick 2338, masses within RNG of
the colonies-02 run). Then fill `findings:`, flip status axes, set
`gate_status: passed` if RSS is bounded. Commit. Do NOT push or merge.
```
```
