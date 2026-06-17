# Mini run brief — colonies-02-parallel-multigen-perf

Autonomous task brief for a headless mini agent (`mct v2ecoli <this file>`).
Branch: `feat/colonies-parallel-multigen`. Run from the v2ecoli worktree on
that branch; use `.venv/bin/python` (bare `python` lacks `unum`). Verify
progress via git commits + `runs/*.parquet`, NOT the buffered `-p` log.

## 0. Setup
- `git fetch && git checkout feat/colonies-parallel-multigen && git pull`
- Ensure `out/cache` exists (symlink to a full ParCa cache if needed —
  `cache_version.json`, `sim_data_cache.dill`, `initial_state.json`).
- `pip install -e` process-bigraph with the `[ray]` extra so `ray:` works.

## 1. Build phase — confirm the jitter/mass fix (FAST, do first)
Run the physics diagnostic and commit the evidence:
```
.venv/bin/python studies/colonies-02-parallel-multigen-perf/sims/diagnose_physics.py --ticks 12
```
Read `diagnostics/physics_diag.csv`. Expected: with `(jitter=0.5, init_mass=None)`
body mass starts ~0.04 and cells move a lot; with `(1e-4, 200)` movement
collapses and mass is coherent (fg). If confirmed, the runner defaults
(jitter=1e-4, init_mass=200) are correct — `git add diagnostics/ && git commit`.
If the numbers disagree, adjust the perturbation defaults in study.yaml and note why.

## 2. Sequential drift run (resolves F-06)
```
.venv/bin/python studies/colonies-02-parallel-multigen-perf/sims/run.py --sim-name seq-1cell-4div
```
One cell, NATURAL division (no --force-divide), 180 min sim → doubles 1→2→4→8.
Commit `runs/runs.parquet` + `runs/ticks.parquet` when done.

## 3. Ray run (does process parallelism lift the GIL ceiling?)
```
.venv/bin/python studies/colonies-02-parallel-multigen-perf/sims/run.py --sim-name ray-1cell-4div
```
Same colony under `ray:EcoliWCM` + `parallel_processes=True`. NOTE: under Ray
the per-cell EcoliWCM timing column is ~0 (update runs in the actor process);
the headline metric is `wall_ms` and RSS. Commit the parquet rows.

## 4. (Optional) static Ray N-sweep
```
.venv/bin/python studies/colonies-02-parallel-multigen-perf/sims/run.py --sim-name ray-nsweep-static
```

## 5. Analysis + write-up
- Derive per-cell wall + RSS by generation (bin ticks by `live_cell_count`
  ∈ {1,2,4,8}). Check the 4 behavior_tests in study.yaml:
  natural-division-2-generations, per-cell-wall-drift-within-20pct,
  per-cell-rss-drift-bounded, ray-lifts-gil-ceiling.
- Fill `findings:` + flip the status axes (simulation_status: ran, etc.) in
  study.yaml. Commit everything. Do NOT merge the PR.

## Smoke test first (recommended)
Before the 180-min runs, sanity-check wiring with a 2-min run each:
`--duration-min 2` on both seq and ray sim-names. A ray smoke that hydrates
≥1 actor and advances confirms the transport plumbing end-to-end.
