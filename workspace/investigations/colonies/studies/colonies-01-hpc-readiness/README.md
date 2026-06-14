# colonies-01-hpc-readiness

Purpose: certify that the pure whole-cell colony composite is ready to scale on
HPC. Two phases:

1. **Build** — fix daughter-process hydration in the colony composite so that
   division's structural `_add` actually instantiates daughter EcoliWCMs through
   the outer engine (no manual workaround). Acceptance: `N=2 → 4 live, ticking
   daughters`.
2. **Decide** — sweep `N ∈ {1, 2, 4, 8}` on this machine with the perf harness,
   record wall time / peak RSS / per-tick latency / per-cell update cost /
   pymunk step cost, and extrapolate a cells-per-HPC-node budget.

See [`study.yaml`](study.yaml) for the full spec (purpose, pipeline gate,
simulation set, behavior tests, requirements).

## Layout

```
studies/colonies-01-hpc-readiness/
├── study.yaml          # study spec (schema_version: 3)
├── README.md           # this file
├── sims/               # per-run logs + the perf harness (run.py)
└── runs.db             # SQLite, written by the harness (gitignored)
```

## Running

After the Build phase fix lands, the harness will be at `sims/run.py`:

```
python studies/colonies-01-hpc-readiness/sims/run.py --sim-name build-smoke-n2
python studies/colonies-01-hpc-readiness/sims/run.py --sim-name nsweep-n1
python studies/colonies-01-hpc-readiness/sims/run.py --sim-name nsweep-n2
python studies/colonies-01-hpc-readiness/sims/run.py --sim-name nsweep-n4
python studies/colonies-01-hpc-readiness/sims/run.py --sim-name nsweep-n8   # best-effort
```

Each run appends one row to `runs` and N rows to `ticks` in `runs.db`. The
Decide-phase report reads both tables.
