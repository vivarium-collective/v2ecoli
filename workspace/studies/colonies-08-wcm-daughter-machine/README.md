# colonies-08 — Whole-cell baseline in a daughter machine (device run)

Runs the **real v2ecoli whole-cell baseline** (full 55-process `EcoliWCM`) as a
single cell inside the **daughter-machine** device geometry, growing and
dividing naturally, then renders:

- `charts/colony.gif` — the whole-cell baseline running inside the device
  (lineage-coloured capsules; shaded band = wash-out boundary)
- `charts/size_at_division.png`, `interdivision_time.png`, `added_size.png` —
  the same phenotype panel as the cheap-agent studies (colonies-05/06)

Run: `.venv/bin/python sims/run.py --max-ticks 5000 --stop-cells 4`

**Compute/RSS-bounded** — a whole-cell division is ~one cell cycle (~2300+
ticks) and colonies leak per-cell RAM (OOM ~gen 3), so this is capped at a few
generations. **Phenotype distributions are preliminary (small n)**; conclusions
held for experimental data, full distributions need HPC (colonies-09). Same
pipeline (`colony_bench.phenotypes` + `colony_bench.viz`) as the simple-agent
device studies, so the tiers are directly comparable.
