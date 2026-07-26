# colonies-09 — Whole-cell baseline in a mother machine (device run)

Runs the **real v2ecoli whole-cell baseline** (full 55-process `EcoliWCM`) as
one whole cell per channel inside the **mother-machine** device geometry
(narrow dead-end channels + flow), growing and dividing naturally, then renders:

- `charts/colony.gif` — whole cells running in the channels (lineage-coloured;
  shaded band = wash-out boundary)
- `charts/size_at_division.png`, `interdivision_time.png`, `added_size.png` —
  the same phenotype panel as colonies-05/06/08

Run: `.venv/bin/python sims/run.py --channels 2 --max-ticks 4000 --stop-cells 6`

**Heavier + compute/RSS-bounded** — N whole cells run at once (one per channel),
so channels are kept small (2) and the run capped at a few generations.
**Phenotype distributions are preliminary (small n)**; conclusions held for
experimental data, full distributions need HPC (colonies-10). Same pipeline
(`colony_bench.phenotypes` + `colony_bench.viz`) as the other device studies.
