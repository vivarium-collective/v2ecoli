# colonies-05 — Mother machine (viva-munk), device run + phenotype distributions

Runs the canonical viva-munk **mother machine** (dead-end channels + flow) with
simple grow/divide agents, then renders:

- `charts/colony.gif` — the device running (cells as capsules, lineage-coloured;
  shaded band = wash-out boundary)
- `charts/size_at_division.png` — length at division distribution
- `charts/interdivision_time.png` — time between birth and division
- `charts/added_size.png` — added size before division (adder plot)

Run: `.venv/bin/python sims/run.py --steps 500 --channels 8`

Simulations run now; **conclusions are held for experimental mother-machine
data** (planned comparison study). Phenotypes via `v2ecoli.colony_bench`.
