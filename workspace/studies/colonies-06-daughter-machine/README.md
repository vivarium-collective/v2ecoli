# colonies-06 — Daughter machine (viva-munk), device run + phenotype distributions

Runs the canonical viva-munk **daughter machine** (chamber + absorbing right
wall) with simple grow/divide agents, then renders:

- `charts/colony.gif` — the device running (cells as capsules, lineage-coloured;
  shaded band = wash-out boundary)
- `charts/size_at_division.png` — length at division distribution
- `charts/interdivision_time.png` — time between birth and division
- `charts/added_size.png` — added size before division (adder plot)

Run: `.venv/bin/python sims/run.py --steps 800`

Simulations run now; **conclusions are held for experimental data** (planned
comparison study). Phenotypes via `v2ecoli.colony_bench`.
