# colonies-04 — Device harness + simple-agent phenotype baseline

Shared infra for the phenotype-quantification pivot. Builds `cell_factory`,
the three geometry builders, and the `phenotype_extractor`, then runs the
mother machine, daughter machine, and free colony with **simple agents** to
validate the pipeline cheaply.

Run: `.venv/bin/python sims/run.py free_colony simple --ticks 60`

Tiers: `simple` (this study), `wcm` (colonies-06), `surrogate` (colonies-05).
