"""colonies-08 driver — the REAL v2ecoli whole-cell BASELINE in a daughter machine.

Runs the full 55-process EcoliWCM as a single cell inside the daughter-machine
device geometry (chamber + absorbing wall), letting it grow and divide
naturally, then renders the running animation and the phenotype figures with the
SAME pipeline the cheap tiers use.

This is compute/RSS-bounded: a whole-cell division is ~one cell cycle (~2300+
ticks), and a multi-cell colony leaks per-cell RAM (OOM ~gen 3), so the run is
capped at a few generations. Phenotype distributions are therefore PRELIMINARY
(small n) — the value is the genuine whole-cell baseline running inside the
device. Full distributions need HPC (see colonies-09).

    .venv/bin/python .../sims/run.py --max-ticks 5000 --stop-cells 4
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(*, out_dir=None, max_ticks: int = 5000, stop_cells: int = 4, env_size: float = 30.0):
    from v2ecoli.colony_bench.devices import run_wcm_device

    study_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(out_dir) if out_dir else study_dir
    return run_wcm_device(
        "daughter_machine", out_dir, max_ticks=max_ticks, stop_cells=stop_cells,
        builder_kwargs={"env_size": env_size}, label="whole-cell daughter machine",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-ticks", type=int, default=5000)
    p.add_argument("--stop-cells", type=int, default=4)
    p.add_argument("--env-size", type=float, default=30.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    r = main(out_dir=a.out, max_ticks=a.max_ticks, stop_cells=a.stop_cells, env_size=a.env_size)
    print(f"whole-cell daughter machine: ticks={r['ticks']} n_final={r['n_final']} "
          f"divisions={r['phenotypes']['n_division_events']}")
