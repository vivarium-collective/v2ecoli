"""colonies-09 driver — the REAL v2ecoli whole-cell BASELINE in a mother machine.

Runs the full 55-process EcoliWCM as one whole cell per channel inside the
mother-machine device geometry (narrow dead-end channels + a flow channel that
washes out cells crossing the top), letting them grow and divide naturally, then
renders the running animation and the phenotype figures with the SAME pipeline
the cheap tiers and the daughter-machine whole-cell run use.

Heavier than colonies-08: N whole cells run simultaneously (one per channel), so
keep ``n_channels`` small (2) and the tick budget bounded — whole-cell colonies
leak per-cell RAM (OOM ~gen 3). Phenotype distributions are PRELIMINARY (small
n); the value is the whole-cell baseline running inside the confined-channel
device. Full distributions need HPC (colonies-10).

    .venv/bin/python .../sims/run.py --channels 2 --max-ticks 4000 --stop-cells 6
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(*, out_dir=None, n_channels: int = 2, max_ticks: int = 4000, stop_cells: int = 6):
    from v2ecoli.colony_bench.devices import run_wcm_device

    study_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(out_dir) if out_dir else study_dir
    return run_wcm_device(
        "mother_machine", out_dir, max_ticks=max_ticks, stop_cells=stop_cells,
        builder_kwargs={"n_channels": n_channels}, label="whole-cell mother machine",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--channels", type=int, default=2)
    p.add_argument("--max-ticks", type=int, default=4000)
    p.add_argument("--stop-cells", type=int, default=6)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    r = main(out_dir=a.out, n_channels=a.channels, max_ticks=a.max_ticks, stop_cells=a.stop_cells)
    print(f"whole-cell mother machine: ticks={r['ticks']} n_final={r['n_final']} "
          f"divisions={r['phenotypes']['n_division_events']}")
