"""colonies-05 driver — viva-munk MOTHER MACHINE with simple agents.

Runs the canonical mother machine (narrow dead-end channels + a flow channel
that washes out cells crossing the top) populated by viva-munk's grow/divide
simple agents, then renders the running animation and the phenotype
distributions (size at division, birth→division time, added size before
division).

Simulations run now; biological conclusions are held for experimental
mother-machine data (a planned comparison study).

    .venv/bin/python .../sims/run.py --steps 500 --channels 8
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(*, n_steps: int = 500, out_dir=None, n_channels: int = 8):
    from v2ecoli.colony_bench.devices import run_device_study

    study_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(out_dir) if out_dir else study_dir
    return run_device_study(
        "mother_machine", out_dir, n_steps=n_steps,
        config={"n_channels": n_channels}, label="mother machine",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    r = main(n_steps=a.steps, out_dir=a.out, n_channels=a.channels)
    ph = r["phenotypes"]
    print(f"mother machine: n_final={r['n_final']} divisions={ph['n_division_events']}")
