"""colonies-06 driver — viva-munk DAUGHTER MACHINE with simple agents.

Runs the canonical daughter machine (a single cell in a chamber with an
absorbing right wall; daughters get pushed out as the lineage grows leftward)
populated by viva-munk's grow/divide simple agents, then renders the running
animation and the phenotype distributions (size at division, birth→division
time, added size before division).

Simulations run now; biological conclusions are held for experimental data
(a planned comparison study).

    .venv/bin/python .../sims/run.py --steps 800
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main(*, n_steps: int = 800, out_dir=None, env_size: float = 30.0):
    from v2ecoli.colony_bench.devices import run_device_study

    study_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(out_dir) if out_dir else study_dir
    return run_device_study(
        "daughter_machine", out_dir, n_steps=n_steps,
        config={"env_size": env_size}, label="daughter machine",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--env-size", type=float, default=30.0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    r = main(n_steps=a.steps, out_dir=a.out, env_size=a.env_size)
    ph = r["phenotypes"]
    print(f"daughter machine: n_final={r['n_final']} divisions={ph['n_division_events']}")
