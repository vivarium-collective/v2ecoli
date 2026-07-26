"""colonies-04 driver — run a geometry+tier, extract the phenotype panel.

Usage:
    .venv/bin/python .../sims/run.py free_colony simple --ticks 60
"""
from __future__ import annotations
import argparse, json, pathlib, sys


def main(*, geometry, tier, n_ticks, out_dir, seed=0, builder_kwargs=None):
    from v2ecoli.colony_bench.harness import run_bench
    out = run_bench(geometry, tier, n_ticks=n_ticks, seed=seed,
                    builder_kwargs=builder_kwargs)
    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phenotypes.json").write_text(json.dumps(out["phenotypes"], indent=2))
    (out_dir / "summary.json").write_text(json.dumps({
        "geometry": geometry, "tier": tier, "n_ticks": n_ticks,
        "n_final": out["n_final"],
        "n_division_events": out["phenotypes"]["n_division_events"],
    }, indent=2))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("geometry"); p.add_argument("tier")
    p.add_argument("--ticks", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    here = pathlib.Path(__file__).resolve().parent.parent
    out_dir = a.out or (here / "runs" / f"{a.geometry}__{a.tier}__seed{a.seed}")
    r = main(geometry=a.geometry, tier=a.tier, n_ticks=a.ticks,
             out_dir=out_dir, seed=a.seed)
    print(f"n_final={r['n_final']} divisions={r['phenotypes']['n_division_events']}")
    sys.exit(0)
