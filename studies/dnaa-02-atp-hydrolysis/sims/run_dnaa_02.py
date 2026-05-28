"""dnaa-02 baseline runner — exercises the IntrinsicHydrolysis + Clamp + Listener.

Outputs a JSON file with per-tick snapshots of:
  - bulk[PD03831[c]]      free DnaA monomer (apo)
  - bulk[MONOMER0-160[c]] DnaA-ATP complex
  - bulk[MONOMER0-4565[c]] DnaA-ADP complex
  - listeners.dnaA_cycle.* (the new listener emit)

Usage:
    python studies/dnaa-02-atp-hydrolysis/sims/run_dnaa_02.py \
        <duration_s> <interval_s> <out_json> \
        [--seed N] [--rate 0.046] [--deterministic] \
        [--clamp-low 0.2 --clamp-high 0.5]
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import numpy as np

if os.environ.get("DNAA02_QUIET"):
    _fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_fd, 2)

from v2ecoli import build_composite
from v2ecoli.composites._helpers import flush_parquet, parquet_emitter
from pbg_superpowers.runner import pbg_runner

STUDY_SLUG = "dnaa-02-atp-hydrolysis"
INVESTIGATION_SLUG = "dnaa-replication"


BULKS = {
    "PD03831[c]":      "apo_count",
    "MONOMER0-160[c]": "atp_count_bulk",
    "MONOMER0-4565[c]": "adp_count_bulk",
}


def _build_bulk_idx(bulk_array):
    return {bid: int(np.where(bulk_array["id"] == bid)[0][0])
            for bid in BULKS
            if (bulk_array["id"] == bid).any()}


def _read_listener(node, field, default=None):
    if node is None:
        return default
    if hasattr(node, "get"):
        v = node.get(field)
        return v if v is not None else default
    if hasattr(node, "dtype") and getattr(node.dtype, "names", None):
        if field in node.dtype.names:
            return node[field]
    return default


def snap(t, cell, bulk_idx):
    bulk = cell["bulk"]
    out = {"time": t}
    for bid, key in BULKS.items():
        idx = bulk_idx.get(bid)
        out[key] = int(bulk[idx]["count"]) if idx is not None else None

    listeners = cell.get("listeners", {})
    dnaa = listeners.get("dnaA_cycle", {})
    # Listener fields
    for f in ["apo_count", "atp_count", "adp_count", "total",
              "atp_fraction", "adp_fraction", "apo_fraction",
              "intrinsic_hydrolysis_events", "clamp_transfer", "clamp_direction"]:
        out[f"listener_{f}"] = _read_listener(dnaa, f)
    # Convert clamp_direction to string if it's a numpy/bytes value
    cd = out.get("listener_clamp_direction")
    if isinstance(cd, (bytes, np.bytes_)):
        out["listener_clamp_direction"] = cd.decode()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("duration", type=int)
    p.add_argument("interval", type=int)
    p.add_argument("out_path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rate", type=float, default=0.046,
                   help="Intrinsic hydrolysis rate per minute (default 0.046)")
    p.add_argument("--deterministic", action="store_true",
                   help="Round expected transfers instead of Poisson draws")
    p.add_argument("--clamp-low", type=float, default=None)
    p.add_argument("--clamp-high", type=float, default=None)
    p.add_argument("--cache-dir", default="out/cache")
    p.add_argument("--sim-name", default=None,
                   help="Label for the dashboard's Simulations tab "
                        "(defaults to dnaa02-seed<seed>-r<rate>).")
    args = p.parse_args()
    sim_name = args.sim_name or f"dnaa02-seed{args.seed}-r{args.rate}"
    params = {
        "seed": args.seed, "duration_s": args.duration,
        "interval_s": args.interval, "rate_per_min": args.rate,
        "deterministic": args.deterministic,
        "clamp_low": args.clamp_low, "clamp_high": args.clamp_high,
    }

    with pbg_runner(study=STUDY_SLUG, name=sim_name, params=params) as run:
        with parquet_emitter(
            out_dir=f"studies/{STUDY_SLUG}/parquet-runs",
            experiment_id=run.run_id,
            study_slug=STUDY_SLUG,
            investigation_slug=INVESTIGATION_SLUG,
        ):
            t0 = time.time()
            composite = build_composite(
                "dnaa_02_with_intrinsic_hydrolysis",
                cache_dir=args.cache_dir,
                seed=args.seed,
                hydrolysis_rate_per_min=args.rate,
                hydrolysis_deterministic=args.deterministic,
                atp_fraction_clamp_low=args.clamp_low,
                atp_fraction_clamp_high=args.clamp_high,
            )
            load_time = time.time() - t0

            cell = composite.state["agents"]["0"]
            bulk_idx = _build_bulk_idx(cell["bulk"])

            snapshots = [snap(0, cell, bulk_idx)]
            t_run = time.time()
            total = 0
            divided = False
            while total < args.duration:
                chunk = min(args.interval, args.duration - total)
                try:
                    composite.run(chunk)
                except Exception:
                    divided = True
                    break
                total += chunk
                cell = composite.state.get("agents", {}).get("0")
                if cell is None:
                    divided = True
                    break
                snapshots.append(snap(total, cell, bulk_idx))
                run.heartbeat(len(snapshots))
            wall_time = time.time() - t_run
            flush_parquet(composite, success=True)
            run.n_steps = len(snapshots)

    # Second-half medians for the key fractions
    nsnap = len(snapshots)
    second_half = snapshots[nsnap // 2:]
    def med(field):
        vals = [s.get(field) for s in second_half if s.get(field) is not None]
        return float(np.median(vals)) if vals else None
    summary = {
        "second_half_median": {
            "atp_count_bulk":      med("atp_count_bulk"),
            "adp_count_bulk":      med("adp_count_bulk"),
            "apo_count_bulk":      med("apo_count"),
            "listener_atp_fraction": med("listener_atp_fraction"),
            "listener_adp_fraction": med("listener_adp_fraction"),
            "listener_apo_fraction": med("listener_apo_fraction"),
            "listener_total":      med("listener_total"),
        },
    }

    result = {
        "composite":  "dnaa_02_with_intrinsic_hydrolysis",
        "seed":       args.seed,
        "rate_per_min": args.rate,
        "deterministic": args.deterministic,
        "clamp":      [args.clamp_low, args.clamp_high] if args.clamp_low is not None else None,
        "load_time":  load_time,
        "wall_time":  wall_time,
        "sim_time":   total,
        "interval":   args.interval,
        "divided":    divided,
        "snapshots":  snapshots,
        "summary":    summary,
    }
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote {args.out_path}")
    print(f"  sim_time={total}s  wall={wall_time:.1f}s  divided={divided}")
    print("  Second-half medians:")
    for k, v in summary["second_half_median"].items():
        if v is None:
            print(f"    {k}: n/a")
        else:
            print(f"    {k}: {v:.3f}")


if __name__ == "__main__":
    main()
