"""dnaa-02f variant E (RIDA-v0) runner.

Builds the ``dnaa_02f_with_rida_v0`` composite — baseline + intrinsic
hydrolysis + RIDA-v0 (fork-active 100× multiplier) + ATP-fraction clamp
+ cycle listener. The equilibrium reaction set is **intact** (no
monkey-patch); this is the variant-E test of whether RIDA outpaces the
equilibrium's MONOMER0-4565_RXN reverse flux on its own.

Outputs:
  - JSON: per-tick DnaA bulk counts + listener emit (apo/atp/adp counts,
    fractions, intrinsic_events, rida_events, rida_active, n_replisomes).
  - Parquet: studies/dnaa-02f-equilibrium-cleanup/parquet-runs/<run_id>/
    (configuration + history + success), via pbg_runner + parquet_emitter.
    The pbg_runner side still writes runs.db.runs_meta for run-registry
    bookkeeping (status, params, heartbeat).

Usage:
    python studies/dnaa-02f-equilibrium-cleanup/sims/run_variant_e.py \\
        <duration_s> <interval_s> <out_json> \\
        [--seed N] [--rida-rate 4.6] [--sim-name <label>]
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

from v2ecoli import build_composite
from v2ecoli.composites._helpers import flush_parquet, parquet_emitter
from pbg_superpowers.runner import pbg_runner

STUDY_SLUG = "dnaa-02f-equilibrium-cleanup"
INVESTIGATION_SLUG = "dnaa-replication"

BULKS = {
    "PD03831[c]":       "apo_count_bulk",
    "MONOMER0-160[c]":  "atp_count_bulk",
    "MONOMER0-4565[c]": "adp_count_bulk",
}


def _build_bulk_idx(bulk_array):
    return {bid: int(np.where(bulk_array["id"] == bid)[0][0])
            for bid in BULKS
            if (bulk_array["id"] == bid).any()}


def snap(t, cell, bulk_idx):
    bulk = cell["bulk"]
    listeners = cell.get("listeners", {})
    cyc = (listeners.get("dnaA_cycle") or {})

    out = {"time": t}
    for bid, key in BULKS.items():
        idx = bulk_idx.get(bid)
        out[key] = int(bulk[idx]["count"]) if idx is not None else None

    out["listener_apo_count"]   = cyc.get("apo_count")
    out["listener_atp_count"]   = cyc.get("atp_count")
    out["listener_adp_count"]   = cyc.get("adp_count")
    out["listener_total"]       = cyc.get("total")
    out["listener_atp_fraction"] = cyc.get("atp_fraction")
    out["listener_adp_fraction"] = cyc.get("adp_fraction")
    out["listener_apo_fraction"] = cyc.get("apo_fraction")
    out["intrinsic_events"]      = cyc.get("intrinsic_hydrolysis_events")
    out["rida_events"]           = cyc.get("rida_events")
    out["rida_active"]           = cyc.get("rida_active")
    out["rida_n_replisomes"]     = cyc.get("rida_n_replisomes")

    # Mass + chromosome state (for biological context).
    mass = listeners.get("mass", {})
    out["cell_mass_fg"] = float(mass.get("cell_mass", 0))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("duration", type=int)
    p.add_argument("interval", type=int)
    p.add_argument("out_path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rida-rate", type=float, default=4.6,
                   help="RIDA-v0 effective rate when ≥1 replisome active "
                        "(default 4.6/min = 100× Sekimizu intrinsic).")
    p.add_argument("--intrinsic-rate", type=float, default=0.046)
    p.add_argument("--clamp-low", type=float, default=None,
                   help="Lower edge of the ATP-fraction clamp band. Pair with "
                        "--clamp-high to enable. Default disabled.")
    p.add_argument("--clamp-high", type=float, default=None)
    p.add_argument("--cache-dir", default="out/cache")
    p.add_argument("--sim-name", default=None,
                   help="Label for the dashboard (default: variant-e-seed<seed>).")
    args = p.parse_args()
    clamp_tag = ""
    if args.clamp_low is not None and args.clamp_high is not None:
        clamp_tag = f"-clamp{args.clamp_low}-{args.clamp_high}"
    sim_name = args.sim_name or (
        f"variant-e-seed{args.seed}-rida{args.rida_rate}{clamp_tag}")
    params = {
        "variant": "E-rida-v0" + ("+clamp" if clamp_tag else ""),
        "seed": args.seed,
        "duration_s": args.duration,
        "interval_s": args.interval,
        "intrinsic_rate_per_min": args.intrinsic_rate,
        "rida_rate_per_min": args.rida_rate,
        "clamp_low": args.clamp_low,
        "clamp_high": args.clamp_high,
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
                "dnaa_02f_with_rida_v0",
                cache_dir=args.cache_dir,
                seed=args.seed,
                hydrolysis_rate_per_min=args.intrinsic_rate,
                rida_rate_per_min=args.rida_rate,
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

            # Quick second-half medians for the gate fields.
            nsnap = len(snapshots)
            second_half = snapshots[nsnap // 2:]
            def med(field):
                vals = [s.get(field) for s in second_half if s.get(field) is not None]
                return float(np.median(vals)) if vals else None
            summary = {
                "second_half_median": {
                    "atp_count_bulk":         med("atp_count_bulk"),
                    "adp_count_bulk":         med("adp_count_bulk"),
                    "apo_count_bulk":         med("apo_count_bulk"),
                    "listener_atp_fraction":  med("listener_atp_fraction"),
                    "listener_total":         med("listener_total"),
                    "rida_active":            med("rida_active"),
                    "rida_n_replisomes":      med("rida_n_replisomes"),
                    "rida_events":            med("rida_events"),
                    "intrinsic_events":       med("intrinsic_events"),
                },
            }

            result = {
                "variant":         "E-rida-v0",
                "composite":       "dnaa_02f_with_rida_v0",
                "seed":            args.seed,
                "rida_rate":       args.rida_rate,
                "intrinsic_rate":  args.intrinsic_rate,
                "load_time":       load_time,
                "wall_time":       wall_time,
                "sim_time":        total,
                "interval":        args.interval,
                "divided":         divided,
                "run_id":          run.run_id,
                "db_path":         str(run.db_path),
                "snapshots":       snapshots,
                "summary":         summary,
            }
            os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
            with open(args.out_path, "w") as f:
                json.dump(result, f, indent=2)

    print(f"\nWrote {args.out_path}")
    print(f"  variant=E (RIDA-v0)  rida_rate={args.rida_rate}/min  "
          f"sim_time={total}s  wall={wall_time:.1f}s")
    print("  Second-half medians:")
    for k, v in summary["second_half_median"].items():
        print(f"    {k}: " + (f"{v:.3f}" if v is not None else "None"))


if __name__ == "__main__":
    main()
