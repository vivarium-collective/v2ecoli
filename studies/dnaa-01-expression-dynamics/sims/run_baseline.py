"""dnaa-01 baseline runner — captures DnaA-specific readouts each minute.

Outputs a JSON file with snapshots of:
  - bulk[PD03831[c]]     — free DnaA monomer
  - bulk[MONOMER0-160[c]] — DnaA-ATP complex
  - bulk[MONOMER0-4565[c]] — additional DnaA complex form
  - listeners.monomer_counts.monomerCounts[3861] — total DnaA monomer count
    (re-aggregated by monomer_counts_listener)
  - listeners.rna_counts.mRNA_counts[idx EG10235_RNA] — dnaA mRNA count
  - cell mass + volume + n_chromosomes + n_forks

Usage:
    python studies/dnaa-01-expression-dynamics/sims/run_baseline.py \
        <duration_s> <interval_s> <out_json> [--seed N]
"""
from __future__ import annotations
import os
import sys
import json
import time
import argparse
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import numpy as np

# Suppress C-level chatter (set DNAA01_QUIET=1 to enable)
if os.environ.get("DNAA01_QUIET"):
    _fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_fd, 2)

from v2ecoli import build_composite
from v2ecoli.composites._helpers import flush_parquet, parquet_emitter
from pbg_superpowers.runner import pbg_runner

STUDY_SLUG = "dnaa-01-expression-dynamics"
INVESTIGATION_SLUG = "dnaa-replication"


DNAA_BULK_IDS = {
    "PD03831[c]": "free_dnaa_monomer",
    "MONOMER0-160[c]": "dnaa_atp_complex",
    "MONOMER0-4565[c]": "dnaa_other_complex",
}
DNAA_MONOMER_COUNT_IDX = 3861   # listeners.monomer_counts.monomerCounts (per YAML)
DNAA_MRNA_TU = "EG10235_RNA"


def _build_bulk_lookup(bulk_array):
    return {bid: int(np.where(bulk_array["id"] == bid)[0][0])
            for bid in DNAA_BULK_IDS
            if (bulk_array["id"] == bid).any()}


def _read_listener_index(node, field_name, idx):
    """Read node[field_name][idx] handling both structured + flat arrays.

    Returns int or None if unavailable / out-of-bounds.
    """
    if node is None:
        return None
    arr = None
    # Structured array with a named field
    if hasattr(node, "dtype") and getattr(node.dtype, "names", None):
        if field_name in node.dtype.names:
            arr = node[field_name]
    # Mapping-like
    elif hasattr(node, "get"):
        arr = node.get(field_name)
    # Plain ndarray: caller passed the field already
    elif hasattr(node, "__len__"):
        arr = node
    if arr is None or not hasattr(arr, "__len__"):
        return None
    if len(arr) <= idx:
        return None
    return int(arr[idx])


def snap(t, cell, bulk_idx, mrna_idx):
    bulk = cell["bulk"]
    listeners = cell.get("listeners", {})
    mc = listeners.get("monomer_counts")
    rna_counts = listeners.get("rna_counts")
    mass = listeners.get("mass", {})

    out = {"time": t}
    for bid, key in DNAA_BULK_IDS.items():
        idx = bulk_idx.get(bid)
        out[key] = int(bulk[idx]["count"]) if idx is not None else None

    out["dnaa_monomer_total"] = _read_listener_index(
        mc, "monomerCounts", DNAA_MONOMER_COUNT_IDX)

    out["dnaa_mrna_count"] = None
    if mrna_idx is not None:
        out["dnaa_mrna_count"] = _read_listener_index(
            rna_counts, "mRNA_counts", mrna_idx)

    out["cell_mass_fg"] = float(mass.get("cell_mass", 0))
    out["dry_mass_fg"] = float(mass.get("dry_mass", 0))
    out["volume_fL"] = float(mass.get("volume", 0))

    unique = cell.get("unique", {})
    fc = unique.get("full_chromosome")
    rep = unique.get("active_replisome")
    n_chrom = 0
    if fc is not None and hasattr(fc, "dtype") and "_entryState" in fc.dtype.names:
        n_chrom = int(fc["_entryState"].sum())
    n_forks = 0
    if rep is not None and hasattr(rep, "dtype") and "_entryState" in rep.dtype.names:
        n_forks = int(rep["_entryState"].sum())
    out["n_chromosomes"] = n_chrom
    out["n_forks"] = n_forks
    return out


def _summarize(snapshots):
    """Median over the second half of the run for each numeric readout."""
    n = len(snapshots)
    if n == 0:
        return {}
    second_half = snapshots[n // 2:]
    keys = [k for k in second_half[0].keys()
            if isinstance(second_half[0][k], (int, float)) and k != "time"]
    summary = {}
    for k in keys:
        vals = [s[k] for s in second_half if s.get(k) is not None]
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        summary[k] = {
            "median_second_half": float(np.median(arr)),
            "mean_second_half": float(np.mean(arr)),
            "min_second_half": float(np.min(arr)),
            "max_second_half": float(np.max(arr)),
        }
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("duration", type=int, help="Simulation duration in seconds")
    p.add_argument("interval", type=int, help="Snapshot interval in seconds")
    p.add_argument("out_path", help="Output JSON path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default="out/cache")
    p.add_argument(
        "--composite", default="baseline",
        help="Composite name to build. Default 'baseline' is the basic "
        "v2ecoli baseline. Pass 'dnaa_stage1_constitutive' to use the "
        "Stage-1 constitutive DnaA source (matches the historical study "
        "setup; pair with the out/cache-stage1-heuristic[-<media>] cache).",
    )
    p.add_argument(
        "--sim-name",
        default=None,
        help="Label for the dashboard's Simulations tab (defaults to "
        "baseline-seed<seed>).",
    )
    args = p.parse_args()
    sim_name = args.sim_name or f"baseline-seed{args.seed}"
    params = {
        "seed": args.seed,
        "duration_s": args.duration,
        "interval_s": args.interval,
    }

    # Open runs.db, log a 'running' row, get a run_id that doubles as the
    # parquet experiment_id. Then wrap build_composite in parquet_emitter()
    # so per-tick state lands in studies/<slug>/parquet-runs/<run_id>/ —
    # rendered by v2ecoli.library.parquet_viz (see scripts/render_study_viz.py).
    with pbg_runner(study=STUDY_SLUG, name=sim_name, params=params) as run:
        with parquet_emitter(
            out_dir=f"studies/{STUDY_SLUG}/parquet-runs",
            experiment_id=run.run_id,
            study_slug=STUDY_SLUG,
            investigation_slug=INVESTIGATION_SLUG,
        ):
            t0 = time.time()
            composite = build_composite(
                args.composite, cache_dir=args.cache_dir, seed=args.seed)
            load_time = time.time() - t0

            cell = composite.state["agents"]["0"]
            bulk_idx = _build_bulk_lookup(cell["bulk"])

            # Resolve dnaA mRNA index from sim_data
            mrna_idx = None
            try:
                sim_data = composite.state["agents"]["0"].get("_sim_data_handle")
            except Exception:
                sim_data = None
            # Fall back: try to read it from a configs cache via the loader
            if mrna_idx is None:
                try:
                    from v2ecoli.core import load_cache_bundle
                    bundle = load_cache_bundle(args.cache_dir)
                    sd = bundle.get("sim_data_handle") or bundle.get("sim_data")
                    if sd is not None and hasattr(sd, "process"):
                        mrna_tu_ids = list(
                            sd.process.transcription.cistron_data["id"])
                        if DNAA_MRNA_TU in mrna_tu_ids:
                            mrna_idx = mrna_tu_ids.index(DNAA_MRNA_TU)
                except Exception:
                    pass

            # First snapshot at t=0 (some listeners not populated yet)
            snapshots = [snap(0, cell, bulk_idx, mrna_idx)]

            t_run = time.time()
            total = 0
            divided = False
            while total < args.duration:
                chunk = min(args.interval, args.duration - total)
                try:
                    composite.run(chunk)
                except Exception as e:
                    divided = True
                    break
                total += chunk
                cell = composite.state.get("agents", {}).get("0")
                if cell is None:
                    divided = True
                    break
                snapshots.append(snap(total, cell, bulk_idx, mrna_idx))
                run.heartbeat(len(snapshots))

            wall_time = time.time() - t_run

            flush_parquet(composite, success=True)

            summary = _summarize(snapshots)
            run.n_steps = len(snapshots)

            result = {
                "engine": "v2ecoli (process-bigraph)",
                "seed": args.seed,
                "load_time": load_time,
                "wall_time": wall_time,
                "sim_time": total,
                "duration_requested": args.duration,
                "interval": args.interval,
                "divided": divided,
                "run_id": run.run_id,
                "db_path": str(run.db_path),
                "snapshots": snapshots,
                "summary_second_half": summary,
                "notes": {
                    "bulk_indices": bulk_idx,
                    "monomer_count_idx": DNAA_MONOMER_COUNT_IDX,
                    "dnaa_mrna_idx": mrna_idx,
                },
            }

            os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
            with open(args.out_path, "w") as f:
                json.dump(result, f, indent=2)

    print(f"\nWrote {args.out_path}")
    print(f"  sim_time: {total}s  wall_time: {wall_time:.1f}s  divided: {divided}")
    print("  second-half medians:")
    for k, stats in summary.items():
        print(f"    {k}: median={stats['median_second_half']:.2f} "
              f"(min={stats['min_second_half']:.1f}, max={stats['max_second_half']:.1f})")


if __name__ == "__main__":
    main()
