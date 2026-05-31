"""dnaa-0 smoke-test runner — succinate baseline, 2-gen single_daughters.

Confirms that the fresh dnaa-replication scaffold works end-to-end before
committing to the heavy 10-generation acceptance run:

  1. Builds the v2ecoli baseline composite against out/cache-succinate.
  2. Runs ~165 min sim time (≈ 2 cycles at τ=82 min) with the parquet
     emitter wired through parquet_emitter(), so the new
     DnaaSteadyStateVisualization can render directly from the
     parquet-runs/ hive via scripts/render_study_viz.py.
  3. Captures snapshots of the three observables the dnaa-0 viz needs:
     listeners.replication_data.number_of_oric, listeners.mass.cell_mass,
     listeners.monomer_counts (aggregated at the dnaA index for total
     DnaA monomer).
  4. Reports DIVISION events and a summary to stdout + JSON sidecar.

Usage::

    python scripts/run_dnaa0_smoke.py [--duration 9900] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

import numpy as np  # noqa: E402

from v2ecoli import build_composite  # noqa: E402
from v2ecoli.composites._helpers import flush_parquet, parquet_emitter  # noqa: E402

STUDY_SLUG = "dnaa-0-parameter-foundation"
INVESTIGATION_SLUG = "dnaa-replication"

# dnaA monomer is PD03831[c] in v2ecoli bulk — the index is conventionally 3861
# but resolve dynamically so it works on any fixture.
DNAA_MONOMER_BULK_ID = "PD03831[c]"


def _resolve_dnaa_monomer_index(cell_state):
    """Find the dnaA monomer index in the bulk['id'] array."""
    bulk = cell_state.get("bulk")
    if bulk is None or not hasattr(bulk, "dtype"):
        return None
    ids = bulk["id"]
    matches = np.where(ids == DNAA_MONOMER_BULK_ID)[0]
    return int(matches[0]) if len(matches) > 0 else None


def _snap(t: float, cell, dnaa_idx):
    if cell is None:
        return {"t": t, "agent_present": False}
    listeners = cell.get("listeners") or {}
    replication = listeners.get("replication_data") or {}
    mass = listeners.get("mass") or {}
    monomers = listeners.get("monomer_counts")

    dnaa = None
    try:
        if monomers is not None and dnaa_idx is not None:
            dnaa = int(monomers[dnaa_idx])
    except (IndexError, TypeError):
        pass

    def _f(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "t": t,
        "agent_present": True,
        "oric_count": int(replication.get("number_of_oric"))
            if replication.get("number_of_oric") is not None else None,
        "cell_mass": _f(mass.get("cell_mass")),
        "dry_mass": _f(mass.get("dry_mass")),
        "dnaa_monomer": dnaa,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--duration", type=int, default=9900,
                   help="Sim seconds (default 9900 ≈ 2 cycles at τ=82 min)")
    p.add_argument("--interval", type=int, default=60,
                   help="Snapshot interval in sim seconds (default 60)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default="out/cache-succinate")
    p.add_argument("--out", default=None,
                   help="JSON output path (default: studies/<slug>/sims/smoke/run.json)")
    p.add_argument("--sim-name", default=None)
    args = p.parse_args()

    sim_name = args.sim_name or f"smoke-seed{args.seed}"
    out_path = args.out or (
        f"studies/{STUDY_SLUG}/sims/smoke/{sim_name}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with parquet_emitter(
        out_dir=f"studies/{STUDY_SLUG}/parquet-runs",
        experiment_id=sim_name,
        study_slug=STUDY_SLUG,
        investigation_slug=INVESTIGATION_SLUG,
        # Disable the background ThreadPoolExecutor — the parquet shim races on
        # filesystem.mv(temp_outfile, outfile) when consecutive batches flush
        # quickly (smoke run hit this at the 1200-row boundary with FileNotFoundError
        # on the prior batch's tmpfile that had already been moved). Serial
        # writes are slightly slower but eliminate the race.
        threaded=False,
    ):
        t_load = time.time()
        composite = build_composite(
            "baseline", cache_dir=args.cache_dir, seed=args.seed)
        load_time = time.time() - t_load

        cell0 = composite.state.get("agents", {}).get("0")
        dnaa_idx = _resolve_dnaa_monomer_index(cell0) if cell0 else None
        print(f"resolved dnaA monomer bulk index: {dnaa_idx}")

        snapshots: list[dict] = []
        snapshots.append({"phase": "initial", **_snap(0.0, cell0, dnaa_idx)})

        division_times: list[float] = []
        last_agent_keys = set(composite.state.get("agents", {}).keys())
        total = 0.0
        t_run = time.time()

        while total < args.duration:
            step = min(args.interval, args.duration - total)
            try:
                composite.run(step)
            except Exception as e:
                snapshots.append({"t": total, "phase": "error", "error": str(e)})
                break
            total += step
            agents = composite.state.get("agents", {}) or {}
            now_keys = set(agents.keys())
            # Detect a division: previous followed agent disappears (or new
            # daughter ids appear).
            if now_keys != last_agent_keys:
                # Mark the time of change as a generation boundary.
                division_times.append(total)
                last_agent_keys = now_keys
            # In single_daughters mode the followed agent's id can change
            # across divisions ('0' → '00' → '000' …). Pick whichever agent
            # still exists for the snapshot — there should be exactly one.
            cell = next(iter(agents.values()), None) if len(agents) > 0 else None
            if cell is None:
                snapshots.append({"t": total, "phase": "no_agents"})
                break
            snapshots.append({"t": total, **_snap(total, cell, dnaa_idx)})

        wall_time = time.time() - t_run
        flush_parquet(composite, success=True)

    # ----- summary -----
    valid = [s for s in snapshots if s.get("agent_present") and s.get("oric_count") is not None]
    oric_vals = [s["oric_count"] for s in valid]
    cm_vals = [s["cell_mass"] for s in valid if s.get("cell_mass") is not None]
    dnaa_vals = [s["dnaa_monomer"] for s in valid if s.get("dnaa_monomer") is not None]
    summary = {
        "n_snapshots": len(snapshots),
        "n_divisions_detected": len(division_times),
        "division_times_s": division_times,
        "oric_range": [min(oric_vals), max(oric_vals)] if oric_vals else None,
        "cell_mass_range_fg": [float(min(cm_vals)), float(max(cm_vals))] if cm_vals else None,
        "dnaa_monomer_range": [min(dnaa_vals), max(dnaa_vals)] if dnaa_vals else None,
    }

    result = {
        "study": STUDY_SLUG,
        "investigation": INVESTIGATION_SLUG,
        "sim_name": sim_name,
        "cache_dir": args.cache_dir,
        "seed": args.seed,
        "load_time": load_time,
        "wall_time": wall_time,
        "sim_time_completed": total,
        "summary": summary,
        "snapshots": snapshots,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote {out_path}")
    print(f"  sim_time: {total:.0f}s ({total/60:.1f} min)  wall: {wall_time:.1f}s")
    print(f"  divisions detected: {len(division_times)}  at t = "
          f"{[f'{t:.0f}s' for t in division_times]}")
    if oric_vals:
        print(f"  oriC range: {summary['oric_range']}")
    if cm_vals:
        print(f"  cell_mass range (fg): {[round(v, 1) for v in summary['cell_mass_range_fg']]}")
    if dnaa_vals:
        print(f"  DnaA monomer range: {summary['dnaa_monomer_range']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
