"""Phase 3 sprint 3: end-to-end likelihood persistence + read-back.

Runs a short PDMP+poisson simulation under SQLiteEmitter, then loads
back the per-tick `listeners.likelihood.{transcript_init,
polypeptide_init, total}` timeseries from the workspace's
.pbg/composite-runs.db. Verifies the Phase-3 inference data pipeline
end-to-end: emission → store → query.

Usage::

    .venv/bin/python scripts/phase3_likelihood_readback.py
"""
from __future__ import annotations

import sys
import sqlite3
import json
from pathlib import Path

sys.path.insert(0, ".")
import numpy as np

from v2ecoli import build_composite
from v2ecoli.composites._helpers import sqlite_emitter


def main():
    duration_s = 30
    seed = 0
    name = f"phase3-sprint3-seed{seed}"

    print(f"Running PDMP+poisson sim, seed={seed}, "
          f"{duration_s}s, under SQLiteEmitter...", flush=True)

    with sqlite_emitter(
        name=name,
        study_slug="pdmp-03-inference",
        investigation_slug="v2ecoli-pdmp",
    ):
        c = build_composite(
            "millard_pdmp_baseline",
            seed=seed,
            with_ref_growth=True,
            ref_growth_flux_source="consumption_matched",
            transcript_initiation_mode="poisson",
            polypeptide_initiation_mode="poisson",
        )
        c.run(duration_s)

    # Find the workspace DB.
    db_path = Path(".pbg/composite-runs.db")
    if not db_path.is_file():
        print(f"ERROR: workspace DB not at {db_path.resolve()}")
        sys.exit(1)
    print(f"\nReading back from {db_path}...")

    con = sqlite3.connect(db_path)
    try:
        # Find this run.
        cur = con.execute(
            "SELECT simulation_id FROM simulations "
            "WHERE name = ? ORDER BY started_at DESC LIMIT 1",
            (name,),
        )
        row = cur.fetchone()
        if not row:
            print(f"ERROR: no simulation row found for name={name!r}")
            sys.exit(1)
        sim_id = row[0]
        print(f"simulation_id: {sim_id}")

        # Pull the time + state rows; the SQLite emitter stores the
        # whole emit-schema-shaped state as a JSON blob per step.
        cur = con.execute(
            "SELECT global_time, state FROM history "
            "WHERE simulation_id = ? ORDER BY global_time",
            (sim_id,),
        )
        ts, ti, pi, tot = [], [], [], []
        for global_time, state_json in cur:
            d = json.loads(state_json) if state_json else {}
            lk = (d.get("listeners") or {}).get("likelihood") or {}
            ts.append(float(global_time))
            ti.append(float(lk.get("transcript_init", float("nan"))))
            pi.append(float(lk.get("polypeptide_init", float("nan"))))
            tot.append(float(lk.get("total", float("nan"))))
    finally:
        con.close()

    ts = np.asarray(ts)
    ti = np.asarray(ti)
    pi = np.asarray(pi)
    tot = np.asarray(tot)

    print(f"\nRead back {len(ts)} rows.")
    print(f"  t range: {ts[0]:.1f} → {ts[-1]:.1f} s")
    print(f"  transcript_init   μ={ti.mean():.2f} σ={ti.std():.2f} "
          f"(finite={np.isfinite(ti).sum()}/{len(ti)})")
    print(f"  polypeptide_init  μ={pi.mean():.2f} σ={pi.std():.2f} "
          f"(finite={np.isfinite(pi).sum()}/{len(pi)})")
    print(f"  total             μ={tot.mean():.2f} σ={tot.std():.2f} "
          f"(finite={np.isfinite(tot).sum()}/{len(tot)})")
    if len(ts) and np.isfinite(tot).all():
        max_diff = np.max(np.abs(tot - (ti + pi)))
        print(f"  collector sum check: max|total − (ti + pi)| = {max_diff:.6f}")
    print("\nEnd-to-end Phase-3 likelihood pipeline working ✓")


if __name__ == "__main__":
    main()
