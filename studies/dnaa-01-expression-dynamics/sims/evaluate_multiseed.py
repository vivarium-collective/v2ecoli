"""Evaluate behavior_tests against the union of all baseline-seed* runs
in runs.db. Concatenates per-step state across seeds so correlation tests
get enough variance to compute Pearson r (the single-seed 10-min window
produces sparse signals for rare events like dnaA mRNA).

Usage:
    .venv/bin/python studies/dnaa-01-expression-dynamics/sims/evaluate_multiseed.py
"""
from __future__ import annotations

import json
import sqlite3
import statistics
import sys
from pathlib import Path

import yaml

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR / "tests"))

from _behaviors import evaluate, _monomer_count, _listener_value, DNAA_MONOMER_PD  # noqa: E402


def _load_seed_runs(db_path: Path, like: str = "baseline-seed%") -> list[tuple[str, list[dict]]]:
    """Return [(run_name, history), ...] for runs matching SQL LIKE pattern."""
    conn = sqlite3.connect(str(db_path))
    sims = conn.execute(
        "SELECT simulation_id, name FROM simulations "
        "WHERE name LIKE ? "
        "ORDER BY name ASC", (like,)
    ).fetchall()
    out = []
    for sim_id, name in sims:
        rows = conn.execute(
            "SELECT step, global_time, state FROM history WHERE simulation_id=? "
            "ORDER BY step ASC", (sim_id,)
        ).fetchall()
        history = [{"step": s, "time": t, "state": json.loads(st)} for s, t, st in rows]
        out.append((name, history))
    conn.close()
    return out


def main() -> int:
    spec = yaml.safe_load((STUDY_DIR / "study.yaml").read_text())
    # Accept an optional SQL LIKE pattern as argv[1] (e.g. "baseline-te50x%").
    pattern = sys.argv[1] if len(sys.argv) > 1 else "baseline-seed%"
    seed_runs = _load_seed_runs(STUDY_DIR / "runs.db", like=pattern)
    if not seed_runs:
        print(f"No runs in runs.db matching LIKE {pattern!r}", file=sys.stderr)
        return 2
    print(f"Filter: name LIKE {pattern!r}")

    print(f"Loaded {len(seed_runs)} seed runs:")
    for name, hist in seed_runs:
        print(f"  {name}: {len(hist)} steps")
    print()

    # Concatenate each sim's SECOND-HALF (not the full sim) so the pool
    # doesn't include startup transients. The behavior_tests' window:
    # second_half then operates on the pool as a whole and gets the
    # steady-state samples only.
    pooled = []
    for _, hist in seed_runs:
        pooled.extend(hist[len(hist)//2:])
    print(f"Pooled second-half history length: {len(pooled)} step-records "
          f"(across {len(seed_runs)} seeds, each contributing its own second half)")
    print()

    # Per-seed sanity for dnaA mRNA — does aggregation add variance?
    series_pooled = []
    for s in pooled:   # pooled is already each sim's second-half concatenated
        v = _listener_value(s["state"], "listeners.rna_counts.mRNA_cistron_counts")
        if isinstance(v, list) and len(v) > 227:
            series_pooled.append(v[227])
    if series_pooled:
        nz = [v for v in series_pooled if v]
        sd = statistics.stdev(series_pooled) if len(series_pooled) > 1 else 0.0
        print(f"dnaA mRNA (pooled second-half, n={len(series_pooled)}): "
              f"min={min(series_pooled)} max={max(series_pooled)} "
              f"mean={statistics.mean(series_pooled):.3f} "
              f"stdev={sd:.3f} "
              f"nonzero={len(nz)} ({100*len(nz)/len(series_pooled):.0f}%)")
        print()

    # Median DnaA across pooled second-half (for the count-in-range finding).
    dnaA = []
    for s in pooled:
        c = _monomer_count(s["state"], DNAA_MONOMER_PD, None)
        if c is not None:
            dnaA.append(c)
    if dnaA:
        print(f"DnaA monomer pooled second-half: median={statistics.median(dnaA)} "
              f"n={len(dnaA)} min={min(dnaA)} max={max(dnaA)}")
        print()

    print("Behavior test results (pooled across seeds):")
    print("-" * 60)
    for t in spec.get("behavior_tests", []):
        if t.get("requires_simulation") != "baseline-steady-state":
            continue
        if t.get("status") == "gated" or t.get("blocked_by_requirements"):
            continue
        passed, msg = evaluate(t, pooled)
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag} ] {t['name']:40s}  {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
