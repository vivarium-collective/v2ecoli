"""Load the latest run from runs.db and evaluate dnaa-01's behavior tests.

Reads:
  - studies/dnaa-01-expression-dynamics/runs.db          (per-step history)
  - studies/dnaa-01-expression-dynamics/study.yaml       (behavior_tests + simulation_set)

Writes (stdout):
  Per-test pass/fail + diagnostic value (e.g. median DnaA count,
  rolling CV, Pearson r), and a single summary line listing the
  v2ecoli_observed value that should populate the dnaA-count-in-range
  test's ``calibration_anchor.v2ecoli_observed`` field.
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

from _behaviors import evaluate, _monomer_count, DNAA_MONOMER_PD  # noqa: E402


def _load_run(db_path: Path, name: str | None = None) -> tuple[str, list[dict]]:
    conn = sqlite3.connect(str(db_path))
    if name:
        row = conn.execute(
            "SELECT simulation_id FROM simulations WHERE name=? "
            "ORDER BY started_at DESC LIMIT 1", (name,)
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT simulation_id FROM simulations "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise RuntimeError("no simulations in runs.db")
    sim_id = row[0]
    rows = conn.execute(
        "SELECT step, global_time, state FROM history WHERE simulation_id=? "
        "ORDER BY step ASC", (sim_id,)
    ).fetchall()
    conn.close()
    return sim_id, [{"step": s, "time": t, "state": json.loads(st)} for s, t, st in rows]


def main() -> int:
    spec = yaml.safe_load((STUDY_DIR / "study.yaml").read_text())
    behavior_tests = spec.get("behavior_tests") or []
    if not behavior_tests:
        print("No behavior_tests in study.yaml", file=sys.stderr)
        return 2

    sim_id, history = _load_run(STUDY_DIR / "runs.db", "baseline-steady-state")
    n_steps = len(history)
    print(f"Loaded sim={sim_id}  ({n_steps} steps; "
          f"t={history[0]['time']:.0f}s → {history[-1]['time']:.0f}s)\n")

    # First: print the v2ecoli_observed calibration value (median DnaA over
    # second half), regardless of pass/fail.
    second_half = history[n_steps // 2:]
    dnaA = [c for c in (_monomer_count(s["state"], DNAA_MONOMER_PD, None) for s in second_half)
            if c is not None]
    v2ecoli_observed = statistics.median(dnaA) if dnaA else None
    print(f"  calibration_anchor.v2ecoli_observed = "
          f"{v2ecoli_observed}  (median DnaA, second-half window)\n")

    passes = fails = skipped = 0
    for t in behavior_tests:
        name = t["name"]
        cls = t.get("classification", "?")
        sim_needed = t.get("requires_simulation", "?")
        if sim_needed != "baseline-steady-state":
            print(f"  [skip ] {name:40s} ({cls}, needs={sim_needed})")
            skipped += 1
            continue
        if t.get("status") == "gated" or t.get("blocked_by_requirements"):
            print(f"  [gated] {name:40s} ({cls}; {t.get('blocked_by_requirements')})")
            skipped += 1
            continue

        passed, msg = evaluate(t, history)
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag} ] {name:40s} ({cls})  {msg}")
        passes += int(passed)
        fails += int(not passed)

    print(f"\nTotals: {passes} pass / {fails} fail / {skipped} skipped")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
