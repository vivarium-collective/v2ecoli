"""
Three-Way E. coli Simulation Comparison (wrapper)

Runs three engines — vEcoli 1.0, vEcoli 2.0, and v2ecoli — in separate
subprocesses to avoid type conflicts, then dispatches to
``V1V2Visualization`` for HTML rendering.

Usage:
    python reports/v1_v2_report.py                    # 2500s default
    python reports/v1_v2_report.py --duration 300     # short comparison
    python reports/v1_v2_report.py --out out/cmp.html
"""

import json
import os
import sys
import time
import argparse
import subprocess as sp

SNAPSHOT_INTERVAL = 50  # seconds between snapshots
REPORT_DIR = "out/comparison"

# ---------------------------------------------------------------------------
# Engine definitions
# ---------------------------------------------------------------------------
# Each entry: (dataset_key, runner_script, temp_result_file)
_RUNNERS = {
    "vecoli_v1":        ("scripts/run_vecoli_v1.py",       "_vecoli_v1_result.json"),
    "vecoli_composite": ("scripts/run_vecoli_composite.py", "_vecoli_composite_result.json"),
    "v2ecoli":          ("scripts/run_v2.py",               "_v2ecoli_result.json"),
}

_EMPTY = {
    "snapshots": [],
    "wall_time": 0,
    "sim_time": 0,
    "speed": 0,
    "load_time": 0,
}


def _launch(key: str, duration: int, base: str, result_paths: dict):
    """Launch a subprocess runner for *key*; return Popen or None."""
    script_rel, result_file = _RUNNERS[key]
    rpath = os.path.join(base, REPORT_DIR, result_file)
    spath = os.path.join(base, script_rel)
    result_paths[key] = rpath
    if os.path.exists(spath):
        return sp.Popen(
            [sys.executable, spath, str(duration), str(SNAPSHOT_INTERVAL), rpath]
        )
    print(f"  {key}: script not found ({spath})")
    return None


def _collect(key: str, proc, result_paths: dict, datasets: dict) -> None:
    """Wait for *proc* and load its JSON output into *datasets[key]*."""
    rpath = result_paths[key]
    if proc is None:
        datasets[key] = {**_EMPTY, "engine": f"{key} (skipped)"}
        return
    proc.wait()
    if os.path.exists(rpath):
        with open(rpath) as f:
            data = json.load(f)
        os.unlink(rpath)
        print(f"  {key}: {data.get('sim_time',0)}s in {data.get('wall_time',0):.1f}s "
              f"({data.get('speed',0):.1f}x)")
        datasets[key] = data
    else:
        print(f"  {key}: FAILED (rc={proc.returncode})")
        datasets[key] = {**_EMPTY, "engine": f"{key} (FAILED)"}


def _datasets_to_histories(datasets: dict) -> tuple[list, list, list]:
    """Convert legacy datasets dict to three snapshot lists for the Step."""
    return (
        datasets.get("vecoli_v1",        {}).get("snapshots", []),
        datasets.get("vecoli_composite", {}).get("snapshots", []),
        datasets.get("v2ecoli",          {}).get("snapshots", []),
    )


def main():
    parser = argparse.ArgumentParser(description="v1 vs v2 three-way comparison")
    parser.add_argument(
        "--duration", type=int, default=2520,
        help="Simulation duration in seconds (default: 2520 = 42 min)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output HTML path (default: out/comparison/comparison.html)",
    )
    args = parser.parse_args()

    # Ensure working directory is the repo root.
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(base)
    os.makedirs(os.path.join(base, REPORT_DIR), exist_ok=True)

    out_path = args.out or os.path.join(base, REPORT_DIR, "comparison.html")

    print("=" * 60)
    print(f"Three-Way Comparison ({args.duration}s)")
    print("=" * 60)

    t0 = time.time()
    datasets: dict = {}
    result_paths: dict = {}

    # Phase 1: composite + v2ecoli in parallel (both use composite branch).
    print("  Launching composite + v2ecoli in parallel...")
    p_comp = _launch("vecoli_composite", args.duration, base, result_paths)
    p_v2   = _launch("v2ecoli",          args.duration, base, result_paths)
    _collect("vecoli_composite", p_comp, result_paths, datasets)
    _collect("v2ecoli",          p_v2,   result_paths, datasets)

    # Phase 2: v1 sequentially (switches vEcoli to master branch).
    print("  Launching v1 (vivarium engine)...")
    p_v1 = _launch("vecoli_v1", args.duration, base, result_paths)
    _collect("vecoli_v1", p_v1, result_paths, datasets)

    total = time.time() - t0

    # Dispatch to V1V2Visualization for rendering.
    from bigraph_schema import allocate_core
    from v2ecoli.visualizations.v1_v2 import V1V2Visualization

    h_v1, h_v2, h_v2ecoli = _datasets_to_histories(datasets)

    # Build per-dataset metadata (wall/sim/load/speed).
    def _meta(key: str) -> dict:
        d = datasets.get(key, {})
        return {
            "wall_time": d.get("wall_time", 0),
            "sim_time":  d.get("sim_time",  0),
            "load_time": d.get("load_time", 0),
            "speed":     d.get("speed",     0),
            "engine":    d.get("engine",    key),
        }

    # Merge per-engine metadata: V1V2Visualization currently accepts a single
    # metadata dict; we use the overall run metadata + the three engine dicts
    # so the performance table can be populated.  The Step renders all three
    # via the datasets dict it rebuilds internally from _meta_from_rows, which
    # is driven by this single metadata pass-through for now.
    meta = {
        "duration_sec": args.duration,
        # Performance for the overview table is embedded in the snapshot
        # lists by the runner scripts.  Pass wall-time totals for reference.
        "total_wall_sec": total,
    }

    viz = V1V2Visualization(
        config={"title": "E. coli Whole-Cell Simulation Comparison"},
        core=allocate_core(),
    )
    result = viz.update({
        "history_v1":      h_v1,
        "history_v2":      h_v2,
        "history_v2ecoli": h_v2ecoli,
        "metadata":        meta,
    })
    html = result["html"]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)

    print(f"\nReport: {out_path}")
    print(f"Total: {total:.0f}s")

    # Mirror to docs/ for GitHub Pages.
    import shutil
    docs_dir = os.path.join(base, "docs")
    if os.path.isdir(docs_dir):
        shutil.copy2(out_path, os.path.join(docs_dir, "v1_v2_comparison.html"))

    # Open in browser if interactive.
    sp.run(["open", out_path], capture_output=True)


if __name__ == "__main__":
    main()
