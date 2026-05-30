"""Benchmark: v2ecoli (new, partitioned) vs vEcoli composite branch.

Runs each engine in a separate subprocess to avoid type conflicts, captures
each trajectory row (load, run, dry_mass, cell_mass), then dispatches to
BenchmarkVisualization for HTML output.

Usage:
    uv run python reports/benchmark_report.py [--out OUT] [--duration SEC]
                                               [--seed SEED] [--cache-dir DIR]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Subprocess script bodies
# ---------------------------------------------------------------------------
# Both scripts print exactly one line of JSON containing timing + mass fields.
# DURATION is filled in by the wrapper before exec.

_V2ECOLI_SCRIPT_TEMPLATE = """\
import time, json, os
os.chdir({workdir!r})
from v2ecoli import build_composite
from v2ecoli.library.quantity_helpers import fg_magnitude

t0 = time.time()
composite = build_composite("baseline", cache_dir={cache_dir!r}, seed={seed})
load_time = time.time() - t0

t0 = time.time()
composite.run({duration})
run_time = time.time() - t0

cell = composite.state['agents']['0']
dm = fg_magnitude(cell.get('listeners', {{}}).get('mass', {{}}).get('dry_mass', 0))
cm = fg_magnitude(cell.get('listeners', {{}}).get('mass', {{}}).get('cell_mass', 0))

print(json.dumps({{'load': load_time, 'run': run_time,
                   'dry_mass': dm, 'cell_mass': cm}}))
"""

_VECOLI_SCRIPT_TEMPLATE = """\
import time, json, os
os.chdir({vecoli_dir!r})
from ecoli.experiments.ecoli_master_sim import EcoliSim
from ecoli.composites.ecoli_composite import build_composite_native
from ecoli.library.bigraph_types import ECOLI_TYPES
from process_bigraph import Composite
from bigraph_schema import allocate_core

sim = EcoliSim.from_cli()
sim.processes = sim._retrieve_processes(
    sim.processes, sim.add_processes, sim.exclude_processes, sim.swap_processes)
sim.topology = sim._retrieve_topology(
    sim.topology, sim.processes, sim.swap_processes, sim.log_updates)
sim.process_configs = sim._retrieve_process_configs(
    sim.process_configs, sim.processes)

core = allocate_core()
core.register_types(ECOLI_TYPES)

t0 = time.time()
state = build_composite_native(core, sim.config)
ecoli = Composite({{'schema': {{}}, 'state': state}}, core=core)
ecoli.to_run = []
load_time = time.time() - t0

t0 = time.time()
ecoli.run({duration})
run_time = time.time() - t0

cell = ecoli.state['agents']['0']
dm = float(cell.get('listeners', {{}}).get('mass', {{}}).get('dry_mass', 0))
cm = float(cell.get('listeners', {{}}).get('mass', {{}}).get('cell_mass', 0))

print(json.dumps({{'load': load_time, 'run': run_time,
                   'dry_mass': dm, 'cell_mass': cm}}))
"""


def _parse_subprocess_output(stdout: str) -> dict | None:
    """Find and return the first JSON object line in subprocess stdout."""
    for line in stdout.strip().split("\n"):
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _run_v2ecoli_subprocess(
    workdir: str,
    cache_dir: str,
    seed: int,
    duration: float,
) -> list[dict]:
    """Run v2ecoli baseline in a subprocess; return a single-row trajectory."""
    script = _V2ECOLI_SCRIPT_TEMPLATE.format(
        workdir=workdir,
        cache_dir=cache_dir,
        seed=seed,
        duration=duration,
    )
    print(f"\n--- v2ecoli (new, partitioned) ---")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        for line in result.stderr.strip().split("\n")[-5:]:
            print(f"  {line}")
        return []
    data = _parse_subprocess_output(result.stdout)
    if data is None:
        print("  FAILED: no valid JSON output")
        print(result.stdout[-500:] if result.stdout else "(empty stdout)")
        return []
    print(f"  Load: {data.get('load', 0):.2f}s, Run: {data.get('run', 0):.2f}s")
    print(f"  dry_mass={data.get('dry_mass', 0):.1f}fg, "
          f"cell_mass={data.get('cell_mass', 0):.1f}fg")
    return [data]


def _run_vecoli_subprocess(
    vecoli_dir: str,
    duration: float,
) -> list[dict]:
    """Run vEcoli in a subprocess; return a single-row trajectory."""
    script = _VECOLI_SCRIPT_TEMPLATE.format(
        vecoli_dir=vecoli_dir,
        duration=duration,
    )
    print(f"\n--- vEcoli (composite branch) ---")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        for line in result.stderr.strip().split("\n")[-5:]:
            print(f"  {line}")
        return []
    data = _parse_subprocess_output(result.stdout)
    if data is None:
        print("  FAILED: no valid JSON output")
        print(result.stdout[-500:] if result.stdout else "(empty stdout)")
        return []
    print(f"  Load: {data.get('load', 0):.2f}s, Run: {data.get('run', 0):.2f}s")
    print(f"  dry_mass={data.get('dry_mass', 0):.1f}fg, "
          f"cell_mass={data.get('cell_mass', 0):.1f}fg")
    return [data]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="out/reports/benchmark.html",
                        help="Output HTML path (default: out/reports/benchmark.html)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default="out/cache")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Seconds of simulation per engine (default: 60)")
    parser.add_argument("--vecoli-dir", default="/Users/eranagmon/code/vEcoli",
                        help="Path to vEcoli checkout (default: /Users/eranagmon/code/vEcoli)")
    parser.add_argument("--workdir",
                        default=str(Path(__file__).parent.parent),
                        help="Working directory for v2ecoli subprocess")
    args = parser.parse_args()

    print("=" * 60)
    print(f"BENCHMARK: {args.duration}s simulation")
    print("=" * 60)

    wall_start = time.time()
    h_ve = _run_vecoli_subprocess(
        vecoli_dir=args.vecoli_dir,
        duration=args.duration,
    )
    ve_wall = time.time() - wall_start
    if h_ve:
        h_ve[-1].setdefault("elapsed_sec", ve_wall)
        print(f"  wall time: {ve_wall:.1f}s ({len(h_ve)} row(s))")

    wall_start = time.time()
    h_v2 = _run_v2ecoli_subprocess(
        workdir=args.workdir,
        cache_dir=args.cache_dir,
        seed=args.seed,
        duration=args.duration,
    )
    v2_wall = time.time() - wall_start
    if h_v2:
        h_v2[-1].setdefault("elapsed_sec", v2_wall)
        print(f"  wall time: {v2_wall:.1f}s ({len(h_v2)} row(s))")

    # Stdout comparison summary (legacy behaviour preserved)
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    v1 = h_ve[-1] if h_ve else None
    v2 = h_v2[-1] if h_v2 else None
    if v1 and v2:
        print(f"  {'':20s} {'vEcoli':>12s} {'v2ecoli':>12s}")
        print(f"  {'Load time':20s} {v1['load']:>10.2f}s {v2['load']:>10.2f}s")
        print(f"  {'Run time':20s} {v1['run']:>10.2f}s {v2['run']:>10.2f}s")
        print(f"  {'dry_mass':20s} {v1['dry_mass']:>10.1f}fg {v2['dry_mass']:>10.1f}fg")
        print(f"  {'Speed':20s} {args.duration/v1['run']:>9.1f}x "
              f"{args.duration/v2['run']:>9.1f}x")
        ratio = v2["run"] / v1["run"]
        print(f"\n  v2ecoli / vEcoli = {ratio:.2f}x")
        mass_diff = (
            abs(v2["dry_mass"] - v1["dry_mass"]) / v1["dry_mass"] * 100
            if v1["dry_mass"]
            else float("inf")
        )
        print(f"  Mass difference: {mass_diff:.1f}%")
        if ratio <= 1.2:
            print("  EXCELLENT: v2ecoli matches vEcoli performance")
        elif ratio <= 1.5:
            print("  GOOD: v2ecoli within 1.5x of vEcoli")
        elif ratio <= 2.0:
            print("  OK: v2ecoli within 2x of vEcoli")
        else:
            print(f"  SLOW: v2ecoli is {ratio:.1f}x slower")
    else:
        for name, data in [("vEcoli", v1), ("v2ecoli", v2)]:
            if data:
                print(f"  {name}: run={data['run']:.2f}s, "
                      f"dry_mass={data['dry_mass']:.1f}fg")
            else:
                print(f"  {name}: FAILED")

    # HTML output via BenchmarkVisualization
    from bigraph_schema import allocate_core
    from v2ecoli.visualizations.benchmark import BenchmarkVisualization

    viz = BenchmarkVisualization(
        config={"title": "v2ecoli vs vEcoli benchmark"},
        core=allocate_core(),
    )
    result = viz.update({
        "history_v2ecoli": h_v2,
        "history_vecoli":  h_ve,
        "metadata": {
            "seed": args.seed,
            "duration_sec": args.duration,
            "cache_dir": args.cache_dir,
            "vecoli_dir": args.vecoli_dir,
        },
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result["html"])
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
