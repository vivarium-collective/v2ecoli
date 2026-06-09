#!/usr/bin/env python3
"""Performance comparison harness: v2ecoli vs vEcoli on the SAME multiseed/
multigen job.

Drives a 2-seed × 2-generation E. coli whole-cell job on both engines and
captures wall-clock, peak RSS, and throughput so we can ask: for the same
workload, which engine runs it faster/cheaper, and where does the time go?

  v2ecoli — process-bigraph runner. `scripts/run_phase0_multigen.py` loops
            seeds in ONE Python process, each running max_generations past
            division (sequential, single-threaded — the GIL story). Per-seed
            wall + steps land in .pbg/runs/phase0-multigen/summary.json.

  vEcoli  — Nextflow DAG. `python -m runscripts.workflow` fans sim tasks
            (one per seed × generation) across processes. Nextflow's trace
            CSV (auto-enabled in its config) records per-task realtime,
            %cpu, rss, peak_rss.

To keep the comparison about SIM execution (not ParCa), vEcoli reuses a
prebuilt simData.cPickle (sim_data_path) and analyses are stripped; v2ecoli
likewise runs off the shared out/cache ParCa build. Both launchers run under
`/usr/bin/time -l` for an independent peak-RSS reading.

Writes reports/perf/perf_results.json (consumed by perf_report.py).

Usage:
  python scripts/perf_compare.py --engine both      # run both, write results
  python scripts/perf_compare.py --engine v2        # v2ecoli only
  python scripts/perf_compare.py --engine vecoli     # vEcoli only
  python scripts/perf_compare.py --seeds 2 --generations 2
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent          # v2e-compare
VECOLI = Path("/Users/eranagmon/code/vEcoli")
OUT_DIR = REPO / "reports" / "perf"
RESULTS = OUT_DIR / "perf_results.json"
LOG_DIR = OUT_DIR / "logs"
VECOLI_SIMDATA = VECOLI / "out" / "kb" / "simData.cPickle"   # prebuilt → skip ParCa


# ── helpers ────────────────────────────────────────────────────────────────

def _maxrss_mb_from_time_l(stderr: str) -> float | None:
    """Parse `/usr/bin/time -l` 'maximum resident set size' (BYTES on macOS)."""
    m = re.search(r"(\d+)\s+maximum resident set size", stderr)
    return round(int(m.group(1)) / (1024 * 1024), 1) if m else None


def _vecoli_env() -> dict:
    """Env for the vEcoli Nextflow subprocess (scoped — no global change).

    Two PATH fixes Nextflow's macOS-local executor needs:
      * JDK 17+ first (Nextflow 24.x rejects Java 11), and
      * vEcoli's OWN .venv/bin first, so the bare `python` inside each
        Nextflow task (create_variants / sim) resolves to the interpreter
        that has vEcoli's deps (fsspec, …) — not some other venv on PATH.
    """
    env = dict(os.environ)
    prepends = []
    venv_bin = VECOLI / ".venv" / "bin"
    if venv_bin.is_dir():
        prepends.append(str(venv_bin))
    for cand in ("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home",
                 "/opt/homebrew/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home"):
        if Path(cand, "bin", "java").is_file():
            env["JAVA_HOME"] = cand
            prepends.append(f"{cand}/bin")
            break
    if prepends:
        env["PATH"] = ":".join(prepends) + ":" + env.get("PATH", "")
    return env


def _timed(cmd: list[str], *, cwd: Path, log: Path, env: dict | None = None) -> dict:
    """Run cmd under /usr/bin/time -l; return {wall_s, maxrss_mb, returncode, log}."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    full = ["/usr/bin/time", "-l", *cmd]
    t0 = time.monotonic()
    proc = subprocess.run(full, cwd=str(cwd), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall = time.monotonic() - t0
    log.write_text(proc.stdout)
    return {
        "wall_s": round(wall, 2),
        "maxrss_mb": _maxrss_mb_from_time_l(proc.stdout),
        "returncode": proc.returncode,
        "cmd": " ".join(cmd),
        "log": str(log.relative_to(REPO)),
    }


# ── v2ecoli ─────────────────────────────────────────────────────────────────

def run_v2ecoli(seeds: int, generations: int, max_steps: int, mode: str = "seq") -> dict:
    """mode='seq' → sequential single-process driver; mode='ray' → Ray-parallel
    driver (one @ray.remote worker per seed, each reporting its own peak RSS)."""
    py = str(REPO / ".venv" / "bin" / "python")
    is_ray = mode == "ray"
    driver = "scripts/_perf_v2_driver_ray.py" if is_ray else "scripts/_perf_v2_driver.py"
    sub = "perf-v2-ray" if is_ray else "perf-v2"
    summary_path = REPO / ".pbg" / "runs" / sub / "summary.json"
    if summary_path.exists():
        summary_path.unlink()
    model = ("parallel Ray tasks (process per seed)" if is_ray
             else "sequential, single-process (GIL-bound)")
    print(f"[v2ecoli/{mode}] {seeds} seeds × ≤{generations} gens (single_daughters) …",
          flush=True)
    timed = _timed(
        [py, driver, "--n-seeds", str(seeds),
         "--max-generations", str(generations), "--max-steps", str(max_steps)],
        cwd=REPO, log=LOG_DIR / "v2ecoli.log",
    )
    out = {"engine": "v2ecoli", "mode": mode, "execution_model": model,
           **timed, "per_cell": [], "ensemble": None}
    if summary_path.exists():
        ens = json.loads(summary_path.read_text())
        out["ensemble"] = ens
        # In ray mode each worker reports its own real peak RSS; the
        # representative engine peak is the max over the concurrent workers
        # (the launcher maxrss from /usr/bin/time excludes the workers).
        seed_rss = [s.get("peak_rss_mb") for s in ens.get("per_seed", [])
                    if isinstance(s, dict) and s.get("peak_rss_mb")]
        if is_ray and seed_rss:
            out["maxrss_mb"] = max(seed_rss)          # real per-worker peak
            out["maxrss_source"] = "max per-seed worker RSS (resource.getrusage)"
        # one "cell" per (seed, generation) actually simulated
        for s in ens.get("per_seed", []):
            if "error" in s:
                continue
            gens = s.get("actual_generations_seen") or list(range(generations))
            wall = s.get("wall_seconds") or s.get("wall")
            n = max(1, len(gens))
            for g in gens:
                out["per_cell"].append({
                    "seed": s["seed"], "generation": g,
                    "wall_s": round((wall or 0) / n, 2) if wall else None,
                    "peak_rss_mb": s.get("peak_rss_mb"),
                    "source": "summary.json (per-seed wall ÷ gens)",
                })
    return out


# ── vEcoli ──────────────────────────────────────────────────────────────────

def _write_vecoli_config(seeds: int, generations: int, exp_id: str) -> Path:
    base = json.loads((VECOLI / "configs" / "two_generations.json").read_text())
    base["experiment_id"] = exp_id
    base["generations"] = generations
    base["n_init_sims"] = seeds
    base["single_daughters"] = True
    base["suffix_time"] = False
    if VECOLI_SIMDATA.is_file():
        base["sim_data_path"] = str(VECOLI_SIMDATA)      # reuse ParCa → skip it
    base["analysis_options"] = {}                         # sim-only headline
    cfg = OUT_DIR / f"vecoli_config_{exp_id}.json"
    cfg.write_text(json.dumps(base, indent=2))
    return cfg


def _parse_nextflow_trace(exp_id: str) -> list[dict]:
    """Find + parse the trace--<exp>--*.csv Nextflow writes (per-task perf)."""
    nf_dir = VECOLI / "out" / exp_id / "nextflow"
    traces = sorted(nf_dir.glob(f"trace--{exp_id}--*.csv")) if nf_dir.is_dir() else []
    if not traces:
        traces = sorted(VECOLI.glob(f"**/trace--{exp_id}--*.csv"))
    if not traces:
        return []
    rows = []
    with open(traces[-1], newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "task": r.get("name"),
                "status": r.get("status"),
                "realtime_s": _dur_to_s(r.get("realtime")),
                "cpu_pct": _num(r.get("%cpu")),
                "rss_mb": _mem_to_mb(r.get("rss")),
                "peak_rss_mb": _mem_to_mb(r.get("peak_rss")),
            })
    return rows


def _dur_to_s(v):
    if v is None or str(v).strip() in ("", "-"):
        return None
    v = str(v).strip()
    # Nextflow trace.raw=true emits a bare integer of MILLISECONDS.
    if re.fullmatch(r"\d+", v):
        return round(int(v) / 1000, 1)
    # Formatted form ("3m 33s", "213ms").
    s = 0.0
    for num, unit in re.findall(r"(\d+\.?\d*)\s*(ms|s|m|h)", v):
        s += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return round(s, 1) if s else None


def _mem_to_mb(v):
    if not v:
        return None
    m = re.match(r"(\d+\.?\d*)\s*([KMGTkmgt]?)B?", v.strip())
    if not m:
        return None
    scale = {"": 1/1e6, "K": 1/1024, "M": 1, "G": 1024, "T": 1024*1024}
    return round(float(m.group(1)) * scale.get(m.group(2).upper(), 1), 1)


def _num(v):
    try:
        return float(re.sub(r"[^\d.]", "", v))
    except (TypeError, ValueError):
        return None


def run_vecoli(seeds: int, generations: int) -> dict:
    exp_id = f"perfcmp_s{seeds}g{generations}"
    cfg = _write_vecoli_config(seeds, generations, exp_id)
    py = str(VECOLI / ".venv" / "bin" / "python")
    if not Path(py).is_file():
        py = sys.executable
    print(f"[vEcoli] Nextflow {seeds} seeds × {generations} gens (exp={exp_id}) …", flush=True)
    timed = _timed(
        [py, "-m", "runscripts.workflow", "--config", str(cfg)],
        cwd=VECOLI, log=LOG_DIR / "vecoli.log", env=_vecoli_env(),
    )
    tasks = _parse_nextflow_trace(exp_id)
    sim_tasks = [t for t in tasks if t.get("task") and re.search(r"sim", t["task"], re.I)]
    out = {"engine": "vEcoli", "execution_model": "parallel Nextflow DAG (process per task)",
           **timed, "experiment_id": exp_id, "config": str(cfg.relative_to(REPO)),
           "tasks": tasks, "n_tasks": len(tasks), "n_sim_tasks": len(sim_tasks)}
    # per-cell = per sim task
    out["per_cell"] = [
        {"task": t["task"], "wall_s": t["realtime_s"], "peak_rss_mb": t["peak_rss_mb"],
         "cpu_pct": t["cpu_pct"], "source": "nextflow trace"}
        for t in sim_tasks
    ]
    return out


# ── orchestration ───────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["both", "v2", "vecoli"], default="both")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--generations", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=8000,
                    help="v2ecoli hard tick cap; generous so max-generations is the binding limit")
    ap.add_argument("--v2-mode", choices=["seq", "ray"], default="seq",
                    help="v2ecoli execution: sequential single-process vs Ray-parallel per seed")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = json.loads(RESULTS.read_text()) if RESULTS.is_file() else {}
    results.setdefault("spec", {})
    results["spec"].update({"seeds": args.seeds, "generations": args.generations,
                            "cells": args.seeds * args.generations,
                            "max_steps": args.max_steps, "v2_mode": args.v2_mode,
                            "condition": "baseline (minimal glucose)"})
    results["provenance"] = {
        "v2ecoli_commit": _git_head(REPO),
        "vecoli_commit": _git_head(VECOLI),
        "host": os.uname().nodename,
    }

    if args.engine in ("both", "v2"):
        results["v2ecoli"] = run_v2ecoli(args.seeds, args.generations, args.max_steps,
                                         mode=args.v2_mode)
        RESULTS.write_text(json.dumps(results, indent=2))
        print(f"[v2ecoli] wall={results['v2ecoli']['wall_s']}s "
              f"rss={results['v2ecoli']['maxrss_mb']}MB rc={results['v2ecoli']['returncode']}")
    if args.engine in ("both", "vecoli"):
        results["vEcoli"] = run_vecoli(args.seeds, args.generations)
        RESULTS.write_text(json.dumps(results, indent=2))
        print(f"[vEcoli] wall={results['vEcoli']['wall_s']}s "
              f"rss={results['vEcoli']['maxrss_mb']}MB rc={results['vEcoli']['returncode']} "
              f"sim_tasks={results['vEcoli']['n_sim_tasks']}")

    RESULTS.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {RESULTS.relative_to(REPO)}")


def _git_head(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


if __name__ == "__main__":
    main()
