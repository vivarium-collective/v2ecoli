"""Runner for the dnag-promoter-availability study.

A thin wrapper over ``scripts/run_condition_multigen_parquet.py``, the same
pattern as ppgpp-off-rescue and replisome-gate-sufficiency.

ONE arm: the unmodified mechanistic lineage. There is no contrast arm because
the contrast is WITHIN the lineage -- dnaG's promoter count at generation 4
against its own generation 1, and against the 41 idx_rprotein peers measured in
the same cells. A second arm would answer a different question.

New runs are unavoidable. promoter_copy_number is empty in every parquet
written before 2026-09-03, because the rna_synth_prob deriver's first-tick guard
required the OPTIONAL n_bound_TF_per_TU and so returned {} on every tick (fixed
in fa71a8d0). So unlike dnag-generational-decay's Phase A there is no reuse path.

Both blocking requirements are in place before this runs:
  req-1  the deriver now emits promoter_copy_number (verified 361/361 timesteps)
  req-2  generation 1 now honours --seed, so these 6 seeds are genuine replicates
         rather than the correlated ones every earlier study used

Concurrency: ONE sim costs ~5.5 GB RSS and a 16 GB machine fits 2 TOTAL --
across all worktrees, not just this one. Check for sims elsewhere (ps) before
raising -j; a run in v2ecoli-aim2-dnaa occupies a slot just as surely as one
here.

Usage::

    python workspace/studies/dnag-promoter-availability/sims/run.py -j 1
    python .../run.py --skip-existing        # resume a partial sweep
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
STUDY_DIR = Path(__file__).resolve().parents[1]
STUDY = STUDY_DIR.name
RUNNER = REPO / "scripts/run_condition_multigen_parquet.py"
CACHE = REPO / "out/cache"
OUT_ROOT = REPO / "out" / STUDY

GATE = "ecoli-chromosome-replication.mechanistic_replisome"

ARMS = {
    "mechanistic": {"seeds": [0, 1, 2, 3, 4, 5],
                    "overrides": [f"{GATE}=true"],
                    "generator": []},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARMS), action="append",
                    help="run only this arm (repeatable); default all")
    ap.add_argument("--seed", type=int, action="append",
                    help="run only this seed (repeatable); default the arm's seeds")
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--max-min", type=float, default=200.0)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip an (arm, seed) whose summary json already exists")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="run this many simulations concurrently (default 1). "
                         "Each run is single-threaded and holds ~4-5.5 GB RSS "
                         "(measured 2026-09-03), so a 16 GB machine fits 2 "
                         "concurrent, not 3.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without running them")
    args = ap.parse_args(argv)

    arms = args.arm or list(ARMS)
    jobs = []
    for arm in arms:
        for seed in (args.seed or ARMS[arm]["seeds"]):
            jobs.append((arm, seed))

    print(f"{len(jobs)} run(s): {', '.join(f'{a}/s{s}' for a, s in jobs)}")
    print(f"cache: {CACHE}")
    t0 = time.time()
    failures: list[str] = []
    lock = threading.Lock()
    done = [0]

    def build(arm, seed):
        """(cmd, out_dir) for one job, or None when it should be skipped."""
        out_dir = OUT_ROOT / arm / f"seed{seed}"
        if args.skip_existing and list(out_dir.glob("*_summary.json")):
            return None
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(RUNNER),
            "--cache-dir", str(CACHE),
            "--out-dir", str(out_dir),
            "--experiment-id", f"{STUDY}__{arm}__seed{seed}",
            "--generations", str(args.generations),
            "--max-min", str(args.max_min),
            "--seed", str(seed),
            "--study-dir", str(STUDY_DIR),
        ]
        for ov in ARMS[arm]["overrides"]:
            cmd += ["--config-override", ov]
        for gp in ARMS[arm].get("generator", []):
            cmd += ["--generator-param", gp]
        return cmd, out_dir

    def execute(job):
        arm, seed = job
        built = build(arm, seed)
        if built is None:
            with lock:
                done[0] += 1
                print(f"[{done[0]}/{len(jobs)}] {arm}/seed{seed} — already done, skipped",
                      flush=True)
            return
        cmd, out_dir = built
        if args.dry_run:
            with lock:
                done[0] += 1
                print(f"[{done[0]}/{len(jobs)}] {arm}/seed{seed}\n$ " + " ".join(cmd),
                      flush=True)
            return
        # Stagger starts so the best-effort runs.db registration (which happens
        # at the START of each run) does not collide across workers.
        with lock:
            time.sleep(2.0)
        log = out_dir / "run.log"
        t = time.time()
        with open(log, "w") as fh:
            r = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        with lock:
            done[0] += 1
            mark = "ok " if r.returncode == 0 else "FAIL"
            print(f"[{done[0]}/{len(jobs)}] {mark} {arm}/seed{seed} "
                  f"({(time.time()-t)/60:.1f} min)  log: {log.relative_to(REPO)}",
                  flush=True)
            if r.returncode != 0:
                failures.append(f"{arm}/seed{seed}")

    n = max(1, args.jobs)
    print(f"running with --jobs {n}\n", flush=True)
    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(execute, jobs))

    mins = (time.time() - t0) / 60
    print(f"\ndone in {mins:.1f} min; {len(failures)} failure(s)"
          + (": " + ", ".join(failures) if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
