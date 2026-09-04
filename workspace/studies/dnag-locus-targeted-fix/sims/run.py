"""Runner for dnag-locus-targeted-fix.

Two arms differing ONLY in which cache they build from:

    lexa-edge-removed   out/cache_lexafix -- one delta_prob entry zeroed,
                        row TU00352 column PC00010 (LexA). basal_prob and all
                        1461 other edges identical to the control.
    unmodified-control  out/cache -- fingerprint ea05ac9b23b1d843.

Unlike every earlier study here the arms use DIFFERENT caches, so the control is
re-run on matched seeds rather than compared against existing numbers (req-2).

Concurrency: one sim costs ~5.5 GB RSS and a 16 GB machine fits 2 TOTAL across
all worktrees. Check `ps` before raising -j.

Usage::

    python workspace/studies/dnag-locus-targeted-fix/sims/run.py -j 1
    python .../run.py --skip-existing
"""
from __future__ import annotations
import argparse, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
STUDY_DIR = Path(__file__).resolve().parents[1]
STUDY = STUDY_DIR.name
RUNNER = REPO / "scripts/run_condition_multigen_parquet.py"
OUT_ROOT = REPO / "out" / STUDY
GATE = "ecoli-chromosome-replication.mechanistic_replisome=true"

ARMS = {
    "lexa-edge-removed":  {"seeds": [0, 1, 2], "cache": REPO / "out/cache_lexafix"},
    "unmodified-control": {"seeds": [0, 1, 2], "cache": REPO / "out/cache"},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARMS), action="append")
    ap.add_argument("--seed", type=int, action="append")
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--max-min", type=float, default=200.0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--jobs", "-j", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    jobs = [(a, s) for a in (args.arm or list(ARMS))
            for s in (args.seed or ARMS[a]["seeds"])]
    print(f"{len(jobs)} run(s); jobs={max(1, args.jobs)}", flush=True)
    lock, done, failures = threading.Lock(), [0], []

    def execute(job):
        arm, seed = job
        out_dir = OUT_ROOT / arm / f"seed{seed}"
        if args.skip_existing and list(out_dir.glob("*_summary.json")):
            with lock:
                done[0] += 1
                print(f"[{done[0]}/{len(jobs)}] {arm}/seed{seed} — done, skipped", flush=True)
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(RUNNER),
               "--cache-dir", str(ARMS[arm]["cache"]),
               "--out-dir", str(out_dir),
               "--experiment-id", f"{STUDY}__{arm}__seed{seed}",
               "--generations", str(args.generations),
               "--max-min", str(args.max_min),
               "--seed", str(seed),
               "--study-dir", str(STUDY_DIR),
               "--config-override", GATE]
        if args.dry_run:
            with lock:
                done[0] += 1
                print(f"[{done[0]}/{len(jobs)}] {arm}/seed{seed}\n$ " + " ".join(cmd), flush=True)
            return
        with lock:
            time.sleep(2.0)
        t = time.time()
        with open(out_dir / "run.log", "w") as fh:
            r = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        with lock:
            done[0] += 1
            print(f"[{done[0]}/{len(jobs)}] {'ok  ' if r.returncode == 0 else 'FAIL'} "
                  f"{arm}/seed{seed} ({(time.time()-t)/60:.1f} min)", flush=True)
            if r.returncode:
                failures.append(f"{arm}/seed{seed}")

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        list(pool.map(execute, jobs))
    print(f"\n{len(failures)} failure(s)" + (": " + ", ".join(failures) if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
