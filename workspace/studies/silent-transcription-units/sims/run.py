"""Runner for the silent-transcription-units study.

Thin wrapper over scripts/run_condition_multigen_parquet.py, the workspace's
existing multi-generation runner, invoked once per seed. Nothing about the cell
cycle, division or emitter handling is reimplemented here.

3 seeds x 3 generations, default (permissive) settings. No config_overrides: the
question is about baseline transcription, so nothing is perturbed. The
transcript elongation listener is emitted by default, which is the primary
observable.

Sized to falsify, not to estimate. The null test only needs to observe a single
non-zero transcript in the cohort to fail, so a replication floor suffices; this
does NOT support a powered claim about transcription rates.

Deliberately NOT run at design time. This study is designed and pre-registered
first, and executed only afterwards.

Usage::

    python workspace/studies/silent-transcription-units/sims/run.py --dry-run
    python .../run.py --jobs 3 --skip-existing
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
SEEDS = [0, 1, 2]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, action="append")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--max-min", type=float, default=200.0)
    ap.add_argument("--jobs", "-j", type=int, default=3,
                    help="concurrent runs; each holds ~1 GB RSS")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    seeds = args.seed or SEEDS
    lock = threading.Lock()
    done = [0]
    failures: list[str] = []
    t0 = time.time()
    print(f"{len(seeds)} run(s): seeds {seeds}\ncache: {CACHE}\n"
          f"running with --jobs {max(1, args.jobs)}\n", flush=True)

    def execute(seed: int) -> None:
        out_dir = OUT_ROOT / f"seed{seed}"
        if args.skip_existing and list(out_dir.glob("*_summary.json")):
            with lock:
                done[0] += 1
                print(f"[{done[0]}/{len(seeds)}] seed{seed} — already done, skipped",
                      flush=True)
            return
        cmd = [sys.executable, str(RUNNER),
               "--cache-dir", str(CACHE), "--out-dir", str(out_dir),
               "--experiment-id", f"{STUDY}__seed{seed}",
               "--generations", str(args.generations),
               "--max-min", str(args.max_min), "--seed", str(seed),
               "--study-dir", str(STUDY_DIR)]
        if args.dry_run:
            # A dry run must be side-effect-free: create nothing.
            with lock:
                done[0] += 1
                print(f"[{done[0]}/{len(seeds)}] seed{seed}\n$ " + " ".join(cmd),
                      flush=True)
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        # Stagger: the runner registers into a shared runs.db at the START of a
        # run, best-effort, so overlapping starts risk only a provenance row --
        # but staggering removes even that.
        with lock:
            time.sleep(2.0)
        log = out_dir / "run.log"
        t = time.time()
        with open(log, "w") as fh:
            r = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
        with lock:
            done[0] += 1
            mark = "ok " if r.returncode == 0 else "FAIL"
            print(f"[{done[0]}/{len(seeds)}] {mark} seed{seed} "
                  f"({(time.time()-t)/60:.1f} min)  log: {log.relative_to(REPO)}",
                  flush=True)
            if r.returncode != 0:
                failures.append(f"seed{seed}")

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        list(pool.map(execute, seeds))

    print(f"\ndone in {(time.time()-t0)/60:.1f} min; {len(failures)} failure(s)"
          + (": " + ", ".join(failures) if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
