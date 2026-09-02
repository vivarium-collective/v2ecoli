"""Runner for the replisome-gate-sufficiency study.

A thin wrapper over ``scripts/run_condition_multigen_parquet.py``, the
workspace's existing multi-generation runner, invoked once per (arm, seed).
Nothing about the cell cycle, division or emitter handling is reimplemented
here, so this study cannot silently diverge from how every other multigen run
in the workspace behaves.

Three arms, all from ONE cache, differing only in runtime config_overrides:

    mechanistic     mechanistic_replisome=true
                    The gate requires >= 6 trimers and >= 2 monomers per oriC.
                    8 seeds.

    permissive      mechanistic_replisome=false
                    Initiation unconditional. Positive control: should complete
                    regardless of subunit supply. 3 seeds.

    dnag-ablation   mechanistic_replisome=true, with EG10239-MONOMER[c] dropped
                    from replisome_monomers_subunits so the gate no longer
                    requires DnaG. Discriminating control for the
                    limiting-resource claim: if DnaG is what the gate waits for,
                    not requiring it should relieve the stall. 5 seeds.

The ablation is applied through the declared ``config_overrides`` parameter, not
by editing the model or building a second cache, so all three arms start from a
byte-identical initial state and differ only in the runtime gate.

Note the == to >= correction is an UNCONDITIONAL patch on this branch
(chromosome_replication.py, commit db3986f3), so there is no ==-arm here. The
==/>= contrast is against the recorded result of mechanistic-replisome-arrest.

Usage::

    python workspace/studies/replisome-gate-sufficiency/sims/run.py
    python .../run.py --arm mechanistic          # one arm
    python .../run.py --generations 12 --dry-run
    python .../run.py --skip-existing            # resume a partial sweep
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
MONOMERS = "ecoli-chromosome-replication.replisome_monomers_subunits"
# The gate's monomer list with DnaG (EG10239-MONOMER[c]) removed.
MONOMERS_NO_DNAG = '["CPLX0-3621[c]", "EG11500-MONOMER[c]", "EG11412-MONOMER[c]"]'

ARMS = {
    "mechanistic": {"seeds": [0, 1, 2, 3, 4, 5, 6, 7],
                    "overrides": [f"{GATE}=true"]},
    "permissive": {"seeds": [0, 1, 2],
                   "overrides": [f"{GATE}=false"]},
    "dnag-ablation": {"seeds": [0, 1, 2, 3, 4],
                      "overrides": [f"{GATE}=true", f"{MONOMERS}={MONOMERS_NO_DNAG}"]},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=sorted(ARMS), action="append",
                    help="run only this arm (repeatable); default all")
    ap.add_argument("--seed", type=int, action="append",
                    help="run only this seed (repeatable); default the arm's seeds")
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--max-min", type=float, default=200.0)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip an (arm, seed) whose summary json already exists")
    ap.add_argument("--jobs", "-j", type=int, default=1,
                    help="run this many simulations concurrently (default 1). "
                         "Each run is single-threaded and holds ~1 GB RSS, so "
                         "the practical ceiling is min(cores, free_GB - 2).")
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
