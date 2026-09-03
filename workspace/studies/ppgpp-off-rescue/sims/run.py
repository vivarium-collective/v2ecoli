"""Runner for the ppgpp-off-rescue study.

A thin wrapper over ``scripts/run_condition_multigen_parquet.py``, the
workspace's existing multi-generation runner, invoked once per (arm, seed) --
the same pattern as replisome-gate-sufficiency, so this study cannot silently
diverge from how every other multigen run in the workspace behaves.

Two arms, both from ONE cache, both with the mechanistic gate ON, differing
ONLY in whether ppGpp-dependent transcription regulation is assembled:

    ppgpp-off       ppgpp_regulation=false (a GENERATOR parameter, not a
                    process config override). Removes the `ppgpp-initiation`
                    step, so nothing writes `ppgpp_state.basal_prob`;
                    TranscriptInitiation's `has_ppgpp` test then falls to the
                    ParCa-fitted `self.basal_prob`. 3 seeds.

    ppgpp-on        Unmodified. The paired positive control: it should still
                    stall at generation 4-5. If it does not, something other
                    than the intervention changed and the contrast is void.
                    3 seeds.

Why a generator parameter and not a config override: `ppgpp_regulation`
decides WHICH processes are built, not the value of a key on an already-built
process. It is in ecoli_baseline's per-cell threaded kwarg set, so the switch
holds across division instead of reverting after generation 1 (verified
2026-09-03: the `ppgpp-initiation` step is absent from the composite with the
flag off, and `ppgpp_state.basal_prob` is empty at t=0 in BOTH arms -- the
ppGpp step populates it during the run, so removing the step leaves it empty
for the whole lineage).

Runs from scratch, no --resume-dill: a dill written under ppGpp-on could carry
a populated `ppgpp_state`, which would make the off arm silently behave like
the control.

Usage::

    python workspace/studies/ppgpp-off-rescue/sims/run.py -j 2
    python .../run.py --arm ppgpp-off             # one arm
    python .../run.py --skip-existing             # resume a partial sweep
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

# Both arms carry the mechanistic gate; `generator` entries become
# --generator-param (top-level composite-generator kwargs), which is what
# makes the off arm a different BUILD rather than a different parameter value.
ARMS = {
    "ppgpp-off": {"seeds": [0, 1, 2],
                  "overrides": [f"{GATE}=true"],
                  "generator": ["ppgpp_regulation=false"]},
    "ppgpp-on": {"seeds": [0, 1, 2],
                 "overrides": [f"{GATE}=true"],
                 "generator": []},
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
    ap.add_argument("--jobs", "-j", type=int, default=2,
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
