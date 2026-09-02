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
import time
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
    failures = []

    for i, (arm, seed) in enumerate(jobs, 1):
        out_dir = OUT_ROOT / arm / f"seed{seed}"
        exp_id = f"{STUDY}__{arm}__seed{seed}"
        if args.skip_existing and list(out_dir.glob("*_summary.json")):
            print(f"\n[{i}/{len(jobs)}] {arm}/seed{seed} — already done, skipped")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(RUNNER),
            "--cache-dir", str(CACHE),
            "--out-dir", str(out_dir),
            "--experiment-id", exp_id,
            "--generations", str(args.generations),
            "--max-min", str(args.max_min),
            "--seed", str(seed),
            "--study-dir", str(STUDY_DIR),
        ]
        for ov in ARMS[arm]["overrides"]:
            cmd += ["--config-override", ov]

        print(f"\n[{i}/{len(jobs)}] {arm}/seed{seed}")
        print("$ " + " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode})", file=sys.stderr)
            failures.append(f"{arm}/seed{seed}")

    mins = (time.time() - t0) / 60
    print(f"\ndone in {mins:.1f} min; {len(failures)} failure(s)"
          + (": " + ", ".join(failures) if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
