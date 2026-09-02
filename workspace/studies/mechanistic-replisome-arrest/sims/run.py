"""Baseline runner for the mechanistic-replisome-arrest study.

Deliberately a thin wrapper. The simulation itself is
``scripts/run_condition_multigen_parquet.py`` — the workspace's existing
multi-generation runner — invoked once per arm. Nothing about the cell
cycle, division, or emitter handling is reimplemented here, so this study
cannot silently diverge from how every other multigen run in the
workspace behaves.

Two arms, identical but for one config override:

    mechanistic   mechanistic_replisome=True    initiation must acquire
                                                6 replisome trimers + 2
                                                monomers per oriC
                                                (chromosome_replication.py:453)
    permissive    mechanistic_replisome=False   initiation is unconditional

Both arms share ONE cache. The flag is applied through the composite
generator's declared ``config_overrides`` parameter
("<process>.<key>": value), not by building two caches. That matters for
the comparison: the two arms then start from a byte-identical initial
state and differ only in the runtime gating, so any divergence is
attributable to the gate rather than to two separately-fitted caches.

Note this is a deliberate choice, not the only route. ``mechanistic_replisome``
is also a LoadSimData/cache-build parameter, where it additionally affects
the *initial* state (replisome protein mass on pre-existing replisomes).
That path is not used here — and is currently broken anyway:
``initial_conditions.py:597`` converts a molar mass (g/mol) directly to
fg/count, which raises DimensionalityError. The correct form lives in
``sim_data.py:729`` (to fg/mol, then divide by Avogadro). Dead code today
because nothing builds a mechanistic cache; recorded as a finding rather
than fixed here.

Usage::

    python workspace/studies/mechanistic-replisome-arrest/sims/run.py
    python .../run.py --generations 12 --seed 0
    python .../run.py --arm mechanistic          # one arm only
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
STUDY_DIR = Path(__file__).resolve().parents[1]

OVERRIDE_KEY = "ecoli-chromosome-replication.mechanistic_replisome"
ARMS = {
    "mechanistic": True,
    "permissive": False,
}


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def simulate(arm: str, cache_dir: Path, out_dir: Path, generations: int,
             seed: int, max_min: float, python: str) -> None:
    run([
        python, "scripts/run_condition_multigen_parquet.py",
        "--cache-dir", str(cache_dir),
        "--out-dir", str(out_dir),
        "--experiment-id", f"mechanistic-replisome-arrest__{arm}__seed{seed}",
        "--generations", str(generations),
        "--seed", str(seed),
        "--max-min", str(max_min),
        "--study-dir", str(STUDY_DIR),
        "--config-override", f"{OVERRIDE_KEY}={str(ARMS[arm]).lower()}",
    ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=12,
                    help="Generations per arm (default 12 — the length at "
                         "which the arrest was originally observed).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-min", type=float, default=200.0,
                    help="Per-generation wall-clock ceiling, passed through to "
                         "the multigen runner.")
    ap.add_argument("--cache-dir", default="out/cache",
                    help="Shared cache for both arms (default out/cache).")
    ap.add_argument("--arm", choices=sorted(ARMS), action="append",
                    help="Run only this arm. Repeatable. Default: both.")
    ap.add_argument("--out-root", default=None,
                    help="Where run outputs land (default out/<study slug>).")
    args = ap.parse_args(argv)

    arms = args.arm or sorted(ARMS)
    python = sys.executable
    cache_dir = (REPO / args.cache_dir if not Path(args.cache_dir).is_absolute()
                 else Path(args.cache_dir))
    if not cache_dir.exists():
        raise SystemExit(
            f"cache not found: {cache_dir}\n"
            f"  build it first:  python scripts/build_cache.py --cache {args.cache_dir}")

    out_root = (Path(args.out_root) if args.out_root
                else REPO / "out" / "mechanistic-replisome-arrest")
    out_root.mkdir(parents=True, exist_ok=True)

    print("study : mechanistic-replisome-arrest")
    print(f"arms  : {', '.join(arms)}")
    print(f"gens  : {args.generations}   seed: {args.seed}")
    print(f"cache : {cache_dir}  (shared by both arms)")
    print(f"out   : {out_root}")

    t0 = time.time()
    for arm in arms:
        out_dir = out_root / arm
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 62}\nARM: {arm}  ({OVERRIDE_KEY}={ARMS[arm]})\n{'=' * 62}")
        simulate(arm, cache_dir, out_dir, args.generations, args.seed,
                 args.max_min, python)

    print(f"\nboth arms finished in {(time.time() - t0) / 60:.1f} min")
    print(f"outputs under {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
