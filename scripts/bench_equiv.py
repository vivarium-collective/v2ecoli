"""Side-by-side equivalence + speed check for in-place perf changes.

Runs the baseline composite for `--duration` seconds and dumps a small
fingerprint dict (dry/cell mass + a few listener fields at fixed sim
times) to a JSON file. Run before and after a change and diff. Exits
non-zero if any key in `--baseline` differs from the current run by more
than the per-key tolerance.

Usage:
    .venv/bin/python scripts/bench_equiv.py --out before.json --duration 600
    # ... make change ...
    .venv/bin/python scripts/bench_equiv.py --out after.json --duration 600 \\
        --baseline before.json
"""

from __future__ import annotations
import argparse
import json
import sys
import time
import warnings
from pathlib import Path


def _grab(state, *path, default=None):
    cur = state
    for p in path:
        if cur is None:
            return default
        cur = cur.get(p) if isinstance(cur, dict) else None
    return default if cur is None else (
        float(cur) if isinstance(cur, (int, float)) else cur
    )


def fingerprint(composite) -> dict:
    cell = composite.state.get("agents", {}).get("0", {})
    mass = cell.get("listeners", {}).get("mass", {})
    ribo = cell.get("listeners", {}).get("ribosome_data", {})
    fba = cell.get("listeners", {}).get("fba_results", {})
    out = {
        "dry_mass": _grab(mass, "dry_mass"),
        "cell_mass": _grab(mass, "cell_mass"),
        "effective_elong_rate": _grab(ribo, "effective_elongation_rate"),
    }
    obj = fba.get("objective_value") if isinstance(fba, dict) else None
    if obj is not None:
        try:
            out["fba_objective"] = float(obj)
        except (TypeError, ValueError):
            pass
    return out


def run_capture(duration: int, sample_every: int, composite_name: str) -> dict:
    warnings.filterwarnings("ignore")
    from v2ecoli import build_composite
    c = build_composite(composite_name)
    samples = {}
    sim_time = 0
    while sim_time < duration:
        chunk = min(sample_every, duration - sim_time)
        c.run(chunk)
        sim_time += chunk
        samples[str(sim_time)] = fingerprint(c)
    return samples


def diff_samples(before: dict, after: dict, tol_rel: float, tol_abs: float):
    failures = []
    for t, b in before.items():
        a = after.get(t)
        if a is None:
            failures.append(f"missing t={t}")
            continue
        for k, bv in b.items():
            av = a.get(k)
            if bv is None or av is None:
                continue
            if not isinstance(bv, (int, float)):
                continue
            diff = abs(av - bv)
            ref = max(abs(bv), 1.0)
            rel = diff / ref
            if diff > tol_abs and rel > tol_rel:
                failures.append(
                    f"t={t} {k}: before={bv:.6g} after={av:.6g} "
                    f"(absdiff={diff:.3g}, rel={rel:.2%})"
                )
    return failures


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=600)
    p.add_argument("--sample-every", type=int, default=120)
    p.add_argument("--out", required=True)
    p.add_argument("--baseline", default=None,
                   help="If given, compare to this prior fingerprint file.")
    p.add_argument("--tol-rel", type=float, default=0.01,
                   help="Allowed relative diff per field (default 1%).")
    p.add_argument("--tol-abs", type=float, default=1e-6)
    p.add_argument("--composite", default="baseline")
    args = p.parse_args()

    t0 = time.perf_counter()
    samples = run_capture(args.duration, args.sample_every, args.composite)
    wall = time.perf_counter() - t0
    payload = {
        "wall_seconds": wall,
        "duration_simulated": args.duration,
        "samples": samples,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"wall: {wall:.2f}s  -> {args.out}")

    if args.baseline:
        base = json.loads(Path(args.baseline).read_text())
        fails = diff_samples(
            base["samples"], samples, args.tol_rel, args.tol_abs,
        )
        speedup = base["wall_seconds"] / wall
        print(f"speedup vs baseline: {speedup:.2f}x  "
              f"(was {base['wall_seconds']:.2f}s, now {wall:.2f}s)")
        if fails:
            print("EQUIVALENCE FAILURES:")
            for f in fails:
                print(" -", f)
            sys.exit(1)
        print(f"equivalence OK at tol_rel={args.tol_rel}")


if __name__ == "__main__":
    main()
