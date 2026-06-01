#!/usr/bin/env python
"""Parity gate for behavior-preserving composite refactors.

The v2ecoli ``baseline`` composite is bit-deterministic run-to-run, so a
refactor that claims to preserve behavior must reproduce a deep state
signature exactly. This script is the standard gate for that claim and MUST
be run after any change to derivers, processes, port schemas, or the
execution flow.

Two independent checks — BOTH are required, because they cover different
failure modes:

  1. signature (default): build with a NULL emitter, run N sim-seconds, emit a
     deep signature (cell_mass, dry_mass, bulk sum, recursive listeners digest,
     per-type unique-molecule _entryState counts). Compare to a golden with
     ``--compare``. Catches behavior/math/scheduling drift.

  2. ``--build-check``: build with the DEFAULT (real) emitter and assert it
     constructs. Catches emitter-schema *resolve* failures that the null
     emitter hides (e.g. stripping ``overwrite[]`` so Quantity can't resolve
     against the emitter's Float wiring). The signature check alone WILL miss
     this — see the units-on-ports gotcha.

Usage:
    # capture a golden (e.g. from a clean origin/main worktree):
    python scripts/parity_check.py --seconds 120 --out golden.json

    # gate a change: signature must match golden AND real-emitter build must pass
    python scripts/parity_check.py --seconds 120 --compare golden.json --build-check

Exit code is non-zero if the signature differs from --compare or the
--build-check build fails, so it can gate CI / a commit step.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402

from v2ecoli import build_composite  # noqa: E402
from v2ecoli.composites._helpers import set_null_emitter_override  # noqa: E402


def _num(x):
    """Coerce a leaf to finite float magnitude, else None."""
    try:
        if hasattr(x, "magnitude"):
            x = x.magnitude
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)):
            f = float(x)
            return f if math.isfinite(f) else None
    except Exception:
        return None
    return None


def _walk(obj, acc):
    """Recursively accumulate a stable numeric digest of nested state."""
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            _walk(obj[k], acc)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _walk(v, acc)
    elif isinstance(obj, np.ndarray):
        if obj.dtype.kind in "fiu":
            arr = obj.astype("float64", copy=False)
            arr = arr[np.isfinite(arr)]
            acc["s"] += float(arr.sum())
            acc["n"] += int(arr.size)
    else:
        f = _num(obj)
        if f is not None:
            acc["s"] += f
            acc["n"] += 1


def signature(composite):
    state = composite.state or {}
    agent = (state.get("agents") or {}).get("0") or {}
    mass = (agent.get("listeners") or {}).get("mass") or {}

    sig = {"cell_mass": _num(mass.get("cell_mass")),
           "dry_mass": _num(mass.get("dry_mass"))}

    bulk = agent.get("bulk")
    if isinstance(bulk, np.ndarray):
        cn = bulk["count"] if bulk.dtype.names and "count" in bulk.dtype.names else bulk
        sig["bulk_sum"] = float(np.asarray(cn).astype("float64", copy=False).sum())

    acc = {"s": 0.0, "n": 0}
    _walk(agent.get("listeners") or {}, acc)
    sig["listeners_sum"] = acc["s"]
    sig["listeners_n"] = acc["n"]

    unique = agent.get("unique") or {}
    sig["unique"] = {
        k: int(np.asarray(unique[k]["_entryState"]).astype("int64", copy=False).sum())
        for k in sorted(unique)
        if isinstance(unique[k], np.ndarray)
        and unique[k].dtype.names and "_entryState" in unique[k].dtype.names
    }
    return sig


def run_signature(composite_name, seconds, chunk):
    set_null_emitter_override(True)
    c = build_composite(composite_name)
    t = 0
    while t < seconds:
        c.run(min(chunk, seconds - t))
        t += chunk
        if ((c.state.get("agents") or {}).get("0")) is None:
            break
    return signature(c)


def build_check(composite_name, cache_dir):
    """Build under the DEFAULT (real) emitter; raise on failure."""
    set_null_emitter_override(False)
    build_composite(composite_name, cache_dir=cache_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--composite", default="baseline")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--cache-dir", default="out/cache")
    ap.add_argument("--out", help="write signature JSON here")
    ap.add_argument("--compare", help="golden JSON to diff against (fail on mismatch)")
    ap.add_argument("--build-check", action="store_true",
                    help="also build under the real emitter (fail on resolve error)")
    args = ap.parse_args()

    failed = False

    if args.build_check:
        try:
            build_check(args.composite, args.cache_dir)
            print(f"[build-check] {args.composite}: real-emitter build OK")
        except Exception as e:
            print(f"[build-check] {args.composite}: FAILED — {type(e).__name__}: {e}")
            failed = True

    sig = run_signature(args.composite, args.seconds, args.chunk)
    blob = json.dumps(sig, sort_keys=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write(blob)
        print(f"[signature] wrote {args.out}")
    print(blob)

    if args.compare:
        with open(args.compare) as f:
            golden = json.load(f)
        diffs = [k for k in golden if golden[k] != sig.get(k)]
        if diffs:
            failed = True
            print(f"[compare] MISMATCH vs {args.compare}:")
            for k in diffs:
                print(f"  {k}: {golden[k]} -> {sig.get(k)}")
        else:
            print(f"[compare] byte-identical to {args.compare}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
