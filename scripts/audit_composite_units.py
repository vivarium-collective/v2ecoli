#!/usr/bin/env python3
"""Audit units coverage across the generated composites.

Units are declared on process port schemas as typed strings (float[uM],
quantity[float,fg], overwrite[array[float[1/s]]], ...) and carried inline in the
resolved composite STATE as `_type` annotations. This walks each composite's
resolved state (reports/composite-state/<id>.json), extracts dotted-path -> unit
via the real resolver (units_resolver.unit_from_type), and compares the
baseline-family composites so we can see whether path-remapping (biological) or
aggregation (population/time_varying) drops unit annotations.

Read-only. Usage:  .venv/bin/python scripts/audit_composite_units.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def walk_units(node, core, unit_from_type, prefix="", out=None):
    """Recursively collect dotted-path -> unit from a resolved-state tree."""
    if out is None:
        out = {}
    if isinstance(node, str):
        # Bare string port type, e.g. "inplace_dict[float[mM]]" or "float[fg]".
        u = unit_from_type(node, core)
        if u and prefix:
            out[prefix] = u
        return out
    if isinstance(node, dict):
        t = node.get("_type")
        if t is not None:
            u = unit_from_type(t, core)
            if u and prefix:
                out[prefix] = u
        for k, v in node.items():
            # Descend into port declarations (_inputs/_outputs carry the unit
            # types) but skip pure leaf-metadata keys (_type/_default/_value/...).
            if isinstance(k, str) and k.startswith("_") and k not in ("_inputs", "_outputs"):
                continue
            child = prefix if k in ("_inputs", "_outputs") else (f"{prefix}.{k}" if prefix else str(k))
            walk_units(v, core, unit_from_type, child, out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk_units(v, core, unit_from_type, f"{prefix}.{i}", out)
    return out


def leafkey(path: str) -> str:
    """Last two dotted segments — stable across composite path-remapping."""
    parts = [p for p in path.split(".") if not p.isdigit()]
    return ".".join(parts[-2:])


def main() -> int:
    ws_root = Path.cwd()
    sys.path.insert(0, str(ws_root))
    from v2ecoli.core import build_core
    from v2ecoli.library.units_resolver import unit_from_type
    core = build_core()

    state_dir = ws_root / "reports" / "composite-state"
    files = sorted(state_dir.glob("v2ecoli.composites.*.json"))

    per_comp: dict[str, dict[str, str]] = {}
    for f in files:
        cid = f.stem
        try:
            state = json.loads(f.read_text()).get("state", {})
        except Exception as e:  # noqa: BLE001
            print(f"  skip {cid}: {e}")
            continue
        per_comp[cid] = walk_units(state, core, unit_from_type)

    # Coverage summary
    print(f"{'composite':52} {'unit-paths':>10} {'units':>6}")
    print("-" * 72)
    for cid in sorted(per_comp):
        idx = per_comp[cid]
        short = cid.replace("v2ecoli.composites.", "")
        print(f"{short:52} {len(idx):>10} {len(set(idx.values())):>6}")

    # Baseline-family propagation check: which baseline unit-readouts (by leaf
    # key) are preserved-with-units in each derived composite?
    base_id = "v2ecoli.composites.ecoli_baseline"
    base = per_comp.get(base_id) or per_comp.get(base_id + ".baseline") or {}
    base_leaves = {leafkey(p): u for p, u in base.items()}
    print(f"\nbaseline reference: {len(base)} unit-paths, {len(base_leaves)} distinct leaf-readouts")

    family = [c for c in per_comp
              if c != base_id and ("baseline" in c or "biological" in c
                                   or "population" in c or "varying" in c)]
    for cid in sorted(family):
        idx = per_comp[cid]
        leaves = {leafkey(p): u for p, u in idx.items()}
        missing = sorted(set(base_leaves) - set(leaves))
        short = cid.replace("v2ecoli.composites.", "")
        status = "OK (all baseline readouts carry units)" if not missing else \
                 f"MISSING {len(missing)} baseline readouts"
        print(f"\n  {short}: {status}")
        for m in missing[:25]:
            print(f"      - {m}  (baseline unit: {base_leaves[m]})")

    # Unit dimension breakdown for baseline (what the atlas enumerated)
    by_unit = collections.Counter(base.values())
    print("\nbaseline units in use:")
    for u, n in by_unit.most_common():
        print(f"  {u:18} x{n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
