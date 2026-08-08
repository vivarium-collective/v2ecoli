"""Per-process divergence attribution: v2ecoli vs genuine vEcoli on one condition.

Steps BOTH engines for N ticks from their condition-specific initial states, accumulates
each process's NET bulk delta per molecule, maps molecules by NAME (the two engines index
bulk differently), aggregates per corresponding process, and ranks processes by how much
their cumulative contribution DIFFERS between the engines. Answers: "which process drives
v2ecoli's <condition> trajectory away from vEcoli's, and via which molecules."

    PYTHONHASHSEED=0 PYTHONPATH=$PWD .venv/bin/python scripts/attribute_divergence.py \
        --condition acetate --steps 2000

The vEcoli side runs its own vivarium Engine (VivariumEcoliProcess's engine); the v2ecoli
side is the native pbg baseline composite. Both on their full condition-complete ParCa.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

UP_CACHE = "out/compare_harness/vecoli_parca"
V2_CACHE = "out/cache_full"


def _canon(name: str) -> str:
    """Canonical process name so v2 ('ecoli-metabolism') and vEcoli ('Metabolism')
    correspond. Strip the 'ecoli-' edge prefix, split on -/_ , PascalCase."""
    n = name[6:] if name.startswith("ecoli-") else name
    if "-" in n or "_" in n:
        return "".join(w.capitalize() for w in n.replace("_", "-").split("-"))
    return n


def _bulk_ids(state) -> np.ndarray:
    arr = np.asarray(state["bulk"])
    return np.array([str(x) for x in arr["id"]])


def _accumulate(contrib: dict, name: str, ids_len: int, bulk_update) -> None:
    """Add a process's emitted bulk delta (list of (idx, val)) into contrib[name]."""
    if bulk_update is None:
        return
    if name not in contrib:
        contrib[name] = np.zeros(ids_len)
    try:
        for e in (bulk_update if isinstance(bulk_update, (list, tuple)) else []):
            if isinstance(e, (list, tuple)) and len(e) == 2:
                idx, val = e
                contrib[name][np.atleast_1d(idx)] += np.atleast_1d(val)
    except Exception:
        pass


def capture_vecoli(condition: str, steps: int) -> tuple[dict, np.ndarray]:
    """{inner_process_name: net_bulk_delta[ids]} for the genuine vEcoli vivarium engine."""
    from v2ecoli.library.vivarium_ecoli_engine import build_vivarium_ecoli
    h = build_vivarium_ecoli(sim_data_path=f"{UP_CACHE}/simData.cPickle", condition=condition,
                             seed=0, exclude_processes=["monomer_counts_listener"])
    eng = h.engine
    ids = _bulk_ids(eng.state.get_value())
    contrib: dict = {}

    def proc_name(step):
        inner = getattr(step, "parameters", {}).get("process") if isinstance(
            getattr(step, "parameters", None), dict) else None
        return _canon(type(inner).__name__ if inner is not None else type(step).__name__)

    def hook(step):
        if not hasattr(step, "next_update"):
            return
        nm = proc_name(step)
        orig = step.next_update
        def traced(ts, states):
            u = orig(ts, states)
            _accumulate(contrib, nm, len(ids), u.get("bulk") if isinstance(u, dict) else None)
            return u
        step.next_update = traced

    def walk(d):
        if isinstance(d, dict):
            for v in d.values():
                walk(v)
        elif hasattr(d, "next_update"):
            hook(d)
    walk(eng.steps)
    for p in (eng.process_paths or {}).values():
        hook(p)
    eng.run_for(float(steps))
    return contrib, ids


def capture_v2(condition: str, steps: int) -> tuple[dict, np.ndarray]:
    """{process_name: net_bulk_delta[ids]} for the native v2ecoli baseline composite."""
    from scripts.run_comparison_ensemble import _build_v2ecoli
    composite = _build_v2ecoli(seed=0, condition=condition, cache_dir=V2_CACHE, overrides=None)
    state = composite.state
    agent = state["agents"]["0"] if "agents" in state else state
    ids = _bulk_ids(agent)
    contrib: dict = {}

    # v2 pbg processes are nodes carrying an 'instance'; hook each instance.update.
    def hook(name, inst):
        if not hasattr(inst, "update"):
            return
        orig = inst.update
        def traced(*a, **k):  # pass args through — emitters/steps have varied signatures
            u = orig(*a, **k)
            _accumulate(contrib, name, len(ids), u.get("bulk") if isinstance(u, dict) else None)
            return u
        inst.update = traced

    seen = set()
    def walk(d):
        if isinstance(d, dict):
            inst = d.get("instance")
            if inst is not None and id(inst) not in seen and hasattr(inst, "update"):
                seen.add(id(inst))
                hook(_canon(getattr(inst, "name", type(inst).__name__)), inst)
            for v in d.values():
                walk(v)
    walk(agent)
    composite.run(steps)
    return contrib, ids


def _per_process_total(contrib: dict, ids: np.ndarray) -> dict:
    """{proc: {mol_name: net_delta}} keyed by molecule NAME."""
    out = {}
    for proc, vec in contrib.items():
        nz = np.nonzero(vec)[0]
        out[proc] = {ids[i]: float(vec[i]) for i in nz}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", default="acetate")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("V2E_VECOLI_DIR", "/Users/eranagmon/code/vEcoli-upstream")

    print(f"[attribute] capturing vEcoli ({args.condition}, {args.steps} steps) ...", flush=True)
    ve_c, ve_ids = capture_vecoli(args.condition, args.steps)
    print(f"[attribute] capturing v2ecoli ({args.condition}, {args.steps} steps) ...", flush=True)
    v2_c, v2_ids = capture_v2(args.condition, args.steps)

    ve = _per_process_total(ve_c, ve_ids)
    v2 = _per_process_total(v2_c, v2_ids)

    # rank corresponding processes by total |Δ net contribution| over shared molecules
    procs = set(ve) | set(v2)
    rows = []
    for p in procs:
        a, b = v2.get(p, {}), ve.get(p, {})
        mols = set(a) | set(b)
        diff = {m: a.get(m, 0.0) - b.get(m, 0.0) for m in mols}
        total = sum(abs(d) for d in diff.values())
        rows.append((p, total, diff, p in v2, p in ve))
    rows.sort(key=lambda r: -r[1])

    print(f"\n=== per-process divergence on {args.condition} (|Σ v2 − vEcoli net bulk Δ|) ===")
    print(f"{'process':30s} {'divergence':>12s}  present")
    for p, total, diff, inv2, inve in rows[:args.top]:
        pres = ("v2" if inv2 else "  ") + "/" + ("vE" if inve else "  ")
        top_mol = sorted(diff.items(), key=lambda x: -abs(x[1]))[:3]
        molstr = ", ".join(f"{m}:{d:+.2g}" for m, d in top_mol)
        print(f"{p:30s} {total:12.4g}  [{pres}]  {molstr}")


if __name__ == "__main__":
    main()
