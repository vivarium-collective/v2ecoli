"""Roll out v2ecoli baseline trajectories that CONTINUE across a division event,
following one daughter lineage across the reset — the extension sm-00 explicitly
did NOT do (it halts at division because "the single-agent assumption breaks
once daughters appear").

The reason sm-00 halts is not a modelling choice but a mechanical one: a single
``Composite`` cannot be ``run()`` past the division tick — the inner Division
step ``_remove``s the mother agent and ``_add``s two daughters, and the
composite's step scheduler then dereferences the vanished mother wiring and
crashes. v2ecoli already solved this for multigeneration runs in
``v2ecoli/workflow/lineage.py``: capture the surviving daughter's biological
state (``select_carry_daughter``) and REBUILD a fresh composite seeded with it
(``apply_carry_state`` + ``seed_mass_listener``). This sampler reuses exactly
that machinery to walk one lineage across a division.

For each trajectory it records, per 1-second transition, the broad observable
vector before/after (reusing observables.PanelLayout), plus a sidecar
``meta.npz`` with per-transition ``spans_division`` / ``tick`` / ``generation``
so the evaluator can split within-generation vs across-division rollout error.
The across-division transition is (mother's last pre-division state -> the
followed daughter's first post-division state); every other transition is a
within-generation step.

Usage:
    .venv/bin/python sample_through_division.py --seeds 0 1 2 3 \
        --max-steps 3200 --out <dir> --groups mass chromosome
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from observables import PanelLayout, build_spec  # noqa: E402

CACHE_DIR = "/Users/eranagmon/code/v2e-hsurrogate/out/cache"


def _cell_view(state: dict) -> dict:
    """A single-agent state view for PanelLayout.extract.

    Every composite we build (gen 0 via build_composite, daughters via the
    rebuild pattern) names its single live cell ``"0"``. Before division there
    is exactly one agent; we never extract from a mid-division state (the
    daughter is read only after its own fresh composite is rebuilt), so keying
    on ``"0"`` is unambiguous.
    """
    agents = state["agents"]
    if "0" in agents:
        return {"agents": {"0": agents["0"]}}
    # defensive: sole surviving agent under any key
    k = sorted(agents.keys())[0]
    return {"agents": {k: agents[k]}}


def _build_gen0(seed, cache_dir):
    from v2ecoli import build_composite
    return build_composite("ecoli_baseline", seed=seed, cache_dir=cache_dir, emitter="null")


def _build_daughter(gen_seed, carry_state, cache_dir):
    """Rebuild a fresh single-cell composite seeded with a carried daughter
    state — the ``LineageProcess._build_generation`` pattern, verbatim."""
    from process_bigraph import Composite
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline, seed_mass_listener
    from v2ecoli.workflow.lineage import apply_carry_state

    core = build_core()
    doc = baseline(core=core, seed=gen_seed, cache_dir=cache_dir, emitter="null")
    agent = doc["state"]["agents"]["0"]
    apply_carry_state(agent, carry_state)
    agent["listeners"]["mass"] = {"dry_mass": 0.0, "cell_mass": 0.0}
    seed_mass_listener(agent, core)
    return Composite(doc, core=core)


def _run_one_step(comp, agent_id="0"):
    """Advance the composite 1 s. Return (divided, agents_before, agents_now,
    mother_snapshot)."""
    from v2ecoli.workflow.lineage import select_carry_daughter  # noqa: F401 (used by caller)

    agents = comp.state.get("agents") or {}
    agents_before = set(agents.keys())
    mother = agents.get(agent_id) or next(iter(agents.values()), {})
    mother_snapshot = (
        {k: mother.get(k) for k in ("bulk", "unique", "environment", "boundary")}
        if isinstance(mother, dict) else None)
    divided = False
    try:
        comp.run(1)
    except Exception as e:  # noqa: BLE001 — division may surface as an exception
        msg = str(e).lower()
        if "divid" in msg or "division" in msg:
            divided = True
        else:
            raise
    agents_now = comp.state.get("agents") or {}
    if agents_before and set(agents_now.keys()) != agents_before:
        divided = True
    survivor = agents_now.get(agent_id) or next(iter(agents_now.values()), {})
    if isinstance(survivor, dict) and survivor.get("divide"):
        divided = True
    return divided, agents_before, agents_now, mother_snapshot


def sample_trajectory(seed, max_steps, layout, cache_dir, groups, tail=120):
    """Step one baseline lineage THROUGH one division; follow the '..0' daughter.

    Returns (Xs, Ys, spans, ticks, gens, layout, div_tick).
    """
    from v2ecoli.workflow.lineage import select_carry_daughter

    comp = _build_gen0(seed, cache_dir)
    comp.run(1)
    if layout is None:
        layout = PanelLayout.discover(comp.state, groups=tuple(groups))

    prev = layout.extract(_cell_view(comp.state))
    Xs, Ys, spans, ticks, gens = [], [], [], [], []
    div_tick = None

    for t in range(max_steps):
        divided, before, now, snap = _run_one_step(comp)
        if not divided:
            cur = layout.extract(_cell_view(comp.state))
            Xs.append(prev); Ys.append(cur); spans.append(0); ticks.append(t + 1); gens.append(0)
            prev = cur
            continue

        # --- DIVISION: carry the daughter, rebuild, cross the boundary ---
        div_tick = t + 1
        carry = select_carry_daughter(before, now, snap)
        if carry is None:
            break  # no recoverable daughter state — stop the lineage here
        comp = _build_daughter((seed + 1) % (2 ** 31), carry, cache_dir)
        comp.run(1)  # one step so the daughter's listeners populate
        cur = layout.extract(_cell_view(comp.state))
        # the across-division transition: mother's last state -> daughter's first
        Xs.append(prev); Ys.append(cur); spans.append(1); ticks.append(div_tick); gens.append(1)
        prev = cur

        # within-generation tail in the daughter, so the division point isn't
        # the trajectory's endpoint (rollout needs post-division continuation)
        for j in range(min(tail, max_steps - t - 1)):
            div2, _, _, _ = _run_one_step(comp)
            if div2:
                break  # a second division — stop; one crossing is the test
            cur = layout.extract(_cell_view(comp.state))
            Xs.append(prev); Ys.append(cur); spans.append(0)
            ticks.append(div_tick + j + 1); gens.append(1)
            prev = cur
        break

    return Xs, Ys, spans, ticks, gens, layout, div_tick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--max-steps", type=int, default=3200,
                    help="hard cap on steps to reach division (first division ~2300-2500)")
    ap.add_argument("--tail", type=int, default=120,
                    help="within-generation steps to collect in the daughter after crossing")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--groups", nargs="+", default=["mass", "chromosome"],
                    help="observable groups (default: the compact mass+chromosome panel)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    from pbg_torch import TransitionDataset

    layout = None
    Xall, Yall, spans_all, ticks_all, gens_all, ids = [], [], [], [], [], []
    div_ticks = {}
    t0 = time.time()
    for traj_index, seed in enumerate(args.seeds):
        ts = time.time()
        if layout is None:
            comp = _build_gen0(seed, args.cache_dir)
            comp.run(1)
            layout = PanelLayout.discover(comp.state, groups=tuple(args.groups))
            del comp
        Xs, Ys, spans, ticks, gens, layout, div_tick = sample_trajectory(
            seed, args.max_steps, layout, args.cache_dir, args.groups, tail=args.tail)
        Xall.extend(Xs); Yall.extend(Ys); spans_all.extend(spans)
        ticks_all.extend(ticks); gens_all.extend(gens)
        ids.extend([traj_index] * len(Xs))
        div_ticks[seed] = div_tick
        print(f"  seed {seed}: {len(Xs)} transitions, division at tick {div_tick} "
              f"({time.time()-ts:.1f}s)", flush=True)

    X = np.asarray(Xall, dtype=np.float64)
    Y = np.asarray(Yall, dtype=np.float64)
    DT = np.ones(X.shape[0], dtype=np.float64)
    traj_id = np.asarray(ids, dtype=np.int64)

    spec = build_spec(layout)
    ds = TransitionDataset(X=X, Y=Y, DT=DT, traj_id=traj_id, spec=spec)
    ds.save(os.path.join(args.out, "transitions.npz"))
    with open(os.path.join(args.out, "layout.json"), "w") as fh:
        json.dump(layout.to_dict(), fh)
    np.savez(os.path.join(args.out, "meta.npz"),
             spans_division=np.asarray(spans_all, dtype=np.int64),
             tick=np.asarray(ticks_all, dtype=np.int64),
             generation=np.asarray(gens_all, dtype=np.int64),
             traj_id=traj_id)

    n_span = int(np.sum(spans_all))
    print(f"\nThrough-division dataset: {X.shape[0]} transitions x {X.shape[1]} observables "
          f"from {len(args.seeds)} trajectories ({time.time()-t0:.1f}s total)")
    print(f"  across-division transitions: {n_span} (one per followed division)")
    print(f"  division ticks: {div_ticks}")
    print(f"  groups: " + ", ".join(
        f"{g}={layout.group_slices[g][1]-layout.group_slices[g][0]}" for g in layout.groups))
    print(f"  saved -> {args.out}/transitions.npz + layout.json + meta.npz")


if __name__ == "__main__":
    main()
