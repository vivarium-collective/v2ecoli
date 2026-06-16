"""Generate colony.gif for colonies-02 — a growing colony with the FIXED physics.

Starts from ONE whole-cell agent and grows it through staged divisions
(1 -> 2 -> 4) with the physics fixes live: jitter_per_second=1e-4 (the old 0.5
flung cells around) + init_mass=200 fg (coherent mass-unit) + the viva-munk
in-place shape update (#11) + bounded inner/outer emitters. Capped at 4 cells /
~95 ticks so it stays well under the multi-cell RAM ceiling (the dominant
colony leak, F-04, only bites at higher counts / long runs).

Renders via viva_munk.plots.multibody_plots.simulation_to_gif.
Output: <study>/colony.gif

    python .../colonies-02-parallel-multigen-perf/sims/make_gif.py
"""
from __future__ import annotations

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent.parent
_WORKTREE_ROOT = STUDY_DIR.parent.parent
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))


def _snapshot(state: dict) -> dict:
    out = {}
    for cid, cell in (state.get("cells") or {}).items():
        if not isinstance(cell, dict):
            continue
        out[cid] = {
            "type":     "segment",
            "location": tuple(cell.get("location") or (0, 0)),
            "length":   float(cell.get("length") or 2.0),
            "radius":   float(cell.get("radius") or 0.5),
            "angle":    float(cell.get("angle")  or 0.0),
            "mass":     float(cell.get("mass")   or 0.0),
        }
    return out


def _force_divide(comp):
    """Trip the inner WCM divide flag on every current cell."""
    for cid in list(comp.state["cells"].keys()):
        inst = comp.state["cells"][cid]["ecoli"]["instance"]
        inst._composite.state["agents"]["0"]["divide"] = True


def main():
    from v2ecoli.colony import make_colony
    from viva_munk.plots.multibody_plots import simulation_to_gif

    env_size = 30
    grow = 28  # ticks per growth phase between divisions

    print("Building N=1 colony (fixed physics)…")
    comp = make_colony(
        n_cells=1, env_size=env_size, cache_dir="out/cache", seed=0,
        jitter_per_second=1e-4, init_mass=200.0, emit_cells=False,
    )

    history = []

    def capture():
        history.append({"agents": _snapshot(comp.state),
                        "time": float(comp.state.get("global_time", 0))})

    capture()
    comp.run(1.0)            # warmup builds the inner WCM
    capture()

    # 1 cell grows -> divide -> 2 grow -> divide -> 4 grow
    for stage, n_before in enumerate([1, 2]):
        print(f"Growth phase {stage} ({len(comp.state['cells'])} cell(s))…")
        for _ in range(grow):
            comp.run(1.0)
            capture()
        print(f"  force-dividing {len(comp.state['cells'])} cell(s)…")
        _force_divide(comp)
        comp.run(1.0)
        capture()
    print(f"Final growth ({len(comp.state['cells'])} cells)…")
    for _ in range(grow):
        comp.run(1.0)
        capture()

    gif_path = STUDY_DIR / "colony.gif"
    print(f"Rendering {len(history)} frames -> {gif_path}…")
    simulation_to_gif(
        history,
        config={"env_size": env_size},
        agents_key="agents",
        filename="colony.gif",
        out_dir=str(STUDY_DIR),
        frame_duration_ms=140,
        show_time_title=True,
        color_by_phylogeny=True,
    )
    print(f"  done. {len(comp.state['cells'])} cells, "
          f"{gif_path.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
