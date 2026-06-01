"""Generate a colony.gif for the study Visualizations tab.

Runs a small pure-WC colony (N=2, force-divide after warmup) for ~40
ticks, captures per-tick cell geometry, and renders an animated GIF
via viva_munk.plots.multibody_plots.simulation_to_gif.

Output: studies/colonies-01-hpc-readiness/colony.gif

Standalone — invoke once after a code change to the colony composite::

    python studies/colonies-01-hpc-readiness/sims/make_gif.py
"""
from __future__ import annotations

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent.parent
_WORKTREE_ROOT = STUDY_DIR.parent.parent
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))


def _snapshot(state: dict) -> dict:
    """Pluck just the geometry fields simulation_to_gif needs from
    sim.state['cells'], so the captured history doesn't drag along
    live Process instances.

    Each entry MUST include a ``type`` key — the draw loop in
    multibody_plots.draw_frame only renders entries with explicit
    ``'segment'`` or ``'circle'`` (it doesn't auto-infer at render time)."""
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


def main():
    from v2ecoli.colony import make_colony
    from viva_munk.plots.multibody_plots import simulation_to_gif

    env_size = 30
    n_warmup = 1
    n_main   = 40

    print("Building N=2 colony…")
    comp = make_colony(n_cells=2, env_size=env_size, cache_dir="out/cache", seed=0)

    history = []  # list of {'agents': {cid: {location, length, radius, angle, mass}}}
    history.append({"agents": _snapshot(comp.state), "time": float(comp.state.get("global_time", 0))})

    print(f"Warmup ({n_warmup} tick)…")
    for _ in range(n_warmup):
        comp.run(1.0)
    history.append({"agents": _snapshot(comp.state), "time": float(comp.state.get("global_time", 0))})

    print("Force-dividing initial cells…")
    for cid in list(comp.state["cells"].keys()):
        comp.state["cells"][cid]["ecoli"]["instance"]._composite.state["agents"]["0"]["divide"] = True

    print(f"Main loop ({n_main} ticks)…")
    for i in range(n_main):
        comp.run(1.0)
        history.append({"agents": _snapshot(comp.state), "time": float(comp.state.get("global_time", 0))})
        if (i + 1) % 10 == 0:
            print(f"  tick {i+1}: {len(comp.state['cells'])} cells")

    gif_path = STUDY_DIR / "colony.gif"
    print(f"Rendering GIF to {gif_path}…")
    simulation_to_gif(
        history,
        config={"env_size": env_size},
        agents_key="agents",
        filename="colony.gif",
        out_dir=str(STUDY_DIR),
        frame_duration_ms=150,
        show_time_title=True,
        color_by_phylogeny=True,  # daughters get hue-shifted variants of mother
    )
    print(f"  done. size = {gif_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
