"""Generate colony.gif for colonies-02 — NATURAL-division colony growth.

Starts from ONE whole-cell agent and lets it grow and divide NATURALLY (no
forced division): each cell elongates as its dry mass accumulates toward the
division threshold (~one cell cycle ≈ 2300+ ticks), then splits into two
daughters that reset and regrow. Runs until ~4 cells (gen 2), subsampling
one frame every `stride` ticks so the gif shows real growth, not pops.

Physics fixes live: jitter_per_second=1e-4 (the old 0.5 flung cells around) +
coherent fg body mass + viva-munk in-place shape update + bounded emitters.
Capped at 4 cells / a tick budget so it stays under the multi-cell RAM ceiling
(F-04 native leak only bites at higher counts / longer runs).

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


def main():
    from v2ecoli.colony import make_colony
    from viva_munk.plots.multibody_plots import simulation_to_gif

    env_size = 30
    stride = 40                 # capture one frame every `stride` sim ticks
    stop_cells = 4              # stop once gen 2 is reached (RAM ceiling)
    post_div_ticks = 600        # observe the 4-cell stage a bit before stopping
    max_ticks = 6000            # hard budget (RAM safety)

    print("Building N=1 colony (natural division, fixed physics)…")
    comp = make_colony(
        n_cells=1, env_size=env_size, cache_dir="out/cache", seed=0,
        jitter_per_second=1e-4, init_mass=200.0, emit_cells=False,
    )

    history = []

    def capture():
        history.append({"agents": _snapshot(comp.state),
                        "time": float(comp.state.get("global_time", 0))})

    comp.run(1.0)               # warmup builds the inner WCM
    capture()

    t = 0
    reached_at = None
    while t < max_ticks:
        comp.run(1.0)
        t += 1
        n = len(comp.state["cells"])
        if t % stride == 0:
            capture()
            masses = [round(c.get("mass", 0)) for c in comp.state["cells"].values()]
            print(f"  tick {t}: {n} cells, masses(fg)={masses}", flush=True)
        if reached_at is None and n >= stop_cells:
            reached_at = t
            print(f"  reached {stop_cells} cells at tick {t}", flush=True)
        if reached_at is not None and t >= reached_at + post_div_ticks:
            break

    gif_path = STUDY_DIR / "colony.gif"
    print(f"Rendering {len(history)} frames -> {gif_path}…")
    simulation_to_gif(
        history,
        config={"env_size": env_size},
        agents_key="agents",
        filename="colony.gif",
        out_dir=str(STUDY_DIR),
        frame_duration_ms=120,
        show_time_title=True,
        color_by_phylogeny=True,
    )
    print(f"  done. {len(comp.state['cells'])} cells, "
          f"{gif_path.stat().st_size/1024:.0f} KB, last tick {t}")


if __name__ == "__main__":
    main()
