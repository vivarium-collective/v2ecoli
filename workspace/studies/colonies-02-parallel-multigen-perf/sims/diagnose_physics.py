"""Build-phase diagnostic: jitter / mass coupling (colonies-02).

Answers the colony.gif "cells move too much" question with numbers.

Hypothesis (from colonies-01 code review):
  * colony.py ran jitter_per_second=0.5 — ~5000x viva-munk's 1e-4 default.
  * The EcoliWCM `mass` output is a DELTA (fg) that accumulates onto a body
    seeded by build_microbe(density=0.02) -> mass ~0.04 in pymunk units.
  * Jitter is a Gaussian impulse; velocity change = impulse / body.mass.
    With body.mass ~0.04 early, jitter=0.5 flings the cell; pymunk-density
    units and femtograms are also summed incoherently.

This script builds a 2-cell colony for a handful of (jitter, init_mass)
settings, runs ~12 ticks each, and prints per-tick body.mass and the
per-tick displacement (how far each cell moved). Compare the legacy setting
(jitter=0.5, init_mass=None) against the fix (jitter=1e-4, init_mass=200).

Usage:
    python studies/colonies-02-parallel-multigen-perf/sims/diagnose_physics.py
    python .../diagnose_physics.py --ticks 20 --cache-dir out/cache

Writes a small CSV (diagnostics/physics_diag.csv) next to the study so the
result is committable evidence for the build_tasks decision.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parent.parent
_WORKTREE_ROOT = STUDY_DIR.parent.parent
if str(_WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKTREE_ROOT))


def _loc(cell):
    loc = cell.get('location', (0.0, 0.0))
    return float(loc[0]), float(loc[1])


def run_setting(jitter, init_mass, ticks, cache_dir, seed=0):
    """Run one (jitter, init_mass) setting; return list of per-tick rows."""
    from v2ecoli.colony import make_colony

    label = f"jitter={jitter:g} init_mass={init_mass}"
    print(f"\n=== {label} ===")
    comp = make_colony(
        n_cells=2, env_size=30, cache_dir=cache_dir, seed=seed,
        jitter_per_second=jitter, init_mass=init_mass,
    )
    # Warmup builds the inner WCM composites lazily.
    comp.run(1.0)

    prev = {cid: _loc(c) for cid, c in comp.state['cells'].items()}
    rows = []
    for tick in range(ticks):
        comp.run(1.0)
        cells = comp.state['cells']
        disps, masses = [], []
        for cid, cell in cells.items():
            x, y = _loc(cell)
            px, py = prev.get(cid, (x, y))
            disp = math.hypot(x - px, y - py)
            prev[cid] = (x, y)
            disps.append(disp)
            masses.append(float(cell.get('mass', 0.0)))
        mean_disp = sum(disps) / len(disps) if disps else 0.0
        max_disp = max(disps) if disps else 0.0
        mean_mass = sum(masses) / len(masses) if masses else 0.0
        rows.append({
            'jitter': jitter, 'init_mass': init_mass, 'tick': tick,
            'mean_body_mass': mean_mass, 'mean_disp_um': mean_disp,
            'max_disp_um': max_disp, 'n_cells': len(cells),
        })
        print(f"  tick {tick:2d}: body_mass~{mean_mass:9.3f}  "
              f"mean_move={mean_disp:7.3f}um  max_move={max_disp:7.3f}um  "
              f"cells={len(cells)}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ticks', type=int, default=12)
    ap.add_argument('--cache-dir', default='out/cache')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)

    # (jitter, init_mass) settings: legacy -> isolate jitter -> isolate mass -> fix
    settings = [
        (0.5,  None),   # colonies-01 legacy: high jitter + tiny density mass
        (1e-4, None),   # lower jitter only (mass still incoherent)
        (0.5,  200.0),  # realistic mass only (does mass alone tame jitter?)
        (1e-4, 200.0),  # proposed fix: low jitter + coherent fg mass
    ]
    all_rows = []
    for jitter, init_mass in settings:
        t0 = time.perf_counter()
        all_rows.extend(run_setting(jitter, init_mass, args.ticks,
                                    args.cache_dir, args.seed))
        print(f"  ({time.perf_counter() - t0:.1f}s)")

    out_dir = STUDY_DIR / 'diagnostics'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'physics_diag.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {out_csv}")
    print("\nInterpretation: if mean_move collapses when jitter drops 0.5->1e-4,"
          " jitter was the cause. If body_mass starts ~0.04 with init_mass=None"
          " and the high-jitter row flings cells, the mass coupling is the"
          " amplifier — prefer the (1e-4, 200) fix for both fidelity and unit"
          " coherence.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
