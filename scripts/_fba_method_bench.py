#!/usr/bin/env python3
"""Measure the per-step FBA cost under primal vs dual simplex.

The metabolism step re-solves the SAME persistent LP each tick with changed
bounds/objective (presolve stays OFF, so GLPK warm-starts from the prior
basis). It defaults to the PRIMAL simplex; after bound changes the prior basis
is often dual-feasible, so the DUAL simplex may re-optimise from the warm basis
in fewer iterations. This benchmark wraps swiglpk.glp_simplex to accumulate
wall time, call count, and iteration count over N real steps, for whichever
method is selected — so we can see if dual is a genuine win before wiring it in.

  python scripts/_fba_method_bench.py --method primal --steps 150
  python scripts/_fba_method_bench.py --method dual   --steps 150
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["primal", "dual", "dualp"], default="primal")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--chunk", type=int, default=10)
    a = ap.parse_args()

    from v2ecoli.composites._helpers import set_null_emitter_override
    set_null_emitter_override(True)

    import swiglpk as glp
    from v2ecoli.processes.parca.wholecell.utils._netflow import nf_glpk

    # --- instrument glp_simplex: time + count + iterations ---------------
    stats = {"t": 0.0, "n": 0, "iters": 0}
    _orig_simplex = glp.glp_simplex

    def _timed_simplex(lp, smcp):
        t0 = time.perf_counter()
        r = _orig_simplex(lp, smcp)
        stats["t"] += time.perf_counter() - t0
        stats["n"] += 1
        try:
            stats["iters"] += glp.glp_get_it_cnt(lp)
        except Exception:
            pass
        return r
    glp.glp_simplex = _timed_simplex
    nf_glpk.glp.glp_simplex = _timed_simplex   # the module imported it as `glp`

    # --- force the chosen simplex method on every GLPK solver ------------
    meth = {"primal": glp.GLP_PRIMAL, "dual": glp.GLP_DUAL, "dualp": glp.GLP_DUALP}[a.method]
    _orig_init = nf_glpk.NetworkFlowGLPK.__init__

    def _patched_init(self, *args, **kw):
        _orig_init(self, *args, **kw)
        self._smcp.meth = meth
    nf_glpk.NetworkFlowGLPK.__init__ = _patched_init

    # --- run the baseline a fixed number of steps -----------------------
    from v2ecoli import build_composite
    from v2ecoli.library.sqlite_run import run_multigen_sqlite
    comp = build_composite("ecoli_baseline", cache_dir="out/cache", seed=0)
    t0 = time.time()
    run_multigen_sqlite(comp, run_id="fbabench", db_file="/tmp/fbabench.db",
                        emit_paths=["listeners.mass.cell_mass"],
                        max_steps=a.steps, max_generations=1, chunk=a.chunk,
                        single_daughters=True)
    wall = time.time() - t0

    n = max(1, stats["n"])
    print(f"method={a.method}  steps={a.steps}")
    print(f"  total wall:        {wall:7.1f}s   ({1000*wall/a.steps:.1f} ms/step)")
    print(f"  glp_simplex time:  {stats['t']:7.1f}s   ({1000*stats['t']/a.steps:.1f} ms/step, "
          f"{100*stats['t']/wall:.0f}% of wall)")
    print(f"  glp_simplex calls: {stats['n']}   ({stats['n']/a.steps:.1f}/step)")
    print(f"  simplex iters:     {stats['iters']}   ({stats['iters']/n:.1f}/solve avg)")


if __name__ == "__main__":
    main()
