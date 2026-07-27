# scripts/compare_biological.py
"""Run baseline vs biological for N steps and emit an HTML comparison of mass /
growth markers. Phase 1: the two are identical by construction; this makes that
visible and is the Phase-2 (tolerant) comparison's entry point.

Usage:
    python scripts/compare_biological.py --steps 100 \
        --cache "$V2ECOLI_CACHE_DIR" --out out/biological_comparison.html
"""
from __future__ import annotations

import argparse
import os

import v2ecoli.library.unit_bridge  # noqa: F401


def _mass(agent_listeners) -> dict:
    m = agent_listeners.get('mass', {})
    def _f(x):
        return float(getattr(x, 'magnitude', x))
    return {k: _f(m[k]) for k in ('cell_mass', 'dry_mass') if k in m}


def run(steps: int, cache: str):
    from process_bigraph import Composite
    from v2ecoli.core import build_core
    from v2ecoli.composites.ecoli_baseline import baseline, load_cache_bundle
    from v2ecoli.composites.biological import biological

    bundle = load_cache_bundle(cache)
    cb = build_core()
    base = Composite(baseline(core=cb, seed=0, bundle=bundle, emitter='null'), core=cb)
    cx = build_core()
    bio = Composite(biological(core=cx, seed=0, bundle=bundle, emitter='null'), core=cx)

    rows = []
    for i in range(steps):
        base.run(1); bio.run(1)
        bm = _mass(base.state['agents']['0']['listeners'])
        xm = _mass(bio.state['agents']['0']['cell']['observables'])
        rows.append((i + 1, bm, xm))
    return rows


def to_html(rows) -> str:
    head = ("<tr><th>step</th><th>baseline cell_mass</th>"
            "<th>biological cell_mass</th><th>Δ</th></tr>")
    body = []
    for step, bm, xm in rows:
        b = bm.get('cell_mass', float('nan'))
        x = xm.get('cell_mass', float('nan'))
        body.append(f"<tr><td>{step}</td><td>{b:.6g}</td>"
                    f"<td>{x:.6g}</td><td>{abs(b - x):.3g}</td></tr>")
    return ("<html><head><meta charset='utf-8'><title>baseline vs biological</title>"
            "<style>table{border-collapse:collapse}td,th{border:1px solid #ccc;"
            "padding:4px 8px;font-family:monospace}</style></head><body>"
            "<h1>Baseline vs Biological — mass markers</h1>"
            "<p>Phase 1 is a pure relabel; Δ should be 0 at every step.</p>"
            f"<table>{head}{''.join(body)}</table></body></html>")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--cache', default=os.environ.get('V2ECOLI_CACHE_DIR', 'out/cache'))
    p.add_argument('--out', default='out/biological_comparison.html')
    args = p.parse_args(argv)

    rows = run(args.steps, args.cache)
    max_delta = max((abs(bm.get('cell_mass', 0) - xm.get('cell_mass', 0))
                     for _, bm, xm in rows), default=0.0)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(to_html(rows))
    print(f"wrote {args.out}; max cell_mass Δ over {args.steps} steps = {max_delta:.3g}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
