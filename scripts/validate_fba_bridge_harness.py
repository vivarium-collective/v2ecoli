"""Validate the Millard FBA-bridge harness — pin fidelity, feasibility, viability.

Runs the ``millard_fba_bridge_harness`` composite (baseline WCM + Millard ODE
flux source + fba-flux-coupler) on M9-glucose for N seconds and reports:

  (a) pin fidelity   — for the reactions the coupler pins, the REALIZED FBA
                       flux (``fba.getReactionFluxes``) vs the ODE-derived
                       pinned bound. Reactions whose hard pin held (not in
                       ``relaxed_reactions``) should track their bound closely;
                       relaxed reactions are reported separately.
  (b) feasibility    — fraction of ticks with a non-empty relaxed_reactions
                       set, and which reactions were relaxed most often.
  (c) viability      — central-metabolite concentrations + dry_mass stay
                       finite and positive over the run.

Writes a committed HTML summary to
``reports/figures/pdmp-01/fba_bridge_flux_pin.html``.

Usage:
    .venv/bin/python scripts/validate_fba_bridge_harness.py [--seconds N]
"""
from __future__ import annotations

import argparse
import html
import math
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np

from process_bigraph import Composite
from pbg_superpowers.composite_generator import _REGISTRY, build_generator

from v2ecoli.core import build_core
from v2ecoli.library.fba_reaction_map import load_reaction_map

REACTION_MAP_FILE = "v2ecoli/data/millard_v2ecoli_reaction_map.yaml"
OUT_HTML = Path("reports/figures/pdmp-01/fba_bridge_flux_pin.html")


def _build():
    """Build the harness Composite, keeping a handle on the live Metabolism
    instance so we can read realized FBA fluxes each tick."""
    core = build_core()
    entry = [e for e in _REGISTRY.values()
             if e.name == "millard_fba_bridge_harness"][0]
    doc = build_generator(
        entry, overrides={"cache_dir": "out/cache", "seed": 0}, core=core)
    cell = doc["state"]["agents"]["0"]
    met_inst = cell["ecoli-metabolism"]["instance"]
    comp = Composite(doc, core=core)
    return comp, met_inst


def _mag(x) -> float:
    """Coerce a scalar / pint Quantity to a plain float magnitude."""
    if x is None:
        return float("nan")
    if hasattr(x, "magnitude"):
        x = x.magnitude
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main(seconds: int) -> None:
    pinned_ids = sorted({
        fba_id
        for entries in load_reaction_map(REACTION_MAP_FILE).values()
        for (fba_id, _scale) in entries
    })

    comp, met = _build()

    # Per-tick records.
    ticks = []                      # tick index
    relaxed_counter: Counter = Counter()
    n_ticks_relaxed = 0
    # fidelity accumulation for HELD pins (pin held => realized should ~ bound)
    held_records = []               # (rid, bound, realized)
    relaxed_records = []            # (rid, bound, realized)
    drymass_series = []
    viability_ok = True
    n_central_nonfinite = 0

    t0 = time.time()
    for t in range(seconds):
        comp.run(1)
        ag = comp.state["agents"]["0"]
        pins = ag.get("pinned_flux_targets") or {}
        relaxed = set(
            ((ag.get("listeners") or {}).get("fba_bridge") or {})
            .get("relaxed_reactions", []) or [])
        for rid in relaxed:
            relaxed_counter[rid] += 1
        if relaxed:
            n_ticks_relaxed += 1

        # Realized FBA fluxes for the pinned reaction ids.
        try:
            realized_raw = met.model.fba.getReactionFluxes(pinned_ids)
            realized = {rid: (None if v is None else float(v))
                        for rid, v in zip(pinned_ids, np.asarray(
                            realized_raw, dtype=object).tolist())}
        except Exception:
            realized = {}

        # Skip tick 0 for fidelity (Millard has not published fluxes yet, so
        # all pins are the 0.0 pre-seed).
        if t >= 1:
            for rid, bound in pins.items():
                b = float(bound)
                if b <= 0.0:
                    continue
                rv = realized.get(rid)
                if rv is None:
                    continue
                if rid in relaxed:
                    relaxed_records.append((rid, b, rv))
                else:
                    held_records.append((rid, b, rv))

        # Viability.
        mass = ((ag.get("listeners") or {}).get("mass") or {})
        dm_f = _mag(mass.get("dry_mass"))
        drymass_series.append(dm_f)
        if not (math.isfinite(dm_f) and dm_f > 0.0):
            viability_ok = False

        central = (ag.get("shared") or {}).get("central_metabolites") or {}
        for v in central.values():
            if not math.isfinite(_mag(v)):
                n_central_nonfinite += 1

        ticks.append(t + 1)

    wall = time.time() - t0

    # ---- Pin fidelity stats (held pins) ----
    def _rel_err(b, rv):
        return abs(rv - b) / max(abs(b), 1e-9)

    held_rel_errs = [_rel_err(b, rv) for (_r, b, rv) in held_records]
    n_held = len(held_records)
    n_relaxed_obs = len(relaxed_records)
    median_held_err = (float(np.median(held_rel_errs))
                       if held_rel_errs else float("nan"))
    frac_held_within_10pct = (
        float(np.mean([e <= 0.10 for e in held_rel_errs]))
        if held_rel_errs else float("nan"))

    # per-reaction held fidelity summary (last observation per reaction)
    per_rxn_last = {}
    for rid, b, rv in held_records:
        per_rxn_last[rid] = (b, rv, _rel_err(b, rv))

    n_central = len((comp.state["agents"]["0"].get("shared") or {})
                    .get("central_metabolites") or {})

    # ---- Print report ----
    print("=" * 72)
    print("Millard FBA-bridge harness validation")
    print("=" * 72)
    print(f"ticks run                : {len(ticks)} s  (wall {wall:.1f}s, "
          f"{wall/max(len(ticks),1):.2f}s/tick)")
    print(f"pinned reaction ids (map): {len(pinned_ids)}")
    print()
    print("(a) PIN FIDELITY  (realized FBA flux vs ODE-pinned bound)")
    print(f"    held-pin observations    : {n_held}")
    print(f"    relaxed-pin observations : {n_relaxed_obs}")
    print(f"    median |rel err| (held)  : {median_held_err:.3g}")
    print(f"    frac held within 10%     : {frac_held_within_10pct:.2%}")
    print("    per-reaction held fidelity (last tick observed):")
    for rid in sorted(per_rxn_last):
        b, rv, e = per_rxn_last[rid]
        print(f"      {rid:30s} bound={b:10.4g} realized={rv:10.4g} "
              f"relerr={e:7.2%}")
    print()
    print("(b) FEASIBILITY")
    frac_relaxed = n_ticks_relaxed / max(len(ticks), 1)
    print(f"    ticks with >=1 relaxed pin: {n_ticks_relaxed}/{len(ticks)} "
          f"({frac_relaxed:.0%})")
    print("    most-relaxed reactions (tick count):")
    for rid, cnt in relaxed_counter.most_common(12):
        print(f"      {rid:30s} {cnt}")
    print()
    print("(c) VIABILITY")
    dm_arr = np.array(drymass_series, dtype=float)
    print(f"    dry_mass finite & positive every tick : {viability_ok}")
    if dm_arr.size:
        print(f"    dry_mass first/last (fg)              : "
              f"{dm_arr[0]:.4g} -> {dm_arr[-1]:.4g}")
    print(f"    central metabolites tracked           : {n_central}")
    print(f"    non-finite central-metabolite samples : {n_central_nonfinite}")
    print("=" * 72)

    # ---- Write HTML ----
    _write_html(
        OUT_HTML, seconds=len(ticks), wall=wall, pinned_ids=pinned_ids,
        n_held=n_held, n_relaxed_obs=n_relaxed_obs,
        median_held_err=median_held_err,
        frac_held_within_10pct=frac_held_within_10pct,
        per_rxn_last=per_rxn_last, frac_relaxed=frac_relaxed,
        n_ticks_relaxed=n_ticks_relaxed, n_total_ticks=len(ticks),
        relaxed_counter=relaxed_counter, viability_ok=viability_ok,
        dm_arr=dm_arr, n_central=n_central,
        n_central_nonfinite=n_central_nonfinite,
    )
    print(f"wrote {OUT_HTML}")


def _fmt(x, spec="{:.4g}"):
    try:
        return spec.format(float(x))
    except (TypeError, ValueError):
        return str(x)


def _write_html(path: Path, **k) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_fid = "".join(
        f"<tr><td>{html.escape(rid)}</td><td>{_fmt(b)}</td>"
        f"<td>{_fmt(rv)}</td><td>{_fmt(e, '{:.2%}')}</td></tr>"
        for rid, (b, rv, e) in sorted(k["per_rxn_last"].items())
    )
    rows_relax = "".join(
        f"<tr><td>{html.escape(rid)}</td><td>{cnt}</td></tr>"
        for rid, cnt in k["relaxed_counter"].most_common(20)
    )
    dm = k["dm_arr"]
    dm_first = _fmt(dm[0]) if dm.size else "n/a"
    dm_last = _fmt(dm[-1]) if dm.size else "n/a"

    # Inline SVG sparkline of dry_mass.
    spark = _svg_spark(dm)

    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Millard FBA-bridge flux-pin validation</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:2rem;color:#1a1a1a;max-width:980px}}
 h1{{font-size:1.4rem}} h2{{font-size:1.1rem;margin-top:1.6rem}}
 table{{border-collapse:collapse;font-size:.85rem;margin:.5rem 0}}
 td,th{{border:1px solid #ccc;padding:3px 8px;text-align:right}}
 td:first-child,th:first-child{{text-align:left}}
 .kpi{{display:inline-block;background:#f3f5f7;border:1px solid #dde;
       border-radius:6px;padding:.4rem .7rem;margin:.2rem .4rem .2rem 0}}
 .ok{{color:#157f3b;font-weight:600}} .warn{{color:#b34700;font-weight:600}}
</style></head><body>
<h1>Millard FBA-bridge flux-pin validation</h1>
<p>Composite <code>millard_fba_bridge_harness</code> — baseline whole-cell
E.&nbsp;coli (real tFBA <code>ecoli-metabolism</code>) with the Millard&nbsp;2017
central-carbon ODE wired in as a flux source that mechanistically pins the
mapped FBA reactions via <code>fba-flux-coupler</code>.</p>
<div>
 <span class="kpi">ticks: {k['seconds']} s</span>
 <span class="kpi">wall: {k['wall']:.1f} s</span>
 <span class="kpi">pinned reactions (map): {len(k['pinned_ids'])}</span>
</div>

<h2>(a) Pin fidelity — realized FBA flux vs ODE-pinned bound</h2>
<div>
 <span class="kpi">held-pin obs: {k['n_held']}</span>
 <span class="kpi">relaxed-pin obs: {k['n_relaxed_obs']}</span>
 <span class="kpi">median |rel.err| (held): {_fmt(k['median_held_err'])}</span>
 <span class="kpi">held within 10%: {_fmt(k['frac_held_within_10pct'],'{:.0%}')}</span>
</div>
<table><thead><tr><th>FBA reaction</th><th>pinned bound</th>
<th>realized flux</th><th>|rel err|</th></tr></thead>
<tbody>{rows_fid}</tbody></table>

<h2>(b) Feasibility</h2>
<div>
 <span class="kpi">ticks with &ge;1 relaxed pin:
   {k['n_ticks_relaxed']}/{k['n_total_ticks']}
   ({k['frac_relaxed']:.0%})</span>
</div>
<table><thead><tr><th>most-relaxed reaction</th><th>tick count</th></tr></thead>
<tbody>{rows_relax}</tbody></table>

<h2>(c) Viability</h2>
<div>
 <span class="kpi {'ok' if k['viability_ok'] else 'warn'}">
   dry_mass finite &amp; positive: {k['viability_ok']}</span>
 <span class="kpi">dry_mass: {dm_first} &rarr; {dm_last} fg</span>
 <span class="kpi">central metabolites: {k['n_central']}</span>
 <span class="kpi {'ok' if k['n_central_nonfinite']==0 else 'warn'}">
   non-finite central samples: {k['n_central_nonfinite']}</span>
</div>
<p>dry_mass over the run:</p>
{spark}
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def _svg_spark(series: np.ndarray, w: int = 600, h: int = 120) -> str:
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return "<p><em>(insufficient dry_mass data for sparkline)</em></p>"
    lo, hi = float(s.min()), float(s.max())
    rng = hi - lo or 1.0
    pts = []
    for i, v in enumerate(s):
        x = w * i / (s.size - 1)
        y = h - (h - 8) * (v - lo) / rng - 4
        pts.append(f"{x:.1f},{y:.1f}")
    return (f'<svg width="{w}" height="{h}" '
            f'style="border:1px solid #ddd;background:#fafafa">'
            f'<polyline fill="none" stroke="#2b6cb0" stroke-width="1.5" '
            f'points="{" ".join(pts)}"/></svg>')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=120,
                    help="simulated seconds (1 tick = 1 s); >=30 acceptable")
    args = ap.parse_args()
    main(args.seconds)
