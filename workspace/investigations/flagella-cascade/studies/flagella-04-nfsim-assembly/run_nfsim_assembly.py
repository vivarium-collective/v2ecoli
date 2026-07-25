"""Maya's Aim 2B — rule-based flagellar assembly via NFsim (pbg-nfsim).

Brings in the existing flagella-assembly example from
https://github.com/vivarium-collective/pbg-nfsim (a process-bigraph wrapper for
BioNetGen/NFsim). The BNGL model encodes hierarchical complexation of ~30 flagellar
proteins through 7 sequential reactions (237 rules): free monomers -> export
apparatus -> motor/basal body -> hook -> complete flagellum. This is the ordered,
conditional assembly that the stochastic Gillespie complexation cannot capture
(Aim 2B rationale).

This driver runs the composed production+complexation workflow (MonomerProduction
feeds monomers; NFSimProcess assembles them) and renders the staged appearance of
the assembly intermediates.

Note: bionetgen 0.8.6 imports the removed `pkg_resources.packaging` on Python 3.12;
we shim it from the standalone `packaging` package before importing pbg_nfsim.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-04-nfsim-assembly/run_nfsim_assembly.py \
        --seconds 1800 --sample 100
"""
import argparse
import os

# --- shim: bionetgen 0.8.6 does `from pkg_resources import packaging` (removed) ---
import pkg_resources
import packaging as _packaging
if not hasattr(pkg_resources, "packaging"):
    pkg_resources.packaging = _packaging
# ----------------------------------------------------------------------------------

import numpy as np
import pbg_nfsim
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

# Ordered assembly chain (coarse stages, in assembly order).
CHAIN = [
    ("flagellar_export_apparatus", "export apparatus", "#1f77b4"),
    ("flagellar_motor", "motor / basal body", "#2ca02c"),
    ("flagellar_hook", "hook", "#ff7f0e"),
    ("flagella", "complete flagellum", "#d62728"),
]
MONOMERS = [
    ("Free_fliF", "FliF (MS-ring)", "#9467bd"),
    ("Free_flgE", "FlgE (hook)", "#8c564b"),
    ("Free_fliC", "FliC (flagellin)", "#17becf"),
]


def _observables(sim):
    for key in ("species", "observables"):
        d = sim.state.get(key)
        if isinstance(d, dict) and d:
            return d
    # fall back: search nested
    return sim.state.get("species", {}) or {}


def run(seconds, sample, n_steps):
    # Dispatch via the REGISTERED composite generator
    # (v2ecoli.composites.flagella_nfsim_assembly) instead of building an ad-hoc
    # in-code document — so this run is a real registered composite in the
    # Simulations DB and opens in the Composite Explorer.
    from v2ecoli.composites.flagella_nfsim_assembly import (
        flagella_nfsim_assembly, _register_nfsim_links,
    )
    core = allocate_core()
    _register_nfsim_links(core)

    doc = flagella_nfsim_assembly(
        core=core, n_steps=n_steps, complexation_interval=float(sample),
        production_interval=1.0, production_rate_scale=1.0,
    )
    sim = Composite({"state": doc}, core=core)

    names = [c[0] for c in CHAIN] + [m[0] for m in MONOMERS]
    rec = {"t": [], **{n: [] for n in names}}

    def snap(t):
        obs = _observables(sim)
        rec["t"].append(t)
        for n in names:
            rec[n].append(float(obs.get(n, 0.0)))

    snap(0.0)
    total = 0.0
    while total < seconds:
        chunk = min(sample, seconds - total)
        sim.run(chunk)
        total += chunk
        snap(total)
    return {k: np.array(v) for k, v in rec.items()}


def figure(rec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    t = rec["t"] / 60.0
    for key, label, color in CHAIN:
        a.plot(t, rec[key], "-o", ms=3, color=color, label=label)
    a.set_title("NFsim rule-based flagellar assembly — staged intermediates")
    a.set_xlabel("time (min)"); a.set_ylabel("count"); a.legend(fontsize=8)

    for key, label, color in MONOMERS:
        b.plot(t, rec[key], "-", color=color, label=label)
    b.plot(t, rec["flagella"], "-o", ms=4, color="#d62728", lw=2, label="complete flagella")
    # Free monomer pools (thousands of FlgE etc.) dwarf the handful of assembled
    # flagella on a linear axis — Maya's "scale seems unfair, can't see the complete
    # flagella." A symlog y-axis keeps the low-count assembled structures legible
    # alongside the large free-monomer pools.
    b.set_yscale("symlog", linthresh=10)
    b.set_title("Free monomer pools vs assembled flagella  (symlog y — pools ≫ flagella)")
    b.set_xlabel("time (min)"); b.set_ylabel("count (symlog)"); b.legend(fontsize=8)
    fig.tight_layout()

    out = f"{STUDY_DIR}/charts/01_nfsim_assembly.svg"
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # Stacked-area view: cumulative assembly hierarchy over time.
    fig2, ax = plt.subplots(figsize=(9, 4.8))
    labels = [c[1] for c in CHAIN]
    colors = [c[2] for c in CHAIN]
    series = [rec[c[0]] for c in CHAIN]
    ax.stackplot(t, *series, labels=labels, colors=colors, alpha=0.85)
    ax.set_title("NFsim flagellar assembly — cumulative hierarchy (stacked)")
    ax.set_xlabel("time (min)"); ax.set_ylabel("count (stacked)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = f"{STUDY_DIR}/charts/02_nfsim_assembly_stacked.svg"
    fig2.savefig(out2, format="svg", bbox_inches="tight")
    plt.close(fig2)
    print("wrote", out2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=1800)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--n-steps", type=int, default=50)
    args = ap.parse_args()

    rec = run(args.seconds, args.sample, args.n_steps)
    figure(rec)
    final = {k: int(rec[k][-1]) for k, _l, _c in CHAIN}
    print("final assembly state:", final)


if __name__ == "__main__":
    main()
