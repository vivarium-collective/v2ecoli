"""Generate 'what happened' figures for the Millard kinetic-metabolism work.

Real short runs of the built composites; PNGs saved under docs/figures/millard/.
"""
import warnings; warnings.filterwarnings("ignore")
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from v2ecoli import build_composite

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures", "millard")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)


def _f(x):
    return float(x.magnitude) if hasattr(x, "magnitude") else float(x)


def _agent(c):
    agents = c.state.get("agents") or {}
    # agent '0' persists for short runs; fall back to first key
    return agents.get("0") or (next(iter(agents.values())) if agents else {})


# ---------------------------------------------------------------------------
# Figures 1 & 2: baseline_millard growth + central fluxes
# ---------------------------------------------------------------------------
def fig_growth_and_fluxes(n_steps=60):
    print(f"[1&2] baseline_millard, {n_steps} steps ...")
    c = build_composite("ecoli_millard", seed=0, cache_dir="out/cache")
    steps, mass = [], []
    flux_keys = ["PTS_4", "CYTBO", "PGI", "PYK"]
    flux_series = {k: [] for k in flux_keys}
    for i in range(n_steps):
        c.run(1)
        ag = _agent(c)
        cm = ag.get("listeners", {}).get("mass", {}).get("cell_mass", 0.0)
        steps.append(i + 1)
        mass.append(_f(cm))
        fl = ag.get("central_fluxes") or {}
        for k in flux_keys:
            flux_series[k].append(_f(fl.get(k, float("nan"))))
    print(f"      cell_mass {mass[0]:.1f} -> {mass[-1]:.1f} fg")
    for k in flux_keys:
        vals = flux_series[k]
        print(f"      {k}: {vals[0]:.4g} -> {vals[-1]:.4g}")

    # Fig 1: mass stability (growth not resolved at this timescale)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, mass, "-o", ms=3, color="#1f6feb", lw=1.8)
    ax.set_xlabel("Simulation step (1 s/step)")
    ax.set_ylabel("Cell mass (fg)")
    ax.set_ylim(mass[0] - 12, mass[0] + 12)  # honest band; 0.1 fg wiggle reads flat
    ax.set_title("Millard-ODE WCM runs stably (baseline_millard)")
    ax.grid(alpha=0.3)
    ax.annotate(
        "Central-carbon metabolism = Millard kinetic ODE (replaces FBA).\n"
        f"cell_mass {mass[0]:.1f}→{mass[-1]:.1f} fg over {n_steps} s "
        f"(Δ={mass[-1]-mass[0]:+.2f} fg, {100*(mass[-1]-mass[0])/mass[0]:+.3f}%).\n"
        "Growth NOT resolved at 60 s (cell cycle ~2700 s); figure shows the\n"
        "WCM stays stable under the metabolism swap, not visible growth.",
        xy=(0.02, 0.98), xycoords="axes fraction", va="top", ha="left",
        fontsize=8.5, bbox=dict(boxstyle="round", fc="#fff8c5", ec="#d4a72c"))
    fig.tight_layout()
    p1 = os.path.join(OUT, "01_metabolism_swap_growth.png")
    fig.savefig(p1, dpi=130); plt.close(fig)
    print("      wrote", p1)

    # Fig 2: central fluxes
    labels = {
        "PTS_4": "PTS_4 (glucose uptake)",
        "CYTBO": "CYTBO (O2 respiration)",
        "PGI": "PGI (glycolysis)",
        "PYK": "PYK (pyruvate kinase)",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in flux_keys:
        ax.plot(steps, flux_series[k], "-", lw=1.8, label=labels[k])
    ax.set_xlabel("Simulation step (1 s/step)")
    ax.set_ylabel("Reaction flux (mM/s, model units)")
    ax.set_title("Millard central-carbon fluxes are live (baseline_millard)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p2 = os.path.join(OUT, "02_central_fluxes.png")
    fig.savefig(p2, dpi=130); plt.close(fig)
    print("      wrote", p2)
    return mass, flux_series


# ---------------------------------------------------------------------------
# Figure 3: step-level glucose responsiveness
# ---------------------------------------------------------------------------
def fig_env_responsiveness():
    print("[3] glucose sweep (step-level) ...")
    from v2ecoli.core import build_core
    from v2ecoli.steps.millard_pdmp_metabolism import MillardPDMPMetabolism
    core = build_core()
    glc_mM = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    uptake = []
    for g in glc_mM:
        s = MillardPDMPMetabolism(config={}, core=core)
        out = s.update(
            {
                "lqr_control": {},
                "bulk": None,
                "listeners_mass": {"cell_mass": 1000.0, "dry_mass": 300.0},
                "external_concentrations": {"GLCx": g},
            },
            1.0,
        )
        uptake.append(abs(_f(out["central_fluxes"]["PTS_4"])))
        print(f"      GLCx={g:>6} mM -> |PTS_4|={uptake[-1]:.4g}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(glc_mM, uptake, "-o", color="#2da44e", lw=1.8, ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("External glucose GLCx (mM, log scale)")
    ax.set_ylabel("Glucose-uptake flux |PTS_4| (mM/s)")
    ax.set_title("Millard kinetics respond to the environment (glucose sweep)")
    ax.grid(alpha=0.3, which="both")
    ax.annotate(
        f"|PTS_4| {uptake[0]:.3g}→{uptake[-1]:.3g}\n"
        "as GLCx 0.05→20 mM",
        xy=(0.02, 0.98), xycoords="axes fraction", va="top", ha="left",
        fontsize=9, bbox=dict(boxstyle="round", fc="#fff8c5", ec="#d4a72c"))
    fig.tight_layout()
    p3 = os.path.join(OUT, "03_env_responsiveness.png")
    fig.savefig(p3, dpi=130); plt.close(fig)
    print("      wrote", p3)
    return glc_mM, uptake


# ---------------------------------------------------------------------------
# Figure 4: reactor O2 consumption
# ---------------------------------------------------------------------------
def fig_reactor_o2(n_steps=60):
    print(f"[4] reactor_bird_coupled_millard, {n_steps} steps ...")
    from v2ecoli.steps.reactor_cell_coupler import O2_EXCHANGE_KEY
    c = build_composite(
        "reactor_bird_coupled_millard", seed=0, cache_dir="out/cache",
        cells_per_agent=1.0e11,
        bird_reactor_config={"gas_flow_rate_Lpm": 2.0},
    )
    steps, do2, sat, o2x = [], [], [], []
    for i in range(n_steps):
        c.run(1)
        r = c.state["reactor"]
        steps.append(i + 1)
        do2.append(_f(r["dissolved_o2"]))
        sat.append(_f(r["o2_saturation"]))
        exch = c.state["agents"]["0"]["environment"]["exchange"]
        o2x.append(float(exch.get(O2_EXCHANGE_KEY, 0.0)))
    print(f"      dissolved_o2 {do2[0]:.4g} -> {do2[-1]:.4g} (sat~{sat[-1]:.4g})")
    print(f"      O2 exchange {o2x[0]:.4g} -> {o2x[-1]:.4g} counts/step")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))
    axL.plot(steps, do2, "-o", ms=3, color="#1f6feb", lw=1.8,
             label="reactor.dissolved_o2")
    axL.plot(steps, sat, "--", color="#888", lw=1.6, label="o2_saturation (C*)")
    axL.set_xlabel("Coupled step (1 s/step)")
    axL.set_ylabel("Dissolved O2 (mg/L)")
    axL.set_title("Reactor DO falls below saturation under cell load")
    axL.grid(alpha=0.3); axL.legend(fontsize=9)

    axR.plot(steps, o2x, "-o", ms=3, color="#cf222e", lw=1.8)
    axR.axhline(0, color="#888", lw=1)
    axR.set_xlabel("Coupled step (1 s/step)")
    axR.set_ylabel("Agent O2 exchange (counts/step, −=uptake)")
    axR.set_title("Millard cell O2 uptake (CYTBO respiration)")
    axR.grid(alpha=0.3)
    fig.suptitle("Millard cell consumes reactor O2 end-to-end "
                 "(reactor_bird_coupled_millard, cells/agent=1e11)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p4 = os.path.join(OUT, "04_reactor_o2_consumption.png")
    fig.savefig(p4, dpi=130); plt.close(fig)
    print("      wrote", p4)
    return do2, sat, o2x


if __name__ == "__main__":
    mass, fluxes = fig_growth_and_fluxes(60)
    glc, uptake = fig_env_responsiveness()
    do2, sat, o2x = fig_reactor_o2(60)
    print("DONE")
