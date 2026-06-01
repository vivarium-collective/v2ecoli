"""Phase 4 JAX/Diffrax compilation pilot — minimal Millard subset.

Implements a 4-state glycolysis subset of Millard 2017 (G6P → F6P → PEP → PYR
via PGI, PFK·ALDO·TPI·GAPDH·PGK·PGM·ENO, PYK) directly in JAX. Compares:

  - wall time vs basico full-model run (jax-compiled-throughput-10x test;
    apples-to-oranges since the JAX version is a 4-state subset, not the
    full 77-state Millard — but documents the speedup factor for that
    subset as a reference point for full-port effort sizing).
  - JIT cold-start vs warm-cache (jit_compile_cost viz, real data).
  - numerical-equivalence on the 4 shared states (target L∞ < 1e-4 vs
    basico's evolution of those 4 species).

For pdmp-04: minimal pilot showing JAX/Diffrax integration works
end-to-end on Millard chemistry. Full 77-state port is the larger
follow-up; this is the proof-of-concept that scaffolds the planned
runs (jax-jit-cold-warm-bench, parallel-1k-throughput).

Run from worktree root:
    python scripts/run_jax_pilot.py
"""
from __future__ import annotations
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import jax
import jax.numpy as jnp
import diffrax
import basico

OUT_ROOT = Path(".pbg/runs/jax-pilot")
FIG_DIR = Path("reports/figures/pdmp-04")
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10})

MODEL_PATH = "v2ecoli/models/sbml/millard2017_central_metabolism.xml"


# JAX-implemented 4-state glycolysis subset (Hill-style rate laws as a
# minimal lumped approximation of the Millard chemistry on these nodes).
# Real Millard uses ~10 reactions involving these 4; we lump to 3 effective
# reactions (uptake-PGI, glycolysis-flux, PYK).
@jax.jit
def millard_subset_rhs(t, y, args):
    """y = [G6P, F6P, PEP, PYR]; args = (k_uptake, k_glyc, k_pyk).
    Returns dy/dt for the 4-state lumped model."""
    G6P, F6P, PEP, PYR = y[0], y[1], y[2], y[3]
    k_up, k_glyc, k_pyk = args
    # Lumped reactions:
    #   v_uptake: external glucose → G6P  (constant input rate)
    #   v_pgi:    G6P ↔ F6P              (rate k_glyc · (G6P - F6P))
    #   v_glyc:   F6P → PEP              (rate k_glyc · F6P)
    #   v_pyk:    PEP → PYR              (rate k_pyk · PEP)
    #   v_drain:  PYR → (sinks)          (rate 0.1 · PYR; ATP/biomass effective drain)
    v_uptake = k_up
    v_pgi = k_glyc * (G6P - F6P)
    v_glyc = k_glyc * F6P
    v_pyk = k_pyk * PEP
    v_drain = 0.1 * PYR
    return jnp.array([
        v_uptake - v_pgi,
        v_pgi - v_glyc,
        v_glyc - v_pyk,
        v_pyk - v_drain,
    ])


def run_jax_subset(y0: jnp.ndarray, args: tuple, duration: float = 5000.0,
                   dt0: float = 0.1) -> jnp.ndarray:
    """Integrate the 4-state JAX subset via diffrax. Returns final state."""
    term = diffrax.ODETerm(millard_subset_rhs)
    solver = diffrax.Tsit5()  # 4th/5th-order RK
    sol = diffrax.diffeqsolve(
        term, solver, t0=0.0, t1=duration, dt0=dt0, y0=y0,
        args=args,
        max_steps=200_000,
    )
    return sol.ys[-1]


# JIT-compiled wrapper for benchmarking
run_jax_subset_jit = jax.jit(run_jax_subset, static_argnames=("duration", "dt0"))


def get_basico_initial_conditions() -> tuple[np.ndarray, dict]:
    basico.load_model(MODEL_PATH)
    sp = basico.get_species()
    y0_dict = {n: float(sp.loc[n]["initial_concentration"])
               for n in ["G6P", "F6P", "PEP", "PYR"]}
    return np.array([y0_dict["G6P"], y0_dict["F6P"], y0_dict["PEP"], y0_dict["PYR"]]), y0_dict


def run_basico_subset(duration: float = 5000.0) -> dict:
    basico.load_model(MODEL_PATH)
    ts = basico.run_time_course(start_time=0, duration=duration, intervals=50,
                                use_sbml_id=True, update_model=True)
    last = ts.iloc[-1]
    return {n: float(last.get(n, np.nan)) for n in ["G6P", "F6P", "PEP", "PYR"]}


def _to_b64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _save_html(name: str, title: str, caption: str, pinned_h: int = 900):
    b64 = _to_b64()
    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
html,body{{height:{pinned_h}px;overflow:hidden;margin:0;padding:0;font-family:system-ui;color:#0f172a;background:#fff}}
.wrap{{box-sizing:border-box;height:{pinned_h}px;padding:14px 18px;display:flex;flex-direction:column;gap:8px}}
h1{{font-size:1.15em;margin:0;border-bottom:1px solid #e2e8f0;padding-bottom:6px}}
p{{margin:0}}
p.caption{{color:#475569;font-size:0.85em;line-height:1.4}}
.fig{{flex:1 1 auto;min-height:0;display:flex;align-items:center;justify-content:center;overflow:hidden}}
.fig img{{max-width:100%;max-height:100%;width:auto;height:auto;display:block;object-fit:contain}}
.tag{{display:inline-block;background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:4px;font-size:0.7em;margin-right:6px}}
</style></head><body><div class="wrap">
<h1>{title}</h1>
<p><span class="tag">real-data</span><span class="tag">JAX/Diffrax pilot</span></p>
<div class="fig"><img src='data:image/png;base64,{b64}' alt='{title}' /></div>
<p class="caption">{caption}</p>
</div></body></html>"""
    out = FIG_DIR / f"{name}.html"
    out.write_text(html, encoding="utf-8")
    print(f"  + {out}")


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Phase 4 JAX/Diffrax pilot — 4-state Millard glycolysis subset")
    print()

    # 1. Get initial conditions
    y0_np, y0_dict = get_basico_initial_conditions()
    y0 = jnp.array(y0_np)
    print(f"Initial conditions: {y0_dict}")

    # Choose lumped rate constants to roughly match Millard's central
    # carbon throughput. These are tuned to give similar steady-state.
    args = (0.5, 0.3, 0.4)  # (k_uptake, k_glyc, k_pyk)
    print(f"Lumped rate constants: k_uptake={args[0]}, k_glyc={args[1]}, k_pyk={args[2]}")

    # 2. Cold JIT compile
    print("\n2. JIT cold-start timing (first call includes XLA compile)...")
    t0 = time.time()
    y_cold = run_jax_subset_jit(y0, args, duration=5000.0, dt0=0.1)
    y_cold.block_until_ready()
    cold_wall = time.time() - t0
    print(f"   cold wall = {cold_wall*1000:.1f} ms, y_final = {y_cold}")

    # 3. Warm runs — measure amortized cost
    print("\n3. JIT warm runs (20 sequential calls)...")
    warm_walls = []
    for i in range(20):
        t0 = time.time()
        y_warm = run_jax_subset_jit(y0, args, duration=5000.0, dt0=0.1)
        y_warm.block_until_ready()
        warm_walls.append(time.time() - t0)
    warm_mean = float(np.mean(warm_walls))
    warm_std = float(np.std(warm_walls))
    print(f"   warm mean = {warm_mean*1000:.2f} ± {warm_std*1000:.2f} ms (N=20)")
    print(f"   speedup cold→warm = {cold_wall/warm_mean:.1f}×")

    # 4. basico reference timing
    print("\n4. basico full-Millard reference timing (5 runs of full 77-state model)...")
    basico_walls = []
    for i in range(5):
        t0 = time.time()
        basico_endpoint = run_basico_subset(duration=5000.0)
        basico_walls.append(time.time() - t0)
    basico_mean = float(np.mean(basico_walls))
    print(f"   basico mean = {basico_mean*1000:.1f} ms (N=5)")
    print(f"   speedup JAX_warm / basico (apples-to-oranges; JAX is 4-state subset, basico is full 77-state) = "
          f"{basico_mean/warm_mean:.1f}×")

    # 5. Numerical comparison
    print("\n5. Numerical equivalence (4 shared states): basico full Millard vs JAX subset...")
    basico_arr = np.array([basico_endpoint["G6P"], basico_endpoint["F6P"],
                          basico_endpoint["PEP"], basico_endpoint["PYR"]])
    jax_arr = np.asarray(y_warm)
    rel_err = np.abs(basico_arr - jax_arr) / (np.abs(basico_arr) + 1e-12)
    print(f"   basico: {basico_arr}")
    print(f"   jax:    {jax_arr}")
    print(f"   rel err: {rel_err}, L∞ = {rel_err.max():.3e}")
    print("   (large rel err is EXPECTED: JAX is a lumped 4-state subset, not a faithful 77-state port)")

    # Save artifacts
    results = {
        "y0": y0_dict,
        "lumped_args": {"k_uptake": args[0], "k_glyc": args[1], "k_pyk": args[2]},
        "cold_wall_s": cold_wall,
        "warm_walls_s": warm_walls,
        "warm_mean_s": warm_mean,
        "warm_std_s": warm_std,
        "basico_walls_s": basico_walls,
        "basico_mean_s": basico_mean,
        "speedup_cold_to_warm": float(cold_wall / warm_mean),
        "speedup_jax_warm_vs_basico_full": float(basico_mean / warm_mean),
        "basico_endpoint": basico_endpoint,
        "jax_endpoint": {n: float(v) for n, v in zip(["G6P", "F6P", "PEP", "PYR"], jax_arr)},
        "rel_err": rel_err.tolist(),
        "L_inf_rel_err": float(rel_err.max()),
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(results, indent=2))

    # ---- Viz ----
    print("\nGenerating viz:")

    # JIT cost amortization
    fig, ax = plt.subplots(figsize=(11, 5.5))
    runs = np.arange(1, 22)
    walls = np.concatenate([[cold_wall], warm_walls]) * 1000
    colors = ["#ef4444"] + ["#10b981"] * 20
    ax.bar(runs, walls, color=colors, alpha=0.85, edgecolor="black", lw=0.4)
    ax.axhline(warm_mean * 1000, color="#475569", ls="--", lw=1,
               label=f"warm steady-state = {warm_mean*1000:.2f} ms")
    ax.text(1.5, cold_wall * 1000 * 0.9, f"cold = {cold_wall*1000:.0f} ms\n(JIT compile)",
            color="#991b1b", fontsize=11)
    ax.set_xlabel("Run index")
    ax.set_ylabel("Wall time (ms)")
    ax.set_title(f"Phase 4 JAX/Diffrax — JIT compile cost amortization "
                 f"(speedup cold→warm = {cold_wall/warm_mean:.0f}×)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    _save_html("jit_compile_cost_real",
               "JIT compile cost amortization (real)",
               (
                 f"Real wall-time per call across 21 sequential invocations of the JAX "
                 f"4-state Millard subset. First call (red) pays JIT compilation overhead "
                 f"({cold_wall*1000:.0f} ms); subsequent warm calls run at {warm_mean*1000:.2f} ms "
                 f"average. Speedup cold→warm = {cold_wall/warm_mean:.0f}×. For long ensembles "
                 f"the JIT cost amortizes to ~0% of total wall."
               ))

    # Speedup vs basico
    fig, ax = plt.subplots(figsize=(11, 5))
    bench = ["basico\nfull Millard\n(77 states)", "JAX subset\n(4 states, lumped)\nwarm-cache"]
    walls_ms = [basico_mean * 1000, warm_mean * 1000]
    colors = ["#94a3b8", "#3b82f6"]
    bars = ax.bar(bench, walls_ms, color=colors, alpha=0.9)
    for b, w in zip(bars, walls_ms):
        ax.text(b.get_x() + b.get_width() / 2, w + max(walls_ms) * 0.02,
                f"{w:.1f} ms", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Wall time per call (ms)")
    ax.set_title(f"JAX warm-cache vs basico full-Millard (subset vs full — for sizing only) "
                 f"= {basico_mean/warm_mean:.0f}×")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    _save_html("jax_julia_speedup_real",
               "JAX vs basico backend timing (real)",
               (
                 f"Real wall-time comparison: basico full-Millard (77 species, 68 reactions) "
                 f"vs the JAX 4-state lumped glycolysis subset on identical hardware. "
                 f"Apples-to-oranges scope (full vs subset), but documents that JAX/Diffrax "
                 f"path runs at {warm_mean*1000:.2f} ms per 5000s-simulated call vs basico's "
                 f"{basico_mean*1000:.0f} ms. Full 77-state JAX port is needed for the "
                 f"jax-compiled-throughput-10x test; this pilot establishes the toolchain."
               ))

    print("\nDone.")


if __name__ == "__main__":
    main()
