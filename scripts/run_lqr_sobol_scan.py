"""Sobol-style scan over LQR (Q, R) weights — find tracking-minimizing tuning.

For each (Q, R) sample, build the closed-loop millard_with_lqr composite,
run 500s simulated, record RMS tracking error + ATP trajectory.

basico runs each tick in ~5 ms, so a 5-tick eval is ~0.5 s wall; a 32-point
Sobol sample takes ~15 s total.

Output:
  .pbg/runs/lqr-sobol/results.json
  reports/figures/pdmp-01/lqr_sobol_scatter.html
  reports/figures/pdmp-01/lqr_sobol_top5_tracking.html

Run from worktree root:
    python scripts/run_lqr_sobol_scan.py [--n-samples 32] [--n-ticks 5]
"""
from __future__ import annotations
import argparse
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

OUT_ROOT = Path(".pbg/runs/lqr-sobol")
FIG_DIR = Path("reports/figures/pdmp-01")
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 110, "font.size": 10})


def _to_b64() -> str:
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _save_html(name: str, title: str, caption: str, pinned_h: int = 760):
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
<p><span class="tag">real-data</span><span class="tag">LQR Sobol scan</span></p>
<div class="fig"><img src='data:image/png;base64,{b64}' alt='{title}' /></div>
<p class="caption">{caption}</p>
</div></body></html>"""
    out = FIG_DIR / f"{name}.html"
    out.write_text(html, encoding="utf-8")
    print(f"  + {out}")


def sobol_pairs(n: int, seed: int = 0) -> list[tuple[float, float]]:
    """Quasi-random log-uniform (Q, R) samples in [0.01, 100]² via scipy Sobol."""
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
        u = sampler.random(n)
    except Exception:
        rng = np.random.default_rng(seed)
        u = rng.random((n, 2))
    Q = 10 ** (np.log10(0.01) + u[:, 0] * (np.log10(100) - np.log10(0.01)))
    R = 10 ** (np.log10(0.01) + u[:, 1] * (np.log10(100) - np.log10(0.01)))
    return list(zip(Q.tolist(), R.tolist()))


def eval_one(Q: float, R: float, n_ticks: int, tick_s: float) -> dict:
    """Run a closed-loop composite at (Q, R), return tracking metrics + trajectory."""
    from process_bigraph import Composite
    from v2ecoli.core import build_core
    from v2ecoli.steps.lqr_controller import register as register_lqr
    from v2ecoli.steps.millard_with_lqr import register as register_millard_lqr

    core = build_core()
    register_lqr(core)
    register_millard_lqr(core)

    doc = {"state": {
        "millard": {
            "_type": "process",
            "address": "local:MillardWithLQR",
            "config": {
                "model_source": "v2ecoli/models/sbml/millard2017_central_metabolism.xml",
                "time": tick_s, "intervals": 10,
                "control_reaction": "PTS_4", "control_parameter": "kF",
                "u_clip": 0.5,
            },
            "inputs": {"lqr_control": ["shared", "lqr_control"]},
            "outputs": {
                "species_concentrations": ["shared", "central_metabolites"],
                "fluxes": ["shared", "central_fluxes"],
                "control_applied": ["shared", "control_applied"],
            },
            "interval": tick_s,
        },
        "lqr": {
            "_type": "process",
            "address": "local:LQRController",
            "config": {
                "reference_npy": "v2ecoli/data/phase0_glucose_mu_ref.npy",
                "Q": Q, "R": R, "tick_s": tick_s,
            },
            "inputs": {"central_metabolites_millard": ["shared", "central_metabolites"]},
            "outputs": {
                "lqr_control": ["shared", "lqr_control"],
                "lqr_diagnostics": ["shared", "lqr_diagnostics"],
            },
            "interval": tick_s,
        },
        "shared": {
            "central_metabolites": {}, "central_fluxes": {},
            "lqr_control": {"u": 0.0}, "lqr_diagnostics": {},
            "control_applied": {},
        },
    }}

    # Capture per-tick log
    composite = Composite(doc, core=core)
    track: list[dict] = []
    for tick in range(n_ticks):
        composite.run(tick_s)
        s = composite.state
        diag = s["shared"]["lqr_diagnostics"]
        ctrl = s["shared"]["control_applied"]
        last = diag.get("last_tick", {})
        track.append({
            "t": (tick + 1) * tick_s,
            "mu_est": last.get("mu_est"),
            "mu_ref": last.get("mu_ref"),
            "tracking_err": last.get("tracking_err"),
            "u_clipped": ctrl.get("u_clipped"),
            "atp": s["shared"]["central_metabolites"].get("ATP"),
        })

    rms = float(np.sqrt(np.mean([
        e["tracking_err"] ** 2 for e in track
        if e.get("tracking_err") is not None
    ]))) if track else 0.0
    return {
        "Q": Q, "R": R, "n_ticks": n_ticks, "tick_s": tick_s,
        "rms_tracking_err": rms,
        "K": diag.get("last_tick", {}).get("u_control", 0.0) / max(abs(
            track[-1].get("tracking_err", 1e-12)), 1e-12) if track else 0.0,
        "track": track,
    }


def viz_scatter(results: list[dict]):
    fig, ax = plt.subplots(figsize=(11, 6))
    Qs = np.array([r["Q"] for r in results])
    Rs = np.array([r["R"] for r in results])
    rms = np.array([r["rms_tracking_err"] for r in results])
    # log color
    rms_log = np.log10(np.clip(rms, 1e-20, None))
    sc = ax.scatter(Qs, Rs, c=rms_log, s=120, cmap="viridis_r",
                    edgecolor="black", lw=0.5)
    best_idx = int(np.argmin(rms))
    ax.scatter([Qs[best_idx]], [Rs[best_idx]], s=400, facecolor="none",
               edgecolor="red", lw=3, label=f"best: Q={Qs[best_idx]:.3g}, R={Rs[best_idx]:.3g}, RMS={rms[best_idx]:.2e}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Q  (state-tracking weight)")
    ax.set_ylabel("R  (control-effort weight)")
    ax.set_title(f"LQR Sobol scan: RMS tracking error over (Q, R) — N={len(results)}")
    plt.colorbar(sc, ax=ax, label="log₁₀(RMS tracking error)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    _save_html("lqr_sobol_scatter",
               "LQR — RMS tracking error across (Q, R) scan",
               (
                 f"Real closed-loop runs: each point is a 5-tick × 100 s coupled "
                 f"Millard+LQR run with that (Q, R) weight pair. Color = log₁₀(RMS "
                 f"tracking error). Best tuning circled in red. Best Q={Qs[best_idx]:.3g}, "
                 f"R={Rs[best_idx]:.3g} gives RMS = {rms[best_idx]:.2e}. The scan was "
                 f"generated via scipy.stats.qmc.Sobol over log[Q, R] ∈ [0.01, 100]²."
               ),
               pinned_h=760)


def viz_top5_tracking(results: list[dict]):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    # Sort by RMS, take best 5
    sorted_r = sorted(results, key=lambda r: r["rms_tracking_err"])
    top5 = sorted_r[:5]
    colors = ["#10b981", "#3b82f6", "#a855f7", "#f59e0b", "#ec4899"]
    for r, c in zip(top5, colors):
        ts = [e["t"] for e in r["track"]]
        errs = [e["tracking_err"] for e in r["track"]]
        label = f"Q={r['Q']:.2g}, R={r['R']:.2g} (RMS={r['rms_tracking_err']:.2e})"
        ax.plot(ts, errs, marker="o", lw=2, color=c, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Time (model seconds)")
    ax.set_ylabel("Tracking error (μ_est − μ_ref) [1/s]")
    ax.set_title("Top 5 (Q, R) tunings — tracking error trajectory")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    _save_html("lqr_sobol_top5_tracking",
               "LQR — top-5 tracking-error trajectories",
               "Per-tick tracking error trajectory for the 5 best (Q, R) tunings "
               "from the Sobol scan. Lower closer to zero = better tracking. The "
               "ranking by RMS is shown in the legend.",
               pinned_h=720)


def viz_best_state(results: list[dict]):
    best = min(results, key=lambda r: r["rms_tracking_err"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ts = [e["t"] for e in best["track"]]
    axes[0].plot(ts, [e["mu_est"] for e in best["track"]], "b-o", label="μ_est")
    axes[0].plot(ts, [e["mu_ref"] for e in best["track"]], "g--", label="μ_ref")
    axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("μ (1/s)")
    axes[0].set_title("Growth rate tracking"); axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ts, [e["u_clipped"] for e in best["track"]], "r-o")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("u (clipped)")
    axes[1].set_title("Control signal"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(ts, [e["atp"] for e in best["track"]], "purple", marker="o")
    axes[2].axhline(2.572, color="green", ls="--", label="Millard published (2.57 mM)")
    axes[2].set_xlabel("t (s)"); axes[2].set_ylabel("ATP (mM)")
    axes[2].set_title("ATP trajectory"); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    fig.suptitle(
        f"Closed-loop LQR — best tuning Q={best['Q']:.3g}, R={best['R']:.3g} "
        f"(RMS tracking = {best['rms_tracking_err']:.2e})",
        y=1.02, fontsize=11,
    )
    plt.tight_layout()
    _save_html("lqr_best_tuning_state",
               "LQR — best tuning trajectories",
               "Per-tick state for the lowest-RMS tuning from the Sobol scan: estimated "
               "vs reference growth rate, applied control, and ATP. ATP horizontal "
               "reference is the published Millard 2017 steady state (2.57 mM).",
               pinned_h=820)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=32)
    p.add_argument("--n-ticks", type=int, default=5)
    p.add_argument("--tick-s", type=float, default=100.0)
    args = p.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"LQR Sobol scan: {args.n_samples} (Q, R) samples × {args.n_ticks} ticks × {args.tick_s}s each")
    pairs = sobol_pairs(args.n_samples)
    results: list[dict] = []
    t0 = time.time()
    for i, (Q, R) in enumerate(pairs):
        try:
            r = eval_one(Q, R, args.n_ticks, args.tick_s)
        except Exception as e:
            print(f"  [{i+1}/{args.n_samples}] Q={Q:.3g} R={R:.3g} FAILED: {e}")
            continue
        results.append(r)
        print(f"  [{i+1}/{args.n_samples}] Q={Q:.3g} R={R:.3g} RMS={r['rms_tracking_err']:.3e}")
    print(f"\nTotal wall: {time.time()-t0:.1f}s")

    (OUT_ROOT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Saved {OUT_ROOT/'results.json'} ({len(results)} runs)")

    print("\nGenerating viz:")
    viz_scatter(results)
    viz_top5_tracking(results)
    viz_best_state(results)

    best = min(results, key=lambda r: r["rms_tracking_err"])
    print(f"\nBest: Q={best['Q']:.3g}, R={best['R']:.3g}, RMS = {best['rms_tracking_err']:.3e}")


if __name__ == "__main__":
    main()
