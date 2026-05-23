"""iML1515 × Beulig 2025 comparison.

Filters the Beulig WT batch-phase (fed-batch time h < 0) trajectories from
references/papers/palsson-2025-supp/process_summary_interpol.csv, runs
iML1515 FBA at glucose / O2 uptake bounds reverse-engineered from the
measured glucose-uptake-rate + OTR per sample, and compares predicted vs
measured μ at each time point.

This is the FAST METABOLISM HALF of the substitutability story — what an
M-model predicts about Beulig's actual fermentation, on a regime where
v2ecoli's full whole-cell sim is impractical to run point-by-point.

Outputs:
* reports/runnable_sims/iml1515_vs_beulig.json — paired (measured, predicted)
* reports/runnable_sims/iml1515_vs_beulig.png  — parity plot + per-time series
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
BEULIG_CSV = WORKSPACE / "references" / "papers" / "palsson-2025-supp" / "process_summary_interpol.csv"
OUT_DIR    = WORKSPACE / "reports" / "runnable_sims"


def _f(v) -> float | None:
    try:
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def load_beulig_batch_phase_samples(max_samples: int = 30) -> list[dict]:
    """Load Beulig's interpolated time-aligned rows, keeping only those with
    enough numeric data to drive iML1515 (need measured glucose-uptake rate
    AND a measured growth rate to compare against).
    """
    out: list[dict] = []
    with open(BEULIG_CSV, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            t = _f(row.get("fed-batch time [h]"))
            mu = _f(row.get("growth rate [1/h]"))
            otr = _f(row.get("OTR [mol/h]"))
            ctr = _f(row.get("CTR [mol/h]"))
            uptake_cmol_h = _f(row.get("D-glucose uptake rate [cmol/h]"))
            od = _f(row.get("OD [-]"))
            glc = _f(row.get("D-glucose [mmol/L]"))
            # The interesting comparison is the BATCH phase (t < 0). At t > 0
            # the population is so dense that iML1515's per-cell μ is no
            # longer the right comparison (see Beulig's maintenance-burden
            # framing — that's what OxidizeME would model).
            if t is None or mu is None or otr is None or uptake_cmol_h is None or od is None:
                continue
            if t > 0:
                continue
            out.append({
                "reactor_id": row.get("reactor_id", ""),
                "time_h": t,
                "mu_measured": mu,
                "OTR_molh": otr,
                "CTR_molh": ctr,
                "glucose_uptake_cmol_h": uptake_cmol_h,
                "OD600": od,
                "glucose_mmolL": glc,
            })
    # Pick samples spanning the available range; if too many, downsample.
    if len(out) > max_samples:
        step = len(out) // max_samples
        out = out[::step]
    return out


def run_iml1515(samples: list[dict]) -> list[dict]:
    """For each measured sample, set iML1515's EX_glc__D_e + EX_o2_e lower
    bounds to the measured uptake rates (converted to per-gDW basis), solve
    FBA, capture predicted μ + acetate flux.

    Bound conversion:
      glucose_uptake_cmol_h is measured in cmol_glucose / h.
        cmol_glucose / 6 = mmol_glucose / h (glucose has 6 C).
        Divide by reactor biomass (≈ OD × od_to_gdw × volume; we don't have
        per-sample volume, so we normalize to per-OD instead and use a
        nominal volume).
      OTR is mol_O2 / h (reactor-scale), same conversion via biomass.

    These conversions are imperfect (paper-level details elide reactor volume
    per sample) — the goal here is to show iML1515 RESPONDS to Beulig's
    actual uptake rates, not to claim quantitative match. Treat the
    comparison as ORDER-OF-MAGNITUDE PARITY, not calibrated overlap.
    """
    import cobra
    print("[iml1515] loading iML1515…")
    model = cobra.io.load_model("iML1515")

    # Nominal conversions
    OD_TO_GDW = 0.34         # paper's value (mSystems Methods)
    NOMINAL_VOLUME_L = 0.05  # AMBR-scale; only used to convert reactor flux → per-gDW
    glc_bound_floor = -25.0  # don't exceed iML1515's effective max glucose uptake

    out: list[dict] = []
    for s in samples:
        biomass_gDW = s["OD600"] * OD_TO_GDW * NOMINAL_VOLUME_L  # g
        if biomass_gDW < 1e-4:
            continue   # too dilute, conversion unreliable
        # Measured glucose uptake → mmol_glucose / (gDW · h)
        glc_mmolh = (s["glucose_uptake_cmol_h"] / 6.0) * 1e3  # cmol → mmol → mmol_glucose
        glc_per_gDW = glc_mmolh / biomass_gDW
        # Measured OTR → mmol_O2 / (gDW · h)
        o2_mmolh = s["OTR_molh"] * 1e3
        o2_per_gDW = o2_mmolh / biomass_gDW
        # Set EX bounds (lower bound = -uptake)
        glc_lb = max(glc_bound_floor, -glc_per_gDW)
        o2_lb  = max(-1000.0,         -o2_per_gDW)

        model.reactions.get_by_id("EX_glc__D_e").lower_bound = glc_lb
        model.reactions.get_by_id("EX_o2_e").lower_bound     = o2_lb
        sol = model.optimize()
        mu_pred = float(sol.objective_value) if sol.objective_value is not None else float("nan")
        # Acetate excretion (positive = excretion)
        try:
            ac_flux = float(sol.fluxes["EX_ac_e"])
        except Exception:
            ac_flux = float("nan")

        out.append({
            **s,
            "biomass_estimate_gDW": biomass_gDW,
            "glc_uptake_per_gDW": glc_per_gDW,
            "o2_uptake_per_gDW": o2_per_gDW,
            "glc_lb_set": glc_lb,
            "o2_lb_set":  o2_lb,
            "mu_predicted": mu_pred,
            "acetate_flux_predicted": ac_flux,
            "solver_status": sol.status,
        })
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[iml1515-vs-beulig] loading Beulig batch-phase samples…")
    samples = load_beulig_batch_phase_samples(max_samples=40)
    print(f"  → {len(samples)} batch-phase samples with measured μ + OTR + uptake")
    if not samples:
        print("  no samples to compare; aborting")
        return 1

    t0 = time.perf_counter()
    pairs = run_iml1515(samples)
    wall = time.perf_counter() - t0
    print(f"[iml1515-vs-beulig] {len(pairs)} FBA solves in {wall:.2f} s")

    out_json = OUT_DIR / "iml1515_vs_beulig.json"
    out_json.write_text(json.dumps(pairs, indent=2, default=str))
    print(f"[iml1515-vs-beulig] wrote {out_json}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[iml1515-vs-beulig] matplotlib unavailable; skipping chart")
        return 0

    mu_m = [p["mu_measured"]    for p in pairs]
    mu_p = [p["mu_predicted"]   for p in pairs]
    ac_p = [p["acetate_flux_predicted"] for p in pairs]
    t    = [p["time_h"]         for p in pairs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(mu_m, mu_p, s=22, alpha=0.7, c="#6366f1", edgecolor="#3730a3")
    lim = max(max(mu_m, default=1.0), max(mu_p, default=1.0)) * 1.15
    axes[0].plot([0, lim], [0, lim], "--", color="#9ca3af", lw=1, label="y = x")
    axes[0].set_xlabel("measured μ (Beulig 2025) [1/h]")
    axes[0].set_ylabel("predicted μ (iML1515 @ matched conditions) [1/h]")
    axes[0].set_title("Predicted vs measured μ — parity plot")
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(t, mu_m, s=14, alpha=0.7, c="#7c3aed", label="measured", edgecolor="#5b21b6")
    axes[1].scatter(t, mu_p, s=14, alpha=0.7, c="#10b981", label="iML1515 predicted", edgecolor="#065f46")
    axes[1].set_xlabel("fed-batch time [h]   (negative = batch phase)")
    axes[1].set_ylabel("μ [1/h]")
    axes[1].set_title("μ over batch phase")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].scatter(t, ac_p, s=14, alpha=0.7, c="#ef4444", edgecolor="#991b1b")
    axes[2].set_xlabel("fed-batch time [h]")
    axes[2].set_ylabel("acetate flux EX_ac_e [mmol/(gDW·h)]")
    axes[2].set_title("iML1515 acetate excretion (overflow signature)")
    axes[2].grid(alpha=0.3)
    axes[2].axhline(0, color="#9ca3af", lw=0.5)

    fig.suptitle(
        f"iML1515 × Beulig 2025 batch-phase comparison · {len(pairs)} samples · {wall:.1f}s",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = OUT_DIR / "iml1515_vs_beulig.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[iml1515-vs-beulig] wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
