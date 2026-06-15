"""Beulig 2025 comparison harness — coupled-arm sim vs Beulig WT batch prefix.

mbp-09 req-3 (also mbp-05's core comparison harness). Runs a coupled reactor arm
(``wcm`` = ``reactor_bird_coupled``; ``millard`` = ``reactor_bird_coupled_millard``),
aligns its observables against the digitized Beulig WT *batch-phase* (fed-batch
time h < 0) trajectories declared in mbp-05's ``comparison_overlays`` spec, grades
each axis with the EXISTING report-card grader
(``v2ecoli.library.card_criteria.grade_axis`` + ``report_card.verdict_json``), and
writes a ``report_card_verdict.json`` (+ overlay charts) at the card path each
study references:

  * ``wcm``     -> ``docs/report_cards/beulig_batch/vs_beulig``        (mbp-05 + mbp-09 WCM)
  * ``millard`` -> ``docs/report_cards/beulig_batch/millard_vs_beulig``(mbp-09 Millard)
  * ``iml1515`` -> ``docs/report_cards/beulig_batch/iml1515_vs_beulig``(mbp-09 genome-scale)

iML1515 arm — NOT apples-to-apples (read this)
----------------------------------------------
The ``iml1515`` arm is the Palsson genome-scale FBA model (cobra ``load_model``),
NOT a process-bigraph composite. It does NOT predict the batch trajectory from
first principles: for each Beulig batch-phase sample it SETS iML1515's glucose +
O2 uptake **lower bounds from the MEASURED Beulig uptake rates** and solves FBA,
so the only genuine PREDICTIONS are growth μ and acetate flux. We grade μ
(predicted-vs-measured, ``r2`` criterion) and mark the uptake INPUTS
(substrate/gas_transfer) honestly ``ungraded`` — grading an input as if it were a
prediction would be dishonest. Acetate is ``ungraded`` too (the Beulig batch
samples carry no measured acetate). This makes the iML1515 arm an
*easier / different* test than the wcm/millard arms (which predict the whole
trajectory from first principles); the 3-way comparison must NOT be read as
apples-to-apples. Reuses ``scripts/runnable_sims/iml1515_vs_beulig.py`` for the
sample loading + FBA bound-setting (no duplicated FBA setup).

SCOPE — SHORT-RUN SMOKE (important)
----------------------------------
This harness + the committed verdict are a SHORT-RUN SMOKE (default a few steps),
NOT the full ~33 h Beulig batch production run. The biological magnitudes are far
below Beulig (v2ecoli plateaus ~mg/L vs Beulig's batch peak ~9.6 g/L — the
documented architectural gap), so the graded axes are EXPECTED to verdict
"mismatch". That is the correct execute-and-report outcome (per
chris_feedback_2026_05_26 §10): divergences feed mbp-06, this is not a biology
gate. The full batch-phase run is a follow-up execution (likely on the mini).

Axis resolution (honest accounting — see ``AXIS_RESOLVERS`` / ``GROUP_OF``)
--------------------------------------------------------------------------
Of the 13 comparison_overlays, the coupled run only exposes a subset of the
declared ``sim_path``s. Resolved (graded) vs ungraded:

  GRADED   growth/OD600   sim population.biomass_concentration_gL vs OD x 0.34
  GRADED   substrate/GUR  sim glucose exchange (env.exchange['GLC'] counts ->
                          mmol/(gDW.h)) vs Beulig process_summary GUR per-biomass
  GRADED   gas_transfer/OTR,CTR  sim reactor.{o2,co2}_transport_delta ->
                          mmol/(L.h) vs Beulig reactor_{OTR,CTR}_data.csv
  GRADED   summary        boolean "comparison executed" (process-level)
  UNGRADED substrate/glucose-conc  sim environment.external_concentrations only
                          carries O2/CO2 (no glucose conc store) -> ungraded
  UNGRADED byproducts/*   no sim byproduct CONCENTRATION store (env.exchange is a
                          per-step COUNT flux, not a concentration); also Beulig
                          has NO acetate CSV -> all ungraded
  UNGRADED gas_transfer/DO  Beulig process_summary has NO dissolved-O2 column
  UNGRADED ph_control/base sim has no cumulative-base-addition store

Adding a resolvable overlay = adding one ``AXIS_RESOLVERS`` entry (the spec stays
the source of the overlay list).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from v2ecoli.library.report_card import grade_card, verdict_json

# --- paths ------------------------------------------------------------------

WS_ROOT = Path(__file__).resolve().parents[1]
BEULIG_DIR = WS_ROOT / "workspace" / "references" / "papers" / "palsson-2025-supp"
MBP05_STUDY = WS_ROOT / "workspace" / "studies" / "mbp-05-palsson-benchmark" / "study.yaml"
CARD_ROOT = WS_ROOT / "docs" / "report_cards" / "beulig_batch"

ARM_COMPOSITE = {
    "wcm": "reactor_bird_coupled",
    "millard": "reactor_bird_coupled_millard",
}
ARM_CARD_DIR = {
    "wcm": "vs_beulig",
    "millard": "millard_vs_beulig",
    "iml1515": "iml1515_vs_beulig",
}
# All gradeable arms (the two PB composite arms + the genome-scale FBA arm).
ALL_ARMS = list(ARM_COMPOSITE) + ["iml1515"]

# --- constants --------------------------------------------------------------

OD_TO_GDW = 0.34                     # Beulig 2025 value
MW_O2 = 31.999                       # g/mol == mg/mmol
MW_CO2 = 44.010
AVOGADRO = 6.02214076e23            # 1/mol
SECONDS_PER_HOUR = 3600.0
FG_PER_GRAM = 1.0e-15
# Nominal AMBR working volume for the per-biomass GUR conversion of the Beulig
# reactor-scale uptake (cmol/h). Mirrors scripts/runnable_sims/iml1515_vs_beulig.py
# (the paper elides per-sample volume); only used to convert reactor flux ->
# per-gDW basis. Approximate by design (does not flip the mismatch verdict).
NOMINAL_VOLUME_L = 0.05
DEFAULT_TOL_REL = 0.25               # execute-and-report band (lenient diagnostic)
DEFAULT_SMOKE_STEPS = 5
DEFAULT_CPA = 1.0e10                 # representative-sampling scale (coupling active)
# iML1515 arm — predicted-vs-measured μ grading band (model-validation style;
# documented diagnostic, NOT tuned to pass). R² vs the identity line y=x.
IML1515_R2_MIN = 0.70                # within_tol at/above
IML1515_R2_DRIFT = 0.30             # drift down to this; mismatch below
DEFAULT_IML1515_SAMPLES = 40         # Beulig batch-phase samples to solve FBA over


# --- low-level numeric helpers ---------------------------------------------

def _f(v):
    """Coerce a possibly pint-Quantity / None scalar to a bare float or None."""
    if v is None:
        return None
    if hasattr(v, "magnitude"):
        v = v.magnitude
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


# --- Beulig CSV loading -----------------------------------------------------

def load_wide_csv_batch_phase(path: str | Path) -> tuple[list[float], list[float]]:
    """Load a WIDE-format Beulig reactor CSV, restricted to the batch phase.

    Column 1 = fed-batch time [h] (header like ``"OD [-], fed-batch [h]"``);
    columns 2+ = per-reactor values. For each row with time < 0 (batch phase),
    average the non-empty reactor columns. Returns ``(times_h, means)`` sorted by
    time. NEGATIVE time = batch phase (mbp-05's scope); fed-batch (t >= 0) is
    dropped.
    """
    times: list[float] = []
    means: list[float] = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            return [], []
        for row in reader:
            if not row:
                continue
            t = _f(row[0])
            if t is None or t >= 0:
                continue
            vals = [x for x in (_f(c) for c in row[1:]) if x is not None]
            if not vals:
                continue
            times.append(t)
            means.append(sum(vals) / len(vals))
    order = sorted(range(len(times)), key=lambda i: times[i])
    return [times[i] for i in order], [means[i] for i in order]


def batch_reference_scalar(means: list[float], mode: str = "peak") -> float | None:
    """Reduce a batch-phase mean trajectory to a single reference scalar.

    ``peak`` = max absolute value (climb metrics: OD, byproducts, OTR/CTR, GUR);
    ``endpoint`` = last batch-phase value (drawdown metrics: glucose).
    """
    if not means:
        return None
    if mode == "endpoint":
        return means[-1]
    return max(means, key=abs)


def load_process_summary_gur_per_biomass(max_rows: int = 0) -> float | None:
    """Beulig GUR per-biomass (mmol/(gDW.h)) peak over the batch phase.

    Read process_summary_interpol.csv (long format) batch-phase rows that carry
    BOTH ``D-glucose uptake rate [cmol/h]`` and ``OD [-]``, convert each to a
    per-biomass basis, and return the peak. Conversion (mirrors
    iml1515_vs_beulig):

        mmol_glucose/h = (cmol/h) / 6 * 1000        # glucose has 6 carbons
        biomass_gDW    = OD * OD_TO_GDW * NOMINAL_VOLUME_L
        GUR            = mmol_glucose/h / biomass_gDW
    """
    path = BEULIG_DIR / "process_summary_interpol.csv"
    if not path.is_file():
        return None
    peak = None
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            t = _f(row.get("fed-batch time [h]"))
            upt = _f(row.get("D-glucose uptake rate [cmol/h]"))
            od = _f(row.get("OD [-]"))
            if t is None or t >= 0 or upt is None or od is None or od <= 0:
                continue
            mmol_h = (upt / 6.0) * 1000.0
            biomass_gDW = od * OD_TO_GDW * NOMINAL_VOLUME_L
            if biomass_gDW <= 0:
                continue
            gur = abs(mmol_h / biomass_gDW)
            if peak is None or gur > peak:
                peak = gur
    return peak


# --- published reference per axis (returns scalar in TARGET units, or None) --

def _pub_od_gdw() -> tuple[float | None, dict]:
    """Beulig batch OD600 peak -> gDW/L (x OD_TO_GDW). Matches the study's
    documented batch peak ~9.6 g/L (OD ~28 x 0.34)."""
    times, means = load_wide_csv_batch_phase(BEULIG_DIR / "reactor_OD_data.csv")
    od_peak = batch_reference_scalar(means, "peak")
    ref = od_peak * OD_TO_GDW if od_peak is not None else None
    return ref, {"od_peak": od_peak, "times": times, "means": means}


def _pub_gur() -> tuple[float | None, dict]:
    ref = load_process_summary_gur_per_biomass()
    return ref, {}


def _pub_transfer(csv_name: str, volume_L: float) -> tuple[float | None, dict]:
    """Beulig OTR/CTR (mol/h, reactor-scale) -> mmol/(L.h): /volume * 1000."""
    times, means = load_wide_csv_batch_phase(BEULIG_DIR / csv_name)
    peak_molh = batch_reference_scalar(means, "peak")
    ref = peak_molh / volume_L * 1000.0 if peak_molh is not None else None
    return ref, {"peak_mol_h": peak_molh, "times": times, "means": means}


# --- arm run + sim observable extraction ------------------------------------

def _agent0(state) -> dict:
    agents = state.get("agents") or {}
    if "0" in agents:
        return agents["0"]
    return next(iter(agents.values()), {}) if agents else {}


def _cell_mass_fg(agent_state) -> float | None:
    try:
        mass = (agent_state.get("listeners") or {}).get("mass") or {}
        cm = mass.get("cell_mass")
        if cm is None:
            return None
        if hasattr(cm, "to") and hasattr(cm, "magnitude"):
            return float(cm.to("femtogram").magnitude)
        return _f(cm)
    except (AttributeError, KeyError, TypeError):
        return None


def run_arm(arm: str, steps: int, cells_per_agent: float = DEFAULT_CPA) -> dict:
    """Build + step a coupled arm, recording the per-step series needed to derive
    each sim axis scalar. Returns a dict of derived sim observables (peaks)."""
    from v2ecoli import build_composite

    composite = ARM_COMPOSITE[arm]
    c = build_composite(composite, seed=0, cache_dir="out/cache",
                        cells_per_agent=cells_per_agent)

    biomass_gL: list[float] = []
    glc_counts: list[float] = []
    cell_mass: list[float] = []
    o2_delta: list[float] = []
    co2_delta: list[float] = []
    gtime: list[float] = []
    for _ in range(steps):
        c.run(1)
        st = c.state
        pop = st.get("population") or {}
        biomass_gL.append(_f(pop.get("biomass_concentration_gL")) or 0.0)
        r = st.get("reactor") or {}
        o2_delta.append(_f(r.get("o2_transport_delta")) or 0.0)
        co2_delta.append(_f(r.get("co2_transport_delta")) or 0.0)
        a0 = _agent0(st)
        exch = ((a0.get("environment") or {}).get("exchange") or {}) if a0 else {}
        glc_counts.append(_f(exch.get("GLC")) or 0.0)
        cell_mass.append(_cell_mass_fg(a0) or 0.0)
        gtime.append(_f(st.get("global_time")) or 0.0)
    volume_L = _f((c.state.get("reactor") or {}).get("volume_L")) or 1.0

    # Mean per-step duration (s) from global_time progression (fallback 1 s).
    dts = [b - a for a, b in zip(gtime, gtime[1:]) if (b - a) > 0]
    dt_s = (sum(dts) / len(dts)) if dts else 1.0

    # Derived sim axis scalars (peaks).
    # growth: peak biomass concentration (gDW/L) — directly comparable to OD x 0.34.
    sim_biomass = max(biomass_gL, default=0.0)

    # substrate/GUR: peak per-cell glucose uptake -> mmol/(gDW.h). Intensive
    # (cells_per_agent cancels). counts<0 = uptake.
    gur_series = []
    for counts, cm in zip(glc_counts, cell_mass):
        if cm <= 0 or counts >= 0:
            continue
        mmol_h = (-counts) / AVOGADRO * 1000.0 * (SECONDS_PER_HOUR / dt_s)
        gur_series.append(mmol_h / (cm * FG_PER_GRAM))
    sim_gur = max(gur_series, default=0.0)

    # gas_transfer: transport deltas (mg/L per 1 s transport interval) ->
    # mmol/(L.h). OTR uses the positive (into-liquid) transfer peak.
    sim_otr = max((d / MW_O2 * SECONDS_PER_HOUR for d in o2_delta), default=0.0)
    sim_ctr = max((abs(d) / MW_CO2 * SECONDS_PER_HOUR for d in co2_delta), default=0.0)

    return {
        "composite": composite,
        "volume_L": volume_L,
        "dt_s": dt_s,
        "steps": steps,
        "sim_biomass_gL": sim_biomass,
        "sim_gur": sim_gur,
        "sim_otr": sim_otr,
        "sim_ctr": sim_ctr,
        "series": {
            "biomass_gL": biomass_gL, "o2_delta": o2_delta,
            "co2_delta": co2_delta, "glc_counts": glc_counts,
        },
    }


# --- axis assembly ----------------------------------------------------------

# Resolvers for the overlays whose sim_path the coupled run actually exposes.
# Keyed by axis id. Each entry: group, label, units, criterion tol, published()
# returning (scalar, meta), and the sim-scalar key in run_arm()'s output.
def build_axes(sim: dict, tol_rel: float) -> dict:
    """Build the report_card `reference` axes dict + the measured `card`.

    Returns (reference, card) where reference is {axes: {path: {group,label,
    units,criterion,...}}} and card is the nested measured tree dig()'d by path.
    Ungraded overlays are still emitted (measured=None) so every group is present.
    """
    od_ref, od_meta = _pub_od_gdw()
    gur_ref, _ = _pub_gur()
    otr_ref, _ = _pub_transfer("reactor_OTR_data.csv", NOMINAL_VOLUME_L)
    ctr_ref, _ = _pub_transfer("reactor_CTR_data.csv", NOMINAL_VOLUME_L)

    def rel(ref):
        return {"type": "rel_tol", "reference": ref, "tol_rel": tol_rel}

    # (path, group, label, units, criterion, measured_value)
    rows = [
        # GRADED
        ("growth.biomass_gDW_per_L", "growth", "Biomass (OD x 0.34)", "gDW/L",
         rel(od_ref), sim["sim_biomass_gL"]),
        ("substrate.GUR", "substrate", "Glucose uptake rate", "mmol/(gDW*h)",
         rel(gur_ref), sim["sim_gur"]),
        ("gas_transfer.OTR", "gas_transfer", "O2 transfer rate (OTR)", "mmol/(L*h)",
         rel(otr_ref), sim["sim_otr"]),
        ("gas_transfer.CTR", "gas_transfer", "CO2 transfer rate (CTR)", "mmol/(L*h)",
         rel(ctr_ref), sim["sim_ctr"]),
        ("summary.comparison_executed", "summary", "Comparison executed (process-level)",
         "", {"type": "boolean"}, True),
        # UNGRADED (sim_path absent and/or no Beulig reference) — emitted so the
        # group is present; measured=None -> grade_axis returns 'ungraded'.
        ("substrate.glucose_conc", "substrate", "Glucose concentration", "g/L",
         rel(None), None),
        ("byproducts.acetate", "byproducts", "Acetate concentration", "g/L",
         rel(None), None),
        ("byproducts.lactate", "byproducts", "Lactate concentration", "mmol/L",
         rel(None), None),
        ("byproducts.formate", "byproducts", "Formate concentration", "mmol/L",
         rel(None), None),
        ("byproducts.ethanol", "byproducts", "Ethanol concentration", "mmol/L",
         rel(None), None),
        ("byproducts.pyruvate", "byproducts", "Pyruvate concentration", "mmol/L",
         rel(None), None),
        ("byproducts.succinate", "byproducts", "Succinate concentration", "mmol/L",
         rel(None), None),
        ("gas_transfer.dissolved_o2", "gas_transfer", "Dissolved O2", "mg/L",
         rel(None), None),
        ("ph_control.base_addition", "ph_control", "Base addition (cumulative)", "mmol",
         rel(None), None),
    ]

    axes: dict[str, dict] = {}
    card: dict = {}
    for path, group, label, units, criterion, value in rows:
        axes[path] = {"group": group, "label": label, "units": units,
                      "criterion": criterion}
        if value is not None:
            node = card
            parts = path.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value if isinstance(value, bool) else {"mean": value}
    reference = {"title": "Beulig 2025 WT batch prefix", "axes": axes}
    return reference, card


# --- chart rendering --------------------------------------------------------

def render_overlay_charts(arm: str, sim: dict, out_dir: Path) -> list[str]:
    """One overlay chart per gradeable comparison (sim scalar vs Beulig batch
    trajectory). Best-effort; returns the written filenames."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written: list[str] = []
    panels = [
        ("growth", "Biomass / OD x 0.34 (gDW/L)", "reactor_OD_data.csv",
         OD_TO_GDW, sim["sim_biomass_gL"]),
        ("OTR", "OTR (mol/h reactor-scale)", "reactor_OTR_data.csv",
         1.0, sim["sim_otr"]),
        ("CTR", "CTR (mol/h reactor-scale)", "reactor_CTR_data.csv",
         1.0, sim["sim_ctr"]),
    ]
    for name, ylab, csv_name, scale, sim_val in panels:
        path = BEULIG_DIR / csv_name
        if not path.is_file():
            continue
        times, means = load_wide_csv_batch_phase(path)
        if not times:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.plot(times, [m * scale for m in means], "o-", ms=3, lw=1,
                color="#2563eb", label="Beulig WT batch (mean of reactors)")
        ax.axhline(sim_val, ls="--", color="#dc2626",
                   label=f"{arm} sim peak = {sim_val:.3g}")
        ax.set_xlabel("fed-batch time [h]  (negative = batch phase)")
        ax.set_ylabel(ylab)
        ax.set_title(f"Beulig batch overlay — {name} ({arm} arm, SMOKE)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out = out_dir / f"overlay_{name}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(out.name)
    return written


# --- iML1515 genome-scale FBA arm -------------------------------------------

def run_iml1515_arm(max_samples: int = DEFAULT_IML1515_SAMPLES) -> list[dict]:
    """Run the iML1515 genome-scale FBA arm over the Beulig batch-phase samples.

    REUSES ``scripts/runnable_sims/iml1515_vs_beulig.py`` for the sample loading
    + the FBA bound-setting / solve (no duplicated FBA setup). Returns the list
    of paired samples (each with ``mu_measured``/``mu_predicted``/
    ``acetate_flux_predicted``/``solver_status`` …).
    """
    # Ensure the repo root is importable when run as a script (python scripts/…),
    # so the sibling `scripts.runnable_sims.*` package resolves (pytest's conftest
    # already does this; direct CLI invocation does not).
    import sys
    if str(WS_ROOT) not in sys.path:
        sys.path.insert(0, str(WS_ROOT))
    from scripts.runnable_sims.iml1515_vs_beulig import (
        load_beulig_batch_phase_samples, run_iml1515)

    samples = load_beulig_batch_phase_samples(max_samples=max_samples)
    return run_iml1515(samples)


def build_iml1515_axes(pairs: list[dict]) -> tuple[dict, dict]:
    """Build the report_card ``reference`` axes + measured ``card`` for iML1515.

    The genuine PREDICTIONS are graded; the FBA INPUTS (uptake bounds set from
    Beulig) are emitted ungraded so the groups are present but nothing is graded
    as a prediction it is not. Returns ``(reference, card)``.
    """
    mu_meas = [_f(p.get("mu_measured")) for p in pairs]
    mu_pred = [_f(p.get("mu_predicted")) for p in pairs]
    mu_meas = [m for m in mu_meas if m is not None]
    mu_pred = [m for m in mu_pred if m is not None]
    n_opt = sum(1 for p in pairs if p.get("solver_status") == "optimal")

    # (path, group, label, units, criterion, measured_value)
    rows = [
        # GRADED — predicted vs measured μ across samples (r2 vs identity y=x).
        ("growth.mu_predicted_vs_measured", "growth",
         "Growth rate μ — predicted (iML1515 @ matched uptake) vs measured", "1/h",
         {"type": "r2", "ref_vector": mu_meas,
          "r2_min": IML1515_R2_MIN, "r2_drift": IML1515_R2_DRIFT},
         {"vector": mu_pred}),
        # GRADED — boolean process-level: iML1515 ran + predicted μ for N samples.
        ("summary.iml1515_ran", "summary",
         f"iML1515 ran + predicted μ for {len(pairs)} Beulig samples "
         f"({n_opt} optimal solves)", "",
         {"type": "boolean"}, bool(pairs) and n_opt > 0),
        # UNGRADED — Beulig batch samples carry no measured acetate; predicted
        # acetate flux has no reference to grade against.
        ("byproducts.acetate", "byproducts",
         "Acetate flux — predicted (no measured acetate in Beulig batch samples)",
         "mmol/(gDW*h)", {"type": "r2", "ref_vector": None}, None),
        # UNGRADED — these are INPUTS: iML1515's glucose / O2 uptake LOWER BOUNDS
        # are SET from the measured Beulig rates, so grading them would grade an
        # input as if it were a prediction. Honestly ungraded.
        ("substrate.glucose_uptake_input", "substrate",
         "Glucose uptake (INPUT: FBA bound set from Beulig, not predicted)",
         "mmol/(gDW*h)", {"type": "rel_tol", "reference": None}, None),
        ("gas_transfer.o2_uptake_input", "gas_transfer",
         "O2 uptake (INPUT: FBA bound set from Beulig, not predicted)",
         "mmol/(gDW*h)", {"type": "rel_tol", "reference": None}, None),
    ]

    axes: dict[str, dict] = {}
    card: dict = {}
    for path, group, label, units, criterion, value in rows:
        axes[path] = {"group": group, "label": label, "units": units,
                      "criterion": criterion}
        if value is not None:
            node = card
            parts = path.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value
    reference = {"title": "Beulig 2025 WT batch prefix (iML1515 genome-scale)",
                 "axes": axes}
    return reference, card


def render_iml1515_scatter(pairs: list[dict], out_dir: Path) -> list[str]:
    """Predicted-vs-measured μ parity scatter -> overlay_growth.png. Best-effort."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    mu_m = [_f(p.get("mu_measured")) or 0.0 for p in pairs]
    mu_p = [_f(p.get("mu_predicted")) or 0.0 for p in pairs]
    if not mu_m:
        return []
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(mu_m, mu_p, s=26, alpha=0.78, c="#6366f1", edgecolor="#3730a3",
               label=f"{len(pairs)} Beulig samples")
    lim = max(max(mu_m, default=0.1), max(mu_p, default=0.1), 0.1) * 1.15
    ax.plot([0, lim], [0, lim], "--", color="#9ca3af", lw=1, label="y = x (perfect)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("measured μ (Beulig 2025) [1/h]")
    ax.set_ylabel("predicted μ (iML1515 @ matched uptake) [1/h]")
    ax.set_title("iML1515 predicted vs measured μ\n"
                 "(uptake bounds SET from Beulig — NOT apples-to-apples vs wcm/millard)",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / "overlay_growth.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return [out.name]


def run_iml1515_harness(max_samples: int = DEFAULT_IML1515_SAMPLES,
                        render: bool = True) -> dict:
    """Run the iML1515 arm, grade vs Beulig, write report_card_verdict.json."""
    t0 = time.perf_counter()
    pairs = run_iml1515_arm(max_samples=max_samples)
    reference, card = build_iml1515_axes(pairs)
    report = grade_card(card, reference)
    verdict = verdict_json(
        report,
        model_ref="iML1515 (Palsson genome-scale FBA, cobra)",
        reference_model="Beulig 2025 WT batch prefix",
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ"),
    )
    n_opt = sum(1 for p in pairs if p.get("solver_status") == "optimal")
    verdict["scope"] = (
        f"iML1515 genome-scale FBA over {len(pairs)} Beulig batch-phase samples "
        f"({n_opt} optimal solves). NOT apples-to-apples with the wcm/millard arms: "
        "iML1515's glucose + O2 uptake LOWER BOUNDS ARE SET FROM the measured Beulig "
        "rates, so the only genuine PREDICTIONS graded are growth μ "
        "(predicted-vs-measured, r2 vs identity y=x). substrate/gas_transfer are "
        "INPUTS -> honestly ungraded; acetate is ungraded (no measured acetate in "
        "the Beulig batch samples). The wcm/millard arms predict the whole batch "
        "trajectory from first principles; iML1515 predicts μ from MEASURED uptake "
        "(a different, arguably easier test). Order-of-magnitude parity, not "
        "calibrated overlap."
    )
    verdict["n_samples"] = len(pairs)
    verdict["n_optimal"] = n_opt
    verdict["wall_s"] = round(time.perf_counter() - t0, 2)

    out_dir = CARD_ROOT / ARM_CARD_DIR["iml1515"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report_card_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8")
    if render:
        verdict["charts"] = render_iml1515_scatter(pairs, out_dir)
        (out_dir / "report_card_verdict.json").write_text(
            json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


# --- top-level harness ------------------------------------------------------

def run_harness(arm: str, steps: int = DEFAULT_SMOKE_STEPS,
                tol_rel: float = DEFAULT_TOL_REL,
                cells_per_agent: float = DEFAULT_CPA,
                render: bool = True,
                max_samples: int = DEFAULT_IML1515_SAMPLES) -> dict:
    """Run one arm, grade vs Beulig, write report_card_verdict.json (+ charts).

    Returns the verdict dict (also written to disk)."""
    if arm == "iml1515":
        # Genome-scale FBA arm — not a PB composite; separate path (reuses the
        # iml1515_vs_beulig FBA harness), graded by the SAME grader.
        return run_iml1515_harness(max_samples=max_samples, render=render)

    if arm not in ARM_COMPOSITE:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ALL_ARMS}")

    t0 = time.perf_counter()
    sim = run_arm(arm, steps, cells_per_agent=cells_per_agent)
    reference, card = build_axes(sim, tol_rel)
    report = grade_card(card, reference)
    verdict = verdict_json(
        report,
        model_ref=sim["composite"],
        reference_model="Beulig 2025 WT batch prefix",
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ"),
    )
    verdict["scope"] = (
        f"SHORT-RUN SMOKE ({steps} steps, cells_per_agent={cells_per_agent:g}); "
        "NOT the full ~33h Beulig batch production run (follow-up execution). "
        "Biological magnitudes are far below Beulig (documented architectural gap) "
        "so graded axes verdict 'mismatch' by design (execute-and-report)."
    )
    verdict["wall_s"] = round(time.perf_counter() - t0, 2)

    out_dir = CARD_ROOT / ARM_CARD_DIR[arm]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report_card_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8")
    if render:
        verdict["charts"] = render_overlay_charts(arm, sim, out_dir)
        (out_dir / "report_card_verdict.json").write_text(
            json.dumps(verdict, indent=2), encoding="utf-8")
    return verdict


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ALL_ARMS + ["both", "all"], default="both",
                    help="'both' = the two PB composite arms (wcm+millard); "
                         "'all' = those plus the iml1515 genome-scale arm")
    ap.add_argument("--steps", type=int, default=DEFAULT_SMOKE_STEPS,
                    help="WCM steps (small for the smoke build)")
    ap.add_argument("--tol-rel", type=float, default=DEFAULT_TOL_REL)
    ap.add_argument("--cells-per-agent", type=float, default=DEFAULT_CPA)
    ap.add_argument("--iml-samples", type=int, default=DEFAULT_IML1515_SAMPLES,
                    help="Beulig batch-phase samples the iml1515 arm solves FBA over")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()

    if args.arm == "both":
        arms = list(ARM_COMPOSITE)
    elif args.arm == "all":
        arms = ALL_ARMS
    else:
        arms = [args.arm]
    for arm in arms:
        v = run_harness(arm, steps=args.steps, tol_rel=args.tol_rel,
                        cells_per_agent=args.cells_per_agent,
                        render=not args.no_render, max_samples=args.iml_samples)
        groups = {g: d["verdict"] for g, d in v["groups"].items()}
        print(f"[{arm}] -> {CARD_ROOT / ARM_CARD_DIR[arm]}/report_card_verdict.json")
        print(f"  overall={v['overall']}  groups={groups}  ({v['wall_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
