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
  GRADED   substrate/glucose-conc  sim reactor.glucose_medium_mM (medium recipe
                          seed drawn down by GLC uptake) vs reactor_glucose_data
                          endpoint (mmol/L drawdown)            [#225 req-3]
  GRADED   gas_transfer/OTR,CTR  sim reactor.{o2,co2}_transport_delta ->
                          mmol/(L.h) vs Beulig reactor_{OTR,CTR}_data.csv
  GRADED   byproducts/{lactate,formate,ethanol,pyruvate,succinate}  sim
                          reactor.<byp>_mM (coupler accumulates secreted
                          env.exchange counts -> mmol/L) vs the matching Beulig
                          reactor_<byp>_data.csv peak             [#225 req-3]
  GRADED   summary        boolean "comparison executed" (process-level)
  UNGRADED byproducts/acetate  sim DOES track reactor.acetate_mM, but there is
                          NO reactor_acetate_data.csv in palsson-2025-supp ->
                          no reference to grade against
  UNGRADED gas_transfer/DO  sim exposes reactor.dissolved_o2 (mg/L) but Beulig
                          has NO dissolved-O2 column anywhere -> no reference
  UNGRADED ph_control/base Beulig reactor_base_data.csv is base VOLUME (ml);
                          grading needs a cumulative-acid sim store + the titrant
                          molarity (ml<->mmol), absent from the supp data ->
                          would require a fabricated constant

The #225 req-3 closure (substrate/glucose-conc + byproducts) is mediated by the
ReactorCellCoupler medium-concentration accumulator (track_medium=True): each
tick it integrates the cell's per-step environment.exchange COUNTS into ADDITIVE
mmol/L deltas on the reactor store (glucose draws down a seeded medium pool;
byproducts accumulate secretion from 0). The harness reads those reactor leaves
and grades them with the EXISTING grader. Magnitudes stay far below Beulig (the
documented architectural gap) so these axes verdict 'mismatch' by design — the
point is they GRADE rather than stay ungraded.

Adding a resolvable overlay = adding one ``MEDIUM_AXES`` entry + the coupler leaf
(the spec stays the source of the overlay list).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path


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
# --- multi-generation PRODUCTION run defaults -------------------------------
# baseline divides ~every 2700 ticks; 3 generations ≈ 8100 ticks. max_steps is a
# hard cap (the runner stops at whichever comes first: max_steps or N divisions).
DEFAULT_MULTIGEN_GENERATIONS = 3
DEFAULT_MULTIGEN_MAX_STEPS = 9000    # ample headroom for 3 divisions
DEFAULT_MULTIGEN_CHUNK = 60          # ticks between emitter updates / division checks
# Followed-agent (under agents/<id>/) emit paths the harness derives sim scalars
# from: per-cell glucose-exchange flux (counts) + cell mass (fg).
MULTIGEN_AGENT_PATHS = [
    "listeners/mass/cell_mass",
    "environment/exchange/GLC",
]
# Composite-root stores the harness reads (population aggregator + reactor):
# biomass concentration, transport deltas, reactor working volume.
MULTIGEN_ROOT_PATHS = [
    "population/biomass_concentration_gL",
    "reactor/o2_transport_delta",
    "reactor/co2_transport_delta",
    "reactor/volume_L",
    # Medium-concentration accumulators (#225 req-3) — root reactor leaves the
    # coupler integrates exchange counts into (glucose drawdown + byproducts).
    "reactor/glucose_medium_mM",
    "reactor/acetate_mM",
    "reactor/lactate_mM",
    "reactor/formate_mM",
    "reactor/ethanol_mM",
    "reactor/pyruvate_mM",
    "reactor/succinate_mM",
]
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


def _pub_concentration(csv_name: str, mode: str = "peak") -> tuple[float | None, dict]:
    """Beulig batch-phase concentration scalar (native CSV units, e.g. mmol/L).

    The byproduct + glucose reactor CSVs are WIDE per-reactor mmol/L trajectories
    (header carries the unit). ``peak`` (byproducts — climb) or ``endpoint``
    (glucose — drawdown). Returns ``(scalar, meta)``; ``(None, ...)`` if the file
    is absent (e.g. there is NO reactor_acetate_data.csv).
    """
    if csv_name is None:
        return None, {}
    path = BEULIG_DIR / csv_name
    if not path.is_file():
        return None, {}
    times, means = load_wide_csv_batch_phase(path)
    ref = batch_reference_scalar(means, mode)
    return ref, {"times": times, "means": means}


# Sim medium-concentration reactor leaves (mmol/L) the coupler accumulates
# (#225 req-3). leaf -> Beulig CSV + reduction mode. Glucose is a remaining-pool
# DRAWDOWN (endpoint); byproducts are accumulated secretion (peak). acetate has
# NO Beulig CSV -> stays ungraded (handled in build_axes).
MEDIUM_AXES = {
    "glucose_medium_mM": ("reactor_glucose_data.csv", "endpoint"),
    "lactate_mM":        ("reactor_lactate_data.csv", "peak"),
    "formate_mM":        ("reactor_formate_data.csv", "peak"),
    "ethanol_mM":        ("reactor_ethanol_data.csv", "peak"),
    "pyruvate_mM":       ("reactor_pyruvate_data.csv", "peak"),
    "succinate_mM":      ("reactor_succinate_data.csv", "peak"),
    # acetate_mM: no reactor_acetate_data.csv in palsson-2025-supp -> ungraded.
    "acetate_mM":        (None, "peak"),
}


def _read_medium_leaves(reactor: dict | None) -> dict[str, float]:
    """Read the coupler's medium-concentration accumulator leaves (mmol/L) from a
    reactor store dict. Missing leaf -> 0.0 (track_medium off / pre-#225 run)."""
    reactor = reactor or {}
    return {leaf: (_f(reactor.get(leaf)) or 0.0) for leaf in MEDIUM_AXES}


# --- arm run + sim observable extraction ------------------------------------

def _agent0(state) -> dict:
    agents = state.get("agents") or {}
    if "0" in agents:
        return agents["0"]
    return next(iter(agents.values()), {}) if agents else {}


def _cell_mass_value_fg(cm) -> float | None:
    """Coerce a raw ``listeners.mass.cell_mass`` value to femtograms.

    Handles a live pint Quantity (``.to('femtogram')``), the JSON round-trip
    forms a SQLite emit can leave (a bare float — the magnitude of the
    ``quantity[float,fg]`` schema is already in fg — or a ``{magnitude,..}``
    dict), and plain numbers. Returns None on anything unrecognized.
    """
    if cm is None:
        return None
    if hasattr(cm, "to") and hasattr(cm, "magnitude"):
        try:
            return float(cm.to("femtogram").magnitude)
        except (AttributeError, ValueError, TypeError):
            return _f(getattr(cm, "magnitude", None))
    if isinstance(cm, dict):
        # serialized quantity forms: {'magnitude': x, 'units': 'fg'} / {'value': x}
        return _f(cm.get("magnitude", cm.get("value")))
    return _f(cm)


def _cell_mass_fg(agent_state) -> float | None:
    try:
        mass = (agent_state.get("listeners") or {}).get("mass") or {}
        return _cell_mass_value_fg(mass.get("cell_mass"))
    except (AttributeError, KeyError, TypeError):
        return None


def _derive_sim_scalars(
    composite: str, steps: int, volume_L: float,
    biomass_gL: list[float], glc_counts: list[float], cell_mass: list[float],
    o2_delta: list[float], co2_delta: list[float], gtime: list[float],
    extra: dict | None = None, dt_override: float | None = None,
) -> dict:
    """Reduce the per-step (or per-chunk) sim series to the graded axis scalars.

    Shared by both the inline single-agent path (:func:`run_arm`) and the
    multi-generation production path (:func:`run_arm_multigen`) so the two routes
    grade identically — only the data SOURCE differs (live state vs emitted db).

    ``dt_override`` (s): the per-TICK duration to use for the per-hour GUR
    conversion. The inline path samples every tick so it infers dt_s≈1 from the
    ``gtime`` deltas; the multigen path samples every ``chunk`` ticks (gtime
    deltas == chunk-span), but the exchange counts it captures are still per-tick
    instantaneous values — so it must pass the per-tick duration here to keep the
    GUR on the same basis as the inline path (else GUR comes out chunk× too low).
    """
    # Mean per-step duration (s) from global_time progression (fallback 1 s).
    if dt_override is not None:
        dt_s = dt_override
    else:
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

    out = {
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
    if extra:
        out.update(extra)
    return out


def run_arm(arm: str, steps: int, cells_per_agent: float = DEFAULT_CPA) -> dict:
    """Build + step a coupled arm, recording the per-step series needed to derive
    each sim axis scalar. Returns a dict of derived sim observables (peaks).

    SMOKE/inline path: steps a freshly-built composite for ``steps`` ticks and
    reads the LIVE state each tick. Agent ``'0'`` is replaced by daughters at the
    first division (~2700 ticks), so this path cannot follow a real lineage past
    one generation — use :func:`run_arm_multigen` for production (multi-gen) runs.
    """
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
    reactor_final = c.state.get("reactor") or {}
    volume_L = _f(reactor_final.get("volume_L")) or 1.0

    sim = _derive_sim_scalars(
        composite, steps, volume_L,
        biomass_gL, glc_counts, cell_mass, o2_delta, co2_delta, gtime,
    )
    # Medium-concentration accumulators (mmol/L) — cumulative, so the FINAL
    # reactor state carries the graded scalar (#225 req-3).
    sim["medium"] = _read_medium_leaves(reactor_final)
    return sim


def run_arm_multigen(
    arm: str,
    max_generations: int = DEFAULT_MULTIGEN_GENERATIONS,
    max_steps: int = DEFAULT_MULTIGEN_MAX_STEPS,
    chunk: int = DEFAULT_MULTIGEN_CHUNK,
    cells_per_agent: float = DEFAULT_CPA,
    single_daughters: bool = True,
    seed: int = 0,
    population_growth_mode: str = "fixed",
) -> dict:
    """PRODUCTION path: run a coupled arm across ``max_generations`` divisions,
    following the lineage with a RAM-safe externally-driven SQLite emitter, then
    read the emitted trajectory back and derive the SAME graded sim scalars as
    the inline smoke path.

    Why SQLite (not run_multigen_xarray): the xarray runner's view only captures
    ``listeners.*`` agent leaves and DROPS composite-root stores; this harness
    grades the ``population`` aggregator + ``reactor`` transport stores (root) and
    the agent ``environment.exchange`` store — none of which the xarray view can
    reach. :func:`run_multigen_sqlite` captures both agent ``emit_paths`` and
    ``extra_root_paths`` and bounds peak memory to one cell via
    ``single_daughters=True`` (the coupled composite otherwise grows 2^N cells —
    62 GB RSS observed in the 2026-05-24 shakedown). Mirrors
    ``scripts/run_mbp_tracked.py``'s mbp-study runner.

    Returns the same dict shape as :func:`run_arm` (so ``build_axes`` is shared),
    plus ``generations`` (list seen), ``n_rows`` (emitted history rows), and
    ``max_steps_reached``.
    """
    from v2ecoli import build_composite
    from v2ecoli.library.sqlite_run import run_multigen_sqlite
    from vivarium_workbench.lib import composite_runs

    from v2ecoli.composites.ecoli_baseline import set_null_emitter_override

    composite = ARM_COMPOSITE[arm]
    # Per-seed run dir so a multi-seed sweep doesn't clobber prior seeds.
    arm_tag = arm if seed == 0 else f"{arm}-seed{seed}"
    out_dir = WS_ROOT / "out" / "mbp_production" / arm_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    db_file = out_dir / "run.db"
    if db_file.exists():
        db_file.unlink()
    run_id = f"mbp-production-{arm_tag}"

    # The external SQLite emitter (run_multigen_sqlite) handles emission, so NULL
    # the composite's internal ParquetEmitter. Otherwise it fires every chunk,
    # writes to .pbg/parquet-runs/<run>/history/experiment_id=…/ (a nested hive
    # dir it doesn't create), and aborts the run at ~gen 2 with FileNotFoundError
    # — and it's the documented RAM trap. Must be set BEFORE build (the emitter
    # step materializes at build time).
    set_null_emitter_override(True)
    try:
        c = build_composite(composite, seed=seed, cache_dir="out/cache",
                            cells_per_agent=cells_per_agent,
                            population_growth_mode=population_growth_mode)
    finally:
        set_null_emitter_override(False)
    result = run_multigen_sqlite(
        c,
        run_id=run_id,
        db_file=str(db_file),
        emit_paths=MULTIGEN_AGENT_PATHS,
        extra_root_paths=MULTIGEN_ROOT_PATHS,
        max_steps=max_steps,
        max_generations=max_generations,
        chunk=chunk,
        initial_agent_id="0",
        single_daughters=single_daughters,
        core=c.core,
    )

    # --- read the emitted multi-gen run back ---------------------------------
    conn = composite_runs.connect(db_file)
    try:
        rows = composite_runs.query_run(conn, run_id=run_id)
    finally:
        conn.close()

    biomass_gL: list[float] = []
    glc_counts: list[float] = []
    cell_mass: list[float] = []
    o2_delta: list[float] = []
    co2_delta: list[float] = []
    gtime: list[float] = []
    volume_L = 1.0
    medium_final: dict[str, float] = {leaf: 0.0 for leaf in MEDIUM_AXES}
    for row in rows:
        st = row.get("state") or {}
        pop = st.get("population") or {}
        biomass_gL.append(_f(pop.get("biomass_concentration_gL")) or 0.0)
        r = st.get("reactor") or {}
        o2_delta.append(_f(r.get("o2_transport_delta")) or 0.0)
        co2_delta.append(_f(r.get("co2_transport_delta")) or 0.0)
        v = _f(r.get("volume_L"))
        if v:
            volume_L = v
        # Cumulative medium accumulators -> keep the latest non-empty reactor row.
        if any(leaf in r for leaf in MEDIUM_AXES):
            medium_final = _read_medium_leaves(r)
        # The followed agent id changes across generations (0 -> 00 -> 000),
        # so take whichever single agent the emitter wrote this row.
        agents = st.get("agents") or {}
        a0 = next(iter(agents.values()), {}) if agents else {}
        exch = ((a0.get("environment") or {}).get("exchange") or {}) if a0 else {}
        glc_counts.append(_f(exch.get("GLC")) or 0.0)
        cm = ((a0.get("listeners") or {}).get("mass") or {}).get("cell_mass")
        cell_mass.append(_cell_mass_value_fg(cm) or 0.0)
        gtime.append(_f(row.get("time")) or 0.0)

    # The emitted rows are `chunk` ticks apart, but the per-cell exchange counts
    # are per-tick instantaneous; recover the per-tick duration so the GUR
    # per-hour conversion matches the inline path's basis (= 1 tick).
    row_deltas = [b - a for a, b in zip(gtime, gtime[1:]) if (b - a) > 0]
    tick_dt = (sum(row_deltas) / len(row_deltas) / chunk) if row_deltas else 1.0

    sim = _derive_sim_scalars(
        composite, result.get("steps", max_steps), volume_L,
        biomass_gL, glc_counts, cell_mass, o2_delta, co2_delta, gtime,
        dt_override=tick_dt,
        extra={
            "generations": result.get("generations"),
            "n_rows": len(rows),
            "max_steps_reached": result.get("steps"),
            "db_file": str(db_file.relative_to(WS_ROOT)),
            "population_growth_mode": population_growth_mode,
        },
    )
    sim["medium"] = medium_final
    return sim


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
    # Medium-concentration refs (mmol/L) — glucose drawdown (endpoint) +
    # byproduct peaks (#225 req-3). acetate -> (None, ...) (no Beulig CSV).
    med = sim.get("medium") or {}
    med_ref = {leaf: _pub_concentration(csv, mode)[0]
               for leaf, (csv, mode) in MEDIUM_AXES.items()}

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
        # GRADED (#225 req-3) — medium-concentration accumulators the coupler
        # integrates from the cell's environment.exchange counts (mmol/L), graded
        # vs the Beulig batch reactor CSVs. Magnitudes are far below Beulig (the
        # documented architectural gap) so these verdict 'mismatch' by design —
        # the point is they GRADE rather than stay ungraded.
        ("substrate.glucose_conc", "substrate", "Glucose concentration (remaining medium)",
         "mmol/L", rel(med_ref["glucose_medium_mM"]), med.get("glucose_medium_mM")),
        ("byproducts.lactate", "byproducts", "Lactate concentration", "mmol/L",
         rel(med_ref["lactate_mM"]), med.get("lactate_mM")),
        ("byproducts.formate", "byproducts", "Formate concentration", "mmol/L",
         rel(med_ref["formate_mM"]), med.get("formate_mM")),
        ("byproducts.ethanol", "byproducts", "Ethanol concentration", "mmol/L",
         rel(med_ref["ethanol_mM"]), med.get("ethanol_mM")),
        ("byproducts.pyruvate", "byproducts", "Pyruvate concentration", "mmol/L",
         rel(med_ref["pyruvate_mM"]), med.get("pyruvate_mM")),
        ("byproducts.succinate", "byproducts", "Succinate concentration", "mmol/L",
         rel(med_ref["succinate_mM"]), med.get("succinate_mM")),
        # UNGRADED — no Beulig reference and/or no honest sim observable:
        #  * acetate: there is NO reactor_acetate_data.csv in palsson-2025-supp
        #    (the sim DOES track reactor.acetate_mM, but there's nothing to grade
        #    against) -> ungraded.
        #  * dissolved_o2: the sim exposes reactor.dissolved_o2 (mg/L) but Beulig's
        #    process_summary_interpol.csv has NO dissolved-O2 column (no DO ref
        #    anywhere in palsson-2025-supp) -> ungraded.
        #  * base_addition: Beulig HAS reactor_base_data.csv but it is base VOLUME
        #    (ml); grading needs a cumulative-acid sim store AND the titrant
        #    molarity to convert ml<->mmol, which the supp data doesn't provide —
        #    a fabricated constant would be required, so honestly ungraded.
        ("byproducts.acetate", "byproducts", "Acetate concentration "
         "(sim tracks reactor.acetate_mM; NO Beulig acetate CSV)", "mmol/L",
         rel(None), None),
        ("gas_transfer.dissolved_o2", "gas_transfer",
         "Dissolved O2 (sim exposes reactor.dissolved_o2; NO Beulig DO column)",
         "mg/L", rel(None), None),
        ("ph_control.base_addition", "ph_control",
         "Base addition (Beulig ref is base VOLUME ml; needs titrant molarity to "
         "grade vs an acid-production store — unmodeled)", "mmol", rel(None), None),
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
                max_samples: int = DEFAULT_IML1515_SAMPLES,
                production: bool = False,
                max_generations: int = DEFAULT_MULTIGEN_GENERATIONS,
                max_steps: int = DEFAULT_MULTIGEN_MAX_STEPS,
                chunk: int = DEFAULT_MULTIGEN_CHUNK,
                single_daughters: bool = True,
                seed: int = 0,
                population_growth_mode: str = "fixed") -> dict:
    """Run one arm, grade vs Beulig, write report_card_verdict.json (+ charts).

    ``production=True`` uses the multi-generation lineage-following runner
    (:func:`run_arm_multigen`) instead of the inline smoke (:func:`run_arm`);
    the card lands at the SAME path so the studies' report_card_axis tests still
    resolve, with the verdict ``scope`` recording the N-generation run.

    Returns the verdict dict (also written to disk)."""
    if arm == "iml1515":
        # Genome-scale FBA arm — not a PB composite; separate path (reuses the
        # iml1515_vs_beulig FBA harness), graded by the SAME grader. Has no
        # multi-generation analogue (FBA over Beulig samples, not a sim).
        return run_iml1515_harness(max_samples=max_samples, render=render)

    if arm not in ARM_COMPOSITE:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ALL_ARMS}")

    t0 = time.perf_counter()
    if production:
        sim = run_arm_multigen(
            arm, max_generations=max_generations, max_steps=max_steps,
            chunk=chunk, cells_per_agent=cells_per_agent,
            single_daughters=single_daughters, seed=seed,
            population_growth_mode=population_growth_mode)
    else:
        sim = run_arm(arm, steps, cells_per_agent=cells_per_agent)
    reference, card = build_axes(sim, tol_rel)
    report = grade_card(card, reference)
    verdict = verdict_json(
        report,
        model_ref=sim["composite"],
        reference_model="Beulig 2025 WT batch prefix",
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ"),
    )
    if production:
        _growing = population_growth_mode == "representative_doubling"
        _growth_note = (
            " Represented population GROWS 2x/generation (representative_doubling): "
            f"biomass accumulates ~2^(g-1) across the {max_generations} generations "
            "instead of plateauing — verdict reflects how far the accumulation "
            "actually reaches (no faked Beulig closure)."
            if _growing else
            " Single-lineage FIXED mode: biomass plateaus (documented architectural "
            "gap) so graded axes verdict 'mismatch' by design (execute-and-report)."
        )
        verdict["scope"] = (
            f"{max_generations}-generation multigen PRODUCTION run "
            f"(run_multigen_sqlite, single_daughters={single_daughters}, "
            f"steps_reached={sim.get('max_steps_reached')}, "
            f"generations={sim.get('generations')}, n_rows={sim.get('n_rows')}, "
            f"cells_per_agent={cells_per_agent:g}, "
            f"population_growth_mode={population_growth_mode}). Real multi-generation "
            "lineage trajectory fed into the grader (vs the short-run smoke)." + _growth_note
        )
        verdict["multigen"] = {
            "max_generations": max_generations,
            "generations_seen": sim.get("generations"),
            "steps_reached": sim.get("max_steps_reached"),
            "n_history_rows": sim.get("n_rows"),
            "single_daughters": single_daughters,
            "population_growth_mode": population_growth_mode,
            "db_file": sim.get("db_file"),
        }
    else:
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
    # --- multi-generation PRODUCTION run -----------------------------------
    ap.add_argument("--production", action="store_true",
                    help="Use the multi-generation lineage-following runner "
                         "(run_multigen_sqlite) instead of the inline smoke. "
                         "Follows daughters past division; RAM-safe.")
    ap.add_argument("--generations", type=int, default=DEFAULT_MULTIGEN_GENERATIONS,
                    help="(production) generations to follow (default 3)")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MULTIGEN_MAX_STEPS,
                    help="(production) hard cap on composite ticks (stops at "
                         "whichever comes first: this or N divisions)")
    ap.add_argument("--chunk", type=int, default=DEFAULT_MULTIGEN_CHUNK,
                    help="(production) ticks between emitter updates / division checks")
    ap.add_argument("--no-single-daughters", action="store_false",
                    dest="single_daughters",
                    help="(production) keep BOTH daughters at each division "
                         "(2^N cells — memory grows; cap --generations)")
    ap.set_defaults(single_daughters=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="(production) RNG seed; per-seed run dir for multi-seed sweeps")
    ap.add_argument("--population-growth-mode",
                    choices=["fixed", "representative_doubling"], default="fixed",
                    help="(production) 'representative_doubling' grows the represented "
                         "population 2x/generation so biomass ACCUMULATES toward the "
                         "Beulig batch scale (breaks the single-lineage plateau); "
                         "'fixed' (default) is the legacy single-lineage behavior")
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
                        render=not args.no_render, max_samples=args.iml_samples,
                        production=args.production,
                        max_generations=args.generations,
                        max_steps=args.max_steps, chunk=args.chunk,
                        single_daughters=args.single_daughters, seed=args.seed,
                        population_growth_mode=args.population_growth_mode)
        groups = {g: d["verdict"] for g, d in v["groups"].items()}
        print(f"[{arm}] -> {CARD_ROOT / ARM_CARD_DIR[arm]}/report_card_verdict.json")
        print(f"  overall={v['overall']}  groups={groups}  ({v['wall_s']}s)")
        if args.production and "multigen" in v:
            print(f"  multigen={v['multigen']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
