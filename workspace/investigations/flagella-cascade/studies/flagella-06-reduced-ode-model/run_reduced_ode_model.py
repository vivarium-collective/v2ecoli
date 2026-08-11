"""Reduced ODE model of the flagella regulatory + assembly cascade —
decoupled from the whole-cell model entirely.

Added 2026-08-10 (v1), revised same day (v2, real-data-validated), revised
again 2026-08-10 (v3, this version) to match the WCM's removal of the
FliT:FlhD4C2 checkpoint. v1 and v2's full history is preserved at
archive/flit-flhdc-regulation-2026-08/code/ (v2 specifically at
run_reduced_ode_model_v2_with_flit.py) for before/after comparison.

WHY v3: on 2026-08-10 the WCM's FliT:FlhD4C2 checkpoint (and its basal-
degradation companion) were removed from the live codebase -- Albanna et
al. (2018, Sci Rep 8:16705) found no significant Delta-fliT phenotype in
E. coli MG1655 (this WCM's exact K-12 strain), unlike in Salmonella where
the mechanism was characterized (Yamamoto & Kutsukake 2006). Full removal
story: archive/flit-flhdc-regulation-2026-08/README.md and flagella-02's
finding flagella-02-flit-checkpoint-removed. This model is kept in sync
with the live WCM (unlike leaving it frozen, which was last session's
interim choice) so it continues to test the SAME mechanism the WCM
actually runs, not a retired one.

WHAT CHANGED FROM v2:
- Removed the FliT:FlhDC checkpoint entirely (no more v/c2 titration term,
  no more DELTA2/K_HALF parameters) -- FlhD4C2 (X) no longer has ANY
  degradation term. This exactly mirrors the real WCM's current state:
  the standard protein_degradation process only handles monomers, never
  assembled complexes, so with both flagella_flhdc_degradation.py and
  flagella_flit_flhdc_checkpoint.py gone, FlhD4C2 has no decay pathway at
  all -- confirmed directly via a WCM smoke test after the removal
  (FlhD4C2 stayed flat over 60s with nothing consuming it).
- Removed the FliT:FliD equilibrium (K_D_FLIT_FLID, the free/bound split)
  -- FLIT-FLID-CPLX_RXN was removed from the WCM's reaction network
  alongside the checkpoint (it existed only to feed the checkpoint's free-
  FliT signal). FliT and FliD are now tracked as independent totals with
  no interaction between them, exactly matching the live WCM.
- FliT is now a genuine dead-end variable: still synthesized under Class
  III control and still degrades at its own real rate (both unrelated to
  the removed checkpoint), but nothing reads it anymore -- kept for
  parity with the real WCM, which still transcribes/translates/dimerizes
  FliT (FLIT-DIMER_RXN was NOT removed), just no longer wires it to
  anything. This matches the "FliT only being there" framing used when
  deciding what to remove.
- K_SYNTH_X's calibration still reuses the same 0.00289/s rate constant
  (Claret & Hughes 2000 estimate) as its basis -- not because that Step
  still exists (it doesn't; the import would now fail since the module
  was deleted), but because the rate estimate itself was always about
  FlhD4C2's synthesis/turnover biology, independent of whether the WCM
  currently chooses to model the turnover half of it. Hardcoded with this
  note rather than imported from a now-nonexistent module.

WHY THIS EXISTS (unchanged from v1/v2): three separate WCM-embedded multi-
generation diagnostics (flagella-02) each got blocked by a different
infrastructure limitation (fliC supply, motor-complex supply, the general
division/mass-homeostasis bug) before the actual regulatory question --
does something bound flagella count? -- could be answered. This sidesteps
all three: a plain system of ODEs, integrated with scipy, needing no
division, metabolism, or FBA.

WHAT IS DIRECTLY IMPORTED (not retyped) FROM THE REAL WCM SOURCE:
- beta, beta_prime, K_flhDC, K_fliA: Kalir & Alon SUM-gate coefficients
  (FlagellaTranscriptionRegulation.config_schema).
- FlgM secretion rate 0.1/s per complete flagellum
  (FlagellaFlgMSecretion.config_schema).
- Filament elongation rate law (a, b) and nucleation_rate
  (FlagellaFilamentElongation / FlagellaFilamentNucleation config_schema).

WHAT IS MEASURED DIRECTLY FROM THE REAL WCM (2026-08-10):
- FliA, FlgM, FliT, FliD monomer degradation rates: sim_data.process.
  translation.monomer_data['deg_rate'] (real ParCa fit).
- Natural (un-overridden) t=0 bulk counts for FlhD4C2, free FliA, FlgM,
  free FliT-dimer, free FliD, all read directly from build_composite()
  with NO diagnostic override applied -- confirmed these are ParCa's own
  basal-condition initialization, not an arbitrary snapshot. All
  complexes were exactly 0 at this natural t=0, confirming these values
  are TOTAL pool sizes.
- K_D for FLGM-FLIA-CPLX_RXN (still active in the WCM, unaffected by the
  FliT removal): measured directly from the running WCM at ~144.5
  count-units (t=2-20s, 4 samples, range 143-146) -- see v2's docstring
  for the full measurement story (the raw tabulated forward_rate=1 in
  equilibrium_reaction_rates.tsv turned out to be a normalization
  convention; only the K_D ratio matters).

WHAT IS STILL A GENUINE SIMPLIFICATION (flagged, not hidden):
1. Filament construction (nucleation -> elongation -> completion) is
   modeled as a mean-field exponential-relaxation process, NOT the WCM's
   genuine per-instance dL/dt=a/(b+L) growth law. See study.yaml's
   open_decisions.
2. Motor-complex supply is NOT modeled (nucleation is not gated by any
   motor-complex availability term) -- this session's own diagnostic
   found motor-complex exhaustion is a real, separate bottleneck.
3. Synthesis rate constants (K_SYNTH_*) are calibrated assuming the real,
   natural ParCa-initialized totals represent a genuine synthesis=
   degradation steady state for each TOTAL pool -- one step removed from
   directly importing ParCa's own basal_prob/translation-efficiency
   machinery. For X specifically, this is now purely a "what does
   synthesis need to be to explain the real reference total, using the
   historical turnover-rate estimate as an anchor" calibration, since
   there is no live degradation term to balance against anymore -- flag
   this explicitly since it's a step further removed than the others.

Usage:
    PYTHONPATH=$PWD .venv/bin/python \
        workspace/investigations/flagella-cascade/studies/flagella-06-reduced-ode-model/run_reduced_ode_model.py \
        --hours 8 --target-length 20000
"""
import argparse
import os

import numpy as np
from scipy.integrate import solve_ivp

STUDY_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# REAL parameters, imported directly from the actual WCM source files.
# ---------------------------------------------------------------------------
from v2ecoli.processes.flagella_transcription_regulation import FlagellaTranscriptionRegulation
from v2ecoli.processes.flagella_flgm_secretion import FlagellaFlgMSecretion
from v2ecoli.processes.flagella_filament_elongation import FlagellaFilamentElongation
from v2ecoli.processes.flagella_filament_nucleation import FlagellaFilamentNucleation

_sumgate = FlagellaTranscriptionRegulation.config_schema
BETA = np.asarray(_sumgate["beta"]["_default"])
BETA_PRIME = np.asarray(_sumgate["beta_prime"]["_default"])
K_FLHDC = _sumgate["K_flhDC"]["_default"]
K_FLIA = _sumgate["K_fliA"]["_default"]
FLIA_IDX = 6  # last entry in beta/beta_prime -- fliA itself, per Kalir & Alon Fig 3

# NOT imported: flagella_flhdc_degradation.py's degradation_rate -- that
# Step was removed 2026-08-10 (archive/flit-flhdc-regulation-2026-08/).
# Hardcoded here purely to anchor K_SYNTH_X's calibration (see docstring
# point 3) -- FlhD4C2 has NO active decay term in this model anymore,
# matching the live WCM exactly.
FLHDC_HISTORICAL_TURNOVER_RATE = 0.00289  # Claret & Hughes 2000 estimate, /s

FLGM_SECRETION_RATE = FlagellaFlgMSecretion.config_schema["secretion_rate"]["_default"]

_elong = FlagellaFilamentElongation.config_schema
RATE_A = _elong["rate_a"]["_default"]
RATE_B = _elong["rate_b"]["_default"]
REAL_TARGET_LENGTH = _elong["target_length"]["_default"]
NUCLEATION_RATE = FlagellaFilamentNucleation.config_schema["nucleation_rate"]["_default"]
FLID_PER_COMPLETION = 5  # real stoichiometry, complexation_reactions_modified.tsv

# ---------------------------------------------------------------------------
# REAL values measured directly from the WCM this session (see docstring).
# ---------------------------------------------------------------------------
DEG_RATE_FLIA = 1.925408834888737e-05   # sim_data monomer_data['deg_rate'], EG11355-MONOMER[c]
DEG_RATE_FLGM = 0.000270549250804038    # G369-MONOMER[c]
DEG_RATE_FLIT = 1.925408834888737e-05   # EG11389-MONOMER[c]
DEG_RATE_FLID = 1.925408834888737e-05   # EG10841-MONOMER[e]

# Natural (un-overridden) t=0 bulk counts -- ParCa's own basal reference,
# read directly with no diagnostic override applied. All complexes were
# exactly 0 at this point (confirmed), so these are TOTAL pool sizes.
X_REF = 335.0
Y_TOTAL_REF = 2487.0
FLGM_TOTAL_REF = 1496.0
FLIT_TOTAL_REF = 974.0
FLID_TOTAL_REF = 2115.0

# K_D for FLGM-FLIA-CPLX_RXN, measured directly (count-scale, 1:1 binding).
# FLIT-FLID-CPLX_RXN's K_D is no longer needed -- that reaction was removed
# from the WCM alongside the checkpoint it fed.
K_D_FLGM_FLIA = 144.5

# FliC's real, directly-measured synthesis rate post-10x-fix (flagella-02,
# 2026-08-06/07) -- used to calibrate the mean-field completion timescale.
FLIC_SYNTHESIS_RATE_MEASURED = 6.0  # molecules/s


def solve_equilibrium_complex(total_a, total_b, k_d):
    """Standard 1:1 mass-action equilibrium: A + B <-> C, K_D=[A][B]/[C].
    Returns C given real totals (conserved: A_free=total_a-C, etc.)."""
    total_a = max(total_a, 0.0)
    total_b = max(total_b, 0.0)
    s = total_a + total_b + k_d
    disc = max(s * s - 4.0 * total_a * total_b, 0.0)
    c = (s - np.sqrt(disc)) / 2.0
    return min(c, total_a, total_b)


# Reference free values, from the real natural totals -- the state the
# fast (sub-minute) FlgM:FliA equilibrium drives the natural ParCa
# initialization toward, before slower transcriptional dynamics act. Used
# only to calibrate synthesis rate constants below.
_c_flgm_flia_ref = solve_equilibrium_complex(Y_TOTAL_REF, FLGM_TOTAL_REF, K_D_FLGM_FLIA)
Y_FREE_REF = Y_TOTAL_REF - _c_flgm_flia_ref
FLGM_FREE_REF = FLGM_TOTAL_REF - _c_flgm_flia_ref

# Class II reference: matches the real Step's own convention exactly --
# p_i_ref uses X_ref with Y taken as 0 (see flagella_transcription_
# regulation.py:187-191, "At reference (X=X_ref, Y=0)...").
X_REF_GATE = X_REF / (K_FLHDC + X_REF)
P_FLIA_REF = (BETA[FLIA_IDX] * X_REF_GATE) / (BETA[FLIA_IDX] + BETA_PRIME[FLIA_IDX])

# Class III reference gate (uses the equilibrium-corrected Y_FREE_REF,
# since Class III's real formula, override=Y*basal_prob, has no
# normalization of its own -- see flagella_transcription_regulation.py:213).
_YG_REF = Y_FREE_REF / (K_FLIA + Y_FREE_REF)

# Synthesis rate constants. K_SYNTH_X now has no decay term to balance
# against (see docstring point 3) -- it's calibrated purely as "the flux
# that would explain X_REF given the historical turnover-rate estimate,"
# an anchor rather than a true steady-state balance, since the model (like
# the live WCM) has no active decay for X anymore.
K_SYNTH_X = FLHDC_HISTORICAL_TURNOVER_RATE * X_REF
K_SYNTH_Y = DEG_RATE_FLIA * Y_TOTAL_REF                       # scaled by p_i/P_FLIA_REF
K_SYNTH_FLGM_BASAL = DEG_RATE_FLGM * FLGM_TOTAL_REF / _YG_REF  # scaled by Y_free/(K_fliA+Y_free)
K_SYNTH_FLIT_BASAL = DEG_RATE_FLIT * FLIT_TOTAL_REF / _YG_REF
K_SYNTH_FLID_BASAL = DEG_RATE_FLID * FLID_TOTAL_REF / _YG_REF


def _class_ii_gate_fliA(X, Y_free):
    """Kalir & Alon SUM-gate p_i for the fliA promoter itself (index 6)."""
    Xg = X / (K_FLHDC + X)
    Yg = Y_free / (K_FLIA + Y_free)
    return (BETA[FLIA_IDX] * Xg + BETA_PRIME[FLIA_IDX] * Yg) / (BETA[FLIA_IDX] + BETA_PRIME[FLIA_IDX])


def rhs(t, state, target_length):
    X, Y, FlgM, FliT, FliD, N_nascent, N_complete = state
    X = max(X, 0.0); Y = max(Y, 0.0); FlgM = max(FlgM, 0.0)
    FliT = max(FliT, 0.0); FliD = max(FliD, 0.0); N_nascent = max(N_nascent, 0.0)

    # --- FlgM:FliA fast equilibrium (unaffected by the FliT removal) ---
    c_flgm_flia = solve_equilibrium_complex(Y, FlgM, K_D_FLGM_FLIA)
    Y_free = Y - c_flgm_flia
    FlgM_free = FlgM - c_flgm_flia

    # --- FlhD4C2: autonomous synthesis, NO decay term ---
    # Matches the live WCM exactly: both flagella_flhdc_degradation.py
    # (basal turnover) and flagella_flit_flhdc_checkpoint.py (FliT-driven
    # extra decay) were removed 2026-08-10, and the standard
    # protein_degradation process never handles assembled complexes -- so
    # FlhD4C2 has no active decay pathway in the current WCM at all. This
    # is a known, accepted tradeoff (see archive/flit-flhdc-regulation-
    # 2026-08/), not a bug in this model.
    dX = K_SYNTH_X

    # --- FliA (total pool): Class II autoregulatory synthesis, degradation ---
    p_fliA = _class_ii_gate_fliA(X, Y_free)
    dY = K_SYNTH_Y * (p_fliA / P_FLIA_REF) - DEG_RATE_FLIA * Y

    # --- Class III gate (shared by FlgM, FliT, FliD promoters) ---
    class_iii = Y_free / (K_FLIA + Y_free)

    # --- FlgM (total pool): Class III synthesis, degradation, secretion ---
    # Real formula (flagella_flgm_secretion.py): a flat per-complete-
    # flagellum export flux, not proportional to available FlgM -- see
    # v2's docstring for the full story of this correction. FlgM/(FlgM+1)
    # reproduces the real flat rate whenever FlgM is non-negligible but
    # smoothly vanishes as FlgM->0, matching the real discrete clamp.
    secretion_flux = FLGM_SECRETION_RATE * N_complete * FlgM / (FlgM + 1.0)
    dFlgM = K_SYNTH_FLGM_BASAL * class_iii - DEG_RATE_FLGM * FlgM - secretion_flux

    # --- FliT (total pool): Class III synthesis, degradation. Dead end --
    # nothing reads FliT anymore (the checkpoint and the FliT:FliD
    # equilibrium that fed it are both gone) -- tracked only for parity
    # with the real WCM, which still makes FliT, just doesn't wire it to
    # anything currently ("FliT only being there").
    dFliT = K_SYNTH_FLIT_BASAL * class_iii - DEG_RATE_FLIT * FliT

    # --- FliD (total pool): Class III synthesis, degradation, consumption on completion ---
    dFliD = K_SYNTH_FLID_BASAL * class_iii - DEG_RATE_FLID * FliD

    # --- filament construction: mean-field nucleation + exponential-relaxation completion ---
    tau_complete = target_length / FLIC_SYNTHESIS_RATE_MEASURED
    completion_rate = N_nascent / tau_complete
    dN_nascent = NUCLEATION_RATE - completion_rate   # motor-complex gate NOT modeled (see docstring)
    dN_complete = completion_rate
    dFliD -= FLID_PER_COMPLETION * completion_rate

    return [dX, dY, dFlgM, dFliT, dFliD, dN_nascent, dN_complete]


def run(hours, target_length, n_flagella_0):
    seconds = hours * 3600.0
    y0 = [X_REF, Y_TOTAL_REF, FLGM_TOTAL_REF, FLIT_TOTAL_REF, FLID_TOTAL_REF, 0.0, float(n_flagella_0)]
    t_eval = np.linspace(0, seconds, int(seconds // 10) + 1)
    sol = solve_ivp(rhs, [0, seconds], y0, args=(target_length,), t_eval=t_eval,
                     method="LSODA", rtol=1e-8, atol=1e-6)
    return sol


def figure(sol, hours, target_length):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})

    t = sol.t / 3600.0
    X, Y, FlgM, FliT, FliD, N_nascent, N_complete = sol.y
    Y_free = np.array([Yi - solve_equilibrium_complex(Yi, Fi, K_D_FLGM_FLIA) for Yi, Fi in zip(Y, FlgM)])

    fig, axs = plt.subplots(1, 4, figsize=(23.0, 4.7))
    a, b, c, d = axs

    a.plot(t, X, "-", color="#1f77b4", label="FlhD4C2 (X), no decay term")
    a.set_title("FlhD4C2 (unbounded -- no checkpoint, no degradation)")
    a.set_xlabel("time (hr)"); a.set_ylabel("count"); a.legend(fontsize=8)

    b.plot(t, Y, "-", color="#2ca02c", alpha=0.5, label="total FliA")
    b.plot(t, Y_free, "-", color="#2ca02c", label="free FliA")
    b.plot(t, FlgM, "-", color="#ff7f0e", label="total FlgM")
    b.set_title("FliA / FlgM")
    b.set_xlabel("time (hr)"); b.set_ylabel("count"); b.legend(fontsize=8)

    c.plot(t, FliT, "-", color="#e377c2", label="total FliT (dead end)")
    c.plot(t, FliD, "-", color="#7f7f7f", label="total FliD")
    c.set_title("FliT (unused) / FliD")
    c.set_xlabel("time (hr)"); c.set_ylabel("count"); c.legend(fontsize=8)

    d.plot(t, N_complete, "-", color="#9467bd", label="complete flagella")
    dd = d.twinx()
    dd.plot(t, N_nascent, "--", color="#8c564b", alpha=0.7, label="n_nascent (mean-field)")
    dd.set_ylabel("n_nascent", color="#8c564b")
    d.axhspan(2, 8, color="#9467bd", alpha=0.08)
    d.set_title("Flagella count (real range: 2-8/cell)")
    d.set_xlabel("time (hr)"); d.set_ylabel("count", color="#9467bd")
    h1, l1 = d.get_legend_handles_labels(); h2, l2 = dd.get_legend_handles_labels()
    d.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    fig.suptitle(f"Reduced ODE model v3, no FliT checkpoint (matches live WCM) -- target_length={target_length}, {hours}hr")
    fig.tight_layout()
    out = f"{STUDY_DIR}/charts/02_reduced_ode_v3_no_flit_target{target_length}_{hours}hr.svg"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--target-length", type=int, default=REAL_TARGET_LENGTH)
    ap.add_argument("--n-flagella-0", type=float, default=0.0)
    args = ap.parse_args()

    print(f"Y_FREE_REF={Y_FREE_REF:.1f} FLGM_FREE_REF={FLGM_FREE_REF:.1f}")
    print(f"K_SYNTH_X={K_SYNTH_X:.4f} K_SYNTH_Y={K_SYNTH_Y:.4f} "
          f"K_SYNTH_FLGM_BASAL={K_SYNTH_FLGM_BASAL:.4f} K_SYNTH_FLIT_BASAL={K_SYNTH_FLIT_BASAL:.4f} "
          f"K_SYNTH_FLID_BASAL={K_SYNTH_FLID_BASAL:.4f}")
    sol = run(args.hours, args.target_length, args.n_flagella_0)
    figure(sol, args.hours, args.target_length)

    X, Y, FlgM, FliT, FliD, N_nascent, N_complete = sol.y
    print(f"\nFINAL (t={args.hours}hr): X={X[-1]:.1f}  Y_total={Y[-1]:.1f}  FlgM_total={FlgM[-1]:.1f}  "
          f"FliT_total={FliT[-1]:.1f}  FliD_total={FliD[-1]:.1f}  "
          f"n_nascent={N_nascent[-1]:.2f}  n_complete={N_complete[-1]:.2f}")
    n = len(sol.t)
    tail = max(2, n // 10)
    rate_tail = (N_complete[-1] - N_complete[-tail]) / (sol.t[-1] - sol.t[-tail])
    rate_overall = (N_complete[-1] - N_complete[0]) / (sol.t[-1] - sol.t[0])
    print(f"completion rate, last 10% of run: {rate_tail:.6f}/s   overall average: {rate_overall:.6f}/s   "
          f"ratio: {rate_tail/rate_overall if rate_overall else float('nan'):.3f} (< 1 suggests slowing/plateauing)")

    np.savez(f"{STUDY_DIR}/reduced_ode_v3_no_flit_target{args.target_length}_{args.hours}hr.npz",
              t=sol.t, X=X, Y=Y, FlgM=FlgM, FliT=FliT, FliD=FliD,
              N_nascent=N_nascent, N_complete=N_complete)
    print(f"wrote {STUDY_DIR}/reduced_ode_v3_no_flit_target{args.target_length}_{args.hours}hr.npz")


if __name__ == "__main__":
    main()
