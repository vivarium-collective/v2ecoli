"""
=========================
DnaA-box equilibrium binding (dnaa-3 Phase 2)
=========================

Fast-equilibrium occupancy of the 315 DnaA boxes (302 chromosomal high-aff +
3 oriC high-aff + 8 oriC low-aff + 2 dnaA-promoter high-aff) by DnaA-ATP and
DnaA-ADP. Pure bookkeeping — does NOT gate replication initiation.

Pool definitions (set in initial_conditions.py and propagated by
chromosome_structure.py on fork passage):

    pool_label = 0  chromosomal_high  302 sites  K_d = 1 nM    ATP or ADP
    pool_label = 1  oriC_high         3 sites    K_d = 1 nM    ATP or ADP
    pool_label = 2  oriC_low          8 sites    K_d = 100 nM  ATP only
    pool_label = 3  promoter_high     2 sites    K_d = 1 nM    ATP or ADP

DnaA_bound_form on each box row:
    0 = free
    1 = bound DnaA-ATP
    2 = bound DnaA-ADP

For ATP+ADP-binding pools (0/1/3) the per-pool equilibrium is a single
saturable site with two competing ligands:

    free_ATP + box ⇌ box.ATP    K_d_ATP
    free_ADP + box ⇌ box.ADP    K_d_ADP

Detailed balance + mass conservation give the closed-form occupancies:

    p_atp = (A / K_d_ATP)   / (1 + A / K_d_ATP + D / K_d_ADP)
    p_adp = (D / K_d_ADP)   / (1 + A / K_d_ATP + D / K_d_ADP)

where A, D are free concentrations of DnaA-ATP and DnaA-ADP. With N >> A_tot
or D_tot the free ≈ total approximation fails; we solve the conservation
equations iteratively (Newton in (A_free, D_free) — see _solve_pool).

For the ATP-only pool (2) the closed form is the standard one-site Langmuir
in A_free alone, solved by the quadratic.

Hydrolysis (DnaA-ATP -> DnaA-ADP) is owned by the equilibrium step (bf8b82e,
k = 0.046 / min) and applies uniformly to bound + free DnaA-ATP per spec.
This step runs AFTER equilibrium so the ATP/ADP form swap has already
happened — see composites/baseline.py layer 2.

dnaa-3 Phase 2b: the Pi / PROTON / WATER byproducts of bound-pool hydrolysis
are now produced via bf8b82e's stoichMatrix (it reads DnaA_boxes optionally
and injects extra DNAA-INTRINSIC-HYDROLYSIS-RXN flux for the bound pool).
This step still computes its own bound-pool hydrolysis count for the
in-place bound ATP → bound ADP form swap on DnaA_box rows; the two
calculations use the same rate (0.046/min) but independent stochastic rounds,
so the bound_form counts will drift by O(sqrt(N)) per tick from the byproduct
flux. Acceptable noise for box bookkeeping; would require a shared-state
channel to make exactly consistent.

Outputs:
    bulk[DnaA-ATP]   updated by the net change in bound-ATP across pools
    bulk[DnaA-ADP]   updated by the net change in bound-ADP across pools
    DnaA_box.DnaA_bound       boolean per active row
    DnaA_box.DnaA_bound_form  int8 per active row (0 / 1 / 2)
"""

from __future__ import annotations

import os
import numpy as np
from scipy.optimize import root as scipy_root

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import attrs, bulk_name_to_idx, counts
from v2ecoli.library.schema_types import DNAA_BOX_ARRAY
from v2ecoli.library.quantity_helpers import as_quantity
from v2ecoli.types.quantity import ureg as units


NAME = "dnaa-box-binding"
TOPOLOGY = {
    "bulk": ("bulk",),
    "DnaA_boxes": ("unique", "DnaA_box"),
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "dnaa_box_binding"),
    # dnaa-3 Phase 2c: shared bound-hydrolysis count consumed by equilibrium.py.
    # Without this, the two steps independently sample the same Poisson process,
    # producing O(sqrt(N)) divergence between the bound-pool form swap and the
    # byproduct accounting.
    "dnaa_hydrolysis": ("process_state", "dnaa_hydrolysis"),
}


# Bulk IDs (hardcoded from sim_data — see AGENTS.md "How to find an ID").
# MONOMER0-160[c]  = DnaA-ATP (active form)
# MONOMER0-4565[c] = DnaA-ADP (inactive form)
# Pi[c], PROTON[c], WATER[c] = hydrolysis byproducts (per the TSV stoich
# for DNAA-INTRINSIC-HYDROLYSIS-RXN: ATP + WATER → ADP + Pi + PROTON).
DNAA_ATP_ID = "MONOMER0-160[c]"
DNAA_ADP_ID = "MONOMER0-4565[c]"
PI_ID = "Pi[c]"
PROTON_ID = "PROTON[c]"
WATER_ID = "WATER[c]"

# Pool labels (must match Phase 1 initial_conditions.py).
POOL_CHROMOSOMAL_HIGH = 0
POOL_ORIC_HIGH = 1
POOL_ORIC_LOW = 2
POOL_PROMOTER_HIGH = 3

# Bound-form enum.
FORM_FREE = 0
FORM_BOUND_ATP = 1
FORM_BOUND_ADP = 2

# K_d values in molar (mol / L).
KD_HIGH_M = 3e-9    # 3 nM — chromosomal_high, oriC_high, promoter_high
KD_LOW_M = 100e-9   # 100 nM — oriC_low (ATP only). Used as KD_LOW_MAX_M when
                    # COOPERATIVE_ORIC_LOW = False (legacy non-cooperative).

# Per-oriC linear-K_d cooperativity at oriC_low (Haochen spec).
# K_d_low decreases linearly with local occupancy on the same chromosome:
#   K_d_low(occ) = max(KD_LOW_MIN_M, KD_LOW_MAX_M
#                                    - (KD_LOW_MAX_M - KD_LOW_MIN_M)
#                                      * occ / COOP_SATURATION_FRAC)
# At occ = 0:   K_d = 100 nM (matches non-cooperative legacy)
# At occ = 7/8: K_d =   1 nM (sites cooperative saturate)
# Above 7/8 the K_d clamps at KD_LOW_MIN_M.
# Cooperativity is per-chromosome (not pooled across chromosomes) — DnaA-DnaA
# contacts at neighbouring oric_low sites are local, so each oriC's 8 sites
# have their own K_d driven by their own local occupancy fraction.
COOPERATIVE_ORIC_LOW = True
KD_LOW_MAX_M = 100e-9  # 100 nM at zero occupancy (= legacy KD_LOW_M)
KD_LOW_MIN_M = 1e-9    # 1 nM at saturation

# Adaptive per-domain K_d relaxation: if a chromosome's oric_low occupancy
# stays at the same integer n_bound for longer than STUCK_THRESHOLD_S, the
# K_d at that domain is multiplied by a relax factor that drops over time
# (STUCK_RELAX_RATE_PER_S per second stuck) to a floor of STUCK_RELAX_MIN.
# Encodes: cooperative DnaA-DnaA contacts develop as the cluster lingers,
# helping the next site to bind across an apparent barrier.
STUCK_THRESHOLD_S = float(os.environ.get("V2ECOLI_DNAA_STUCK_THRESHOLD_S", "60.0"))
STUCK_RELAX_RATE_PER_S = 0.05  # 5%/s K_d decay once stuck — fast tightening
STUCK_RELAX_MIN = float(os.environ.get("V2ECOLI_DNAA_STUCK_RELAX_MIN", "0.01"))  # floor on relax

# Minimum n_bound the cluster must have reached before the relax dial can
# trigger. Biologically, cooperativity is a property of an already-nucleated
# cluster — if no DnaA has even arrived yet (n=0), there is nothing for the
# new molecules to be cooperative WITH, and the relax dial should not fire.
# Set ≥1 to require at least one bound DnaA-ATP before cooperative help kicks in.
STUCK_RELAX_MIN_N_BOUND = int(os.environ.get(
    "V2ECOLI_DNAA_STUCK_RELAX_MIN_N", "1"))

# Minimum bulk DnaA-ATP concentration (in nM) required for the relax dial to
# fire. Biological motivation: cooperative cluster assembly (DiaA/IHF help)
# requires a meaningful supply of DnaA-ATP available to the cluster. When
# bulk DnaA-ATP is depleted (e.g., post-initiation when daughter clusters
# inherit but bulk is crashed by sequestration), cooperativity should not
# engage because there is no DnaA available to be cooperative ABOUT.
# Naturally distinguishes pre-init parent (bulk peaks ~30 nM, relax fires)
# from post-init daughters (bulk drops to ~0 nM, relax cannot fire).
STUCK_RELAX_MIN_BULK_NM = float(os.environ.get(
    "V2ECOLI_DNAA_STUCK_RELAX_MIN_BULK_NM", "0.0"))

# Pre-init bulk-gate on the relax dial. When enabled, the bulk DnaA-ATP
# threshold that gates the relax dial is set dynamically at initiation time:
# we track the running max of bulk DnaA-ATP during the pre-init phase, and
# at initiation we freeze that value as the post-init threshold. Idea:
# daughter clusters should not fire unless bulk exceeds what was needed
# to fire the parent cluster. Replaces STUCK_RELAX_MIN_BULK_NM after init
# when on.
PREINIT_BULK_GATE = os.environ.get(
    "V2ECOLI_DNAA_PREINIT_BULK_GATE", "0") in ("1", "true", "True")

# Peak-detection gate on the relax dial. When enabled, the relax dial fires
# only after the running max of bulk DnaA-ATP within the current generation
# has been "passed" — i.e., current bulk has dropped below PEAK_FRACTION ×
# running_max. Once init fires, the gate closes for the rest of the
# generation so daughter clusters cannot trigger cooperativity. Adapts the
# trigger to each generation's natural bulk profile (no fixed nM threshold).
# Phenomenological proxy for intrinsic concentration-dependent cooperativity
# (real biology drives the peak via cooperative sequestration; we use the
# peak as a signal to trigger cooperativity).
PEAK_DETECT_GATE = os.environ.get(
    "V2ECOLI_DNAA_PEAK_DETECT_GATE", "0") in ("1", "true", "True")
PEAK_FRACTION = float(os.environ.get(
    "V2ECOLI_DNAA_PEAK_FRACTION", "0.9"))
# Minimum running max (nM) before peak detection arms — avoids false
# positives from tick-to-tick noise at very low bulk values.
PEAK_MIN_NM = float(os.environ.get(
    "V2ECOLI_DNAA_PEAK_MIN_NM", "5.0"))

# Positive-gradient gate on the relax dial. When enabled, the relax dial only
# fires while bulk DnaA-ATP is rising over the past GRADIENT_WINDOW_S seconds.
# Biological motivation: cooperative cluster assembly is a property of the
# growth phase of the cell cycle (synthesis > consumption). After initiation,
# bulk DnaA-ATP first spikes from fork passage then decreases as the system
# settles; during the decreasing phase, daughter clusters should fill only
# through natural binding, not through artificial cooperative boost.
# Distinguishes "rising bulk → parent cluster fills cooperatively → triggers
# init" from "falling bulk → daughters fill (or don't) on physical binding".
GRADIENT_GATE = os.environ.get(
    "V2ECOLI_DNAA_GRADIENT_GATE", "0") in ("1", "true", "True")
GRADIENT_WINDOW_S = float(os.environ.get(
    "V2ECOLI_DNAA_GRADIENT_WINDOW_S", "60.0"))
# Minimum positive slope (nM / s) to count as rising; smaller positive slopes
# are treated as flat (gates off). Prevents noise from spuriously enabling
# the relax dial during quasi-steady-state phases.
GRADIENT_MIN_SLOPE_NM_PER_S = float(os.environ.get(
    "V2ECOLI_DNAA_GRADIENT_MIN_SLOPE_NM_PER_S", "0.0"))

# When stuck > threshold, snap relax to STUCK_RELAX_MIN immediately rather than
# decaying gradually. Default linear K_d at n=2 is 75 nM; snapping to 0.01
# gives K_d ≈ 0.75 nM — competitive with the chromosomal buffer (K_d=1 nM)
# and small enough that even modest free A_f drives cluster to full saturation.
# Idea: "when stuck, lower K_d enough to GUARANTEE filling."

# Cluster dissolution: if n_bound drops > DISSOLUTION_DROPOFF below max_seen,
# the cooperative cluster is losing structure. Relax recovers toward 1.0 at
# RELAX_RECOVERY_RATE_PER_S (slower than decay — clusters take longer to
# dissolve than to mature). When relax > RELAX_RECOVERED_THRESHOLD, max_seen
# resets to current n (the cluster has effectively dissolved).
DISSOLUTION_DROPOFF = 4        # n_bound must be ≥ 4 below max_seen to trigger
                                # (≈ half the cluster collapsed, not flicker)
RELAX_RECOVERY_RATE_PER_S = 0.001  # 0.1%/s recovery — 50× slower than decay
                                    # (cluster takes minutes to dissolve)
RELAX_RECOVERED_THRESHOLD = 0.9    # above this, declare cluster dissolved

COOP_NUCLEATION_THRESHOLD = 0.75  # 6/8 sites — ENTRY into the committed
                                   # (hysteresis-locked) cooperative state.
                                   # Combined with the v8 stuck-time mechanism
                                   # below: a domain commits if it reaches 6/8
                                   # via the linear-Kd build-up. Once committed,
                                   # its K_d is locked at KD_LOW_MIN regardless
                                   # of stuck-time / relax — structural lock-in
                                   # like the report-baseline hysteresis.
COOP_EXIT_THRESHOLD = 0.375       # 3/8 sites — once a chromosome is in the
                                  # cooperative state, it only EXITS when
                                  # occupancy drops below this much lower
                                  # threshold. Asymmetric hysteresis (enter
                                  # at 6/8, exit at 3/8) ensures that once
                                  # cooperative saturation is reached, the
                                  # state is held until fork passage / strong
                                  # dilution forces it out. (Tested EXIT=1/8;
                                  # produced identical dynamics because the
                                  # cooperative state never drops below 3/8
                                  # in practice — fork release is what
                                  # actually destroys the state.)
COOP_SATURATION_FRAC = 7.0 / 8.0  # occupancy at which K_d hits the floor

# Asymmetric K_d (KNF-style ratchet) toggle. When True, the per-oriC
# equilibrium treats bound vs empty sites with DIFFERENT K_ds:
#   - Bound sites:  K_d = KD_LOW_MIN_M  (very stable, slow unbinding)
#   - Empty sites:  K_d = _kd_low_cooperative(n, N)  (linear with occ)
# This breaks the symmetry of the standard equilibrium and collapses the
# bistability — the cluster monotonically climbs to saturation as bulk grows.
# When False (default), uses the symmetric model (single K_d(n) for all
# sites), which has bistable low-n and high-n basins.

# Stochastic kinetics at oriC_low toggle. When True, replaces the equilibrium
# solve for oriC_low with per-site Bernoulli sampling at each tick. Each
# domain's sites are independently sampled from the (asymmetric) K_d-derived
# binding probability. Breaks the symmetry between chromosomes (each cluster
# gets a different random draw), so when bulk DnaA-ATP is in the transition
# regime, one chromosome can cross to high-n while another stays at low-n
# (competitive exclusion via stochastic timing). Equilibrium-solving is still
# used for the high-affinity pools (no bistability there).
STOCHASTIC_ORIC_LOW = os.environ.get("V2ECOLI_DNAA_STOCHASTIC_ORIC_LOW", "0") in ("1", "true", "True")

# Kinetic oriC_low binding. When enabled, the equilibrium solver still finds
# the fixed point (the "target" occupancy at current conditions), but each
# tick we RELAX toward that target with time constant tau = 1/k_off — instead
# of jumping to it instantly. Same K_d curve, same cooperativity, but binding
# takes real biological time to happen. Prevents daughter clusters from
# snap-filling 0→8 in one 2-second tick.
#   dN/dt = k_on × A_free × (N_max − N) − k_off × N
# At equilibrium: N_eq × K_d = A_f × (N_max − N_eq)  ← what solver finds
# Kinetic step: N_new = N_prev + (N_eq − N_prev) × (1 − exp(−dt × k_off))
KINETIC_ORIC_LOW = os.environ.get("V2ECOLI_DNAA_KINETIC_ORIC_LOW", "0") in ("1", "true", "True")
# k_off in 1/s. Real DnaA-oriC dissociation is ~0.001-0.1/s in vitro.
# Default 0.01/s → τ ≈ 100s. Cluster reaches 90% of eq in ~230s (~4 min).
KINETIC_KOFF_PER_S = float(os.environ.get("V2ECOLI_DNAA_KINETIC_KOFF_PER_S", "0.01"))

# Hill K_d cooperativity toggle. When True, replaces the linear K_d(n) formula
# with a Hill function:
#   K_d(n) = K_d_min + (K_d_max - K_d_min) × K_half^h / (K_half^h + n^h)
# Default K_half = 4 (half-saturation occupancy), h = 4 (cooperativity exponent).
# This is steeper than linear — K_d stays near K_d_max at low n and drops
# sharply around K_half. Captures real cooperative binding where the
# transition happens in a narrow occupancy range rather than gradually.
# Adair stepwise binding constants. Cluster of N sites has N sequential
# dissociation constants K_d,1 > K_d,2 > ... > K_d,N (positive cooperativity:
# each bound site makes next binding easier). Geometric interpolation between
# ADAIR_KD_MAX_NM (first binding) and ADAIR_KD_MIN_NM (last binding):
#   K_d,i = K_d_max × (K_d_min/K_d_max)^((i-1)/(N-1))
# Expected occupancy computed via Adair partition function — smooth sigmoidal
# response to bulk concentration, NO bistability, single-valued equilibrium.
# Cluster cooperativity captured by the K_d,i sequence.
ADAIR_KD = os.environ.get("V2ECOLI_DNAA_ADAIR_KD", "0") in ("1", "true", "True")
ADAIR_KD_MAX_nM = float(os.environ.get("V2ECOLI_DNAA_ADAIR_KD_MAX_NM", "100.0"))
ADAIR_KD_MIN_nM = float(os.environ.get("V2ECOLI_DNAA_ADAIR_KD_MIN_NM", "1.0"))
ADAIR_KD_MAX_M = ADAIR_KD_MAX_nM * 1e-9
ADAIR_KD_MIN_M = ADAIR_KD_MIN_nM * 1e-9
# Explicit per-site Adair K_d list (comma-separated nM values, one per site).
# When set, overrides the geometric interpolation between MAX/MIN. Standard
# Adair parameterization (Hb-style). Example: "100,100,100,30,30,3,3,3" for
# a stepped K_d representing IHF-unlock at n=3-5.
ADAIR_KDS_nM_STR = os.environ.get("V2ECOLI_DNAA_ADAIR_KDS_NM", "").strip()
if ADAIR_KDS_nM_STR:
    ADAIR_KDS_M = tuple(float(v) * 1e-9 for v in ADAIR_KDS_nM_STR.split(","))
else:
    ADAIR_KDS_M = None

# Gradient-gated cooperativity: when enabled, Hill K_d cooperativity only
# engages while bulk DnaA-ATP is actively accumulating (positive gradient).
# When gradient is not positive, K_d is pinned to K_d_max regardless of n
# — the cluster behaves as if no cooperativity has taken hold. Biologically
# this approximates the idea that cooperative oligomerization at oriC
# requires DnaA-ATP to be arriving. Combined with a fixed K_half (no
# adaptive), this gives a two-gate cooperativity: K_half=X AND rising bulk.
COOP_GRADIENT_GATE = os.environ.get(
    "V2ECOLI_DNAA_COOP_GRADIENT_GATE", "0") in ("1", "true", "True")


def _kd_low_cooperative(n_bound: float, n_total: float, relax: float = 1.0,
                         coop_engaged: bool = True) -> float:
    """Per-oriC K_d for the Langmuir (linear-K_d) fallback path.

    K_d drops linearly from K_d_max at n=0 to K_d_min at n=N_total. Gradual
    transition across the whole range. Used as the non-Adair fallback and
    inside the ASYMMETRIC / stochastic / kinetic oric_low branches. The
    stepped Adair ladder (ADAIR_KD=1) is the primary path and does not
    use this function.
    """
    if n_total <= 0:
        return KD_LOW_MAX_M
    # Gradient-gated cooperativity: if the gate is enabled and cooperativity
    # is NOT currently engaged (bulk not rising), pin K_d to K_d_max regardless
    # of n. Cluster behaves as if no cooperative oligomerization has taken hold.
    if COOP_GRADIENT_GATE and not coop_engaged:
        return KD_LOW_MAX_M
    occ = max(0.0, min(1.0, n_bound / n_total))
    base_kd = KD_LOW_MAX_M + (KD_LOW_MIN_M - KD_LOW_MAX_M) * occ
    # Clamp effective K_d at K_d_min so the relax dial cannot push K_d below
    # the natural fully-cooperative value. The relax represents "the cluster
    # behaves cooperatively, like fully bound sites" — not an artificial
    # super-tight binding that exceeds biology.
    return max(KD_LOW_MIN_M, base_kd * relax)

# DnaA-ATP intrinsic hydrolysis rate. Bound DnaA-ATP hydrolyzes at the same
# rate as free DnaA-ATP (Sekimizu 1987, k = 0.046 / min). bf8b82e's equilibrium
# step hydrolyzes the FREE pool only; this step additionally hydrolyzes the
# BOUND pool, in-place (bound box-ATP → bound box-ADP on the same row).
HYDROLYSIS_RATE_PER_MIN = float(os.environ.get(
    "V2ECOLI_DNAA_HYDROLYSIS_RATE_PER_MIN", "0.025"))

# Per-domain post-init K_d ladder unlock. Fresh daughter oriCs (born after a
# fork event) start with K_d clamped to K_d_max (100 nM) for all 8 sites —
# i.e., cooperativity is disengaged. They can only "unlock" the K_d ladder
# after seeing a SUSTAINED positive bulk-DnaA-ATP gradient for
# POST_INIT_UNLOCK_S seconds. Any negative-gradient tick resets the counter.
# Purpose: prevent daughter oriCs from firing on the transient positive-slope
# artifact when mother's oriC releases 8 DnaA-ATP into bulk at fork passage.
# Set to 0 to disable.
POST_INIT_UNLOCK_S = float(os.environ.get(
    "V2ECOLI_DNAA_POST_INIT_UNLOCK_S", "0.0"))


class DnaABoxBinding(Step):
    """DnaA-box equilibrium binding step.

    Pure equilibrium bookkeeping — does not gate replication initiation.
    See module docstring for the maths.
    """

    description = (
        "DnaA-box equilibrium binding — closed-form Langmuir / "
        "competitive-Langmuir solve per pool.\n"
        "    p_atp = (A/Kd_A) / (1 + A/Kd_A + D/Kd_D)   competitive\n"
        "    p_adp = (D/Kd_D) / (1 + A/Kd_A + D/Kd_D)   competitive\n"
        "    p_atp = (A/Kd_A) / (1 + A/Kd_A)            ATP-only\n"
        "Free DnaA-ATP / DnaA-ADP solved iteratively (Newton) from"
        " conservation; bound-form counts rounded stochastically."
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'cell_density': {'_type': 'float[g/L]', '_default': 1100.0},
        'n_avogadro': {'_type': 'float[1/mol]', '_default': 6.02214076e23},
        'kd_high_M': {'_type': 'float', '_default': KD_HIGH_M},
        'kd_low_M': {'_type': 'float', '_default': KD_LOW_M},
        'seed': {'_type': 'integer', '_default': 0},
        'time_step': {'_type': 'integer[s]', '_default': 1},
        'bulk_mass_data': {'_type': 'quantity[array[float],g/mol]', '_default': None},
        'bulk_molecule_ids': {'_type': 'array[string]', '_default': None},
        'submass_indices': {'_type': 'map[integer]', '_default': {}},
    }

    def initialize(self, config):
        self.cell_density = self.parameters["cell_density"]
        self.n_avogadro = self.parameters["n_avogadro"]
        self.kd_high_M = float(self.parameters.get("kd_high_M", KD_HIGH_M))
        self.kd_low_M = float(self.parameters.get("kd_low_M", KD_LOW_M))
        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Per-molecule mass for DnaA-ATP and DnaA-ADP, indexed by submass type.
        # Used to update DnaA_box.massDiff_* fields so cell mass tracks the
        # DnaA moved bulk → box on binding (same pattern as tf_binding for TFs
        # binding promoters; see tf_binding.py:362-369).
        self.submass_indices = dict(self.parameters.get("submass_indices") or {})
        self._mass_lookup = None  # shape (3, n_submass); resolved on first update
        self._bulk_mass_data = self.parameters.get("bulk_mass_data")
        self._bulk_molecule_ids = self.parameters.get("bulk_molecule_ids")

        # Indices resolved on first update.
        self._atp_idx = None
        self._adp_idx = None
        self._pi_idx = None
        self._proton_idx = None
        self._water_idx = None

        # Memory-aware cooperativity (asymmetric hysteresis): set of
        # domain_index values whose oric_low subpool is currently in the
        # cooperative state. Once a domain enters (occ ≥ NUCLEATION) its
        # K_d stays at KD_LOW_MIN_M until occ drops below COOP_EXIT_THRESHOLD.
        # Domains that disappear (fork release) are pruned each tick.
        self._cooperative_domains = set()

        # Adaptive per-domain stuck-state tracking (Option 2: progress-based).
        # _stuck_n[dom]: MAX n_bound this domain has reached so far in its lifetime
        # _stuck_secs[dom]: seconds spent at-or-below max without upward progress
        # _kd_relax[dom]: multiplicative K_d factor (1.0 default; drops while
        #                 stuck beyond STUCK_THRESHOLD_S, resets only on progress
        #                 — i.e., n_bound exceeding the previous max)
        # Biological rationale: cooperative cluster maturity is preserved across
        # hydrolysis flicker (single-molecule turnover); only true upward
        # progress (cluster growth) resets the "stuck" accumulator.
        self._stuck_n = {}
        self._stuck_secs = {}
        self._kd_relax = {}

        # Per-cluster K_half state for the adaptive-K_half mechanism. Persisted
        # tick-by-tick so the listener can emit per-domain K_half traces —
        # domains that spent time stuck at a low n show a K_half floor equal to
        # that stuck n, matching the "K_d curve unlocked at stuck occupancy"
        # semantic. Reset when the domain vanishes (fork release).

        # Positive-gradient gate state: rolling window of recent bulk DnaA-ATP
        # observations as (time_s, nM) tuples. Gate fires only while the
        # current bulk concentration is greater than the oldest value within
        # the window — i.e. bulk is rising.
        self._bulk_atp_history = []

        # Pre-init bulk-gate / peak-detect-gate state. Tracks running max of
        # bulk DnaA-ATP and whether a "bulk peak" has been detected this gen.
        # Shared by PREINIT_BULK_GATE and PEAK_DETECT_GATE.
        self._gen_running_max_bulk_nM = 0.0
        self._preinit_bulk_max_nM = None
        self._gen_peak_detected = False
        self._gen_init_fired = False
        self._prev_noric = None

        # Solver warm-start state — pure-linear-K_d mode (no hysteresis flag,
        # no relax dial). Cooperative K_d has intrinsic bistability; the
        # equilibrium solver picks an arbitrary basin per tick when started
        # cold, which produces tick-by-tick flicker. Warm-starting from the
        # previous tick's converged bound counts gives the solver "memory" —
        # a cluster that landed in the high-n basin stays there until a real
        # release event (fork passage clears the domain key from cache).
        # Stochastic kinetics would give this naturally; here we approximate.
        self._prev_A_free = None
        self._prev_D_free = None
        self._prev_n_bound_by_dom = {}  # domain_key → bound DnaA-ATP from last tick

        # Per-domain post-init K_d ladder unlock (see POST_INIT_UNLOCK_S).
        # `_seen_domains` records every dom_key ever encountered; the first
        # time we see any domain, we treat it as "existing" (unlocked, since
        # it was present at cell birth / dill resume). Domains that first
        # appear on later ticks are post-fork daughters — they start locked.
        self._seen_domains = set()
        self._dom_kd_ladder_unlocked = {}
        self._dom_pos_grad_secs = {}

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'DnaA_boxes': {'_type': DNAA_BOX_ARRAY, '_default': []},
            'listeners': {
                'mass': {
                    'cell_mass': {'_type': 'quantity[float,fg]', '_default': 0},
                    # Volume computed by mass_deriver as cell_mass / cell_density.
                    # Reading from the listener keeps a single source of truth.
                    'volume': {'_type': 'quantity[float,fL]', '_default': 0.0},
                },
                # Used by PREINIT_BULK_GATE to detect initiation events
                # (number_of_oric step) and freeze the pre-init bulk max.
                'replication_data': {
                    'number_of_oric': {'_type': 'integer', '_default': 0},
                },
            },
            'global_time': {'_type': 'float[s]', '_default': 0.0},
            'timestep': {'_type': 'float[s]', '_default': 1.0},
            'next_update_time': {
                '_type': 'overwrite[float[s]]', '_default': 1.0},
            'dnaa_hydrolysis': {
                'bound_count': {'_type': 'integer', '_default': 0},
            },
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'DnaA_boxes': DNAA_BOX_ARRAY,
            'next_update_time': 'overwrite[float[s]]',
            'dnaa_hydrolysis': {
                'bound_count': 'overwrite[integer]',
            },
        }

    def update_condition(self, timestep, states):
        if states["next_update_time"] <= states["global_time"]:
            return True
        return False

    # ------------------------------------------------------------------
    # Equilibrium maths
    # ------------------------------------------------------------------
    @staticmethod
    def _solve_competitive_pool(
        n_sites: int,
        atp_total_molecules: float,
        adp_total_molecules: float,
        kd_atp_molecules: float,
        kd_adp_molecules: float,
    ) -> tuple[float, float]:
        """Solve the competitive single-site equilibrium for one pool.

        Each box has at most one bound ligand (ATP or ADP). Three conservation
        constraints + two equilibria give:

            n_atp_bound = N * a / (1 + a + d)
            n_adp_bound = N * d / (1 + a + d)

        where a = A_free / Kd_atp, d = D_free / Kd_adp. The free counts must
        also satisfy

            A_free = A_total - n_atp_bound
            D_free = D_total - n_adp_bound

        We work in molecules throughout (all K_d arguments are pre-converted
        to "molecules" by multiplying by V * N_A) and solve by fixed-point
        iteration on (A_free, D_free) — converges in <20 iter for biological
        ratios (binding fraction is monotone in A/D).
        """
        if n_sites == 0:
            return 0.0, 0.0
        # Initial guess: assume all DnaA is free (overestimate of A_free, D_free).
        A_free = float(atp_total_molecules)
        D_free = float(adp_total_molecules)
        damping = 0.5
        n_atp_bound = 0.0
        n_adp_bound = 0.0
        for _ in range(500):
            a = A_free / kd_atp_molecules if kd_atp_molecules > 0 else 0.0
            d = D_free / kd_adp_molecules if kd_adp_molecules > 0 else 0.0
            denom = 1.0 + a + d
            n_atp_bound = n_sites * a / denom
            n_adp_bound = n_sites * d / denom
            target_A = max(0.0, atp_total_molecules - n_atp_bound)
            target_D = max(0.0, adp_total_molecules - n_adp_bound)
            # Damped update — straight substitution can oscillate when the
            # system is saturated (bound ~ available).
            new_A = A_free + damping * (target_A - A_free)
            new_D = D_free + damping * (target_D - D_free)
            if (abs(new_A - A_free) < 1e-6 * max(1.0, atp_total_molecules)
                    and abs(new_D - D_free) < 1e-6 * max(1.0, adp_total_molecules)):
                A_free, D_free = new_A, new_D
                break
            A_free, D_free = new_A, new_D
        return n_atp_bound, n_adp_bound

    @staticmethod
    def _solve_atp_only_pool(
        n_sites: int,
        atp_total_molecules: float,
        kd_atp_molecules: float,
    ) -> float:
        """Solve A_free + box ⇌ box.A for a single-ligand pool.

        Mass conservation: A_total = A_free + N * A_free / (Kd + A_free)
            => A_free^2 + (Kd + N - A_total) * A_free - Kd * A_total = 0
        Quadratic in A_free; take the positive root.
        """
        if n_sites == 0 or atp_total_molecules <= 0:
            return 0.0
        b = kd_atp_molecules + n_sites - atp_total_molecules
        c = -kd_atp_molecules * atp_total_molecules
        disc = b * b - 4.0 * c
        if disc < 0:
            disc = 0.0
        A_free = 0.5 * (-b + np.sqrt(disc))
        if A_free < 0:
            A_free = 0.0
        if A_free > atp_total_molecules:
            A_free = atp_total_molecules
        n_atp_bound = atp_total_molecules - A_free
        if n_atp_bound > n_sites:
            n_atp_bound = float(n_sites)
        return n_atp_bound

    def _stochastic_round(self, x: float) -> int:
        if x <= 0:
            return 0
        floor_x = int(np.floor(x))
        frac = x - floor_x
        if self.random_state.random_sample() < frac:
            floor_x += 1
        return floor_x

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, states, interval=None):
        # First-tick bulk index resolution.
        if self._atp_idx is None:
            bulk_ids = states["bulk"]["id"]
            self._atp_idx = bulk_name_to_idx(DNAA_ATP_ID, bulk_ids)
            self._adp_idx = bulk_name_to_idx(DNAA_ADP_ID, bulk_ids)
            self._pi_idx = bulk_name_to_idx(PI_ID, bulk_ids)
            self._proton_idx = bulk_name_to_idx(PROTON_ID, bulk_ids)
            self._water_idx = bulk_name_to_idx(WATER_ID, bulk_ids)
            # Build the bound-form → submass mass lookup once. Row 0 = free
            # (zero mass added to the box); rows 1/2 = mass per molecule of
            # DnaA-ATP / DnaA-ADP. Shape (3, n_submass), units fg/molecule.
            if (self._bulk_mass_data is not None
                    and self._bulk_molecule_ids is not None):
                ids_arr = np.asarray(self._bulk_molecule_ids)
                atp_bulk_idx = int(np.where(ids_arr == DNAA_ATP_ID)[0][0])
                adp_bulk_idx = int(np.where(ids_arr == DNAA_ADP_ID)[0][0])
                # bulk_mass_data arrives as pint Quantity in g/mol; strip
                # units, divide by N_A to get g per molecule, scale to fg.
                bm = self._bulk_mass_data
                bm_g_per_mol = (
                    bm.to("g/mol").magnitude if hasattr(bm, "to")
                    else np.asarray(bm))
                masses_per_mol_fg = bm_g_per_mol / float(self.n_avogadro) * 1e15
                n_submass = masses_per_mol_fg.shape[1]
                self._mass_lookup = np.zeros((3, n_submass), dtype=np.float64)
                self._mass_lookup[FORM_BOUND_ATP] = masses_per_mol_fg[atp_bulk_idx]
                self._mass_lookup[FORM_BOUND_ADP] = masses_per_mol_fg[adp_bulk_idx]

        # Pull active DnaA_box attributes.
        boxes = states["DnaA_boxes"]
        n_active = int(boxes["_entryState"].sum())
        if n_active == 0:
            # No chromosome yet — nothing to do.
            return {
                "next_update_time": states["global_time"] + states["timestep"],
            }

        pool_label, bound_form, domain_index = attrs(
            boxes, ["pool_label", "DnaA_bound_form", "domain_index"])

        # Snapshot bulk counts.
        atp_bulk_count = int(counts(states["bulk"], self._atp_idx))
        adp_bulk_count = int(counts(states["bulk"], self._adp_idx))

        # Current bound counts (from previous tick's bookkeeping).
        prev_bound_atp = int(np.count_nonzero(bound_form == FORM_BOUND_ATP))
        prev_bound_adp = int(np.count_nonzero(bound_form == FORM_BOUND_ADP))

        # Hydrolyze bound DnaA-ATP at the same per-molecule rate as bf8b82e's
        # DNAA-INTRINSIC-HYDROLYSIS-RXN on the free pool. The TSV stores the
        # bimolecular rate (1.4E-5 M⁻¹s⁻¹); pseudo-first-order rate with
        # [WATER]≈55 M is k = 0.046/min. bf8b82e covers the free pool, this
        # step covers the bound pool — together they apply the TSV reaction
        # uniformly to (free + bound).
        dt_min = float(states["timestep"]) / 60.0
        delta_h_bound = self._stochastic_round(
            HYDROLYSIS_RATE_PER_MIN * prev_bound_atp * dt_min)
        if delta_h_bound > prev_bound_atp:
            delta_h_bound = prev_bound_atp

        total_atp = atp_bulk_count + prev_bound_atp - delta_h_bound
        total_adp = adp_bulk_count + prev_bound_adp + delta_h_bound

        # Cell volume (L) — read directly from the mass_deriver's volume
        # listener rather than recomputing from cell_mass / cell_density.
        cell_volume_L = as_quantity(
            states["listeners"]["mass"]["volume"], units.fL
        ).to(units.L).magnitude

        # Maintain rolling window of bulk DnaA-ATP concentration for the
        # positive-gradient gate (used inside the relax-fire check below).
        # Window is pruned to GRADIENT_WINDOW_S seconds.
        if GRADIENT_GATE:
            t_now_s = float(states["global_time"])
            tick_bulk_atp_nM = (atp_bulk_count
                                / (cell_volume_L * self.n_avogadro)
                                * 1e9)
            self._bulk_atp_history.append((t_now_s, tick_bulk_atp_nM))
            cutoff = t_now_s - GRADIENT_WINDOW_S
            while (len(self._bulk_atp_history) > 1
                   and self._bulk_atp_history[0][0] < cutoff):
                self._bulk_atp_history.pop(0)

        # Track running max of bulk DnaA-ATP (shared by PREINIT_BULK_GATE and
        # PEAK_DETECT_GATE), peak detection, and initiation detection.
        if PREINIT_BULK_GATE or PEAK_DETECT_GATE:
            tick_bulk_atp_nM = (atp_bulk_count
                                / (cell_volume_L * self.n_avogadro)
                                * 1e9)
            self._gen_running_max_bulk_nM = max(
                self._gen_running_max_bulk_nM, tick_bulk_atp_nM)
            cur_noric = int(states["listeners"]["replication_data"]
                            .get("number_of_oric", 0))
            if (self._prev_noric is not None
                    and cur_noric > self._prev_noric
                    and self._preinit_bulk_max_nM is None):
                # Initiation just fired. Freeze the pre-init max as the
                # threshold for the rest of the cycle and lock the peak gate.
                self._preinit_bulk_max_nM = self._gen_running_max_bulk_nM
                self._gen_init_fired = True
            self._prev_noric = cur_noric

            # Peak detection: arm once running_max exceeds PEAK_MIN_NM, fire
            # once current bulk has dropped to PEAK_FRACTION × running_max.
            if (PEAK_DETECT_GATE
                    and not self._gen_peak_detected
                    and self._gen_running_max_bulk_nM >= PEAK_MIN_NM
                    and tick_bulk_atp_nM
                        < PEAK_FRACTION * self._gen_running_max_bulk_nM):
                self._gen_peak_detected = True

        # K_d in molecules (so all per-pool maths stay integer-scale).
        # Kd[mol/L] * V[L] * N_A[1/mol] = molecules.
        kd_high_molecules = self.kd_high_M * cell_volume_L * self.n_avogadro
        kd_low_molecules = self.kd_low_M * cell_volume_L * self.n_avogadro

        # Per-pool site counts.
        pool_masks = {p: pool_label == p for p in (
            POOL_CHROMOSOMAL_HIGH, POOL_ORIC_HIGH,
            POOL_ORIC_LOW, POOL_PROMOTER_HIGH)}
        n_sites = {p: int(pool_masks[p].sum()) for p in pool_masks}

        # ATP-binding sites (chromosomal_high + oriC_high + oriC_low +
        # promoter_high) all compete for the same free DnaA-ATP pool.
        # ADP-binding sites (chromosomal_high + oriC_high + promoter_high)
        # all compete for the same free DnaA-ADP pool.
        # We treat the four pools as a single effective competitive
        # equilibrium: one big high-aff pool (302 + 3 + 2 = 307 sites at
        # K_d_high, ATP or ADP) plus the low-aff pool (8 sites at K_d_low,
        # ATP only) that runs in parallel sharing the same A_free.
        # This is the right physics because all pools see the same free
        # cytoplasmic [DnaA-ATP] and [DnaA-ADP].
        n_high_total = (n_sites[POOL_CHROMOSOMAL_HIGH]
                        + n_sites[POOL_ORIC_HIGH]
                        + n_sites[POOL_PROMOTER_HIGH])
        n_low_total = n_sites[POOL_ORIC_LOW]

        # Solve the global equilibrium for high-aff + low-aff pools using
        # scipy.optimize.root on the algebraic mass-balance system.
        #
        # High-aff pool: K_h, both ATP and ADP, sites = N_h (chrom+oric_hi+prom)
        # Low-aff pool(s): K_l(occ) per-chromosome cooperativity (Haochen spec).
        #   Each chromosome has its own oric_low subpool (typically 8 sites);
        #   its K_l depends on its OWN local occupancy. All subpools compete
        #   for the same A_free, but each has its own K_l(n_bound_g / 8).
        #
        # Variables for scipy.root:
        #   x[0] = A_free, x[1] = D_free
        #   x[2..2+N_groups-1] = n_bound on each oric_low subpool
        #
        # Residuals (mass balance):
        #   r0 = A_T - A_free - N_h*a/(1+a+d) - Σ x[2+g]
        #   r1 = D_T - D_free - N_h*d/(1+a+d)
        #   r[2+g] = x[2+g] - N_l_g * A_free / (K_l_g(x[2+g]) + A_free)
        #
        # MINPACK hybr (Powell hybrid, Newton-like) converges in ~5-15 iter.
        K_h = kd_high_molecules
        N_h = n_high_total
        A_T = float(total_atp)
        D_T = float(total_adp)

        # Per-chromosome oric_low subpool partitioning (cooperativity is local).
        oric_low_mask_full = pool_masks[POOL_ORIC_LOW]
        if COOPERATIVE_ORIC_LOW and oric_low_mask_full.any():
            oric_low_doms = domain_index[oric_low_mask_full]
            unique_doms = np.unique(oric_low_doms)
            group_n_sites = np.array(
                [int((oric_low_doms == d).sum()) for d in unique_doms],
                dtype=np.float64)
            n_groups = int(len(unique_doms))

            # Update per-chromosome stuck-state tracking (Option 2).
            # Track MAX n_bound seen for each domain. Reset stuck state only
            # on upward progress (n exceeding previous max). Hydrolysis flicker
            # — single-molecule turnover dropping n by 1 — accumulates as stuck
            # time, because the cooperative cluster's higher-order structure
            # is preserved across transient single-molecule loss.
            current_doms_set = set()
            n_bound_prev_by_dom = {}
            for i, dom in enumerate(unique_doms):
                dom_key = int(dom)
                current_doms_set.add(dom_key)
                dom_mask = (domain_index == dom) & oric_low_mask_full
                n_sites_in_dom = float(dom_mask.sum())
                if n_sites_in_dom <= 0:
                    continue
                n_bound_prev = int(np.count_nonzero(
                    bound_form[dom_mask] == FORM_BOUND_ATP))
                n_bound_prev_by_dom[dom_key] = n_bound_prev
                # v8 stuck-time relax (restored): track MAX n_bound per
                # domain; if cluster stays at the same n for STUCK_THRESHOLD_S,
                # multiply K_d by a relax factor that decays with stuck time.
                # Resets on upward progress (cluster grew past previous max) or
                # dissolution (cluster fell ≥ DISSOLUTION_DROPOFF below max).
                prev_max_n = self._stuck_n.get(dom_key)
                # Reset stuck timer only on SIGNIFICANT upward progress (≥2
                # jump), not on every +1 fluctuation. With +1 reset rule,
                # natural binding/unbinding fluctuations between adjacent
                # low-n values (e.g. 0↔1) keep resetting the timer, preventing
                # relax from ever firing. The ≥2 jump rule lets the timer
                # accumulate during low-n fluctuation, so relax reliably
                # triggers once per cycle when the cluster is genuinely stuck
                # at a low max.
                if prev_max_n is None:
                    # New domain — initialize state
                    self._stuck_n[dom_key] = n_bound_prev
                    self._stuck_secs[dom_key] = 0.0
                    self._kd_relax[dom_key] = 1.0
                elif n_bound_prev > prev_max_n:
                    # Upward progress (cluster grew) — update max, zero the
                    # stuck timer, AND reset relax to 1.0. Cooperativity must
                    # be re-engaged each cycle by the bulk-DnaA-ATP gate —
                    # without carry-forward, once the cluster has filled it
                    # falls back to natural K_d, and the next cycle requires
                    # bulk to cross the gate threshold again. This prevents
                    # the cluster from staying locked across post-init bulk
                    # crashes (which is what allows daughter cluster firing
                    # under the carry-forward semantics).
                    self._stuck_n[dom_key] = n_bound_prev
                    self._stuck_secs[dom_key] = 0.0
                    self._kd_relax[dom_key] = 1.0
                elif (prev_max_n - n_bound_prev) >= DISSOLUTION_DROPOFF:
                    self._stuck_n[dom_key] = n_bound_prev
                    self._stuck_secs[dom_key] = 0.0
                    self._kd_relax[dom_key] = 1.0
                else:
                    # Guard: cooperativity is a property of an already-nucleated
                    # AND currently-bound cluster. Check the CURRENT occupancy
                    # (n_bound_prev), not the historical max (prev_max_n). If
                    # the cluster transiently reached n=1 in the past but has
                    # since dropped back to n=0, there is nothing currently
                    # bound for cooperativity to help — relax must NOT fire.
                    # The historical max being ≥ MIN_N is a necessary but not
                    # sufficient condition; the cluster must ALSO currently
                    # have ≥ MIN_N bound.
                    if n_bound_prev >= STUCK_RELAX_MIN_N_BOUND:
                        self._stuck_secs[dom_key] += dt_min * 60.0
                        # Bulk DnaA-ATP concentration guard: relax only fires
                        # if the cell has enough bulk DnaA-ATP available for
                        # the cooperative cluster to draw from. This natural
                        # mechanism prevents daughter clusters from firing
                        # post-init when bulk has been depleted.
                        bulk_atp_nM = (atp_bulk_count
                                       / (cell_volume_L * self.n_avogadro)
                                       * 1e9)
                        # Dynamic bulk-gate (optional): once init has fired,
                        # the threshold becomes the pre-init bulk max.
                        # Daughters only fire if bulk exceeds parent's peak.
                        bulk_gate_threshold = STUCK_RELAX_MIN_BULK_NM
                        if (PREINIT_BULK_GATE
                                and self._preinit_bulk_max_nM is not None):
                            bulk_gate_threshold = max(
                                bulk_gate_threshold,
                                self._preinit_bulk_max_nM)
                        # Positive-gradient gate (optional): only fire when
                        # bulk DnaA-ATP is rising over the past GRADIENT_WINDOW_S
                        # seconds. Disabled by default — see GRADIENT_GATE.
                        gradient_ok = True
                        if GRADIENT_GATE and len(self._bulk_atp_history) >= 2:
                            t_now = float(states["global_time"])
                            t_old = self._bulk_atp_history[0][0]
                            nM_old = self._bulk_atp_history[0][1]
                            window_s = max(t_now - t_old, 1e-6)
                            slope_nM_per_s = (
                                (bulk_atp_nM - nM_old) / window_s)
                            gradient_ok = (
                                slope_nM_per_s > GRADIENT_MIN_SLOPE_NM_PER_S)
                        # Peak-detection gate: fire only after the bulk peak
                        # has been detected this generation AND before init
                        # fires (lock daughters out). Overrides the bulk
                        # concentration check when on.
                        peak_gate_ok = True
                        if PEAK_DETECT_GATE:
                            peak_gate_ok = (self._gen_peak_detected
                                            and not self._gen_init_fired)
                        if (self._stuck_secs[dom_key] > STUCK_THRESHOLD_S
                                and bulk_atp_nM >= bulk_gate_threshold
                                and gradient_ok
                                and peak_gate_ok):
                            # Gradual decay of relax dial while cluster is stuck.
                            new_relax = (1.0 - STUCK_RELAX_RATE_PER_S * (dt_min * 60.0)) \
                                * self._kd_relax.get(dom_key, 1.0)
                            self._kd_relax[dom_key] = max(STUCK_RELAX_MIN, new_relax)
                    else:
                        # Cluster currently below the nucleation threshold —
                        # keep stuck timer at zero so the 60-second settling
                        # window restarts whenever natural binding pushes
                        # n_bound back to ≥ STUCK_RELAX_MIN_N_BOUND.
                        self._stuck_secs[dom_key] = 0.0
            # Prune any vanished domains (fork release). Clear warm-start
            # cache + stuck-time state for vanished keys so daughter domains
            # (different keys) start fresh.
            for stale in list(self._prev_n_bound_by_dom.keys()):
                if stale not in current_doms_set:
                    self._prev_n_bound_by_dom.pop(stale, None)
            for stale in list(self._stuck_n.keys()):
                if stale not in current_doms_set:
                    self._stuck_n.pop(stale, None)
                    self._stuck_secs.pop(stale, None)
                    self._kd_relax.pop(stale, None)
            # Prune post-init-unlock state for vanished domains.
            if POST_INIT_UNLOCK_S > 0:
                for stale in list(self._dom_kd_ladder_unlocked.keys()):
                    if stale not in current_doms_set:
                        self._dom_kd_ladder_unlocked.pop(stale, None)
                        self._dom_pos_grad_secs.pop(stale, None)
                        self._seen_domains.discard(stale)
        else:
            unique_doms = np.array([], dtype=np.int64)
            group_n_sites = np.array([], dtype=np.float64)
            n_groups = 0
            # No oric_low groups → clear hysteresis state.
            self._cooperative_domains.clear()

        # Non-cooperative fallback K_l (used when COOPERATIVE_ORIC_LOW=False
        # or when grouping is degenerate). Matches legacy behaviour.
        K_l_legacy = kd_low_molecules
        N_l_legacy = n_low_total

        if (A_T + D_T) > 0 and (N_h + N_l_legacy) > 0:
            if COOPERATIVE_ORIC_LOW and n_groups > 0:
                # Precompute per-group K_d relax factors from stuck-tracking.
                # v8 dynamic adjustment: K_d is multiplied by relax<1 when the
                # cluster has been stuck for STUCK_THRESHOLD_S seconds. Only
                # affects the Langmuir (linear-K_d) fallback path; the stepped
                # Adair ladder ignores relax and reads its K_d,i list directly.
                group_relax = [
                    self._kd_relax.get(int(unique_doms[i]), 1.0)
                    for i in range(n_groups)
                ]
                # Bulk-ATP gradient state (used by the Adair GRADIENT gate and
                # by the POST_INIT_UNLOCK_S post-init ladder unlock below).
                # gradient_rising = True → bulk DnaA-ATP is accumulating over
                # the current window (or GRADIENT_GATE disabled); False → bulk
                # is flat or falling, so cooperative loading is not permitted.
                gradient_rising = True
                if GRADIENT_GATE and len(self._bulk_atp_history) >= 2:
                    t_now_g = float(states["global_time"])
                    t_old_g = self._bulk_atp_history[0][0]
                    nM_old_g = self._bulk_atp_history[0][1]
                    nM_now_g = self._bulk_atp_history[-1][1]
                    window_s_g = max(t_now_g - t_old_g, 1e-6)
                    slope_g = (nM_now_g - nM_old_g) / window_s_g
                    gradient_rising = (slope_g > GRADIENT_MIN_SLOPE_NM_PER_S)

                # Per-domain K_d ladder unlock tracking. New domains (post-fork
                # daughters) start locked; they must accumulate POST_INIT_UNLOCK_S
                # of continuous positive bulk gradient to unlock. Any negative-
                # gradient tick resets the counter. Existing domains (present at
                # first tick / dill resume) start unlocked.
                if POST_INIT_UNLOCK_S > 0:
                    is_first_tick = (len(self._seen_domains) == 0)
                    for dom_key in current_doms_set:
                        if dom_key not in self._seen_domains:
                            self._seen_domains.add(dom_key)
                            # First-tick domains are treated as "existing" and
                            # start unlocked; later-appearing domains start locked.
                            self._dom_kd_ladder_unlocked[dom_key] = is_first_tick
                            self._dom_pos_grad_secs[dom_key] = 0.0
                        if not self._dom_kd_ladder_unlocked.get(dom_key, False):
                            if gradient_rising:
                                self._dom_pos_grad_secs[dom_key] = (
                                    self._dom_pos_grad_secs.get(dom_key, 0.0)
                                    + dt_min * 60.0)
                                if (self._dom_pos_grad_secs[dom_key]
                                        >= POST_INIT_UNLOCK_S):
                                    self._dom_kd_ladder_unlocked[dom_key] = True
                            else:
                                self._dom_pos_grad_secs[dom_key] = 0.0

                group_is_coop = [False] * n_groups

                # Per-chromosome equilibrium residuals. For each oriC-low group
                # the target occupancy is computed from the stepped Adair
                # ladder when ADAIR_KD=1 (primary path used by the milestone
                # config), or from the Langmuir linear-K_d fallback otherwise.
                def _residuals(x, A_T=A_T, D_T=D_T, N_h=N_h, K_h=K_h,
                               n_groups=n_groups, group_n_sites=group_n_sites,
                               group_relax=group_relax,
                               group_is_coop=group_is_coop,
                               cell_volume_L=cell_volume_L,
                               n_avogadro=self.n_avogadro):
                    A_f = max(x[0], 0.0)
                    D_f = max(x[1], 0.0)
                    if K_h > 0:
                        a = A_f / K_h
                        d = D_f / K_h
                        denom = 1.0 + a + d
                        A_h = N_h * a / denom
                        D_h = N_h * d / denom
                    else:
                        A_h = D_h = 0.0
                    sum_A_l = 0.0
                    group_res = [0.0] * n_groups
                    for i in range(n_groups):
                        n_b_g = max(x[2 + i], 0.0)
                        n_s_g = group_n_sites[i]
                        if ADAIR_KD:
                            # Adair stepwise binding: K_d,i = K_d_max × (K_d_min/K_d_max)^((i-1)/(N-1))
                            # Partition function: Z = Σ s_i, s_i = x_1 × x_2 × ... × x_i, x_i = A/K_d,i
                            # <n> = Σ i × s_i / Z. Smooth sigmoidal, no bistability.
                            # When COOP_GRADIENT_GATE is on and bulk gradient is not
                            # rising, collapse to independent-site binding at K_d_max
                            # (cooperativity requires DnaA-ATP to be arriving).
                            A_f_M = A_f / (cell_volume_L * n_avogadro)
                            N_sites = int(n_s_g)
                            if N_sites > 0 and A_f_M > 0:
                                # Two gates that can collapse Adair to Langmuir:
                                # 1. GRADIENT gate: bulk isn't rising → no coop
                                # 2. UNLOCK gate: fresh daughter domain hasn't
                                #    accumulated POST_INIT_UNLOCK_S of continuous
                                #    positive gradient yet → K_d clamped at K_d_max
                                _dom_key_a = int(unique_doms[i])
                                _kd_unlocked = (POST_INIT_UNLOCK_S <= 0
                                    or self._dom_kd_ladder_unlocked.get(
                                        _dom_key_a, False))
                                if (COOP_GRADIENT_GATE and not gradient_rising) or not _kd_unlocked:
                                    # Non-cooperative: independent sites at K_d_max
                                    P_bound = A_f_M / (ADAIR_KD_MAX_M + A_f_M)
                                    target_g = n_s_g * P_bound
                                else:
                                    cumprod = 1.0
                                    Z = 1.0
                                    n_avg_num = 0.0
                                    if ADAIR_KDS_M is not None:
                                        # Explicit per-site K_d list (stepped
                                        # or arbitrary Adair). Clamp to N_sites.
                                        kd_list = ADAIR_KDS_M[:N_sites] if len(ADAIR_KDS_M) >= N_sites \
                                            else tuple(ADAIR_KDS_M) + (ADAIR_KD_MIN_M,) * (N_sites - len(ADAIR_KDS_M))
                                        for k in range(1, N_sites + 1):
                                            x_k = A_f_M / kd_list[k - 1]
                                            cumprod *= x_k
                                            Z += cumprod
                                            n_avg_num += k * cumprod
                                    else:
                                        # Geometric interpolation MAX → MIN.
                                        ratio = (ADAIR_KD_MIN_M / ADAIR_KD_MAX_M) ** (1.0 / max(N_sites - 1, 1))
                                        K_di = ADAIR_KD_MAX_M
                                        for k in range(1, N_sites + 1):
                                            x_k = A_f_M / K_di
                                            cumprod *= x_k
                                            Z += cumprod
                                            n_avg_num += k * cumprod
                                            K_di *= ratio
                                    target_g = n_avg_num / Z if Z > 0 else 0.0
                            else:
                                target_g = 0.0
                        elif group_is_coop[i]:
                            # Committed domain — K_d locked at floor (hysteresis).
                            kd_g_M = KD_LOW_MIN_M
                            kd_g_mol = kd_g_M * cell_volume_L * n_avogadro
                            denom_g = kd_g_mol + A_f
                            target_g = (n_s_g * A_f / denom_g) if denom_g > 0 else 0.0
                        else:
                            # Symmetric (single K_d for all sites in cluster).
                            kd_g_M = _kd_low_cooperative(
                                n_b_g, n_s_g, group_relax[i],
                                coop_engaged=gradient_rising)
                            kd_g_mol = kd_g_M * cell_volume_L * n_avogadro
                            denom_g = kd_g_mol + A_f
                            target_g = (n_s_g * A_f / denom_g) if denom_g > 0 else 0.0
                        group_res[i] = n_b_g - target_g
                        sum_A_l += n_b_g
                    return [A_T - A_f - A_h - sum_A_l, D_T - D_f - D_h, *group_res]

                # Warm-start initial guess. Cooperative K_d has intrinsic
                # bistability (low-n basin + high-n basin); using the previous
                # tick's converged solution as x0 keeps the system in whichever
                # basin it landed in, instead of letting scipy.root choose
                # arbitrarily. Without warm-start the solver flickers between
                # basins tick-to-tick (verified in gen-1 tick-compare analysis).
                # Warm-start floors removed. The old max(prev, 1e-3) guard was
                # a numerical divide-by-zero protection, but at near-empty
                # bulk-ATP states it re-seeded A_free ≈ 1e-3 molecules ≈ 2 nM
                # per tick. That is above K_d,8 = 2 nM for the stepped Adair,
                # so the solver could climb into the high-n basin from x0 even
                # when the true equilibrium is low-n. Result: tick-to-tick
                # basin-hop flicker. Warm-start now preserves the previous
                # tick's basin exactly.
                if self._prev_A_free is not None and self._prev_D_free is not None:
                    A0 = float(self._prev_A_free)
                    D0 = float(self._prev_D_free)
                else:
                    # Fresh sim / post-resume — use the current bulk pool as
                    # the A_free / D_free guess. At near-empty bulk states
                    # (dill loaded with most DnaA bound), atp_bulk_count is
                    # tiny → x0 near true equilibrium. The old A_T/2 default
                    # placed A_free 100× above equilibrium, causing the solver
                    # to over-saturate the cluster on the very first tick.
                    A0 = float(atp_bulk_count)
                    D0 = float(adp_bulk_count)
                x0 = [A0, D0]
                for i in range(n_groups):
                    dom_key = int(unique_doms[i])
                    prev_n = self._prev_n_bound_by_dom.get(dom_key)
                    if prev_n is not None:
                        x0.append(float(prev_n))
                    elif dom_key in n_bound_prev_by_dom:
                        # Fresh sim / post-resume — use the current DNA-bound
                        # state as x0. This matches the dill's actual bound
                        # count so the solver equilibrates from the loaded
                        # state rather than from an arbitrary A_T/2 cold start
                        # (which would place A_free above K_d,8 and let the
                        # cluster saturate immediately on tick 1).
                        x0.append(float(n_bound_prev_by_dom[dom_key]))
                    else:
                        # New domain (post-fork) — start at low-n basin guess.
                        x0.append(float(group_n_sites[i]) * 0.1)
                sol = scipy_root(_residuals, x0, method="hybr", tol=1e-9)
                # DEBUG: log first ~5 solver calls to inspect what's happening
                if os.environ.get("V2ECOLI_DNAA_SOLVER_DEBUG", "0") == "1":
                    if not hasattr(self, "_debug_call_count"):
                        self._debug_call_count = 0
                    if self._debug_call_count < 5:
                        residuals_at_sol = _residuals(list(sol.x))
                        print(f"[SOLVER_DBG call {self._debug_call_count}] "
                              f"A_T={A_T:.3f} D_T={D_T:.3f} "
                              f"cell_vol_L={cell_volume_L:.3e}",
                              flush=True)
                        print(f"  x0=[A0={x0[0]:.4f}, D0={x0[1]:.4f}, "
                              f"x_cluster={x0[2]:.4f}]", flush=True)
                        print(f"  sol.x=[A_f={sol.x[0]:.4f}, "
                              f"D_f={sol.x[1]:.4f}, x_cluster={sol.x[2]:.4f}]",
                              flush=True)
                        print(f"  sol.success={sol.success} "
                              f"residuals={[f'{r:.3e}' for r in residuals_at_sol]}",
                              flush=True)
                        print(f"  atp_bulk_count={atp_bulk_count} "
                              f"n_bound_prev_by_dom={n_bound_prev_by_dom}",
                              flush=True)
                        self._debug_call_count += 1
                if not sol.success:
                    # Fallback: retry from the current bulk pool + actual
                    # DNA-bound state. Same physical x0 as the primary path,
                    # nudged by 10% toward the low basin for cluster targets
                    # to give the solver a slightly different starting point.
                    # The old A_T/2 fallback re-introduced the pathological
                    # 100× overshoot on A_free.
                    x0_fb = [float(atp_bulk_count), float(adp_bulk_count)]
                    for i in range(n_groups):
                        dom_key = int(unique_doms[i])
                        n0 = float(n_bound_prev_by_dom.get(
                            dom_key, group_n_sites[i] * 0.1))
                        x0_fb.append(n0 * 0.9)  # nudge slightly toward low basin
                    sol = scipy_root(_residuals, x0_fb, method="hybr", tol=1e-9)
                    if os.environ.get("V2ECOLI_DNAA_SOLVER_DEBUG", "0") == "1":
                        print(f"[SOLVER_DBG] FALLBACK TRIGGERED "
                              f"sol.success={sol.success} "
                              f"sol.x[cluster]={sol.x[2]:.4f}", flush=True)
                A_free = max(float(sol.x[0]), 0.0)
                D_free = max(float(sol.x[1]), 0.0)
                group_bound_atp = np.maximum(
                    np.asarray(sol.x[2:2 + n_groups], dtype=np.float64), 0.0)

                if STOCHASTIC_ORIC_LOW:
                    # Override the solver's oric_low result with per-site
                    # sequential Bernoulli sampling with within-tick chain
                    # reaction. Bound sites can stay bound or unbind (K_d_min);
                    # empty sites are sampled in sequence, and each site's
                    # binding decreases the K_d seen by SUBSEQUENT sites this
                    # same tick (nearest-neighbor cooperativity propagation).
                    #
                    # Effect: a fresh cluster that starts to nucleate can
                    # cascade within a tick IF bulk_ATP is high enough; if it
                    # can't get past the first few bindings (low-P region at
                    # K_d~100 nM), it stops. Prevents deterministic snap-fill
                    # but allows cascade-to-full when conditions favor it.
                    A_free_M = A_free / (cell_volume_L * self.n_avogadro)
                    for i in range(n_groups):
                        dom_key = int(unique_doms[i])
                        dom_n_sites = int(group_n_sites[i])
                        n_b_prev = self._prev_n_bound_by_dom.get(dom_key, 0.0)
                        n_bound_initial = int(round(n_b_prev))
                        # 1) Sample the already-bound sites (K_d = K_d_min).
                        n_bound_running = 0
                        for _ in range(n_bound_initial):
                            kd_M = KD_LOW_MIN_M
                            P_bound = A_free_M / (A_free_M + kd_M) \
                                if (A_free_M + kd_M) > 0 else 0.0
                            if self.random_state.random_sample() < P_bound:
                                n_bound_running += 1
                        # 2) Sample the empty sites in sequence. Each site's
                        # K_d is computed from the CURRENT running n_bound so
                        # a within-tick chain reaction can propagate.
                        n_empty = max(0, dom_n_sites - n_bound_initial)
                        for _ in range(n_empty):
                            kd_M = _kd_low_cooperative(
                                float(n_bound_running), float(dom_n_sites),
                                1.0, coop_engaged=gradient_rising)
                            P_bound = A_free_M / (A_free_M + kd_M) \
                                if (A_free_M + kd_M) > 0 else 0.0
                            if self.random_state.random_sample() < P_bound:
                                n_bound_running += 1
                        group_bound_atp[i] = float(n_bound_running)
                    # Restore A_T conservation: adjust A_free for the change
                    # in oric_low bound counts vs what the solver assumed.
                    solver_oric_low = float(np.asarray(
                        sol.x[2:2 + n_groups], dtype=np.float64).sum())
                    stochastic_oric_low = float(group_bound_atp.sum())
                    A_free = max(A_free + (solver_oric_low - stochastic_oric_low), 0.0)

                if KINETIC_ORIC_LOW:
                    # True kinetic ODE: dN/dt = k_on(N)·[A_f]·(N_max−N) − k_off·N
                    # where k_on(N) = k_off / K_d(N) — cooperativity affects
                    # both the destination AND the rate. At low N, K_d is high
                    # → k_on small → slow nucleation. Past K_half, K_d drops →
                    # k_on rises → binding accelerates through the transition.
                    #
                    # For each cluster, freeze K_d at K_d(n_prev) over the
                    # timestep (small-dt approximation) and integrate exactly:
                    #   n_new = n_eq_local + (n_prev − n_eq_local)·exp(−c·dt)
                    # where c = k_on·[A_f] + k_off, n_eq_local = N_max·A_f/(A_f+K_d)
                    dt_s = float(states["timestep"])
                    A_free_M = A_free / (cell_volume_L * self.n_avogadro)
                    kinetic_bound = np.zeros(n_groups, dtype=np.float64)
                    for i in range(n_groups):
                        dom_key = int(unique_doms[i])
                        n_prev = float(self._prev_n_bound_by_dom.get(
                            dom_key, 0.0))
                        cluster_n_sites = float(group_n_sites[i])
                        kd_M = _kd_low_cooperative(
                            n_prev, cluster_n_sites, group_relax[i],
                            coop_engaged=gradient_rising)
                        # Rate constants at current n_prev
                        k_on = KINETIC_KOFF_PER_S / kd_M if kd_M > 0 else 0.0
                        a = k_on * A_free_M  # forward rate (1/s)
                        b = KINETIC_KOFF_PER_S  # reverse rate (1/s)
                        c = a + b  # total relaxation rate (1/s)
                        n_eq_local = (cluster_n_sites * A_free_M
                                      / (A_free_M + kd_M)
                                      if (A_free_M + kd_M) > 0 else 0.0)
                        alpha_local = 1.0 - np.exp(-dt_s * c) if c > 0 else 0.0
                        kinetic_bound[i] = (n_prev
                                            + alpha_local * (n_eq_local - n_prev))
                    group_bound_atp = np.clip(kinetic_bound, 0.0,
                                              np.asarray(group_n_sites,
                                                         dtype=np.float64))
                    # Restore A_T conservation: unbound DnaA-ATP (from
                    # kinetic under-fill vs equilibrium) stays in bulk.
                    solver_targets = np.asarray(
                        sol.x[2:2 + n_groups], dtype=np.float64)
                    solver_targets = np.maximum(solver_targets, 0.0)
                    solver_oric_low_total = float(solver_targets.sum())
                    kinetic_oric_low_total = float(group_bound_atp.sum())
                    A_free = max(A_free + (solver_oric_low_total
                                           - kinetic_oric_low_total), 0.0)

                # Cache for next tick's warm-start.
                self._prev_A_free = A_free
                self._prev_D_free = D_free
                for i in range(n_groups):
                    self._prev_n_bound_by_dom[int(unique_doms[i])] = float(
                        group_bound_atp[i])
            else:
                # Legacy non-cooperative pooled solve.
                def _residuals(x, A_T=A_T, D_T=D_T, N_h=N_h, N_l=N_l_legacy,
                               K_h=K_h, K_l=K_l_legacy):
                    A_f = max(x[0], 0.0)
                    D_f = max(x[1], 0.0)
                    if K_h > 0:
                        a = A_f / K_h
                        d = D_f / K_h
                        denom = 1.0 + a + d
                        A_h = N_h * a / denom
                        D_h = N_h * d / denom
                    else:
                        A_h = D_h = 0.0
                    if K_l > 0 and N_l > 0:
                        A_l = N_l * A_f / (K_l + A_f)
                    else:
                        A_l = 0.0
                    return [A_T - A_f - A_h - A_l, D_T - D_f - D_h]

                x0 = [max(A_T / 2.0, 1e-3), max(D_T / 2.0, 1e-3)]
                sol = scipy_root(_residuals, x0, method="hybr", tol=1e-9)
                if not sol.success:
                    sol = scipy_root(_residuals, [A_T, D_T], method="hybr", tol=1e-9)
                A_free = max(float(sol.x[0]), 0.0)
                D_free = max(float(sol.x[1]), 0.0)
                # Legacy distributed-as-pool target; recomputed below.
                group_bound_atp = np.array([], dtype=np.float64)
        else:
            A_free = A_T
            D_free = D_T
            group_bound_atp = np.array([], dtype=np.float64)

        if K_h > 0 and N_h > 0:
            a_h = A_free / K_h
            d_h = D_free / K_h
            denom_h = 1.0 + a_h + d_h
            high_bound_atp = N_h * a_h / denom_h
            high_bound_adp = N_h * d_h / denom_h
        else:
            high_bound_atp = high_bound_adp = 0.0
        # Low-aff total bound count: sum over per-chromosome subpools when
        # cooperative; closed-form Langmuir when legacy.
        if COOPERATIVE_ORIC_LOW and n_groups > 0:
            low_bound_atp = float(group_bound_atp.sum())
        elif K_l_legacy > 0 and N_l_legacy > 0:
            low_bound_atp = N_l_legacy * A_free / (K_l_legacy + A_free)
        else:
            low_bound_atp = 0.0

        # Re-solve once more with converged free, distribute bound across
        # the three high-aff sub-pools (chromosomal / oriC_high / promoter)
        # proportional to their site counts (all share the same Kd, so the
        # binding probability is identical → counts split by N_i / N_total).
        if n_high_total > 0:
            frac_chrom = n_sites[POOL_CHROMOSOMAL_HIGH] / n_high_total
            frac_orichi = n_sites[POOL_ORIC_HIGH] / n_high_total
            frac_prom = n_sites[POOL_PROMOTER_HIGH] / n_high_total
        else:
            frac_chrom = frac_orichi = frac_prom = 0.0

        per_pool_targets = {
            POOL_CHROMOSOMAL_HIGH: {
                FORM_BOUND_ATP: high_bound_atp * frac_chrom,
                FORM_BOUND_ADP: high_bound_adp * frac_chrom,
            },
            POOL_ORIC_HIGH: {
                FORM_BOUND_ATP: high_bound_atp * frac_orichi,
                FORM_BOUND_ADP: high_bound_adp * frac_orichi,
            },
            POOL_PROMOTER_HIGH: {
                FORM_BOUND_ATP: high_bound_atp * frac_prom,
                FORM_BOUND_ADP: high_bound_adp * frac_prom,
            },
            POOL_ORIC_LOW: {
                FORM_BOUND_ATP: low_bound_atp,
                FORM_BOUND_ADP: 0.0,
            },
        }

        # Stochastic round each per-pool target to integers, and assign
        # boxes within each pool. New DnaA_bound_form for every active row.
        new_bound_form = np.zeros(n_active, dtype=np.int8)

        for pool_id, mask in pool_masks.items():
            if pool_id == POOL_ORIC_LOW and COOPERATIVE_ORIC_LOW and n_groups > 0:
                # Per-chromosome distribution: each oriC's 8 sites get their
                # own stochastic-rounded target from the cooperative solve.
                # Guard: solver may have been skipped (e.g., zero-DnaA cold
                # start), leaving group_bound_atp empty. Treat as target=0.
                if group_bound_atp.size < n_groups:
                    continue
                for g_idx, dom in enumerate(unique_doms):
                    group_mask = mask & (domain_index == dom)
                    pool_idx = np.nonzero(group_mask)[0]
                    n_pool = pool_idx.size
                    if n_pool == 0:
                        continue
                    target_atp = self._stochastic_round(
                        float(group_bound_atp[g_idx]))
                    target_atp = min(target_atp, n_pool)
                    target_atp = max(0, target_atp)
                    if target_atp > 0:
                        chosen = self.random_state.choice(
                            pool_idx, size=target_atp, replace=False)
                        new_bound_form[chosen] = FORM_BOUND_ATP
                continue
            pool_idx = np.nonzero(mask)[0]
            n_pool = pool_idx.size
            if n_pool == 0:
                continue
            target_atp = self._stochastic_round(
                per_pool_targets[pool_id][FORM_BOUND_ATP])
            target_adp = self._stochastic_round(
                per_pool_targets[pool_id][FORM_BOUND_ADP])
            # Defensive clamps.
            if target_atp + target_adp > n_pool:
                # Trim the larger one first.
                overflow = target_atp + target_adp - n_pool
                if target_atp >= target_adp:
                    target_atp -= overflow
                else:
                    target_adp -= overflow
                target_atp = max(0, target_atp)
                target_adp = max(0, target_adp)
            # Random selection within the pool.
            if target_atp + target_adp > 0:
                chosen = self.random_state.choice(
                    pool_idx, size=target_atp + target_adp, replace=False)
                new_bound_form[chosen[:target_atp]] = FORM_BOUND_ATP
                new_bound_form[chosen[target_atp:]] = FORM_BOUND_ADP

        # Per-pool actual bound counts (post stochastic rounding).
        actual_bound_atp = int(np.count_nonzero(new_bound_form == FORM_BOUND_ATP))
        actual_bound_adp = int(np.count_nonzero(new_bound_form == FORM_BOUND_ADP))

        # Cap by the total ATP/ADP available — if rounding overshot, drop
        # back to free (DnaA stays apo for one tick rather than going
        # negative on the bulk pool).
        if actual_bound_atp > total_atp:
            drop = actual_bound_atp - total_atp
            atp_rows = np.nonzero(new_bound_form == FORM_BOUND_ATP)[0]
            drop_rows = self.random_state.choice(
                atp_rows, size=drop, replace=False)
            new_bound_form[drop_rows] = FORM_FREE
            actual_bound_atp -= drop
        if actual_bound_adp > total_adp:
            drop = actual_bound_adp - total_adp
            adp_rows = np.nonzero(new_bound_form == FORM_BOUND_ADP)[0]
            drop_rows = self.random_state.choice(
                adp_rows, size=drop, replace=False)
            new_bound_form[drop_rows] = FORM_FREE
            actual_bound_adp -= drop

        # New DnaA_bound boolean.
        new_bound = (new_bound_form != FORM_FREE)

        # Net bulk deltas. Account for the bound-pool hydrolysis flux: bound
        # ATP was reduced by delta_h_bound (which became bound ADP), so the
        # ATP pool released into bulk is (prev_bound_atp - delta_h_bound),
        # and the ADP pool released into bulk is (prev_bound_adp + delta_h_bound).
        delta_atp_bulk = (prev_bound_atp - delta_h_bound) - actual_bound_atp
        delta_adp_bulk = (prev_bound_adp + delta_h_bound) - actual_bound_adp

        bulk_update = []
        if delta_atp_bulk != 0:
            bulk_update.append((self._atp_idx, int(delta_atp_bulk)))
        if delta_adp_bulk != 0:
            bulk_update.append((self._adp_idx, int(delta_adp_bulk)))
        # dnaa-3 Phase 2b: Pi / PROTON / WATER for bound-pool hydrolysis are
        # NOW produced by the equilibrium step (bf8b82e). Writing them here
        # was bypassing FBA accounting → metabolism over-allocated biomass
        # synthesis → cells ballooned 2-3× and stopped dividing. The in-place
        # ATP→ADP bound-form swap (delta_h_bound rows above) is still owned
        # here — only the byproducts moved.

        # massDiff_* per-row updates so cell-mass accounting reflects the
        # DnaA moved bulk → bound on each box. Same pattern as tf_binding for
        # TFs binding promoters (tf_binding.py:362-369). Now that bf8b82e
        # owns the Pi/PROTON/WATER byproduct accounting (Phase 2b), enabling
        # massDiff here doesn't conflict with FBA's mass balance.
        box_set = {
            "DnaA_bound": new_bound,
            "DnaA_bound_form": new_bound_form,
        }
        if self._mass_lookup is not None and self.submass_indices:
            old_mass = self._mass_lookup[bound_form]      # (n_active, n_submass)
            new_mass = self._mass_lookup[new_bound_form]  # (n_active, n_submass)
            mass_delta = new_mass - old_mass
            for submass_field, idx in self.submass_indices.items():
                current = attrs(boxes, [submass_field])[0]
                box_set[submass_field] = current + mass_delta[:, idx]

        # dnaa-4 autoregulation: publish dnaA-promoter occupancy fraction so
        # transcript_initiation.py can repress the dnaA TU when DnaA has bound
        # the promoter sites (negative feedback). f = bound / total over the
        # POOL_PROMOTER_HIGH boxes in this cell (2 sites per chromosome; 2-4
        # total over the cell cycle as the fork passes the promoter region).
        prom_mask = pool_masks[POOL_PROMOTER_HIGH]
        n_prom_total = int(prom_mask.sum())
        if n_prom_total > 0:
            n_prom_bound = int((new_bound_form[prom_mask] != FORM_FREE).sum())
            promoter_fraction = float(n_prom_bound) / float(n_prom_total)
        else:
            promoter_fraction = 0.0

        update = {
            "DnaA_boxes": {"set": box_set},
            "next_update_time": states["global_time"] + states["timestep"],
            # dnaa-3 Phase 2c: publish the bound-pool hydrolysis count so
            # equilibrium.py can route the byproducts (Pi/PROTON/WATER) for the
            # SAME hydrolysis events rather than independently re-sampling.
            "dnaa_hydrolysis": {
                "bound_count": int(delta_h_bound),
                "promoter_fraction": promoter_fraction,
            },
        }
        if bulk_update:
            update["bulk"] = bulk_update
        return update
