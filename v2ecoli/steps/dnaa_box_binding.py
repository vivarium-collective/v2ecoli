"""
=========================
DnaA-box equilibrium binding
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
equations for (A_free, D_free) with scipy.optimize.root.

For the ATP-only pool (2), oriC_low occupancy is set by a per-domain stepped
Adair ladder (K_d,1 ≥ K_d,2 ≥ … ≥ K_d,N via V2ECOLI_DNAA_ADAIR_KDS_NM), coupled
to the same (A_free, D_free) solve.

Hydrolysis (DnaA-ATP -> DnaA-ADP; k = 0.046 / min per Sekimizu 1987) is split
between the equilibrium step (owns the free pool + Pi/PROTON/WATER byproduct
accounting via DNAA-INTRINSIC-HYDROLYSIS-RXN) and this step (owns the in-place
bound ATP → bound ADP form swap on DnaA_box rows). The shared bound-hydrolysis
count is published on the process_state channel `dnaa_hydrolysis.bound_count`
so the equilibrium step can route the byproducts for the SAME Poisson events
instead of independently re-sampling.

Outputs:
    bulk[DnaA-ATP]   updated by the net change in bound-ATP across pools
    bulk[DnaA-ADP]   updated by the net change in bound-ADP across pools
    DnaA_box.DnaA_bound       boolean per active row
    DnaA_box.DnaA_bound_form  int8 per active row (0 / 1 / 2)
    dnaa_hydrolysis.bound_count        int, consumed by equilibrium for
                                       byproduct routing
    dnaa_hydrolysis.promoter_fraction  float in [0,1], consumed by
                                       transcript_initiation for dnaA
                                       autoregulation
"""

from __future__ import annotations

import os
import numpy as np
from scipy.optimize import root as scipy_root

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import attrs, bulk_name_to_idx, counts
from v2ecoli.library.schema_types import DNAA_BOX_ARRAY, ACTIVE_REPLISOME_ARRAY
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
    # Shared bound-hydrolysis count consumed by the equilibrium step for
    # byproduct routing — avoids independently sampling the same Poisson process.
    "dnaa_hydrolysis": ("process_state", "dnaa_hydrolysis"),
    # For RIDA: read active replisome coordinates → count fork pairs.
    "active_replisomes": ("unique", "active_replisome"),
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
# RIDA (Regulatory Inactivation of DnaA) is a per-fork-pair mechanism.
# Catalyzes DnaA-ATP + WATER → DnaA-ADP + Pi + PROTON. Chemically identical
# to the intrinsic hydrolysis reaction (H+ balance matters for the metabolic
# proton pool); we use the standard ATP-hydrolysis stoichiometry even though
# EcoCyc's RXN0-7444 entry happens to omit PROTON.

# Pool labels (must match initial_conditions.py).
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
KD_LOW_M = 100e-9   # 100 nM — oriC_low (ATP only), legacy non-cooperative K_d.
                    # Consumed only by the K_l_legacy path (fallback when
                    # n_groups == 0); Adair ladder ignores it.

COOPERATIVE_ORIC_LOW = True

# Positive-gradient gate on the Adair cooperativity path. Enables the
# gradient_rising computation over a GRADIENT_WINDOW_S rolling window of
# bulk DnaA-ATP concentrations. Feeds COOP_GRADIENT_GATE and the
# POST_INIT_UNLOCK_S SeqA-proxy unlock logic.
GRADIENT_GATE = os.environ.get(
    "V2ECOLI_DNAA_GRADIENT_GATE", "0") in ("1", "true", "True")
GRADIENT_WINDOW_S = float(os.environ.get(
    "V2ECOLI_DNAA_GRADIENT_WINDOW_S", "60.0"))
# Minimum positive slope (nM / s) to count as rising; smaller positive slopes
# are treated as flat.
GRADIENT_MIN_SLOPE_NM_PER_S = float(os.environ.get(
    "V2ECOLI_DNAA_GRADIENT_MIN_SLOPE_NM_PER_S", "0.0"))

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

# Gradient-gated cooperativity: when enabled, the Adair ladder only engages
# while bulk DnaA-ATP is actively accumulating (positive gradient). When the
# gradient is not positive, the cluster is pinned to the K_d_max floor for
# all sites — cooperative loading only proceeds while DnaA-ATP is arriving.
COOP_GRADIENT_GATE = os.environ.get(
    "V2ECOLI_DNAA_COOP_GRADIENT_GATE", "0") in ("1", "true", "True")


# DnaA-ATP intrinsic hydrolysis rate. Bound DnaA-ATP hydrolyzes at the same
# rate as free DnaA-ATP (Sekimizu 1987, k = 0.046 / min). The equilibrium step
# hydrolyzes the FREE pool only; this step additionally hydrolyzes the BOUND
# pool, in-place (bound box-ATP → bound box-ADP on the same row).
HYDROLYSIS_RATE_PER_MIN = float(os.environ.get(
    "V2ECOLI_DNAA_HYDROLYSIS_RATE_PER_MIN", "0.025"))

# RIDA (Regulatory Inactivation of DnaA) — Hda-β-clamp complex catalyses
# DnaA-ATP + WATER → DnaA-ADP + Pi (RXN0-7444) on the free/bulk pool.
# Rate is hardcoded per Haochen's spec (Katayama 2017 / Riber 2016): 40 events
# per minute per replication fork pair. Gated on active replisome count so it
# only fires during S-phase. Pi/WATER byproducts are written directly to bulk
# here (bypasses FBA — small stoichiometric drift). Toggle whole mechanism
# with V2ECOLI_DNAA_RIDA_ENABLED=1 (default 0 = off, backward compatible).
RIDA_RATE_PER_MIN_PER_FORK_PAIR = 40.0
RIDA_ENABLED = os.environ.get(
    "V2ECOLI_DNAA_RIDA_ENABLED", "0") in ("1", "true", "True")

# Chromosomal DnaA-box blocking perturbation.
# Randomly select CHROM_BLOCK_FRAC of the chromosomal high-affinity DnaA boxes
# (pool_label = POOL_CHROMOSOMAL_HIGH, ~302 per chromosome) and remove them
# from the binding pool. The blocked set is picked ONCE per sim using the
# process seed, then applied stably (by genomic coordinate) for the whole run.
# oriC and promoter boxes are never blocked. Toggle via
# V2ECOLI_DNAA_CHROM_BLOCK_FRAC (default 0.0 = no blocking).
CHROM_BLOCK_FRAC = float(os.environ.get(
    "V2ECOLI_DNAA_CHROM_BLOCK_FRAC", "0.0"))
# When set, the block RNG is seeded from this value in every process (i.e.,
# every generation of a lineage) so the same 302×frac chromosomal coordinates
# are blocked for the whole run. When unset, falls back to the per-process
# seed (which changes across generations because daughters re-seed).
_CHROM_BLOCK_SEED_ENV = os.environ.get("V2ECOLI_DNAA_CHROM_BLOCK_SEED")
CHROM_BLOCK_SEED = int(_CHROM_BLOCK_SEED_ENV) if _CHROM_BLOCK_SEED_ENV else None

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
        "DnaA-box equilibrium binding.\n"
        "  High-affinity pools (chromosomal/oriC_high/promoter):\n"
        "    p_atp = (A/Kd_A) / (1 + A/Kd_A + D/Kd_D)   competitive\n"
        "    p_adp = (D/Kd_D) / (1 + A/Kd_A + D/Kd_D)   competitive\n"
        "  oriC_low (ATP-only, per-domain): stepped Adair ladder,\n"
        "    K_d,1 ≥ K_d,2 ≥ … ≥ K_d,N (from V2ECOLI_DNAA_ADAIR_KDS_NM).\n"
        "  Free (A, D) solved by scipy.optimize.root; bound-form counts"
        " rounded stochastically."
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

        # Positive-gradient gate state: rolling window of recent bulk DnaA-ATP
        # observations as (time_s, nM) tuples used to compute gradient_rising.
        self._bulk_atp_history = []

        # Chromosomal DnaA-box block set (populated on first update from the
        # first tick's unique chromosomal coords). np.int64 array of blocked
        # genomic coordinates, or None if not yet populated.
        self._blocked_chrom_coords = None

        # Solver warm-start state. Warm-starting scipy.root from the previous
        # tick's converged (A_free, D_free, per-domain bound counts) speeds
        # convergence and, more importantly, resists tick-by-tick flicker in
        # the transition regime. Cache is pruned on fork passage (domain key
        # disappears), so daughter oriCs start from the cold basin.
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
            'listeners': {
                'replication_data': {
                    'gradient_rising': 'overwrite[boolean]',
                    'bulk_atp_slope_nM_per_s': 'overwrite[float]',
                    'bulk_atp_nM_current': 'overwrite[float]',
                    'rida_events': 'overwrite[integer]',
                },
            },
        }

    def update_condition(self, timestep, states):
        if states["next_update_time"] <= states["global_time"]:
            return True
        return False

    def _smoothed_gradient_rising(self, t_now_s: float) -> tuple[bool, float]:
        """Return (rising, slope_nM_per_s) using half-window averages.

        Smoothing rationale: the sim runs with 1s ticks; hydrolysis + equilibrium
        release ~10-20 nM into bulk each tick and the box-binding step consumes
        it in the same tick. The instantaneous newest sample oscillates fast
        enough that comparing raw endpoints of the 120s window would alternate
        rising/falling every couple of ticks. Averaging over 60s halves collapses
        that intra-tick noise so the gate reflects the multi-minute trend.
        """
        hist = self._bulk_atp_history
        if len(hist) < 2:
            return True, 0.0
        t_split = t_now_s - GRADIENT_WINDOW_S / 2.0
        older = [nM for t, nM in hist if t <= t_split]
        newer = [nM for t, nM in hist if t > t_split]
        if not older or not newer:
            # Window hasn't filled a full half yet — fall back to raw endpoints.
            slope = (hist[-1][1] - hist[0][1]) / max(
                hist[-1][0] - hist[0][0], 1e-6)
            return slope > GRADIENT_MIN_SLOPE_NM_PER_S, slope
        older_mean = sum(older) / len(older)
        newer_mean = sum(newer) / len(newer)
        # Midpoint of newer half ≈ t_now - WINDOW/4; midpoint of older ≈
        # t_now - 3·WINDOW/4. Delta = WINDOW/2.
        slope = (newer_mean - older_mean) / (GRADIENT_WINDOW_S / 2.0)
        return slope > GRADIENT_MIN_SLOPE_NM_PER_S, slope

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

        # Chromosomal DnaA-box blocking perturbation. On first call, pick a
        # stable random subset of chromosomal box coordinates to block using
        # the process seed. Same coordinates are excluded from the binding
        # pool every subsequent tick (persistent by genomic coordinate).
        if CHROM_BLOCK_FRAC > 0.0:
            (coords_this_tick,) = attrs(boxes, ["coordinates"])
            if self._blocked_chrom_coords is None:
                chrom_mask_init = pool_label == POOL_CHROMOSOMAL_HIGH
                unique_chrom_coords = np.unique(coords_this_tick[chrom_mask_init])
                # Use fixed CHROM_BLOCK_SEED env if set (stable across all
                # generations of a lineage); otherwise fall back to per-process
                # seed (changes per generation).
                block_seed = (CHROM_BLOCK_SEED if CHROM_BLOCK_SEED is not None
                              else int(self.parameters.get("seed", 0)))
                rng = np.random.RandomState(block_seed)
                n_total = len(unique_chrom_coords)
                n_block = int(round(n_total * CHROM_BLOCK_FRAC))
                if n_block > 0 and n_total > 0:
                    blocked = rng.choice(unique_chrom_coords, size=n_block, replace=False)
                    self._blocked_chrom_coords = np.sort(blocked.astype(np.int64))
                else:
                    self._blocked_chrom_coords = np.array([], dtype=np.int64)
                print(f"[CHROM_BLOCK] frac={CHROM_BLOCK_FRAC}  "
                      f"block_seed={block_seed}  "
                      f"n_chrom_boxes={n_total}  n_blocked={len(self._blocked_chrom_coords)}",
                      flush=True)
            # Effectively re-label blocked chromosomal boxes so they leave the
            # binding pool. We do this by flipping pool_label to a sentinel
            # value (-1) that is not in any of the four POOL_* enums, so all
            # downstream pool_masks skip them.
            if len(self._blocked_chrom_coords) > 0:
                blocked_mask = np.isin(coords_this_tick, self._blocked_chrom_coords)
                chrom_and_blocked = (pool_label == POOL_CHROMOSOMAL_HIGH) & blocked_mask
                pool_label = pool_label.copy()
                pool_label[chrom_and_blocked] = -1

        # Snapshot bulk counts.
        atp_bulk_count = int(counts(states["bulk"], self._atp_idx))
        adp_bulk_count = int(counts(states["bulk"], self._adp_idx))

        # Current bound counts (from previous tick's bookkeeping).
        prev_bound_atp = int(np.count_nonzero(bound_form == FORM_BOUND_ATP))
        prev_bound_adp = int(np.count_nonzero(bound_form == FORM_BOUND_ADP))

        # Hydrolyze bound DnaA-ATP at the same per-molecule rate as
        # DNAA-INTRINSIC-HYDROLYSIS-RXN applies to the free pool. The TSV
        # stores the bimolecular rate (1.4E-5 M⁻¹s⁻¹); the pseudo-first-order
        # rate with [WATER]≈55 M is k = 0.046/min. The equilibrium step covers
        # the free pool; this step covers the bound pool.
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
        diag_bulk_atp_nM_current = float(
            atp_bulk_count / (cell_volume_L * self.n_avogadro) * 1e9)
        if GRADIENT_GATE:
            t_now_s = float(states["global_time"])
            self._bulk_atp_history.append((t_now_s, diag_bulk_atp_nM_current))
            cutoff = t_now_s - GRADIENT_WINDOW_S
            while (len(self._bulk_atp_history) > 1
                   and self._bulk_atp_history[0][0] < cutoff):
                self._bulk_atp_history.pop(0)

        # Diagnostic: recompute gradient state upfront so it's emitted even on
        # the else-branch paths. Uses the same half-window smoothed calc as
        # the Adair block.
        diag_gradient_rising = True
        diag_slope_nM_per_s = 0.0
        if GRADIENT_GATE and len(self._bulk_atp_history) >= 2:
            diag_gradient_rising, diag_slope_nM_per_s = (
                self._smoothed_gradient_rising(float(states["global_time"])))

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
        # Low-aff pool(s): per-chromosome stepped Adair ladder on oric_low.
        #   Each chromosome has its own 8-site subpool with its own K_d,i
        #   sequence; all subpools compete for the same A_free but track
        #   occupancy independently.
        #
        # Variables for scipy.root:
        #   x[0] = A_free, x[1] = D_free
        #   x[2..2+N_groups-1] = n_bound on each oric_low subpool
        #
        # Residuals (mass balance):
        #   r0 = A_T - A_free - N_h*a/(1+a+d) - Σ x[2+g]
        #   r1 = D_T - D_free - N_h*d/(1+a+d)
        #   r[2+g] = x[2+g] - <n>_Adair(A_free; K_d,1..K_d,N)
        #
        # where <n>_Adair is the mean occupancy from the Adair partition
        # function Z = Σᵢ (A/K_d,1)(A/K_d,2)…(A/K_d,i).
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

            # Snapshot per-domain state: current domain keys + previous-tick
            # bound counts (for solver warm-start on fresh sims / post-resume).
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
            # Prune warm-start cache for vanished domains (fork release), so
            # daughter domains (different keys) start with a fresh solver x0.
            for stale in list(self._prev_n_bound_by_dom.keys()):
                if stale not in current_doms_set:
                    self._prev_n_bound_by_dom.pop(stale, None)
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

        # Pooled Langmuir K_l / N_l used only when n_groups == 0 (no oriC-low
        # sites present — edge case; per-domain Adair path is the primary).
        K_l_legacy = kd_low_molecules
        N_l_legacy = n_low_total

        if (A_T + D_T) > 0 and (N_h + N_l_legacy) > 0:
            if COOPERATIVE_ORIC_LOW and n_groups > 0:
                # Bulk-ATP gradient state (used by the Adair GRADIENT gate and
                # by the POST_INIT_UNLOCK_S post-init ladder unlock below).
                # gradient_rising = True → bulk DnaA-ATP is accumulating over
                # the current window (or GRADIENT_GATE disabled); False → bulk
                # is flat or falling, so cooperative loading is not permitted.
                gradient_rising = True
                if GRADIENT_GATE and len(self._bulk_atp_history) >= 2:
                    gradient_rising, _ = self._smoothed_gradient_rising(
                        float(states["global_time"]))

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

                # Per-chromosome equilibrium residuals. For each oriC-low group
                # the target occupancy is computed from the stepped Adair
                # ladder (primary path; ADAIR_KD=0 leaves the group empty).
                def _residuals(x, A_T=A_T, D_T=D_T, N_h=N_h, K_h=K_h,
                               n_groups=n_groups, group_n_sites=group_n_sites,
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
                        else:
                            # ADAIR_KD=0: leave the group empty (no fallback).
                            target_g = 0.0
                        group_res[i] = n_b_g - target_g
                        sum_A_l += n_b_g
                    return [A_T - A_f - A_h - sum_A_l, D_T - D_f - D_h, *group_res]

                # Warm-start initial guess from the previous tick's converged
                # (A_free, D_free). Preserves basin selection in the bistable
                # transition regime — without this, scipy.root picks basins
                # arbitrarily per tick and produces flicker. No floors on x0:
                # a small warm-start seed near true equilibrium is safer than
                # a fixed lower bound, which can seed A_free above K_d,N and
                # push the solver into the wrong basin.
                if self._prev_A_free is not None and self._prev_D_free is not None:
                    A0 = float(self._prev_A_free)
                    D0 = float(self._prev_D_free)
                else:
                    # Fresh sim / post-resume — seed from the current bulk
                    # pool so x0 tracks the loaded state.
                    A0 = float(atp_bulk_count)
                    D0 = float(adp_bulk_count)
                x0 = [A0, D0]
                for i in range(n_groups):
                    dom_key = int(unique_doms[i])
                    prev_n = self._prev_n_bound_by_dom.get(dom_key)
                    if prev_n is not None:
                        x0.append(float(prev_n))
                    elif dom_key in n_bound_prev_by_dom:
                        # Fresh sim / post-resume — seed from the actual
                        # bound count so the solver starts near the loaded
                        # equilibrium instead of a cold cluster guess.
                        x0.append(float(n_bound_prev_by_dom[dom_key]))
                    else:
                        # New domain (post-fork) — start at low-n basin guess.
                        x0.append(float(group_n_sites[i]) * 0.1)
                sol = scipy_root(_residuals, x0, method="hybr", tol=1e-9)
                if not sol.success:
                    # Retry from the current bulk pool + actual DNA-bound
                    # state, nudged 10% toward the low basin to give scipy
                    # a slightly different starting point.
                    x0_fb = [float(atp_bulk_count), float(adp_bulk_count)]
                    for i in range(n_groups):
                        dom_key = int(unique_doms[i])
                        n0 = float(n_bound_prev_by_dom.get(
                            dom_key, group_n_sites[i] * 0.1))
                        x0_fb.append(n0 * 0.9)  # nudge slightly toward low basin
                    sol = scipy_root(_residuals, x0_fb, method="hybr", tol=1e-9)
                A_free = max(float(sol.x[0]), 0.0)
                D_free = max(float(sol.x[1]), 0.0)
                group_bound_atp = np.maximum(
                    np.asarray(sol.x[2:2 + n_groups], dtype=np.float64), 0.0)

                # Cache for next tick's warm-start.
                self._prev_A_free = A_free
                self._prev_D_free = D_free
                for i in range(n_groups):
                    self._prev_n_bound_by_dom[int(unique_doms[i])] = float(
                        group_bound_atp[i])
            else:
                # n_groups == 0 fallback: pooled Langmuir on high-aff pool
                # only (no oriC-low sites present — degenerate edge case).
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
                # No oriC-low subpools in this branch.
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
        # Low-aff total bound count: sum over per-chromosome subpools on the
        # primary path; closed-form Langmuir on the n_groups==0 fallback.
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

        # RIDA (Regulatory Inactivation of DnaA): Hda-β-clamp complex catalyzed
        # hydrolysis of free/bulk DnaA-ATP. Rate is hardcoded 40/min per active
        # fork pair (Katayama 2017 / Riber 2016 synthesis of Kurokawa 1999).
        # Fork pairs = len(active_replisomes) // 2. Capped by post-bound-pool
        # free ATP available. Pi produced, WATER consumed — written to bulk
        # directly (bypasses FBA).
        n_rida_events = 0
        n_fork_pairs = 0
        if RIDA_ENABLED:
            (fork_coords,) = attrs(states["active_replisomes"], ["coordinates"])
            n_fork_pairs = len(fork_coords) // 2
            if n_fork_pairs > 0:
                atp_bulk_after_bound_hydr = atp_bulk_count + delta_atp_bulk
                n_rida_events = self._stochastic_round(
                    RIDA_RATE_PER_MIN_PER_FORK_PAIR * n_fork_pairs * dt_min)
                # Cap by available bulk DnaA-ATP (post bound-pool release).
                if n_rida_events > atp_bulk_after_bound_hydr:
                    n_rida_events = max(0, int(atp_bulk_after_bound_hydr))
                if n_rida_events > 0:
                    delta_atp_bulk -= n_rida_events
                    delta_adp_bulk += n_rida_events

        bulk_update = []
        if delta_atp_bulk != 0:
            bulk_update.append((self._atp_idx, int(delta_atp_bulk)))
        if delta_adp_bulk != 0:
            bulk_update.append((self._adp_idx, int(delta_adp_bulk)))
        # Pi / PROTON / WATER for bound-pool hydrolysis are produced by the
        # equilibrium step (via DNAA-INTRINSIC-HYDROLYSIS-RXN in the FBA
        # stoichMatrix). Writing them here would bypass FBA accounting.
        # This step still owns the in-place bound-form ATP → ADP swap
        # (delta_h_bound above) — only the byproducts moved.
        # RIDA Pi/PROTON/WATER byproducts written directly to bulk (bypasses
        # FBA — small drift, revisit if RIDA rate is scaled up substantially).
        # Full ATP-hydrolysis stoichiometry: ATP + WATER → ADP + Pi + PROTON.
        if n_rida_events > 0:
            bulk_update.append((self._pi_idx, int(n_rida_events)))
            bulk_update.append((self._proton_idx, int(n_rida_events)))
            bulk_update.append((self._water_idx, -int(n_rida_events)))

        # massDiff_* per-row updates so cell-mass accounting reflects the
        # DnaA moved bulk → bound on each box. Same pattern as tf_binding for
        # TFs binding promoters (tf_binding.py:362-369).
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

        # dnaA autoregulation: publish dnaA-promoter occupancy fraction so
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
            # Publish the bound-pool hydrolysis count so the equilibrium step
            # routes Pi/PROTON/WATER byproducts for the SAME hydrolysis events
            # instead of independently re-sampling.
            "dnaa_hydrolysis": {
                "bound_count": int(delta_h_bound),
                "promoter_fraction": promoter_fraction,
            },
            "listeners": {
                "replication_data": {
                    "gradient_rising": bool(diag_gradient_rising),
                    "bulk_atp_slope_nM_per_s": float(diag_slope_nM_per_s),
                    "bulk_atp_nM_current": float(diag_bulk_atp_nM_current),
                    "rida_events": int(n_rida_events),
                },
            },
        }
        if bulk_update:
            update["bulk"] = bulk_update
        return update
