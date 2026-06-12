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
# dnaa-4: loosened from 1→3 nM so the dnaA-promoter de-occupies at low [DnaA],
# enabling the negative-feedback signal to de-repress transcription.
KD_LOW_M = 100e-9   # 100 nM — oriC_low (ATP only)

def _promoter_fraction(pool_label: np.ndarray, bound_form: np.ndarray) -> float:
    """Bound fraction of the dnaA-promoter sites — the autoregulation signal.

    Published on the dnaa_hydrolysis port each tick so transcript_initiation
    can read it as the negative-feedback input for dnaA self-autoregulation.
    """
    prom_mask = (pool_label == POOL_PROMOTER_HIGH)
    n_total = int(prom_mask.sum())
    if n_total == 0:
        return 0.0
    n_bound = int((bound_form[prom_mask] != FORM_FREE).sum())
    return float(n_bound) / float(n_total)


# DnaA-ATP intrinsic hydrolysis rate. Bound DnaA-ATP hydrolyzes at the same
# rate as free DnaA-ATP (Sekimizu 1987, k = 0.046 / min). bf8b82e's equilibrium
# step hydrolyzes the FREE pool only; this step additionally hydrolyzes the
# BOUND pool, in-place (bound box-ATP → bound box-ADP on the same row).
HYDROLYSIS_RATE_PER_MIN = 0.046


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

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'DnaA_boxes': {'_type': DNAA_BOX_ARRAY, '_default': []},
            'listeners': {
                'mass': {
                    'cell_mass': {'_type': 'quantity[float,fg]', '_default': 0},
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
                'promoter_fraction': 'overwrite[float]',
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

        pool_label, bound_form = attrs(boxes, ["pool_label", "DnaA_bound_form"])

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

        # Cell volume (L).
        cell_mass_g = as_quantity(
            states["listeners"]["mass"]["cell_mass"], units.fg
        ).to(units.g).magnitude
        cell_volume_L = cell_mass_g / self.cell_density

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
        # 4 pools, 2 free ligand species (A_free, D_free):
        #   high-aff: K_h, both ATP and ADP, sites = N_h (chrom+oric_hi+prom)
        #   low-aff:  K_l, ATP only, sites = N_l (oriC_low)
        #
        # Per-box occupancy from competitive Langmuir:
        #   p_atp_h = a/(1+a+d),  a=A_free/K_h, d=D_free/K_h
        #   p_atp_l = (A_free/K_l) / (1 + A_free/K_l)
        #
        # Residuals (mass balance):
        #   r0 = A_T - A_free - N_h*a/(1+a+d) - N_l*p_atp_l
        #   r1 = D_T - D_free - N_h*d/(1+a+d)
        #
        # MINPACK hybr (Powell hybrid, Newton-like) converges in ~5 iter.
        K_h = kd_high_molecules
        K_l = kd_low_molecules
        N_h = n_high_total
        N_l = n_low_total
        A_T = float(total_atp)
        D_T = float(total_adp)

        if (A_T + D_T) > 0 and (N_h + N_l) > 0:
            def _residuals(x, A_T=A_T, D_T=D_T, N_h=N_h, N_l=N_l,
                           K_h=K_h, K_l=K_l):
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

            # Initial guess: half-bound, half-free for each.
            x0 = [max(A_T / 2.0, 1e-3), max(D_T / 2.0, 1e-3)]
            sol = scipy_root(_residuals, x0, method="hybr", tol=1e-9)
            if not sol.success:
                # Fallback: warm-start at the totals (all-free guess).
                sol = scipy_root(_residuals, [A_T, D_T], method="hybr", tol=1e-9)
            A_free = max(float(sol.x[0]), 0.0)
            D_free = max(float(sol.x[1]), 0.0)
        else:
            A_free = A_T
            D_free = D_T

        if K_h > 0 and N_h > 0:
            a_h = A_free / K_h
            d_h = D_free / K_h
            denom_h = 1.0 + a_h + d_h
            high_bound_atp = N_h * a_h / denom_h
            high_bound_adp = N_h * d_h / denom_h
        else:
            high_bound_atp = high_bound_adp = 0.0
        if K_l > 0 and N_l > 0:
            low_bound_atp = N_l * A_free / (K_l + A_free)
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

        # dnaa-4: promoter occupancy signal for dnaA self-autoregulation.
        promoter_fraction = _promoter_fraction(pool_label, new_bound_form)

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

        update = {
            "DnaA_boxes": {"set": box_set},
            "next_update_time": states["global_time"] + states["timestep"],
            # dnaa-3 Phase 2c: publish the bound-pool hydrolysis count so
            # equilibrium.py can route the byproducts (Pi/PROTON/WATER) for the
            # SAME hydrolysis events rather than independently re-sampling.
            "dnaa_hydrolysis": {
                "bound_count": int(delta_h_bound),
                # dnaa-4: promoter occupancy for dnaA negative feedback.
                "promoter_fraction": promoter_fraction,
            },
        }
        if bulk_update:
            update["bulk"] = bulk_update
        return update
