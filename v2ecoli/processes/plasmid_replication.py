"""
======================
Plasmid Replication
======================

Adapted from chromosome replication. Performs initiation, elongation, and
termination of active plasmid molecules that replicate independently of the
chromosome (ColE1 / pBR322).

Copy-number control is the full Brendel & Perelson 1993 ODE system
(J Mol Biol 229:860, eqns 1a-1j): 10 species (7 plasmid-DNA states
D / D_tII / D_lII / D_p / D_starc / D_c / D_M, plus free R_I, R_II, M)
integrated each ``_prepare`` step with RK4. Replication initiations are
discretized from the continuous ``k_D × D_p`` flux via a fractional
accumulator; the integer count is exposed to ``_evolve`` as
``n_rna_initiations`` and gated by replisome-subunit availability.
The cell-growth term μ in BP's eqns is set to 0 — vEcoli handles
volume increase and division externally.

Runs as a plain Step, matching the chromosome replication architecture in
v2ecoli. When ``has_plasmid=False`` the unique molecule ports are empty
and this process is a no-op.
"""

import numpy as np

from v2ecoli.library.schema import (
    numpy_schema,
    counts,
    attrs,
    bulk_name_to_idx,
    listener_schema,
)

from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.unit_bridge import unum_to_pint
from wholecell.utils.polymerize import buildSequences, polymerize, computeMassIncrease

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema_types import (
    FULL_PLASMID_ARRAY,
    PLASMID_DOMAIN_ARRAY,
    ORIV_ARRAY,
    PLASMID_ACTIVE_REPLISOME_ARRAY,
)


NAME = "ecoli-plasmid-replication"
TOPOLOGY = {
    "bulk": ("bulk",),
    "plasmid_active_replisomes": ("unique", "plasmid_active_replisome"),
    "oriVs": ("unique", "oriV"),
    "plasmid_domains": ("unique", "plasmid_domain"),
    "full_plasmids": ("unique", "full_plasmid"),
    "listeners": ("listeners",),
    "environment": ("environment",),
    "plasmid_rna_control": ("process_state", "plasmid_rna_control"),
    "timestep": ("timestep",),
    "global_time": ("global_time",),
}


class PlasmidReplication(Step):
    """Plasmid Replication Step (ColE1 / pBR322).

    Replisome subunits and dNTPs are shared with chromosome replication;
    because v2ecoli's ChromosomeReplication runs as a plain Step without
    allocator arbitration, this process runs similarly. Resource contention
    is not mediated here.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'time_step': {'_type': 'integer', '_default': 1},
        'D_period': {'_type': 'any', '_default': np.array([], dtype=float)},
        'basal_elongation_rate': {'_type': 'integer[nt/s]', '_default': 967},
        'dntps': {'_type': 'list[string]', '_default': []},
        'emit_unique': {'_type': 'boolean', '_default': False},
        'make_elongation_rates': {'_type': 'method', '_default': None},
        'mechanistic_replisome': {'_type': 'boolean', '_default': True},
        'no_child_place_holder': {'_type': 'integer', '_default': -1},
        'polymerized_dntp_weights': {'_type': 'any', '_default': []},
        'ppi': {'_type': 'list[string]', '_default': []},
        'replichore_lengths': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'replisome_monomers_subunits': {'_type': 'list[string]', '_default': []},
        'replisome_protein_mass': {'_type': 'float[fg]', '_default': 0},
        'replisome_trimers_subunits': {'_type': 'list[string]', '_default': []},
        'seed': {'_type': 'integer', '_default': 0},
        'sequences': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        # Copy-number control — full BP1993 ODE system (eqns 1a-1j),
        # pBR322 rom+ parameterization from Brendel & Perelson 1993
        # (J Mol Biol 229:860-872) Table 1. Rate constants are stored in
        # BP's native units (min⁻¹ for unimolecular; M⁻¹·min⁻¹ for
        # bimolecular k_1, k_3) and converted in ``initialize`` using
        # V_c (BP's reported cytoplasmic volume, 6.25e-16 L) and
        # Avogadro's number to obtain count-based per-second rates.
        # We do NOT couple to v2ecoli's dynamic volume: V_c is a fixed
        # BP parameter; the planned bulk-RNAP refactor (project memory
        # `plasmid_mechanistic_target`) eliminates V_c entirely.
        'use_rna_control': {'_type': 'boolean', '_default': True},
        # Volume used to convert BP's molarity into molecule counts.
        'V_c_L': {'_type': 'float', '_default': 6.25e-16},
        'n_avogadro': {'_type': 'float', '_default': 6.022e23},
        # Bimolecular rates (M⁻¹·min⁻¹).
        'k_1': {'_type': 'float', '_default': 1.5e8},
        'k_3': {'_type': 'float', '_default': 1.7e8},
        # Unimolecular rates (min⁻¹).
        'k_neg1': {'_type': 'float', '_default': 48.0},
        'k_2':    {'_type': 'float', '_default': 44.0},
        'k_neg2': {'_type': 'float', '_default': 0.085},
        'k_neg3': {'_type': 'float', '_default': 0.17},
        'k_4':    {'_type': 'float', '_default': 34.0},
        'k_l':    {'_type': 'float', '_default': 12.0},
        'k_negl': {'_type': 'float', '_default': 4.3},
        'k_p':    {'_type': 'float', '_default': 4.3},
        'k_D':    {'_type': 'float', '_default': 5.0},
        'k_negc': {'_type': 'float', '_default': 17.0},
        'k_I':    {'_type': 'float', '_default': 6.0},
        'k_II':   {'_type': 'float', '_default': 0.25},
        'k_M':    {'_type': 'float', '_default': 4.0},
        'eps_I':  {'_type': 'float', '_default': 0.35},
        'eps_II': {'_type': 'float', '_default': 0.35},
        'eps_M':  {'_type': 'float', '_default': 0.14},
        # Number of RK4 sub-steps per process timestep.
        'n_substeps': {'_type': 'integer', '_default': 10},
    }

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'plasmid_active_replisomes': {'_type': PLASMID_ACTIVE_REPLISOME_ARRAY, '_default': []},
            'oriVs': {'_type': ORIV_ARRAY, '_default': []},
            'plasmid_domains': {'_type': PLASMID_DOMAIN_ARRAY, '_default': []},
            'full_plasmids': {'_type': FULL_PLASMID_ARRAY, '_default': []},
            'environment': {
                'media_id': {'_type': 'string', '_default': ''},
            },
            'plasmid_rna_control': {
                'D':       {'_type': 'float', '_default': 1.0},
                'D_tII':   {'_type': 'float', '_default': 0.0},
                'D_lII':   {'_type': 'float', '_default': 0.0},
                'D_p':     {'_type': 'float', '_default': 0.0},
                'D_starc': {'_type': 'float', '_default': 0.0},
                'D_c':     {'_type': 'float', '_default': 0.0},
                'D_M':     {'_type': 'float', '_default': 0.0},
                'R_I':     {'_type': 'float', '_default': 17.0},
                'R_II':    {'_type': 'float', '_default': 0.0},
                'M':       {'_type': 'float', '_default': 0.0},
                'repl_accum':        {'_type': 'float',   '_default': 0.0},
                'n_rna_initiations': {'_type': 'integer', '_default': 0},
            },
            'timestep': {'_type': 'integer[s]', '_default': 1},
            'global_time': {'_type': 'float', '_default': 0.0},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'plasmid_active_replisomes': PLASMID_ACTIVE_REPLISOME_ARRAY,
            'oriVs': ORIV_ARRAY,
            'plasmid_domains': PLASMID_DOMAIN_ARRAY,
            'full_plasmids': FULL_PLASMID_ARRAY,
            'plasmid_rna_control': {
                'D':       'overwrite[float]',
                'D_tII':   'overwrite[float]',
                'D_lII':   'overwrite[float]',
                'D_p':     'overwrite[float]',
                'D_starc': 'overwrite[float]',
                'D_c':     'overwrite[float]',
                'D_M':     'overwrite[float]',
                'R_I':     'overwrite[float]',
                'R_II':    'overwrite[float]',
                'M':       'overwrite[float]',
                'repl_accum':        'overwrite[float]',
                'n_rna_initiations': 'overwrite[integer]',
            },
            'listeners': {
                'replication_data': {
                    'plasmid_critical_mass_per_oriV': {'_type': 'overwrite[float]', '_default': 0.0},
                },
            },
        }

    def initialize(self, config):
        self.replichore_lengths = self.parameters["replichore_lengths"]
        self.sequences = self.parameters["sequences"]
        self.polymerized_dntp_weights = unum_to_pint(
            self.parameters["polymerized_dntp_weights"]
        )
        self.D_period = self.parameters["D_period"]
        self.replisome_protein_mass = self.parameters["replisome_protein_mass"]
        self.no_child_place_holder = self.parameters["no_child_place_holder"]
        self.basal_elongation_rate = self.parameters["basal_elongation_rate"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]

        self.mechanistic_replisome = self.parameters["mechanistic_replisome"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.emit_unique = self.parameters.get("emit_unique", True)

        self.replisome_trimers_subunits = self.parameters["replisome_trimers_subunits"]
        self.replisome_monomers_subunits = self.parameters["replisome_monomers_subunits"]
        self.dntps = self.parameters["dntps"]
        self.ppi = self.parameters["ppi"]

        self.ppi_idx = None

        self.use_rna_control = self.parameters["use_rna_control"]
        if self.use_rna_control:
            # BP1993 rates: convert min⁻¹ → s⁻¹ (unimolecular) and
            # M⁻¹·min⁻¹ → (molecule·s)⁻¹ (bimolecular k_1, k_3).
            V_c = self.parameters["V_c_L"]
            N_A = self.parameters["n_avogadro"]
            self.bp_k_neg1 = self.parameters["k_neg1"] / 60.0
            self.bp_k_2    = self.parameters["k_2"]    / 60.0
            self.bp_k_neg2 = self.parameters["k_neg2"] / 60.0
            self.bp_k_neg3 = self.parameters["k_neg3"] / 60.0
            self.bp_k_4    = self.parameters["k_4"]    / 60.0
            self.bp_k_l    = self.parameters["k_l"]    / 60.0
            self.bp_k_negl = self.parameters["k_negl"] / 60.0
            self.bp_k_p    = self.parameters["k_p"]    / 60.0
            self.bp_k_D    = self.parameters["k_D"]    / 60.0
            self.bp_k_negc = self.parameters["k_negc"] / 60.0
            self.bp_k_I    = self.parameters["k_I"]    / 60.0
            self.bp_k_II   = self.parameters["k_II"]   / 60.0
            self.bp_k_M    = self.parameters["k_M"]    / 60.0
            self.bp_eps_I  = self.parameters["eps_I"]  / 60.0
            self.bp_eps_II = self.parameters["eps_II"] / 60.0
            self.bp_eps_M  = self.parameters["eps_M"]  / 60.0
            # Bimolecular: k [M⁻¹·min⁻¹] / (N_A × V_c [L]) / 60
            #            = (mol·L) per (M·count) → per (count·s)
            bimol_factor = 1.0 / (N_A * V_c) / 60.0
            self.bp_k_1 = self.parameters["k_1"] * bimol_factor
            self.bp_k_3 = self.parameters["k_3"] * bimol_factor
            self.n_substeps = self.parameters["n_substeps"]

    def update(self, states, interval=None):
        self._rna_control_update, self._n_rna_initiations = self._prepare(states)
        return self._evolve(states)

    def _prepare(self, states):
        timestep = states["timestep"]
        if self.ppi_idx is None:
            self.ppi_idx = bulk_name_to_idx(self.ppi, states["bulk"]["id"])
            self.replisome_trimers_idx = bulk_name_to_idx(
                self.replisome_trimers_subunits, states["bulk"]["id"]
            )
            self.replisome_monomers_idx = bulk_name_to_idx(
                self.replisome_monomers_subunits, states["bulk"]["id"]
            )
            self.dntps_idx = bulk_name_to_idx(self.dntps, states["bulk"]["id"])

        # Copy-number control — full BP1993 ODE system (eqns 1a-1j).
        # Integrate with RK4 sub-stepping over the process timestep,
        # then discretize the continuous replication flux d(sum_D)/dt
        # = k_D * D_p into integer initiation events via a fractional
        # accumulator. After integration the D-state pool is rescaled
        # so its sum equals n_full_plasmids - n_active_replisomes, the
        # number of plasmids "available" to be in a BP D-state.
        rna_control_update = {}
        n_rna_initiations = 0
        if self.use_rna_control:
            n_plasmids = int(states["full_plasmids"]["_entryState"].sum())
            n_active = int(
                states["plasmid_active_replisomes"]["_entryState"].sum()
            )
            n_idle = max(0, n_plasmids - n_active)

            ctrl = states["plasmid_rna_control"]
            y = np.array([
                ctrl["D"], ctrl["D_tII"], ctrl["D_lII"], ctrl["D_p"],
                ctrl["D_starc"], ctrl["D_c"], ctrl["D_M"],
                ctrl["R_I"], ctrl["R_II"], ctrl["M"],
            ], dtype=float)
            repl_accum = float(ctrl["repl_accum"])

            # Renormalize D-pool to match n_idle (handles division and
            # termination, where n_full_plasmids changed externally).
            sum_D = y[0:7].sum()
            if n_idle > 0:
                if sum_D <= 0:
                    y[0] = float(n_idle)
                else:
                    y[0:7] *= n_idle / sum_D
            else:
                y[0:7] = 0.0

            if n_idle > 0:
                y = self._bp_rk4(y, float(timestep), self.n_substeps)
                # Discretize replication: d(sum_D)/dt = k_D * D_p, so
                # the post-step growth in sum_D equals the continuous
                # number of replication events that occurred.
                new_sum = y[0:7].sum()
                delta = new_sum - n_idle
                if delta > 0:
                    repl_accum += delta
                    # Renormalize back to n_idle so D-pool stays
                    # consistent until Module 1/3 update n_full_plasmids
                    # and n_active_replisomes.
                    y[0:7] *= n_idle / new_sum
                n_rna_initiations = int(repl_accum)
                if n_rna_initiations > 0:
                    repl_accum -= n_rna_initiations

            rna_control_update = {
                "D":       float(y[0]),
                "D_tII":   float(y[1]),
                "D_lII":   float(y[2]),
                "D_p":     float(y[3]),
                "D_starc": float(y[4]),
                "D_c":     float(y[5]),
                "D_M":     float(y[6]),
                "R_I":     float(y[7]),
                "R_II":    float(y[8]),
                "M":       float(y[9]),
                "repl_accum":        float(repl_accum),
                "n_rna_initiations": int(n_rna_initiations),
            }

        # Compute elongation rates for the current timestep (used by _evolve).
        n_active_replisomes = states["plasmid_active_replisomes"]["_entryState"].sum()
        if n_active_replisomes > 0:
            self.elongation_rates = self.make_elongation_rates(
                self.random_state,
                len(self.sequences),
                self.basal_elongation_rate,
                timestep,
            )

        return rna_control_update, n_rna_initiations

    def _bp_deriv(self, y):
        """BP1993 eqns 1a-1j (μ=0). y in molecule counts; returns dy/dt in /s.

        Only short RNA II (D_tII, 110-360 nt) is susceptible to R_I binding;
        once elongated past 360 nt to D_lII, the loop region is no longer
        accessible to R_I (paper §2, eqns 1b/1c/1e/1h).
        """
        D, D_tII, D_lII, D_p, D_starc, D_c, D_M, R_I, R_II, M = y
        f_kiss = self.bp_k_1 * R_I * D_tII
        f_rom  = self.bp_k_3 * M  * D_starc
        return np.array([
            # dD/dt — eqn 1a
            2.0 * self.bp_k_D * D_p - self.bp_k_II * D
            + self.bp_k_negl * D_lII + self.bp_k_negc * D_c,
            # dD_tII/dt — eqn 1b
            self.bp_k_II * D - self.bp_k_l * D_tII - f_kiss
            + self.bp_k_neg1 * D_starc,
            # dD_lII/dt — eqn 1c (no R_I term: long RNA II is past inhibition window)
            self.bp_k_l * D_tII
            - (self.bp_k_negl + self.bp_k_p) * D_lII,
            # dD_p/dt — eqn 1d
            self.bp_k_p * D_lII - self.bp_k_D * D_p,
            # dD_starc/dt — eqn 1e (only f_kiss with D_tII)
            f_kiss
            - (self.bp_k_neg1 + self.bp_k_2) * D_starc - f_rom
            + self.bp_k_neg2 * D_c + self.bp_k_neg3 * D_M,
            # dD_c/dt — eqn 1f
            self.bp_k_2 * D_starc - (self.bp_k_neg2 + self.bp_k_negc) * D_c
            + self.bp_k_4 * D_M,
            # dD_M/dt — eqn 1g
            f_rom - (self.bp_k_neg3 + self.bp_k_4) * D_M,
            # dR_I/dt — eqn 1h (sink is k_1·D_tII, not D_lII)
            self.bp_k_I * D - (self.bp_k_1 * D_tII + self.bp_eps_I) * R_I
            + self.bp_k_neg1 * D_starc,
            # dR_II/dt — eqn 1i (+k_D·D_p: primer released at replication)
            self.bp_k_negl * D_lII + self.bp_k_D * D_p
            - self.bp_eps_II * R_II,
            # dM/dt — eqn 1j
            self.bp_k_M * D - (self.bp_k_3 * D_starc + self.bp_eps_M) * M
            + (self.bp_k_neg3 + self.bp_k_4) * D_M,
        ])

    def _bp_rk4(self, y, dt, n_substeps):
        """Fixed-step RK4 with non-negative clamping."""
        h = dt / n_substeps
        for _ in range(n_substeps):
            k1 = self._bp_deriv(y)
            k2 = self._bp_deriv(y + 0.5 * h * k1)
            k3 = self._bp_deriv(y + 0.5 * h * k2)
            k4 = self._bp_deriv(y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            np.maximum(y, 0.0, out=y)
        return y

    def _evolve(self, states):
        update = {
            "bulk": [],
            "plasmid_active_replisomes": {},
            "oriVs": {},
            "plasmid_domains": {},
            "full_plasmids": {},
            "listeners": {"replication_data": {}},
        }
        if self._rna_control_update:
            update["plasmid_rna_control"] = self._rna_control_update

        n_active_replisomes = states["plasmid_active_replisomes"]["_entryState"].sum()
        n_oriV = states["oriVs"]["_entryState"].sum()
        n_full_plasmids = states["full_plasmids"]["_entryState"].sum()

        # If there are no plasmids, return immediately (RNA control has been
        # written; nothing else to do).
        if n_full_plasmids == 0:
            return update

        # Module 1: Replication initiation
        (fork_coordinates, domain_index_replisome) = attrs(
            states["plasmid_active_replisomes"], ["coordinates", "domain_index"]
        )

        domain_index_existing_domain, child_domains = attrs(
            states["plasmid_domains"], ["domain_index", "child_domains"]
        )
        (domain_index_existing_plasmid,) = attrs(
            states["full_plasmids"], ["domain_index"]
        )
        # Plasmid domains that currently have no active replisome.
        idle_plasmid_domains = np.setdiff1d(
            domain_index_existing_plasmid, domain_index_replisome
        )

        initiate_replication = False
        max_new_replisomes = 0
        n_rna_initiations = self._n_rna_initiations
        if len(idle_plasmid_domains) > 0:
            # Gate 1: replisome subunit availability. In mechanistic mode
            # the per-oriV replisome needs the full subunit complement
            # (DnaG, pol III core, β-clamp, DnaB, HolA) — so depletion of
            # any one caps new forks via ``max_new_replisomes``. In
            # permissive mode (mechanistic_replisome=False) subunits don't
            # gate plasmid replication, so we set the cap to the number
            # of idle domains instead of the subunit-limited value. Without
            # this override the fork-count cap at ``min(..., max_new_replisomes, ...)``
            # below quietly forces n_new_replisome to 0 when DnaG hits 0,
            # even though the subunits_ok gate says we're free to initiate —
            # which was the gen-5/6 stall signature: many RNA II
            # initiation events fire, zero new plasmids actually appear.
            if self.mechanistic_replisome:
                n_replisome_trimers = counts(states["bulk"], self.replisome_trimers_idx)
                n_replisome_monomers = counts(states["bulk"], self.replisome_monomers_idx)
                min_trimers = int(np.min(n_replisome_trimers))
                min_monomers = int(np.min(n_replisome_monomers))
                max_by_trimers = min_trimers // 3
                max_by_monomers = min_monomers // 1
                max_new_replisomes = min(max_by_trimers, max_by_monomers)
                subunits_ok = max_new_replisomes != 0
            else:
                max_new_replisomes = len(idle_plasmid_domains)
                subunits_ok = True

            # Gate 2: RNA II copy number control (Brendel-Perelson 1993).
            if self.use_rna_control:
                rna_ok = n_rna_initiations > 0
            else:
                n_rna_initiations = len(idle_plasmid_domains)
                rna_ok = True

            initiate_replication = subunits_ok and rna_ok

        if initiate_replication:
            (domain_index_existing_oriv,) = attrs(states["oriVs"], ["domain_index"])

            new_parent_domains = np.where(
                np.isin(domain_index_existing_domain, domain_index_existing_oriv)
            )[0]

            n_new_replisome = 0
            n_new_domain = 0
            domain_index_new = np.array([], dtype=np.int32)

            if max_new_replisomes != 0:
                n_new_replisome = min(
                    len(idle_plasmid_domains),
                    max_new_replisomes,
                    n_rna_initiations,
                )
                n_new_domain = 2 * n_new_replisome

                max_domain_index = domain_index_existing_domain.max()
                domain_index_new = np.arange(
                    max_domain_index + 1,
                    max_domain_index + 2 * n_new_replisome + 1,
                    dtype=np.int32,
                )

            if len(domain_index_new) > 0:
                if n_oriV > n_new_replisome:
                    n_excess_orivs = n_oriV - n_new_replisome
                    domain_index_new_oriv = np.concatenate(
                        (
                            domain_index_existing_oriv[-n_excess_orivs:],
                            domain_index_new,
                        )
                    )
                    update["oriVs"]["set"] = {
                        "domain_index": domain_index_new_oriv[:n_oriV]
                    }
                    update["oriVs"]["add"] = {
                        "domain_index": domain_index_new_oriv[n_oriV:],
                    }
                else:
                    update["oriVs"]["set"] = {
                        "domain_index": domain_index_new[:n_oriV]
                    }
                    update["oriVs"]["add"] = {
                        "domain_index": domain_index_new[n_oriV:],
                    }

            # Plasmid replication is unidirectional: one replisome per oriV.
            coordinates_replisome = np.zeros(n_new_replisome, dtype=np.int64)
            right_replichore = np.full(n_new_replisome, False, dtype=bool)
            domain_index_new_replisome = idle_plasmid_domains[:n_new_replisome]
            massDiff_protein_new_replisome = np.full(
                n_new_replisome,
                self.replisome_protein_mass if self.mechanistic_replisome else 0.0,
            )

            update["plasmid_active_replisomes"]["add"] = {
                "coordinates": coordinates_replisome,
                "right_replichore": right_replichore,
                "domain_index": domain_index_new_replisome,
                "massDiff_protein": massDiff_protein_new_replisome,
            }

            if n_new_domain != 0:
                new_child_domains = np.full(
                    (n_new_domain, 2), self.no_child_place_holder, dtype=np.int32
                )
                new_domains_update = {
                    "add": {
                        "domain_index": domain_index_new,
                        "child_domains": new_child_domains,
                    }
                }
                if new_parent_domains.size > 0:
                    if new_parent_domains.size != n_new_replisome:
                        new_parent_domains = new_parent_domains[:n_new_replisome]
                    child_domains[new_parent_domains] = domain_index_new.reshape(-1, 2)

                existing_domains_update = {"set": {"child_domains": child_domains}}
                update["plasmid_domains"].update(
                    {**new_domains_update, **existing_domains_update}
                )

            if self.mechanistic_replisome:
                update["bulk"].append(
                    (self.replisome_trimers_idx, -3 * n_new_replisome)
                )
                update["bulk"].append(
                    (self.replisome_monomers_idx, -1 * n_new_replisome)
                )

        # Module 2: replication elongation
        if n_active_replisomes == 0:
            return update

        dNtpCounts = counts(states["bulk"], self.dntps_idx)

        (
            domain_index_replisome,
            right_replichore,
            coordinates_replisome,
        ) = attrs(
            states["plasmid_active_replisomes"],
            ["domain_index", "right_replichore", "coordinates"],
        )

        sequence_length = np.abs(np.repeat(coordinates_replisome, 2))
        sequence_indexes = np.tile(np.arange(2), n_active_replisomes)

        sequences = buildSequences(
            self.sequences, sequence_indexes, sequence_length, self.elongation_rates
        )

        reactionLimit = dNtpCounts.sum()
        active_elongation_rates = self.elongation_rates[sequence_indexes]

        result = polymerize(
            sequences,
            dNtpCounts,
            reactionLimit,
            self.random_state,
            active_elongation_rates,
        )

        sequenceElongations = result.sequenceElongation
        dNtpsUsed = result.monomerUsages

        mass_increase_dna = computeMassIncrease(
            sequences,
            sequenceElongations,
            self.polymerized_dntp_weights.to(units.fg).magnitude,
        )

        added_dna_mass = mass_increase_dna[0::2] + mass_increase_dna[1::2]

        updated_length = sequence_length + sequenceElongations
        updated_coordinates = updated_length[0::2]

        (current_dna_mass,) = attrs(
            states["plasmid_active_replisomes"], ["massDiff_DNA"]
        )

        update["plasmid_active_replisomes"].update(
            {
                "set": {
                    "coordinates": updated_coordinates,
                    "massDiff_DNA": current_dna_mass + added_dna_mass,
                }
            }
        )

        update["bulk"].append((self.dntps_idx, -dNtpsUsed))
        update["bulk"].append((self.ppi_idx, dNtpsUsed.sum()))

        # Module 3: replication termination
        terminated_replisomes = updated_coordinates >= self.replichore_lengths

        if terminated_replisomes.sum() > 0:
            terminated_domains = np.unique(
                domain_index_replisome[terminated_replisomes]
            )

            (
                domain_index_domains,
                child_domains,
            ) = attrs(
                states["plasmid_domains"], ["domain_index", "child_domains"]
            )
            (domain_index_full_plasmid,) = attrs(
                states["full_plasmids"], ["domain_index"]
            )

            replisomes_to_delete = np.zeros_like(
                domain_index_replisome, dtype=np.bool_
            )
            n_new_plasmids = 0
            domain_index_new_full_plasmid = []

            for terminated_domain_index in terminated_domains:
                terminated_domain_matching_replisomes = np.logical_and(
                    domain_index_replisome == terminated_domain_index,
                    terminated_replisomes,
                )

                # Only a single fork per plasmid domain (unidirectional).
                if terminated_domain_matching_replisomes.any():
                    replisomes_to_delete = np.logical_or(
                        replisomes_to_delete, terminated_domain_matching_replisomes
                    )

                    domain_mask = domain_index_domains == terminated_domain_index
                    child_domains_this_domain = child_domains[
                        np.where(domain_mask)[0][0], :
                    ]

                    domain_index_full_plasmid = domain_index_full_plasmid.copy()
                    domain_index_full_plasmid[
                        np.where(
                            domain_index_full_plasmid == terminated_domain_index
                        )[0]
                    ] = child_domains_this_domain[0]

                    n_new_plasmids += 1
                    domain_index_new_full_plasmid.append(
                        child_domains_this_domain[1]
                    )

            update["plasmid_active_replisomes"]["delete"] = np.where(
                replisomes_to_delete
            )[0]

            if n_new_plasmids > 0:
                plasmid_add_update = {
                    "add": {
                        "domain_index": domain_index_new_full_plasmid,
                        "division_time": states["global_time"] + self.D_period,
                        "has_triggered_division": False,
                    }
                }
                plasmid_existing_update = {
                    "set": {"domain_index": domain_index_full_plasmid}
                }
                update["full_plasmids"].update(
                    {**plasmid_add_update, **plasmid_existing_update}
                )

            if self.mechanistic_replisome:
                update["bulk"].append(
                    (self.replisome_trimers_idx, 3 * replisomes_to_delete.sum())
                )
                update["bulk"].append(
                    (self.replisome_monomers_idx, replisomes_to_delete.sum())
                )

        return update


def test_plasmid_replication():
    test_config = {}
    process = PlasmidReplication(test_config)
    assert process is not None


if __name__ == "__main__":
    test_plasmid_replication()
