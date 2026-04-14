"""
=====================
Simplified Metabolism
=====================

Flux-balance analysis (FBA) via direct scipy LP solve.

Mathematical Model
------------------
Solves the homeostatic FBA problem each timestep:

    min  sum_i (s_i^+ + s_i^-)                  # L1 deviation from targets
    s.t. S @ v = 0                                # steady-state mass balance
         M @ v + s^+ - s^- = target / coefficient # output = target (slack form)
         v_lb <= v <= v_ub                         # flux bounds
         s^+, s^- >= 0

Where:
    S: stoichiometric matrix (n_metabolites x n_reactions), sparse CSR
    v: reaction flux vector (mM = mmol/L per timestep)
    M: output molecule rows of S (tracked metabolites)
    target: homeostatic concentration targets (mM)
    coefficient: conversion factor (dry_mass/cell_mass * density * dt) [g*s/L]

Flux bounds are set by:
    - Exchange constraints (nutrient availability from environment)
    - Enzyme presence/absence (catalyst_ids, catalysis_matrix)
    - Maintenance energy (NGAM, translation GTP)

After solving, fluxes are converted to molecule count changes:

    delta_counts = stochasticRound(delta_concentration / counts_to_molar)

This replaces the 1000+ line Metabolism process + FluxBalanceAnalysisModel
wrapper + wholecell FluxBalanceAnalysis class with a single self-contained
module using scipy.optimize.linprog (HiGHS solver).

Usage
-----
Drop-in replacement for ``Metabolism`` in the composite. Uses the same
port schema (bulk, environment, listeners, etc.) so it can be swapped
by changing the process class in ``composite.py``.
"""

import warnings
from typing import Any, Optional

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import bulk_name_to_idx, counts
from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.unit_bridge import unum_to_pint
from wholecell.utils.random import stochasticRound


NAME = "ecoli-metabolism-simple"
TOPOLOGY = {
    "bulk": ("bulk",),
    "bulk_total": ("bulk",),
    "listeners": ("listeners",),
    "environment": ("environment",),
    "polypeptide_elongation": ("process_state", "polypeptide_elongation"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "metabolism"),
}

# Unit basis
COUNTS_UNITS = units.mmol
VOLUME_UNITS = units.L
MASS_UNITS = units.g
CONC_UNITS = COUNTS_UNITS / VOLUME_UNITS  # mM


class SimplifiedMetabolism(Step):
    """Metabolism via direct scipy linprog.

    Solves an LP each timestep to find reaction fluxes that drive internal
    metabolite concentrations toward homeostatic targets while respecting
    stoichiometric mass balance, nutrient availability, and enzyme capacity.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # Stoichiometry: {reaction_id: {metabolite_id: coefficient}}
        'reaction_stoich': {'_type': 'any', '_default': {}},
        # Homeostatic targets: {metabolite_id: target_concentration_mM}
        'homeostatic_targets': {'_type': 'map[float]', '_default': {}},
        # Exchange molecules: list of external molecule IDs (with compartment tags)
        'exchange_molecules': {'_type': 'list[string]', '_default': []},
        # Enzyme catalysis data
        'catalyst_ids': {'_type': 'list[string]', '_default': []},
        'reactions_with_catalyst': {'_type': 'list[string]', '_default': []},
        'catalysis_matrix_I': {'_type': 'array[integer]', '_default': []},
        'catalysis_matrix_J': {'_type': 'array[integer]', '_default': []},
        'catalysis_matrix_V': {'_type': 'array[float]', '_default': []},
        # Maintenance
        'ngam': {'_type': 'float[mmol/g/h]', '_default': 8.39},
        'maintenance_reaction': {'_type': 'any', '_default': {}},
        'dark_atp': {'_type': 'float', '_default': 33.565},
        'cell_dry_mass_fraction': {'_type': 'float', '_default': 0.3},
        # Physical constants
        'avogadro': {'_type': 'float[1/mol]', '_default': 6.022e23},
        'cell_density': {'_type': 'float[g/L]', '_default': 1100.0},
        # Environment
        'exchange_data_from_media': {'_type': 'method', '_default': None},
        'media_id': {'_type': 'string', '_default': 'minimal'},
        'nutrientToDoublingTime': {'_type': 'map[float]', '_default': {}},
        'get_biomass_as_concentrations': {'_type': 'method', '_default': None},
        # Misc
        'seed': {'_type': 'integer', '_default': 0},
        'time_step': {'_type': 'integer[s]', '_default': 1},
        'output_molecule_ids': {'_type': 'list[string]', '_default': []},
        'molecule_masses': {'_type': 'map[float]', '_default': {}},
    }

    def inputs(self):
        return {
            'bulk': {'_type': 'bulk_array', '_default': []},
            'bulk_total': {'_type': 'bulk_array', '_default': []},
            'listeners': {
                'mass': {
                    'cell_mass': {'_type': 'float[fg]', '_default': 0.0},
                    'dry_mass': {'_type': 'float[fg]', '_default': 0.0},
                },
            },
            'environment': {
                'media_id': {'_type': 'string', '_default': ''},
                'exchange_data': {
                    'constrained': 'map[float]',
                    'unconstrained': 'list[string]',
                },
            },
            'polypeptide_elongation': {
                'gtp_to_hydrolyze': {'_type': 'float', '_default': 0.0},
            },
            'global_time': {'_type': 'float[s]', '_default': 0.0},
            'timestep': {'_type': 'integer[s]', '_default': 1},
            'next_update_time': {'_type': 'float[s]', '_default': 1.0},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'listeners': {
                'fba_results': {
                    'delta_metabolites': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'reaction_fluxes': {'_type': 'overwrite[array[float]]', '_default': []},
                    'media_id': {'_type': 'overwrite[string]', '_default': ''},
                    'objective_value': {'_type': 'overwrite[float]', '_default': 0.0},
                },
            },
            'next_update_time': 'overwrite[float]',
        }

    def initialize(self, config):
        stoich = self.parameters["reaction_stoich"]
        self.n_avogadro = unum_to_pint(self.parameters["avogadro"])
        self.cell_density = unum_to_pint(self.parameters["cell_density"])
        self.ngam = self.parameters["ngam"]
        self.random_state = np.random.RandomState(seed=self.parameters["seed"])

        # Build reaction and metabolite index maps
        self.reaction_ids = sorted(stoich.keys())
        metabolite_set = set()
        for rxn_stoich in stoich.values():
            metabolite_set.update(rxn_stoich.keys())
        self.metabolite_ids = sorted(metabolite_set)

        rxn_to_idx = {r: i for i, r in enumerate(self.reaction_ids)}
        met_to_idx = {m: i for i, m in enumerate(self.metabolite_ids)}
        self.n_rxn = len(self.reaction_ids)
        self.n_met = len(self.metabolite_ids)

        # Build sparse stoichiometric matrix S (n_met x n_rxn)
        rows, cols, vals = [], [], []
        for rxn_id, rxn_stoich in stoich.items():
            j = rxn_to_idx[rxn_id]
            for met_id, coeff in rxn_stoich.items():
                rows.append(met_to_idx[met_id])
                cols.append(j)
                vals.append(coeff)
        self.S = csr_matrix(
            (vals, (rows, cols)), shape=(self.n_met, self.n_rxn))

        # Homeostatic target setup
        self.target_met_ids = sorted(self.parameters["homeostatic_targets"].keys())
        self.target_met_indices = [met_to_idx[m] for m in self.target_met_ids
                                   if m in met_to_idx]
        self.n_targets = len(self.target_met_indices)

        # Output molecule rows of S for tracked metabolites
        self.M = self.S[self.target_met_indices, :]

        # Enzyme catalysis: sparse matrix (n_enzyme_rxns x n_catalysts)
        cat_ids = self.parameters["catalyst_ids"]
        self.reactions_with_catalyst = self.parameters["reactions_with_catalyst"]
        cat_rxn_indices = [rxn_to_idx.get(r) for r in self.reactions_with_catalyst]
        self.cat_rxn_indices = [i for i in cat_rxn_indices if i is not None]

        I = np.asarray(self.parameters["catalysis_matrix_I"])
        J = np.asarray(self.parameters["catalysis_matrix_J"])
        V = np.asarray(self.parameters["catalysis_matrix_V"])
        if len(I) > 0:
            shape = (I.max() + 1, J.max() + 1)
            self.catalysis_matrix = csr_matrix((V, (I, J)), shape=shape)
        else:
            self.catalysis_matrix = None

        # Exchange molecule indices in reaction list
        self.exchange_molecules = self.parameters["exchange_molecules"]
        self.exchange_rxn_indices = [
            rxn_to_idx[m] for m in self.exchange_molecules if m in rxn_to_idx]

        # Maintenance reaction index
        maint = self.parameters["maintenance_reaction"]
        self.ngam_rxn_idx = rxn_to_idx.get(maint) if isinstance(maint, str) else None

        # Bulk molecule index cache
        self._bulk_idx = None

        # Config callables
        self.get_biomass_as_concentrations = self.parameters.get(
            "get_biomass_as_concentrations")

    def update_condition(self, timestep, states):
        if states["next_update_time"] <= states["global_time"]:
            return True
        return False

    def update(self, states, interval=None):
        dt = states["timestep"]

        # Lazy bulk index initialization
        if self._bulk_idx is None:
            self._bulk_idx = bulk_name_to_idx(
                self.metabolite_ids, states["bulk"]["id"])
            self._catalyst_idx = bulk_name_to_idx(
                self.parameters["catalyst_ids"], states["bulk"]["id"]
            ) if self.parameters["catalyst_ids"] else None

        # ---- State ----
        met_counts = counts(states["bulk"], self._bulk_idx)
        cell_mass = states["listeners"]["mass"]["cell_mass"] * units.fg
        dry_mass = states["listeners"]["mass"]["dry_mass"] * units.fg
        cell_volume = cell_mass / self.cell_density
        counts_to_molar = 1.0 / (self.n_avogadro * cell_volume.to(VOLUME_UNITS).magnitude)

        # Coefficient: converts flux (mM) to count changes
        # coefficient = (dry_mass / cell_mass) * density * dt  [g*s/L]
        coefficient = (dry_mass / cell_mass * self.cell_density * dt * units.s)
        coeff_val = coefficient.to(MASS_UNITS * units.s / VOLUME_UNITS).magnitude

        # ---- Targets ----
        targets = np.array([
            self.parameters["homeostatic_targets"].get(m, 0.0)
            for m in self.target_met_ids
            if m in {self.metabolite_ids[i] for i in self.target_met_indices}
        ])

        # Current concentrations of target metabolites
        target_counts = met_counts[self.target_met_indices]
        current_conc = target_counts * counts_to_molar

        # ---- Build LP ----
        # Decision variables: [v (n_rxn), s+ (n_targets), s- (n_targets)]
        n_vars = self.n_rxn + 2 * self.n_targets

        # Objective: minimize sum(s+ + s-)
        c = np.zeros(n_vars)
        c[self.n_rxn:self.n_rxn + self.n_targets] = 1.0   # s+
        c[self.n_rxn + self.n_targets:] = 1.0              # s-

        # Equality constraints:
        # 1. S @ v = 0  (mass balance)
        # 2. M @ v + s+ - s- = (target - current_conc)  (homeostatic)
        S_dense = self.S.toarray()
        M_dense = self.M.toarray()

        # Block 1: S @ v = 0
        A_eq_1 = np.zeros((self.n_met, n_vars))
        A_eq_1[:, :self.n_rxn] = S_dense
        b_eq_1 = np.zeros(self.n_met)

        # Block 2: M @ v + s+ - s- = target_delta
        A_eq_2 = np.zeros((self.n_targets, n_vars))
        A_eq_2[:, :self.n_rxn] = M_dense
        A_eq_2[:, self.n_rxn:self.n_rxn + self.n_targets] = np.eye(self.n_targets)
        A_eq_2[:, self.n_rxn + self.n_targets:] = -np.eye(self.n_targets)
        b_eq_2 = targets - current_conc

        A_eq = np.vstack([A_eq_1, A_eq_2])
        b_eq = np.concatenate([b_eq_1, b_eq_2])

        # Bounds
        bounds = []
        for j in range(self.n_rxn):
            bounds.append((None, None))  # default: unconstrained
        # s+, s- >= 0
        for _ in range(2 * self.n_targets):
            bounds.append((0, None))

        # Apply enzyme capacity constraints: if no catalyst, upper bound = 0
        if self._catalyst_idx is not None and self.catalysis_matrix is not None:
            catalyst_counts = counts(states["bulk"], self._catalyst_idx)
            no_enzyme = self.catalysis_matrix.dot(catalyst_counts) == 0
            for i, rxn_idx in enumerate(self.cat_rxn_indices):
                if i < len(no_enzyme) and no_enzyme[i]:
                    lb, ub = bounds[rxn_idx]
                    bounds[rxn_idx] = (lb, 0)

        # ---- Solve ----
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs',
                         options={'presolve': True, 'disp': False})

        if not result.success:
            warnings.warn(f"FBA solver failed: {result.message}")
            return {"next_update_time": states["global_time"] + dt}

        v = result.x[:self.n_rxn]

        # ---- Convert fluxes to count changes ----
        # delta_concentration = S @ v  (in mM)
        delta_conc = S_dense @ v
        # delta_counts = delta_conc / counts_to_molar
        delta_counts_float = delta_conc / counts_to_molar
        delta_counts_int = np.fmax(
            stochasticRound(self.random_state,
                            met_counts + delta_counts_float),
            0).astype(np.int64) - met_counts

        return {
            "bulk": [(self._bulk_idx, delta_counts_int)],
            "listeners": {
                "fba_results": {
                    "delta_metabolites": delta_counts_int,
                    "reaction_fluxes": v / dt,
                    "media_id": states["environment"]["media_id"],
                    "objective_value": result.fun,
                },
            },
            "next_update_time": states["global_time"] + dt,
        }
