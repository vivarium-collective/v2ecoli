"""
==========
Metabolism
==========

Encodes molecular simulation of microbial metabolism using flux-balance
analysis (FBA).

Mathematical Model
------------------
**Flux Balance Analysis**

The core optimization problem solved each timestep:

    max  c^T v
    s.t. S v = 0        (steady-state mass balance)
         v_lb <= v <= v_ub   (flux bounds)

where:
    - S: stoichiometric matrix (metabolites x reactions)
    - v: reaction flux vector (mmol/g_DCW/h)
    - c: objective coefficients (biomass production)
    - v_lb, v_ub: lower/upper flux bounds from:
      * environment exchange constraints (nutrient availability)
      * enzyme capacity constraints (kinetic limits)
      * maintenance energy requirements (NGAM)

**Unit conversion chain**

Fluxes from FBA (mmol/g_DCW/h) are converted to molecule count changes:

    delta_counts = stochasticRound(
        flux * dry_mass * dt / (molecular_weight * conversion_factor)
    )

The conversion constants defined at module level:
    - COUNTS_UNITS = mmol
    - VOLUME_UNITS = L
    - MASS_UNITS = g
    - TIME_UNITS = s
    - CONC_UNITS = mmol/L  (concentration)
    - GDCW_BASIS = mmol/g/h  (FBA flux basis)

**ppGpp regulation** (optional)

When include_ppgpp=True, ppGpp concentration reduces the growth rate
objective, coupling transcriptional regulation to metabolic capacity.

**Kinetic constraints** (optional)

When USE_KINETICS=True, amino acid supply rates from polypeptide
elongation are used as additional flux constraints, coupling
translation demand to metabolic output.

NOTE:
- Metabolism runs after all other processes have completed and internal
  states have been updated (deriver-like, no partitioning necessary)
"""

from typing import Any, Optional
import warnings

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from unum import Unum
from v2ecoli.library.unit_defs import units as vivunits

from v2ecoli.library.ecoli_step import EcoliStep as Step
# topology_registry removed
from v2ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts, listener_schema
from wholecell.utils import units
from wholecell.utils.random import stochasticRound
from wholecell.utils.modular_fba import FluxBalanceAnalysis
REVERSE_TAG = " (reverse)"


# Register default topology for this process, associating it with process name
NAME = "ecoli-metabolism"
TOPOLOGY = {
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "environment": ("environment",),
    # All per-step pre-FBA work (counts_to_molar, coefficient, conc_updates,
    # aa_uptake_package, bulk-count reads, aa_targets drift, ...) has moved
    # to MetabolicKinetics, which writes this store before we run.
    "metabolism_inputs": ("process_state", "metabolism_inputs"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
    "next_update_time": ("next_update_time", "metabolism"),
}

# Unit conversion constants for FBA flux -> molecule count conversion:
#   flux (mmol/gDCW/h) * dry_mass (g) * dt (s) / (3600 s/h) -> mmol
#   mmol * N_A -> molecule count
COUNTS_UNITS = units.mmol          # internal amount basis
VOLUME_UNITS = units.L             # volume basis
MASS_UNITS = units.g               # mass basis (for dry cell weight)
TIME_UNITS = units.s               # simulation time basis
CONC_UNITS = COUNTS_UNITS / VOLUME_UNITS   # mmol/L = mM
CONVERSION_UNITS = MASS_UNITS * TIME_UNITS / VOLUME_UNITS  # g*s/L
GDCW_BASIS = units.mmol / units.g / units.h  # FBA flux units

USE_KINETICS = True


class Metabolism(Step):
    """Metabolism Process

    Encodes molecular simulation of microbial metabolism using FBA.
    Runs as a time-driven process (not partitioned).
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'aa_exchange_names': {'_type': 'list[string]', '_default': []},
        'aa_names': {'_type': 'list[string]', '_default': []},
        'aa_targets_not_updated': {'_type': 'any', '_default': set()},
        'amino_acid_ids': {'_type': 'map', '_default': {}},
        'avogadro': {'_type': 'unum', '_default': 6.02214076e+23},
        'base_reaction_ids': {'_type': 'list[string]', '_default': []},
        'cell_density': {'_type': 'unum', '_default': 1100.0},
        'cell_dry_mass_fraction': {'_type': 'float', '_default': 0.3},
        'dark_atp': {'_type': 'unum', '_default': 33.565052868380675},
        'doubling_time': {'_type': 'unum', '_default': 44.0},
        'exchange_data_from_media': {'_type': 'method', '_default': None},
        'exchange_molecules': {'_type': 'list[string]', '_default': []},
        'fba_reaction_ids_to_base_reaction_ids': {'_type': 'list[string]', '_default': []},
        'get_biomass_as_concentrations': {'_type': 'method', '_default': None},
        'get_import_constraints': {'_type': 'method', '_default': None},
        'get_masses': {'_type': 'method', '_default': None},
        'get_ppGpp_conc': {'_type': 'method', '_default': None},
        'import_constraint_threshold': {'_type': 'integer', '_default': 0},
        'imports': {'_type': 'map[node]', '_default': {}},
        'include_ppgpp': {'_type': 'boolean', '_default': False},
        'mechanistic_aa_transport': {'_type': 'boolean', '_default': False},
        'media_id': {'_type': 'string', '_default': 'minimal'},
        'metabolism': {'_type': 'map[node]', '_default': {}},
        'ngam': {'_type': 'unum', '_default': 8.39},
        'nutrientToDoublingTime': {'_type': 'map[float]', '_default': {}},
        'ppgpp_id': {'_type': 'string', '_default': 'ppgpp'},
        'removed_aa_uptake': {'_type': 'list[string]', '_default': []},
        'seed': {'_type': 'integer', '_default': 0},
        'time_step': {'_type': 'integer', '_default': 1},
        'use_trna_charging': {'_type': 'boolean', '_default': False},
    }

    def inputs(self):
        return {
            # bulk is still read at t=0 to resolve the metabolite_idx used
            # for writing back FBA deltas. Runtime counts come in via the
            # metabolism_inputs port.
            'bulk': {'_type': 'bulk_array', '_default': []},
            'environment': {
                'exchange_data': {
                    'constrained': 'map[float]',
                    'unconstrained': 'list[string]',
                },
            },
            # Match MetabolicKinetics.outputs() — overwrite semantics so
            # reads see the upstream step's fresh values, not an
            # accumulating sum.
            'metabolism_inputs': {
                'current_media_id': {'_type': 'overwrite[string]', '_default': ''},
                'counts_to_molar_mM': {'_type': 'overwrite[float]', '_default': 1.0},
                'coefficient_gsL': {'_type': 'overwrite[float]', '_default': 0.0},
                'translation_gtp': {'_type': 'overwrite[float]', '_default': 0.0},
                'conc_updates_mM': {'_type': 'overwrite[map[float]]', '_default': {}},
                'aa_uptake_present': {'_type': 'overwrite[boolean]', '_default': False},
                'aa_uptake_rates': {'_type': 'overwrite[array[float]]', '_default': []},
                'aa_uptake_names': {'_type': 'overwrite[list[string]]', '_default': []},
                'aa_uptake_force': {'_type': 'overwrite[boolean]', '_default': True},
                'metabolite_counts': {'_type': 'overwrite[array[integer]]', '_default': []},
                'catalyst_counts': {'_type': 'overwrite[array[integer]]', '_default': []},
                'kinetic_enzyme_counts': {'_type': 'overwrite[array[integer]]', '_default': []},
                'kinetic_substrate_counts': {'_type': 'overwrite[array[integer]]', '_default': []},
                'disallowed_imports': {'_type': 'overwrite[list[string]]', '_default': []},
            },
            'global_time': {'_type': 'float[s]', '_default': 0.0},
            'timestep': {'_type': 'integer[s]', '_default': 1},
            'next_update_time': {'_type': 'float[s]', '_default': 1.0},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            # Overwrite semantics: environment_update reads this as a
            # per-step snapshot of counts exchanged with the environment.
            # The default add-merge for `map[float]` would accumulate
            # every step's counts and cause runaway depletion.
            'environment': {'exchange': {
                '_type': 'overwrite[map[float]]', '_default': {}}},
            'listeners': {
                'fba_results': {
                    # Coefficient for flux→delta conversion (g*s/L)
                    'coefficient': {'_type': 'overwrite[float[g*s/L]]', '_default': 0.0},
                    # GTP from polypeptide elongation (count, dimensionless)
                    'translation_gtp': {'_type': 'overwrite[float]', '_default': 0.0},
                    # Concentration updates per molecule (mM = mmol/L)
                    'conc_updates': {'_type': 'overwrite[array[float[mM]]]', '_default': []},
                    # Homeostatic target concentrations (mM)
                    'target_concentrations': {'_type': 'overwrite[array[float[mM]]]', '_default': []},
                    # FBA solver outputs (mostly mmol/g/h flux units, but
                    # stored without units in the listener history)
                    'reaction_fluxes': {'_type': 'overwrite[array[float]]', '_default': []},
                    'external_exchange_fluxes': {'_type': 'overwrite[array[float]]', '_default': []},
                    'base_reaction_fluxes': {'_type': 'overwrite[array[float]]', '_default': []},
                    'objective_value': {'_type': 'overwrite[float]', '_default': 0.0},
                    'shadow_prices': {'_type': 'overwrite[array[float]]', '_default': []},
                    'reduced_costs': {'_type': 'overwrite[array[float]]', '_default': []},
                    'homeostatic_objective_values': {'_type': 'overwrite[array[float]]', '_default': []},
                    'kinetic_objective_values': {'_type': 'overwrite[array[float]]', '_default': []},
                    'fba_mass_exchange_out': {'_type': 'overwrite[float]', '_default': 0.0},
                    # Counts (dimensionless)
                    'catalyst_counts': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'delta_metabolites': {'_type': 'overwrite[array[integer]]', '_default': []},
                    # Identifiers and constraint sets — flexible
                    'media_id': {'_type': 'overwrite[string]', '_default': ''},
                    'unconstrained_molecules': {'_type': 'overwrite[list[string]]', '_default': []},
                    'constrained_molecules': {'_type': 'overwrite[map[float]]', '_default': {}},
                    'uptake_constraints': {'_type': 'overwrite[array[float]]', '_default': []},
                },
                'enzyme_kinetics': {
                    # Counts→molar conversion (mmol/L = mM)
                    'counts_to_molar': {'_type': 'overwrite[float[mM]]', '_default': 1.0},
                    # Counts (dimensionless)
                    'metabolite_counts_init': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'metabolite_counts_final': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'enzyme_counts_init': {'_type': 'overwrite[array[integer]]', '_default': []},
                    # Fluxes (mmol/L/s) — leave plain for now since
                    # the listener stores per-timestep values, not rates
                    'actual_fluxes': {'_type': 'overwrite[array[float]]', '_default': []},
                    'target_fluxes': {'_type': 'overwrite[array[float]]', '_default': []},
                    'target_fluxes_upper': {'_type': 'overwrite[array[float]]', '_default': []},
                    'target_fluxes_lower': {'_type': 'overwrite[array[float]]', '_default': []},
                    'target_aa_conc': {'_type': 'overwrite[array[float[mM]]]', '_default': []},
                },
            },
            'next_update_time': 'overwrite[float]',
        }



    def initialize(self, config):
        # FBA-only state. Kinetics/pre-FBA inputs come in via the
        # metabolism_inputs port (see MetabolicKinetics).
        self.get_import_constraints = self.parameters["get_import_constraints"]
        self.current_timeline = self.parameters["current_timeline"]
        self.media_id = self.parameters["media_id"]

        self.model = FluxBalanceAnalysisModel(
            self.parameters,
            timeline=self.current_timeline,
            include_ppgpp=self.parameters["include_ppgpp"],
        )

        # Mechanistic biomass mass-balance: bound the homeostatic
        # `quadFractionFromUnity` slack pseudofluxes so the LP can
        # undershoot a target (cell produces less biomass than wanted
        # when starved) but cannot overshoot by conjuring metabolites.
        # See modular_fba.py:596 for where the slacks are created with
        # [-inf, +inf] bounds by default, and the nutrient-growth
        # report's "How is biomass being manufactured?" section for
        # the derivation.
        # Experimental: capping the aboveUnity homeostatic slack makes
        # the LP INFEASIBLE at t=0 in the current wholecell model (cell's
        # initial internal metabolite state is far enough from the
        # biomass targets that some pseudoFlux > 1 is required to
        # rebalance). Default off until we have a more flexible
        # objective. See report's "How is biomass being manufactured?"
        # section + the user-facing discussion of LP-alternative paths.
        self._mass_balance_enforced = False
        if self.parameters.get("enforce_biomass_mass_balance", False):
            self._enforce_biomass_mass_balance()

        # Fixed list of molecules reported in fba_results.conc_updates. The
        # set depends only on ParCa state, not on runtime, so we compute it
        # once here even though conc_updates themselves arrive through a port.
        nutrient_to_dt = self.parameters["nutrientToDoublingTime"]
        doubling_time = nutrient_to_dt.get(
            self.media_id, nutrient_to_dt["minimal"]
        )
        update_molecules = list(
            self.model.getBiomassAsConcentrations(doubling_time).keys()
        )
        if self.parameters["use_trna_charging"]:
            aa_not_updated = self.parameters["aa_targets_not_updated"]
            update_molecules += [
                aa for aa in self.parameters["aa_names"]
                if aa not in aa_not_updated
            ]
            update_molecules += list(
                self.parameters["linked_metabolites"].keys())
        if self.parameters["include_ppgpp"]:
            update_molecules += [self.model.ppgpp_id]
        self.conc_update_molecules = sorted(update_molecules)

        self.aa_names = self.parameters["aa_names"]  # listener ordering only

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Lazy bulk index for writing back delta_metabolites.
        self.metabolite_idx = None

        # Get conversion matrix to compile individual fluxes in the FBA
        # solution to the fluxes of base reactions
        self.fba_reaction_ids = self.model.fba.getReactionIDs()
        self.base_reaction_ids = self.parameters["base_reaction_ids"]
        fba_reaction_ids_to_base_reaction_ids = self.parameters[
            "fba_reaction_ids_to_base_reaction_ids"
        ]
        self.externalMoleculeIDs = self.model.fba.getExternalMoleculeIDs()
        # ID → index map for zeroing out disallowed imports in O(1).
        self._ext_id_to_idx = {
            mid: i for i, mid in enumerate(self.externalMoleculeIDs)
        }
        self.outputMoleculeIDs = self.model.fba.getOutputMoleculeIDs()
        self.kineticTargetFluxNames = self.model.fba.getKineticTargetFluxNames()
        self.homeostaticTargetMolecules = self.model.fba.getHomeostaticTargetMolecules()
        fba_reaction_id_to_index = {
            rxn_id: i for (i, rxn_id) in enumerate(self.fba_reaction_ids)
        }
        base_reaction_id_to_index = {
            rxn_id: i for (i, rxn_id) in enumerate(self.base_reaction_ids)
        }
        base_rxn_indexes = []
        fba_rxn_indexes = []
        v = []

        for fba_rxn_id in self.fba_reaction_ids:
            base_rxn_id = fba_reaction_ids_to_base_reaction_ids[fba_rxn_id]
            base_rxn_indexes.append(base_reaction_id_to_index[base_rxn_id])
            fba_rxn_indexes.append(fba_reaction_id_to_index[fba_rxn_id])
            if fba_rxn_id.endswith(REVERSE_TAG):
                v.append(-1)
            else:
                v.append(1)

        base_rxn_indexes = np.array(base_rxn_indexes)
        fba_rxn_indexes = np.array(fba_rxn_indexes)
        v = np.array(v)
        shape = (len(self.base_reaction_ids), len(self.fba_reaction_ids))

        self.reaction_mapping_matrix = csr_matrix(
            (v, (base_rxn_indexes, fba_rxn_indexes)), shape=shape
        )

    def _enforce_biomass_mass_balance(self):
        """Cap the homeostatic "above unity" slack pseudofluxes at 0.

        In modular_fba's homeostatic formulation with a linear
        objective (``quadratic_objective=False``):

            pseudoFlux = 1 + aboveUnity − belowUnity

        where ``pseudoFlux`` is the rate at which a target metabolite
        is consumed into biomass. ``aboveUnity`` represents over-
        shooting the target ("produce more biomass than asked");
        ``belowUnity`` represents undershooting. Both default to
        ``[0, +inf]`` on flux.

        Over-shooting is where the "free mass" comes from — the LP
        inflates biomass production without a mass-balanced carbon
        source. Capping ``aboveUnity`` at 0 forces ``pseudoFlux ≤ 1``,
        i.e. the cell can only meet or miss a target, never exceed
        it. Biomass production naturally trails off under
        nutrient starvation.

        ``belowUnity`` is left at its default bound — the cell is
        expected to produce less biomass than the homeostatic target
        when starved, which is the biologically correct behaviour.
        """
        fba = self.model.fba
        solver = fba._solver
        above_prefix = type(fba)._generatedID_fractionAboveUnityOut
        targets = fba.getHomeostaticTargetMolecules()
        known_flows = set(solver.getFlowNames())
        bounded = skipped = 0
        for mol in targets:
            flux_id = above_prefix + mol
            if flux_id not in known_flows:
                skipped += 1
                continue
            try:
                solver.setFlowBounds(flux_id, lowerBound=0.0, upperBound=0.0)
                bounded += 1
            except Exception:
                skipped += 1
        self._mass_balance_enforced = True
        self._mass_balance_info = {
            "bounded_slacks": bounded, "skipped_slacks": skipped,
            "target_molecules": len(targets),
        }

    def __getstate__(self):
        return self.parameters

    def __setstate__(self, state):
        self.__init__(state)
    def update_condition(self, timestep, states):
        """
        See :py:meth:`~ecoli.processes.partition.Requester.update_condition`.
        """
        if states["next_update_time"] <= states["global_time"]:
            if states["next_update_time"] < states["global_time"]:
                warnings.warn(
                    f"{self.name} updated at t="
                    f"{states['global_time']} instead of t="
                    f"{states['next_update_time']}. Decrease the "
                    "timestep for the global clock process for more "
                    "accurate timekeeping."
                )
            return True
        return False

    def update(self, states, interval=None):
        timestep = states.get('timestep', 1)
        return self._do_update(timestep, states)

    def _do_update(self, timestep, states):
        # Lazy bulk index for writing back delta_metabolites. Everything
        # else comes in via the metabolism_inputs port.
        if self.metabolite_idx is None:
            self.metabolite_idx = bulk_name_to_idx(
                self.model.metaboliteNamesFromNutrients, states["bulk"]["id"]
            )

        timestep = states["timestep"]
        mi = states["metabolism_inputs"]

        # Reconstruct Unum quantities at the boundary; the upstream step
        # passes plain floats to keep the port schema simple.
        counts_to_molar = float(mi["counts_to_molar_mM"]) * CONC_UNITS
        coefficient = float(mi["coefficient_gsL"]) * CONVERSION_UNITS
        current_media_id = mi["current_media_id"]

        unconstrained = set(states["environment"]["exchange_data"]["unconstrained"])
        constrained = states["environment"]["exchange_data"]["constrained"]

        metabolite_counts_init = np.asarray(mi["metabolite_counts"])
        catalyst_counts = np.asarray(mi["catalyst_counts"])
        kinetic_enzyme_counts = np.asarray(mi["kinetic_enzyme_counts"])
        kinetic_substrate_counts = np.asarray(mi["kinetic_substrate_counts"])

        conc_updates = dict(mi["conc_updates_mM"])

        if mi.get("aa_uptake_present", False):
            aa_uptake_package = (
                np.asarray(mi["aa_uptake_rates"]),
                np.asarray(mi["aa_uptake_names"]),
                bool(mi.get("aa_uptake_force", True)),
            )
        else:
            aa_uptake_package = None

        # Drive the FBA model: levels → bounds → targets → solve.
        self.model.set_molecule_levels(
            metabolite_counts_init,
            counts_to_molar,
            coefficient,
            current_media_id,
            unconstrained,
            constrained,
            conc_updates,
            aa_uptake_package,
        )

        # Mechanistic carbon-balance enforcement: after set_molecule_levels
        # has set the import-availability array, explicitly zero out every
        # FBA external molecule NOT on the wholecell's importExchange
        # allowlist. This blocks the LP from using obscure boundary
        # molecules (H2, CO2 fixation proxies, nucleotide scavenging, ...)
        # as free carbon/energy sources — biology doesn't permit them,
        # FBA stoichiometry alone does.
        disallowed = mi.get("disallowed_imports", []) or []
        if disallowed:
            idx = [self._ext_id_to_idx[m] for m in disallowed
                   if m in self._ext_id_to_idx]
            if idx:
                self.model.fba.setExternalMoleculeLevels(
                    np.zeros(len(idx)),
                    molecules=[self.externalMoleculeIDs[i] for i in idx])
        self.model.set_reaction_bounds(
            catalyst_counts,
            counts_to_molar,
            coefficient,
            float(mi["translation_gtp"]),
        )
        targets, upper_targets, lower_targets = self.model.set_reaction_targets(
            kinetic_enzyme_counts,
            kinetic_substrate_counts,
            counts_to_molar,
            timestep * units.s,
        )

        fba = self.model.fba
        fba.solve(3)

        delta_metabolites = (1 / counts_to_molar) * (
            CONC_UNITS * fba.getOutputMoleculeLevelsChange()
        )
        metabolite_counts_final = np.fmax(
            stochasticRound(
                self.random_state,
                metabolite_counts_init + delta_metabolites.asNumber(),
            ),
            0,
        ).astype(np.int64)
        delta_metabolites_final = metabolite_counts_final - metabolite_counts_init

        exchange_fluxes = CONC_UNITS * fba.getExternalExchangeFluxes()
        converted_exchange_fluxes = (exchange_fluxes / coefficient).asNumber(
            GDCW_BASIS
        )
        delta_nutrients = (
            ((1 / counts_to_molar) * exchange_fluxes).asNumber().astype(int)
        )

        unconstrained_out, constrained_out, uptake_constraints = (
            self.get_import_constraints(unconstrained, constrained, GDCW_BASIS)
        )

        reaction_fluxes = fba.getReactionFluxes() / timestep

        # Probe the FBA's built-in exchange-mass accounting. massExchangeOut
        # is wholecell's pseudoflux that tracks net mass in − out via
        # *exchange reactions only*. It does NOT reflect mass introduced
        # via the homeostatic quadFractionFromUnity slack fluxes, which
        # the LP uses freely to satisfy biomass targets without carbon
        # input. Surfacing it lets the report compare exchange-mass flow
        # to actual dry-mass growth; the gap = slack-induced "free mass".
        fba_mass_exchange_out = 0.0
        try:
            raw = fba._solver.getFlowRates(fba._massExchangeOutName)
            fba_mass_exchange_out = float(np.asarray(raw).flatten()[0])
        except Exception:
            pass

        return {
            "bulk": [(self.metabolite_idx, delta_metabolites_final)],
            "environment": {
                "exchange": {
                    str(molecule[:-3]): delta_nutrients[index]
                    for index, molecule in enumerate(self.externalMoleculeIDs)
                }
            },
            "listeners": {
                "fba_results": {
                    "media_id": current_media_id,
                    "conc_updates": [
                        conc_updates.get(m, 0) for m in self.conc_update_molecules
                    ],
                    "catalyst_counts": catalyst_counts,
                    "translation_gtp": float(mi["translation_gtp"]),
                    "coefficient": coefficient.asNumber(CONVERSION_UNITS),
                    "unconstrained_molecules": unconstrained_out,
                    "constrained_molecules": constrained_out,
                    "uptake_constraints": uptake_constraints,
                    "delta_metabolites": delta_metabolites_final,
                    "reaction_fluxes": reaction_fluxes,
                    "external_exchange_fluxes": converted_exchange_fluxes,
                    "objective_value": fba.getObjectiveValue(),
                    "shadow_prices": fba.getShadowPrices(
                        self.model.metaboliteNamesFromNutrients
                    ),
                    "reduced_costs": fba.getReducedCosts(fba.getReactionIDs()),
                    "target_concentrations": [
                        self.model.homeostatic_objective[mol]
                        for mol in fba.getHomeostaticTargetMolecules()
                    ],
                    "homeostatic_objective_values":
                        fba.getHomeostaticObjectiveValues(),
                    "kinetic_objective_values": fba.getKineticObjectiveValues(),
                    "base_reaction_fluxes": self.reaction_mapping_matrix.dot(
                        reaction_fluxes
                    ),
                    # Built-in wholecell mass accounting (exchange only).
                    # Mismatch vs. real dry-mass growth = slack-induced
                    # "free mass" from homeostatic pseudofluxes.
                    "fba_mass_exchange_out": fba_mass_exchange_out,
                },
                "enzyme_kinetics": {
                    "metabolite_counts_init": metabolite_counts_init,
                    "metabolite_counts_final": metabolite_counts_final,
                    "enzyme_counts_init": kinetic_enzyme_counts,
                    "counts_to_molar": counts_to_molar.asNumber(CONC_UNITS),
                    "actual_fluxes": fba.getReactionFluxes(
                        self.model.kinetics_constrained_reactions
                    ) / timestep,
                    "target_fluxes": targets / timestep,
                    "target_fluxes_upper": upper_targets / timestep,
                    "target_fluxes_lower": lower_targets / timestep,
                    # aa_targets now live on MetabolicKinetics; report zeros
                    # here since this listener is not the authoritative source.
                    "target_aa_conc": [0.0] * len(self.aa_names),
                },
            },
            "next_update_time": states["global_time"] + states["timestep"],
        }

class FluxBalanceAnalysisModel(object):
    """
    Metabolism model that solves an FBA problem with modular_fba.
    """

    def __init__(
        self,
        parameters: dict[str, Any],
        timeline: tuple[tuple[int, str], ...],
        include_ppgpp: bool = True,
    ):
        """
        Args:
            parameters: parameters from simulation data
            timeline: timeline for nutrient changes during simulation
                (time of change, media ID), by default [(0.0, 'minimal')]
            include_ppgpp: if True, ppGpp is included as a concentration target
        """
        nutrients = timeline[0][1]

        # Local sim_data references
        metabolism = parameters["metabolism"]
        self.stoichiometry = metabolism.reaction_stoich
        self.maintenance_reaction = metabolism.maintenance_reaction

        # Load constants
        self.ngam = parameters["ngam"]
        gam = parameters["dark_atp"] * parameters["cell_dry_mass_fraction"]

        self.exchange_constraints = metabolism.exchange_constraints

        self._biomass_concentrations = {}  # type: dict
        self.getBiomassAsConcentrations = parameters["get_biomass_as_concentrations"]

        # Include ppGpp concentration target in objective if not handled
        # kinetically in other processes
        self.ppgpp_id = parameters["ppgpp_id"]
        self.getppGppConc = parameters["get_ppGpp_conc"]

        # go through all media in the timeline and add to metaboliteNames
        metaboliteNamesFromNutrients = set()
        conc_from_nutrients = (
            metabolism.concentration_updates.concentrations_based_on_nutrients
        )
        if include_ppgpp:
            metaboliteNamesFromNutrients.add(self.ppgpp_id)
        for time, media_id in timeline:
            exchanges = parameters["exchange_data_from_media"](media_id)
            metaboliteNamesFromNutrients.update(
                conc_from_nutrients(imports=exchanges["importExchangeMolecules"])
            )
        self.metaboliteNamesFromNutrients = list(sorted(metaboliteNamesFromNutrients))
        exchange_molecules = sorted(parameters["exchange_molecules"])
        molecule_masses = dict(
            zip(
                exchange_molecules,
                parameters["get_masses"](exchange_molecules).asNumber(
                    MASS_UNITS / COUNTS_UNITS
                ),
            )
        )

        # Setup homeostatic objective concentration targets
        # Determine concentrations based on starting environment
        conc_dict = conc_from_nutrients(
            media_id=nutrients, imports=parameters["imports"]
        )
        doubling_time = parameters["doubling_time"]
        conc_dict.update(self.getBiomassAsConcentrations(doubling_time))
        if include_ppgpp:
            conc_dict[self.ppgpp_id] = self.getppGppConc(doubling_time)
        self.homeostatic_objective = dict(
            (key, conc_dict[key].asNumber(CONC_UNITS)) for key in conc_dict
        )

        # Include all concentrations that will be present in a sim for constant
        # length listeners
        for met in self.metaboliteNamesFromNutrients:
            if met not in self.homeostatic_objective:
                self.homeostatic_objective[met] = 0.0

        # Data structures to compute reaction bounds based on enzyme
        # presence/absence
        self.catalyst_ids = metabolism.catalyst_ids
        self.reactions_with_catalyst = metabolism.reactions_with_catalyst

        i = metabolism.catalysis_matrix_I
        j = metabolism.catalysis_matrix_J
        v = metabolism.catalysis_matrix_V
        shape = (i.max() + 1, j.max() + 1)
        self.catalysis_matrix = csr_matrix((v, (i, j)), shape=shape)

        # Function to compute reaction targets based on kinetic parameters and
        # molecule concentrations
        self.get_kinetic_constraints = metabolism.get_kinetic_constraints

        # Remove disabled reactions so they don't get included in the FBA
        # problem setup
        kinetic_constraint_reactions = metabolism.kinetic_constraint_reactions
        constraintsToDisable = metabolism.constraints_to_disable
        self.active_constraints_mask = np.array(
            [(rxn not in constraintsToDisable) for rxn in kinetic_constraint_reactions]
        )
        self.kinetics_constrained_reactions = list(
            np.array(kinetic_constraint_reactions)[self.active_constraints_mask]
        )

        self.kinetic_constraint_enzymes = metabolism.kinetic_constraint_enzymes
        self.kinetic_constraint_substrates = metabolism.kinetic_constraint_substrates

        # Set solver and kinetic objective weight (lambda)
        solver = metabolism.solver
        kinetic_objective_weight = metabolism.kinetic_objective_weight
        kinetic_objective_weight_in_range = metabolism.kinetic_objective_weight_in_range

        # Disable kinetics completely if weight is 0 or specified in file above
        if not USE_KINETICS or kinetic_objective_weight == 0:
            objective_type = "homeostatic"
            self.use_kinetics = False
            kinetic_objective_weight = 0
        else:
            objective_type = "homeostatic_kinetics_mixed"
            self.use_kinetics = True

        # Set up FBA solver
        # reactionRateTargets value is just for initialization, it gets reset
        # each timestep during evolveState
        fba_options = {
            "reactionStoich": metabolism.reaction_stoich,
            "externalExchangedMolecules": exchange_molecules,
            "objective": self.homeostatic_objective,
            "objectiveType": objective_type,
            "objectiveParameters": {
                "kineticObjectiveWeight": kinetic_objective_weight,
                "kinetic_objective_weight_in_range": kinetic_objective_weight_in_range,
                "reactionRateTargets": {
                    reaction: 1 for reaction in self.kinetics_constrained_reactions
                },
                "oneSidedReactionTargets": [],
            },
            "moleculeMasses": molecule_masses,
            # The "inconvenient constant"--limit secretion (e.g., of CO2)
            "secretionPenaltyCoeff": metabolism.secretion_penalty_coeff,
            "solver": solver,
            "maintenanceCostGAM": gam.asNumber(COUNTS_UNITS / MASS_UNITS),
            "maintenanceReaction": metabolism.maintenance_reaction,
        }
        self.fba = FluxBalanceAnalysis(**fba_options)

        self.metabolite_names = {
            met: i for i, met in enumerate(self.fba.getOutputMoleculeIDs())
        }
        self.aa_names_no_location = [x[:-3] for x in parameters["amino_acid_ids"]]

    def update_external_molecule_levels(
        self,
        objective: dict[str, Unum],
        metabolite_concentrations: Unum,
        external_molecule_levels: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Limit amino acid uptake to what is needed to meet concentration
        objective to prevent use as carbon source, otherwise could be used
        as an infinite nutrient source.

        Args:
            objective: homeostatic objective for internal
                molecules (molecule ID: concentration in counts/volume units)
            metabolite_concentrations: concentration for each
                molecule in metabolite_names
            external_molecule_levels: current limits on
                external molecule availability

        Returns:
            Updated limits on external molecule availability

        TODO(wcEcoli):
            determine rate of uptake so that some amino acid uptake can
            be used as a carbon/nitrogen source
        """

        external_exchange_molecule_ids = self.fba.getExternalMoleculeIDs()
        for aa in self.aa_names_no_location:
            if aa + "[p]" in external_exchange_molecule_ids:
                idx = external_exchange_molecule_ids.index(aa + "[p]")
            elif aa + "[c]" in external_exchange_molecule_ids:
                idx = external_exchange_molecule_ids.index(aa + "[c]")
            else:
                continue

            conc_diff = objective[aa + "[c]"] - metabolite_concentrations[
                self.metabolite_names[aa + "[c]"]
            ].asNumber(CONC_UNITS)
            if conc_diff < 0:
                conc_diff = 0

            if external_molecule_levels[idx] > conc_diff:
                external_molecule_levels[idx] = conc_diff

        return external_molecule_levels

    def set_molecule_levels(
        self,
        metabolite_counts: npt.NDArray[np.int64],
        counts_to_molar: Unum,
        coefficient: Unum,
        current_media_id: str,
        unconstrained: set[str],
        constrained: set[str],
        conc_updates: dict[str, Unum],
        aa_uptake_package: Optional[
            tuple[npt.NDArray[np.float64], npt.NDArray[np.str_], bool]
        ] = None,
    ):
        """
        Set internal and external molecule levels available to the FBA solver.

        Args:
            metabolite_counts: counts for each metabolite with a concentration target
            counts_to_molar: conversion from counts to molar (counts/volume)
            coefficient: coefficient to convert from mmol/g DCW/hr to mM basis
                (mass*time/volume)
            current_media_id: ID of current media
            unconstrained: molecules that have unconstrained import
            constrained: molecules (keys) and their limited max uptake rates
                (mol / mass / time)
            conc_updates: updates to concentrations targets for molecules (mmol/L)
            aa_uptake_package: (uptake rates, amino acid names, force levels),
                determines whether to set hard uptake rates
        """

        # Update objective from media exchanges
        external_molecule_levels, objective = self.exchange_constraints(
            self.fba.getExternalMoleculeIDs(),
            coefficient,
            CONC_UNITS,
            current_media_id,
            unconstrained,
            constrained,
            conc_updates,
        )
        # exchange_constraints occasionally emits objective entries for
        # molecules that weren't part of the FBA's static homeostatic
        # target registry (built at __init__ from the full media
        # timeline). update_homeostatic_targets raises on those; filter
        # them out here. This is defensive — only the pre-registered
        # targets are driving biomass anyway.
        if not hasattr(self, "_homeostatic_target_set"):
            self._homeostatic_target_set = set(
                self.fba.getHomeostaticTargetMolecules())
        objective = {k: v for k, v in objective.items()
                     if k in self._homeostatic_target_set}
        self.fba.update_homeostatic_targets(objective)
        self.homeostatic_objective = {**self.homeostatic_objective, **objective}

        # Internal concentrations
        metabolite_conc = counts_to_molar * metabolite_counts
        self.fba.setInternalMoleculeLevels(metabolite_conc.asNumber(CONC_UNITS))

        # External concentrations
        external_molecule_levels = self.update_external_molecule_levels(
            objective, metabolite_conc, external_molecule_levels
        )
        self.fba.setExternalMoleculeLevels(external_molecule_levels)

        if aa_uptake_package:
            levels, molecules, force = aa_uptake_package
            self.fba.setExternalMoleculeLevels(
                levels, molecules=molecules, force=force, allow_export=True
            )

    def set_reaction_bounds(
        self,
        catalyst_counts: npt.NDArray[np.int64],
        counts_to_molar: Unum,
        coefficient: Unum,
        gtp_to_hydrolyze: float,
    ):
        """
        Set reaction bounds for constrained reactions in the FBA object.

        Args:
            catalyst_counts: counts of enzyme catalysts
            counts_to_molar: conversion from counts to molar (counts/volume)
            coefficient: coefficient to convert from mmol/g DCW/hr to mM basis
                (mass*time/volume)
            gtp_to_hydrolyze: number of GTP molecules to hydrolyze to
                account for consumption in translation
        """

        # Maintenance reactions
        # Calculate new NGAM
        flux = (self.ngam * coefficient).asNumber(CONC_UNITS)
        self.fba.setReactionFluxBounds(
            self.fba._reactionID_NGAM,
            lowerBounds=flux,
            upperBounds=flux,
        )

        # Calculate GTP usage based on how much was needed in polypeptide
        # elongation in previous step.
        flux = (counts_to_molar * gtp_to_hydrolyze).asNumber(CONC_UNITS)
        self.fba.setReactionFluxBounds(
            self.fba._reactionID_polypeptideElongationEnergy,
            lowerBounds=flux,
            upperBounds=flux,
        )

        # Set hard upper bounds constraints based on enzyme presence
        # (infinite upper bound) or absence (upper bound of zero)
        reaction_bounds = np.inf * np.ones(len(self.reactions_with_catalyst))
        no_rxn_mask = self.catalysis_matrix.dot(catalyst_counts) == 0
        reaction_bounds[no_rxn_mask] = 0
        self.fba.setReactionFluxBounds(
            self.reactions_with_catalyst,
            upperBounds=reaction_bounds,
            raiseForReversible=False,
        )

    def set_reaction_targets(
        self,
        kinetic_enzyme_counts: npt.NDArray[np.int64],
        kinetic_substrate_counts: npt.NDArray[np.int64],
        counts_to_molar: Unum,
        time_step: Unum,
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        """
        Set reaction targets for constrained reactions in the FBA object.

        Args:
            kinetic_enzyme_counts: counts of enzymes used in kinetic constraints
            kinetic_substrate_counts: counts of substrates used in kinetic
                constraints
            counts_to_molar: conversion from counts to molar (counts/volume)
            time_step: current time step (time)

        Returns:
            3-element tuple containing

            - **mean_targets**: mean target for each constrained reaction
            - **upper_targets**: upper target limit for each constrained reaction
            - **lower_targets**: lower target limit for each constrained reaction
        """

        if self.use_kinetics:
            enzyme_conc = counts_to_molar * kinetic_enzyme_counts
            substrate_conc = counts_to_molar * kinetic_substrate_counts

            # Set target fluxes for reactions based on their most relaxed
            # constraint
            reaction_targets = self.get_kinetic_constraints(enzyme_conc, substrate_conc)

            # Calculate reaction flux target for current time step
            targets = (time_step * reaction_targets).asNumber(CONC_UNITS)[
                self.active_constraints_mask, :
            ]
            lower_targets = targets[:, 0]
            mean_targets = targets[:, 1]
            upper_targets = targets[:, 2]

            # Set kinetic targets only if kinetics is enabled
            self.fba.set_scaled_kinetic_objective(time_step.asNumber(units.s))
            self.fba.setKineticTarget(
                self.kinetics_constrained_reactions,
                mean_targets,
                lower_targets=lower_targets,
                upper_targets=upper_targets,
            )
        else:
            lower_targets = np.zeros(len(self.kinetics_constrained_reactions))
            mean_targets = np.zeros(len(self.kinetics_constrained_reactions))
            upper_targets = np.zeros(len(self.kinetics_constrained_reactions))

        return mean_targets, upper_targets, lower_targets


def test_metabolism_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    sim = EcoliSim.from_file()
    sim.max_duration = 2
    sim.raw_output = False
    sim.build_ecoli()
    sim.run()
    data = sim.query()
    reaction_fluxes = data["agents"]["0"]["listeners"]["fba_results"]["reaction_fluxes"]
    assert isinstance(reaction_fluxes[0], list)
    assert isinstance(reaction_fluxes[1], list)


if __name__ == "__main__":
    test_metabolism_listener()
