"""
Frozen partitioned process classes from commit 5d03dea.

This module contains the exact Logic + Requester + Evolver class definitions
for all 11 partitioned processes as they existed in the partitioned architecture.
These classes are preserved for comparison and backward-compatibility testing.

DO NOT MODIFY -- this is an archival snapshot.
"""

from typing import Any, Callable, Optional, Tuple, cast

from numba import njit
import numpy as np
import numpy.typing as npt
import scipy.sparse
import warnings
from scipy.integrate import solve_ivp
from stochastic_arrow import StochasticSystem
from unum import Unum

from process_bigraph import Step
from bigraph_schema.schema import Float, Overwrite

from v2ecoli.library.fitting import normalize
from v2ecoli.library.polymerize import buildSequences, polymerize, computeMassIncrease
from v2ecoli.library.random import stochasticRound
from v2ecoli.library.schema import (
    create_unique_indices,
    counts,
    attrs,
    bulk_name_to_idx,
    MetadataArray,
    zero_listener,
)
from v2ecoli.library.units import units
from v2ecoli.processes.metabolism import CONC_UNITS, TIME_UNITS
from v2ecoli.steps.partition import _protect_state, deep_merge, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from vivarium.library.units import units as vivunits


# ======================================================================
# equilibrium
# ======================================================================

class EquilibriumLogic:
    """Equilibrium — shared state container for Requester/Evolver.

    molecule_names: list of molecules that are being iterated over size:94
    """

    name = "ecoli-equilibrium"
    topology = {"listeners": ("listeners",), "bulk": ("bulk",), "timestep": ("timestep",)}
    defaults = {
        "jit": False,
        "n_avogadro": 0.0,
        "cell_density": 0.0,
        "stoichMatrix": [[]],
        "fluxesAndMoleculesToSS": lambda counts, volume, avogadro, random, jit: (
            [],
            [],
        ),
        "moleculeNames": [],
        "seed": 0,
        "complex_ids": [],
        "reaction_ids": [],
    }

    # Constructor
    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

        # Simulation options
        # utilized in the fluxes and molecules function
        self.jit = self.parameters["jit"]

        # Get constants
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Create matrix and method
        # stoichMatrix: (94, 33), molecule counts are (94,).
        self.stoichMatrix = self.parameters["stoichMatrix"]

        # fluxesAndMoleculesToSS: solves ODES to get to steady state based off
        # of cell density, volumes and molecule counts
        self.fluxesAndMoleculesToSS = self.parameters["fluxesAndMoleculesToSS"]

        self.product_indices = [
            idx for idx in np.where(np.any(self.stoichMatrix > 0, axis=1))[0]
        ]

        # Build views
        # moleculeNames: list of molecules that are being iterated over size: 94
        self.moleculeNames = self.parameters["moleculeNames"]
        self.molecule_idx = None

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.complex_ids = self.parameters["complex_ids"]
        self.reaction_ids = self.parameters["reaction_ids"]


class EquilibriumRequester(_SafeInvokeMixin, Step):
    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'timestep': Float(_default=self.process.parameters.get('timestep', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('timestep', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}
        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        p = self.process

        # --- inlined from calculate_request ---
        # At t=0, convert all strings to indices
        if p.molecule_idx is None:
            p.molecule_idx = bulk_name_to_idx(
                p.moleculeNames, state["bulk"]["id"]
            )

        # Get molecule counts
        moleculeCounts = counts(state["bulk"], p.molecule_idx)

        # Get cell mass and volume
        cellMass = (state["listeners"]["mass"]["cell_mass"] * units.fg).asNumber(
            units.g
        )
        cellVolume = cellMass / p.cell_density

        # Solve ODEs to steady state
        p.rxnFluxes, p.req = p.fluxesAndMoleculesToSS(
            moleculeCounts,
            cellVolume,
            p.n_avogadro,
            p.random_state,
            jit=p.jit,
        )

        # Request counts of molecules needed
        request = {"bulk": [(p.molecule_idx, p.req.astype(int))]}
        # --- end inlined ---

        p.request_set = True
        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request
        return result


class EquilibriumEvolver(_SafeInvokeMixin, Step):
    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('timestep', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('timestep', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}
        state = _protect_state(state)
        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)
        if not self.process.request_set:
            return {}
        timestep = state.get('timestep', 1.0)
        p = self.process

        # --- inlined from evolve_state ---
        # Get molecule counts
        moleculeCounts = counts(state["bulk"], p.molecule_idx)

        # Get counts of molecules allocated to this process
        rxnFluxes = p.rxnFluxes.copy()

        # If we didn't get allocated all the molecules we need, make do with
        # what we have (decrease reaction fluxes so that they make use of what
        # we have, but not more). Reduces at least one reaction every iteration
        # so the max number of iterations is the number of reactions that were
        # originally expected to occur + 1 to reach the break statement.
        max_iterations = int(np.abs(rxnFluxes).sum()) + 1
        for it in range(max_iterations):
            # Check if any metabolites will have negative counts with current reactions
            negative_metabolite_idxs = np.where(
                np.dot(p.stoichMatrix, rxnFluxes) + moleculeCounts < 0
            )[0]
            if len(negative_metabolite_idxs) == 0:
                break

            # Reduce reactions that consume metabolites with negative counts
            limited_rxn_stoich = p.stoichMatrix[negative_metabolite_idxs, :]
            fwd_rxn_idxs = np.where(
                np.logical_and(limited_rxn_stoich < 0, rxnFluxes > 0)
            )[1]
            rev_rxn_idxs = np.where(
                np.logical_and(limited_rxn_stoich > 0, rxnFluxes < 0)
            )[1]
            rxnFluxes[fwd_rxn_idxs] -= 1
            rxnFluxes[rev_rxn_idxs] += 1
            rxnFluxes[fwd_rxn_idxs] = np.fmax(0, rxnFluxes[fwd_rxn_idxs])
            rxnFluxes[rev_rxn_idxs] = np.fmin(0, rxnFluxes[rev_rxn_idxs])
        else:
            raise ValueError(
                "Could not get positive counts in equilibrium with allocated molecules."
            )

        # Increment changes in molecule counts
        deltaMolecules = np.dot(p.stoichMatrix, rxnFluxes).astype(int)

        update = {
            "bulk": [(p.molecule_idx, deltaMolecules)],
            "listeners": {
                "equilibrium_listener": {
                    "reaction_rates": deltaMolecules[p.product_indices]
                    / state["timestep"]
                }
            },
        }
        # --- end inlined ---

        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update


# ======================================================================
# two_component_system
# ======================================================================

class TwoComponentSystemLogic:
    """Two Component System — shared state container for Requester/Evolver."""

    name = "ecoli-two-component-system"
    topology = {
    "listeners": ("listeners",),
    "bulk": ("bulk",),
    "timestep": ("timestep",),
}
    defaults = {
        "jit": False,
        "n_avogadro": 0.0,
        "cell_density": 0.0,
        "moleculesToNextTimeStep": (
            lambda counts, volume, avogadro, timestep, random, method, min_step, jit: (
                [],
                [],
            )
        ),
        "moleculeNames": [],
        "seed": 0,
    }

    # Constructor
    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

        # Simulation options
        self.jit = self.parameters["jit"]

        # Get constants
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Create method
        self.moleculesToNextTimeStep = self.parameters["moleculesToNextTimeStep"]

        # Build views
        self.moleculeNames = self.parameters["moleculeNames"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.molecule_idx = None


class TwoComponentSystemRequester(_SafeInvokeMixin, Step):
    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'timestep': Float(_default=self.process.parameters.get('timestep', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('timestep', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}
        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        p = self.process

        # --- inlined from calculate_request ---
        # At t=0, convert all strings to indices
        if p.molecule_idx is None:
            p.molecule_idx = bulk_name_to_idx(
                p.moleculeNames, state["bulk"]["id"]
            )

        # Get molecule counts
        moleculeCounts = counts(state["bulk"], p.molecule_idx)

        # Get cell mass and volume
        cellMass = (state["listeners"]["mass"]["cell_mass"] * units.fg).asNumber(
            units.g
        )
        p.cellVolume = cellMass / p.cell_density

        # Solve ODEs to next time step using the BDF solver through solve_ivp.
        # Note: the BDF solver has been empirically tested to be the fastest
        # solver for this setting among the list of solvers that can be used
        # by the scipy ODE suite.
        p.molecules_required, p.all_molecule_changes = (
            p.moleculesToNextTimeStep(
                moleculeCounts,
                p.cellVolume,
                p.n_avogadro,
                state["timestep"],
                p.random_state,
                method="BDF",
                jit=p.jit,
            )
        )
        request = {"bulk": [(p.molecule_idx, p.molecules_required.astype(int))]}
        # --- end inlined ---

        p.request_set = True
        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request
        return result


class TwoComponentSystemEvolver(_SafeInvokeMixin, Step):
    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('timestep', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('timestep', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}
        state = _protect_state(state)
        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)
        if not self.process.request_set:
            return {}
        timestep = state.get('timestep', 1.0)
        p = self.process

        # --- inlined from evolve_state ---
        moleculeCounts = counts(state["bulk"], p.molecule_idx)
        # Check if any molecules were allocated fewer counts than requested
        if (p.molecules_required > moleculeCounts).any():
            _, p.all_molecule_changes = p.moleculesToNextTimeStep(
                moleculeCounts,
                p.cellVolume,
                p.n_avogadro,
                10000,
                p.random_state,
                method="BDF",
                min_time_step=state["timestep"],
                jit=p.jit,
            )
        # Increment changes in molecule counts
        update = {"bulk": [(p.molecule_idx, p.all_molecule_changes.astype(int))]}
        # --- end inlined ---

        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update


# ======================================================================
# rna_maturation
# ======================================================================

class RnaMaturationLogic:
    """RnaMaturation — shared state container for Requester/Evolver."""

    name = "ecoli-rna-maturation"
    topology = {"bulk": ("bulk",), "bulk_total": ("bulk",), "listeners": ("listeners",)}
    defaults = {}

    # Constructor
    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False
        # Get matrices and vectors that describe maturation reactions
        self.stoich_matrix = self.parameters["stoich_matrix"]
        self.enzyme_matrix = self.parameters["enzyme_matrix"]
        self.n_required_enzymes = self.parameters["n_required_enzymes"]
        self.degraded_nt_counts = self.parameters["degraded_nt_counts"]
        self.n_ppi_added = self.parameters["n_ppi_added"]

        # Calculate number of NMPs that should be added when consolidating rRNA
        # molecules
        self.main_23s_rRNA_id = self.parameters["main_23s_rRNA_id"]
        self.main_16s_rRNA_id = self.parameters["main_16s_rRNA_id"]
        self.main_5s_rRNA_id = self.parameters["main_5s_rRNA_id"]

        self.variant_23s_rRNA_ids = self.parameters["variant_23s_rRNA_ids"]
        self.variant_16s_rRNA_ids = self.parameters["variant_16s_rRNA_ids"]
        self.variant_5s_rRNA_ids = self.parameters["variant_5s_rRNA_ids"]

        self.delta_nt_counts_23s = self.parameters["delta_nt_counts_23s"]
        self.delta_nt_counts_16s = self.parameters["delta_nt_counts_16s"]
        self.delta_nt_counts_5s = self.parameters["delta_nt_counts_5s"]

        # Bulk molecule IDs
        self.unprocessed_rna_ids = self.parameters["unprocessed_rna_ids"]
        self.mature_rna_ids = self.parameters["mature_rna_ids"]
        self.rna_maturation_enzyme_ids = self.parameters["rna_maturation_enzyme_ids"]
        self.fragment_bases = self.parameters["fragment_bases"]
        self.ppi = self.parameters["ppi"]
        self.water = self.parameters["water"]
        self.nmps = self.parameters["nmps"]
        self.proton = self.parameters["proton"]

        # Numpy indices for bulk molecules
        self.ppi_idx = None


class RnaMaturationRequester(_SafeInvokeMixin, Step):
    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('timestep', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}
        state = _protect_state(state)
        timestep = 1.0  # RnaMaturation has no timestep in topology
        p = self.process

        # --- inlined from calculate_request ---
        # Get bulk indices
        if p.ppi_idx is None:
            bulk_ids = state["bulk"]["id"]
            p.unprocessed_rna_idx = bulk_name_to_idx(
                p.unprocessed_rna_ids, bulk_ids
            )
            p.mature_rna_idx = bulk_name_to_idx(p.mature_rna_ids, bulk_ids)
            p.rna_maturation_enzyme_idx = bulk_name_to_idx(
                p.rna_maturation_enzyme_ids, bulk_ids
            )
            p.fragment_base_idx = bulk_name_to_idx(p.fragment_bases, bulk_ids)
            p.ppi_idx = bulk_name_to_idx(p.ppi, bulk_ids)
            p.water_idx = bulk_name_to_idx(p.water, bulk_ids)
            p.nmps_idx = bulk_name_to_idx(p.nmps, bulk_ids)
            p.proton_idx = bulk_name_to_idx(p.proton, bulk_ids)
            p.main_23s_rRNA_idx = bulk_name_to_idx(p.main_23s_rRNA_id, bulk_ids)
            p.main_16s_rRNA_idx = bulk_name_to_idx(p.main_16s_rRNA_id, bulk_ids)
            p.main_5s_rRNA_idx = bulk_name_to_idx(p.main_5s_rRNA_id, bulk_ids)
            p.variant_23s_rRNA_idx = bulk_name_to_idx(
                p.variant_23s_rRNA_ids, bulk_ids
            )
            p.variant_16s_rRNA_idx = bulk_name_to_idx(
                p.variant_16s_rRNA_ids, bulk_ids
            )
            p.variant_5s_rRNA_idx = bulk_name_to_idx(
                p.variant_5s_rRNA_ids, bulk_ids
            )

        unprocessed_rna_counts = counts(state["bulk_total"], p.unprocessed_rna_idx)
        variant_23s_rRNA_counts = counts(
            state["bulk_total"], p.variant_23s_rRNA_idx
        )
        variant_16s_rRNA_counts = counts(
            state["bulk_total"], p.variant_16s_rRNA_idx
        )
        variant_5s_rRNA_counts = counts(state["bulk_total"], p.variant_5s_rRNA_idx)
        p.enzyme_availability = counts(
            state["bulk_total"], p.rna_maturation_enzyme_idx
        ).astype(bool)

        # Determine which maturation reactions to turn off based on enzyme
        # availability
        reaction_is_off = (
            p.enzyme_matrix.dot(p.enzyme_availability) < p.n_required_enzymes
        )
        unprocessed_rna_counts[reaction_is_off] = 0

        # Calculate NMPs, water, and proton needed to balance mass
        n_added_bases_from_maturation = np.dot(
            p.degraded_nt_counts.T, unprocessed_rna_counts
        )
        n_added_bases_from_consolidation = (
            p.delta_nt_counts_23s.T.dot(variant_23s_rRNA_counts)
            + p.delta_nt_counts_16s.T.dot(variant_16s_rRNA_counts)
            + p.delta_nt_counts_5s.T.dot(variant_5s_rRNA_counts)
        )
        n_added_bases = n_added_bases_from_maturation + n_added_bases_from_consolidation
        n_total_added_bases = int(n_added_bases.sum())

        # Request all unprocessed RNAs, ppis that need to be added to the
        # 5'-ends of mature RNAs, all variant rRNAs, and NMPs/water/protons
        # needed to balance mass
        request = {
            "bulk": [
                (p.unprocessed_rna_idx, unprocessed_rna_counts),
                (p.ppi_idx, p.n_ppi_added.dot(unprocessed_rna_counts)),
                (p.variant_23s_rRNA_idx, variant_23s_rRNA_counts),
                (p.variant_16s_rRNA_idx, variant_16s_rRNA_counts),
                (p.variant_5s_rRNA_idx, variant_5s_rRNA_counts),
                (p.nmps_idx, np.abs(-n_added_bases).astype(int)),
            ]
        }

        if n_total_added_bases > 0:
            request["bulk"].append((p.water_idx, n_total_added_bases))
        else:
            request["bulk"].append((p.proton_idx, -n_total_added_bases))
        # --- end inlined ---

        p.request_set = True
        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request
        return result


class RnaMaturationEvolver(_SafeInvokeMixin, Step):
    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('timestep', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}
        state = _protect_state(state)
        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)
        if not self.process.request_set:
            return {}
        timestep = 1.0
        p = self.process

        # --- inlined from evolve_state ---
        # Create copy of bulk counts so can update in real-time
        state["bulk"] = counts(state["bulk"], range(len(state["bulk"])))

        # Get counts of unprocessed RNAs
        unprocessed_rna_counts = counts(state["bulk"], p.unprocessed_rna_idx)

        # Calculate numbers of mature RNAs and fragment bases that are generated
        # upon maturation
        n_mature_rnas = p.stoich_matrix.dot(unprocessed_rna_counts)
        n_added_bases_from_maturation = np.dot(
            p.degraded_nt_counts.T, unprocessed_rna_counts
        )

        state["bulk"][p.mature_rna_idx] += n_mature_rnas
        state["bulk"][p.unprocessed_rna_idx] += -unprocessed_rna_counts
        ppi_update = p.n_ppi_added.dot(unprocessed_rna_counts)
        state["bulk"][p.ppi_idx] += -ppi_update
        update = {
            "bulk": [
                (p.mature_rna_idx, n_mature_rnas),
                (p.unprocessed_rna_idx, -unprocessed_rna_counts),
                (p.ppi_idx, -ppi_update),
            ],
            "listeners": {
                "rna_maturation_listener": {
                    "total_maturation_events": unprocessed_rna_counts.sum(),
                    "total_degraded_ntps": n_added_bases_from_maturation.sum(dtype=int),
                    "unprocessed_rnas_consumed": unprocessed_rna_counts,
                    "mature_rnas_generated": n_mature_rnas,
                    "maturation_enzyme_counts": counts(
                        state["bulk_total"], p.rna_maturation_enzyme_idx
                    ),
                }
            },
        }

        # Get counts of variant rRNAs
        variant_23s_rRNA_counts = counts(state["bulk"], p.variant_23s_rRNA_idx)
        variant_16s_rRNA_counts = counts(state["bulk"], p.variant_16s_rRNA_idx)
        variant_5s_rRNA_counts = counts(state["bulk"], p.variant_5s_rRNA_idx)

        # Calculate number of NMPs that should be added to balance out the mass
        # difference during the consolidation
        n_added_bases_from_consolidation = (
            p.delta_nt_counts_23s.T.dot(variant_23s_rRNA_counts)
            + p.delta_nt_counts_16s.T.dot(variant_16s_rRNA_counts)
            + p.delta_nt_counts_5s.T.dot(variant_5s_rRNA_counts)
        )

        # Evolve states
        update["bulk"].extend(
            [
                (p.main_23s_rRNA_idx, variant_23s_rRNA_counts.sum()),
                (p.main_16s_rRNA_idx, variant_16s_rRNA_counts.sum()),
                (p.main_5s_rRNA_idx, variant_5s_rRNA_counts.sum()),
                (p.variant_23s_rRNA_idx, -variant_23s_rRNA_counts),
                (p.variant_16s_rRNA_idx, -variant_16s_rRNA_counts),
                (p.variant_5s_rRNA_idx, -variant_5s_rRNA_counts),
            ]
        )

        # Consume or add NMPs to balance out mass
        n_added_bases = (
            n_added_bases_from_maturation + n_added_bases_from_consolidation
        ).astype(int)
        n_total_added_bases = n_added_bases.sum()

        update["bulk"].extend(
            [
                (p.nmps_idx, n_added_bases),
                (p.water_idx, -n_total_added_bases),
                (p.proton_idx, n_total_added_bases),
            ]
        )
        # --- end inlined ---

        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update


# ======================================================================
# complexation
# ======================================================================

class ComplexationLogic:
    """Complexation — shared state container for Requester/Evolver."""

    name = "ecoli-complexation"
    topology = {"bulk": ("bulk",), "listeners": ("listeners",), "timestep": ("timestep",)}
    defaults = {
        "stoichiometry": np.array([[]]),
        "rates": np.array([]),
        "molecule_names": [],
        "seed": 0,
        "reaction_ids": [],
        "complex_ids": [],
        "time_step": 1,
    }

    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

        self.stoichiometry = self.parameters["stoichiometry"]
        self.rates = self.parameters["rates"]
        self.molecule_names = self.parameters["molecule_names"]
        self.molecule_idx = None
        self.reaction_ids = self.parameters["reaction_ids"]
        self.complex_ids = self.parameters["complex_ids"]

        self.randomState = np.random.RandomState(seed=self.parameters["seed"])
        self.seed = self.randomState.randint(2**31)
        self.system = StochasticSystem(self.stoichiometry, random_seed=self.seed)


class ComplexationRequester(_SafeInvokeMixin, Step):
    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('timestep', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('timestep', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}
        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        p = self.process

        # --- inlined from calculate_request ---
        timestep = state["timestep"]
        if p.molecule_idx is None:
            p.molecule_idx = bulk_name_to_idx(
                p.molecule_names, state["bulk"]["id"]
            )

        moleculeCounts = counts(state["bulk"], p.molecule_idx)

        result = p.system.evolve(timestep, moleculeCounts, p.rates)
        updatedMoleculeCounts = result["outcome"]
        request = {}
        request["bulk"] = [
            (p.molecule_idx, np.fmax(moleculeCounts - updatedMoleculeCounts, 0))
        ]
        # --- end inlined ---

        p.request_set = True
        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request
        return result


class ComplexationEvolver(_SafeInvokeMixin, Step):
    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('timestep', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('timestep', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}
        state = _protect_state(state)
        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)
        if not self.process.request_set:
            return {}
        timestep = state.get('timestep', 1.0)
        p = self.process

        # --- inlined from evolve_state ---
        timestep = state["timestep"]
        substrate = counts(state["bulk"], p.molecule_idx)

        result = p.system.evolve(timestep, substrate, p.rates)
        complexationEvents = result["occurrences"]
        outcome = result["outcome"] - substrate

        # Write outputs to listeners
        update = {
            "bulk": [(p.molecule_idx, outcome)],
            "listeners": {
                "complexation_listener": {
                    "complexation_events": complexationEvents.astype(int)
                }
            },
        }
        # --- end inlined ---

        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update


# ======================================================================
# protein_degradation
# ======================================================================

class ProteinDegradationLogic:
    """Protein degradation — shared state container for Requester/Evolver."""

    name = "ecoli-protein-degradation"
    topology = {"bulk": ("bulk",), "timestep": ("timestep",)}
    defaults = {
        "raw_degradation_rate": [],
        "water_id": "h2o",
        "amino_acid_ids": [],
        "amino_acid_counts": [],
        "protein_ids": [],
        "protein_lengths": [],
        "seed": 0,
        "time_step": 1,
    }

    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

        self.raw_degradation_rate = self.parameters["raw_degradation_rate"]
        self.water_id = self.parameters["water_id"]
        self.amino_acid_ids = self.parameters["amino_acid_ids"]
        self.amino_acid_counts = self.parameters["amino_acid_counts"]
        self.metabolite_ids = self.amino_acid_ids + [self.water_id]
        self.amino_acid_indexes = np.arange(0, len(self.amino_acid_ids))
        self.water_index = self.metabolite_ids.index(self.water_id)
        self.protein_ids = self.parameters["protein_ids"]
        self.protein_lengths = self.parameters["protein_lengths"]
        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)
        self.metabolite_idx = None

        # Build S matrix
        self.degradation_matrix = np.zeros(
            (len(self.metabolite_ids), len(self.protein_ids)), np.int64
        )
        self.degradation_matrix[self.amino_acid_indexes, :] = np.transpose(
            self.amino_acid_counts
        )
        self.degradation_matrix[self.water_index, :] = -(
            np.sum(self.degradation_matrix[self.amino_acid_indexes, :], axis=0) - 1
        )


class ProteinDegradationRequester(_SafeInvokeMixin, Step):
    """Reads bulk to compute degradation request. Writes to request store."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        p = self.process

        # --- inlined from calculate_request ---
        if p.metabolite_idx is None:
            p.water_idx = bulk_name_to_idx(p.water_id, state["bulk"]["id"])
            p.protein_idx = bulk_name_to_idx(p.protein_ids, state["bulk"]["id"])
            p.metabolite_idx = bulk_name_to_idx(
                p.metabolite_ids, state["bulk"]["id"]
            )

        protein_data = counts(state["bulk"], p.protein_idx)
        nProteinsToDegrade = np.fmin(
            p.random_state.poisson(
                p.raw_degradation_rate * timestep * protein_data
            ),
            protein_data,
        )
        nReactions = np.dot(p.protein_lengths, nProteinsToDegrade)

        request = {
            "bulk": [
                (p.protein_idx, nProteinsToDegrade),
                (p.water_idx, nReactions - np.sum(nProteinsToDegrade)),
            ]
        }
        # --- end inlined ---

        p.request_set = True
        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request
        return result


class ProteinDegradationEvolver(_SafeInvokeMixin, Step):
    """Reads allocation, writes bulk updates."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)

        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)

        if not self.process.request_set:
            return {}

        p = self.process
        timestep = state.get('timestep', 1.0)

        # --- inlined from evolve_state ---
        allocated_proteins = counts(state["bulk"], p.protein_idx)
        metabolites_delta = np.dot(p.degradation_matrix, allocated_proteins)

        update = {
            "bulk": [
                (p.metabolite_idx, metabolites_delta),
                (p.protein_idx, -allocated_proteins),
            ]
        }
        # --- end inlined ---

        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update


# ======================================================================
# rna_degradation
# ======================================================================

class RnaDegradationLogic:
    """RNA Degradation — shared state container for Requester/Evolver."""

    name = "ecoli-rna-degradation"
    topology = {
    "bulk": ("bulk",),
    "RNAs": ("unique", "RNA"),
    "active_ribosome": ("unique", "active_ribosome"),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
}
    defaults = {
        "rna_ids": [],
        "mature_rna_ids": [],
        "cistron_ids": [],
        "cistron_tu_mapping_matrix": [],
        "mature_rna_cistron_indexes": [],
        "all_rna_ids": [],
        "n_total_RNAs": 0,
        "n_avogadro": 0.0,
        "cell_density": 1100 * units.g / units.L,
        "endoRNase_ids": [],
        "exoRNase_ids": [],
        "kcat_exoRNase": np.array([]) / units.s,
        "Kcat_endoRNases": np.array([]) / units.s,
        "charged_trna_names": [],
        "uncharged_trna_indexes": [],
        "rna_deg_rates": [],
        "is_mRNA": np.array([]),
        "is_rRNA": np.array([]),
        "is_tRNA": np.array([]),
        "is_miscRNA": np.array([]),
        "degrade_misc": False,
        "rna_lengths": np.array([]),
        "nt_counts": np.array([[]]),
        "polymerized_ntp_ids": [],
        "water_id": "h2o",
        "ppi_id": "ppi",
        "proton_id": "h+",
        "nmp_ids": [],
        "rrfa_idx": 0,
        "rrla_idx": 0,
        "rrsa_idx": 0,
        "ribosome30S": "ribosome30S",
        "ribosome50S": "ribosome50S",
        "Kms": np.array([]) * units.mol / units.L,
        "seed": 0,
        "emit_unique": False,
        "time_step": 1,
    }

    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

        self.rna_ids = self.parameters["rna_ids"]
        self.mature_rna_ids = self.parameters["mature_rna_ids"]
        self.n_transcribed_rnas = len(self.rna_ids)
        self.mature_rna_exists = len(self.mature_rna_ids) > 0
        self.cistron_ids = self.parameters["cistron_ids"]
        self.cistron_tu_mapping_matrix = self.parameters["cistron_tu_mapping_matrix"]
        self.mature_rna_cistron_indexes = self.parameters["mature_rna_cistron_indexes"]
        self.all_rna_ids = self.parameters["all_rna_ids"]
        self.n_total_RNAs = self.parameters["n_total_RNAs"]

        # Load constants
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Load RNase kinetic data
        self.endoRNase_ids = self.parameters["endoRNase_ids"]
        self.exoRNase_ids = self.parameters["exoRNase_ids"]
        self.kcat_exoRNase = self.parameters["kcat_exoRNase"]
        self.Kcat_endoRNases = self.parameters["Kcat_endoRNases"]

        # Load information about uncharged/charged tRNA
        self.uncharged_trna_indexes = self.parameters["uncharged_trna_indexes"]
        self.charged_trna_names = self.parameters["charged_trna_names"]

        # Load first-order RNA degradation rates
        # (estimated by mRNA half-life data)
        self.rna_deg_rates = self.parameters["rna_deg_rates"]

        self.is_mRNA = self.parameters["is_mRNA"]
        self.is_rRNA = self.parameters["is_rRNA"]
        self.is_tRNA = self.parameters["is_tRNA"]

        # NEW to vivarium-ecoli
        self.is_miscRNA = self.parameters["is_miscRNA"]
        self.degrade_misc = self.parameters["degrade_misc"]

        self.rna_lengths = self.parameters["rna_lengths"]
        self.nt_counts = self.parameters["nt_counts"]

        # Build stoichiometric matrix
        self.polymerized_ntp_ids = self.parameters["polymerized_ntp_ids"]
        self.nmp_ids = self.parameters["nmp_ids"]
        self.water_id = self.parameters["water_id"]
        self.ppi_id = self.parameters["ppi_id"]
        self.proton_id = self.parameters["proton_id"]

        self.end_cleavage_metabolite_ids = self.polymerized_ntp_ids + [
            self.water_id,
            self.ppi_id,
            self.proton_id,
        ]
        nmp_idx = list(range(4))
        water_idx = self.end_cleavage_metabolite_ids.index(self.water_id)
        ppi_idx = self.end_cleavage_metabolite_ids.index(self.ppi_id)
        proton_idx = self.end_cleavage_metabolite_ids.index(self.proton_id)
        self.endo_degradation_stoich_matrix = np.zeros(
            (len(self.end_cleavage_metabolite_ids), self.n_total_RNAs), np.int64
        )
        self.endo_degradation_stoich_matrix[nmp_idx, :] = self.nt_counts.T
        self.endo_degradation_stoich_matrix[water_idx, :] = 0
        self.endo_degradation_stoich_matrix[ppi_idx, :] = 1
        self.endo_degradation_stoich_matrix[proton_idx, :] = 0

        # Load Michaelis-Menten constants fitted to recapitulate
        # first-order RNA decay model
        self.Kms = self.parameters["Kms"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Numpy indices for bulk molecules
        self.water_idx = None

    def _calculate_total_n_to_degrade(
        self, timestep, specificity, total_kcat_endornase
    ):
        """
        Calculate the total number of RNAs to degrade for a specific class of
        RNAs, based on the specificity of endoRNases on that specific class and
        the total kcat value of the endoRNases.

        Args:
            specificity: Sum of fraction of active endoRNases for all RNAs
                in a given class
            total_kcat_endornase: The summed kcat of all existing endoRNases
        Returns:
            Total number of RNAs to degrade for the given class of RNAs
        """
        return np.round(
            (specificity * total_kcat_endornase * (units.s * timestep)).asNumber()
        )

    def _get_rnas_to_degrade(self, n_total_rnas_to_degrade, rna_deg_probs, rna_counts):
        """
        Distributes the total count of RNAs to degrade for each class of RNAs
        into individual RNAs, based on the given degradation probabilities
        of individual RNAs. The upper bound is set by the current count of the
        specific RNA.

        Args:
            n_total_rnas_to_degrade: Total number of RNAs to degrade for the
                given class of RNAs (integer, scalar)
            rna_deg_probs: Degradation probabilities of each RNA (vector of
                equal length to the total number of different RNAs)
            rna_counts: Current counts of each RNA molecule (vector of equal
                length to the total number of different RNAs)
        Returns:
            Vector of equal length to rna_counts, specifying the number of
            molecules to degrade for each RNA
        """
        n_rnas_to_degrade = np.zeros_like(rna_counts)
        remaining_rna_counts = rna_counts

        while (
            n_rnas_to_degrade.sum() < n_total_rnas_to_degrade
            and remaining_rna_counts.sum() != 0
        ):
            n_rnas_to_degrade += np.fmin(
                self.random_state.multinomial(
                    n_total_rnas_to_degrade - n_rnas_to_degrade.sum(), rna_deg_probs
                ),
                remaining_rna_counts,
            )
            remaining_rna_counts = rna_counts - n_rnas_to_degrade

        return n_rnas_to_degrade


class RnaDegradationRequester(_SafeInvokeMixin, Step):
    """Reads stores to compute RNA degradation request."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'RNAs': UniqueNumpyUpdate(),
            'active_ribosome': UniqueNumpyUpdate(),
            'listeners': ListenerStore(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
            'listeners': ListenerStore(),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from calculate_request ---
        if p.water_idx is None:
            bulk_ids = state["bulk"]["id"]
            p.charged_trna_idx = bulk_name_to_idx(p.charged_trna_names, bulk_ids)
            p.bulk_rnas_idx = bulk_name_to_idx(p.all_rna_ids, bulk_ids)
            p.nmps_idx = bulk_name_to_idx(p.nmp_ids, bulk_ids)
            p.fragment_metabolites_idx = bulk_name_to_idx(
                p.end_cleavage_metabolite_ids, bulk_ids
            )
            p.fragment_bases_idx = bulk_name_to_idx(
                p.polymerized_ntp_ids, bulk_ids
            )
            p.endoRNase_idx = bulk_name_to_idx(p.endoRNase_ids, bulk_ids)
            p.exoRNase_idx = bulk_name_to_idx(p.exoRNase_ids, bulk_ids)
            p.water_idx = bulk_name_to_idx(p.water_id, bulk_ids)
            p.proton_idx = bulk_name_to_idx(p.proton_id, bulk_ids)

        # Compute factor that convert counts into concentration, and vice versa
        cell_mass = state["listeners"]["mass"]["cell_mass"] * units.fg
        cell_volume = cell_mass / p.cell_density
        counts_to_molar = 1 / (p.n_avogadro * cell_volume)

        # Get total counts of RNAs including free rRNAs, uncharged and charged tRNAs, and
        # active (translatable) unique mRNAs
        bulk_RNA_counts = counts(state["bulk"], p.bulk_rnas_idx)
        bulk_RNA_counts[p.uncharged_trna_indexes] += counts(
            state["bulk"], p.charged_trna_idx
        )

        TU_index, can_translate, is_full_transcript = attrs(
            state["RNAs"], ["TU_index", "can_translate", "is_full_transcript"]
        )

        TU_index_translatable_mRNAs = TU_index[can_translate]
        unique_RNA_counts = np.bincount(
            TU_index_translatable_mRNAs, minlength=p.n_total_RNAs
        )
        total_RNA_counts = bulk_RNA_counts + unique_RNA_counts

        # Compute RNA concentrations
        rna_conc_molar = counts_to_molar * total_RNA_counts

        # Get counts of endoRNases
        endoRNase_counts = counts(state["bulk"], p.endoRNase_idx)
        total_kcat_endoRNase = units.dot(p.Kcat_endoRNases, endoRNase_counts)

        # Calculate the fraction of active endoRNases for each RNA based on
        # Michaelis-Menten kinetics
        frac_endoRNase_saturated = (
            rna_conc_molar / p.Kms / (1 + units.sum(rna_conc_molar / p.Kms))
        ).asNumber()

        # Calculate difference in degradation rates from first-order decay
        # and the number of EndoRNases per one molecule of RNA
        total_endoRNase_counts = np.sum(endoRNase_counts)
        diff_relative_first_order_decay = units.sum(
            units.abs(
                p.rna_deg_rates * total_RNA_counts
                - total_kcat_endoRNase * frac_endoRNase_saturated
            )
        )
        endoRNase_per_rna = total_endoRNase_counts / np.sum(total_RNA_counts)

        request = {"listeners": {"rna_degradation_listener": {}}}
        request["listeners"]["rna_degradation_listener"][
            "fraction_active_endoRNases"
        ] = np.sum(frac_endoRNase_saturated)
        request["listeners"]["rna_degradation_listener"][
            "diff_relative_first_order_decay"
        ] = diff_relative_first_order_decay.asNumber()
        request["listeners"]["rna_degradation_listener"]["fract_endo_rrna_counts"] = (
            endoRNase_per_rna
        )

        # Dissect RNAse specificity into mRNA, tRNA, and rRNA
        # NEW to vivarium-ecoli: Degrade miscRNAs and mRNAs together
        if p.degrade_misc:
            is_transient_rna = p.is_mRNA | p.is_miscRNA
            mrna_specificity = np.dot(frac_endoRNase_saturated, is_transient_rna)
        else:
            mrna_specificity = np.dot(frac_endoRNase_saturated, p.is_mRNA)
        trna_specificity = np.dot(frac_endoRNase_saturated, p.is_tRNA)
        rrna_specificity = np.dot(frac_endoRNase_saturated, p.is_rRNA)

        n_total_mrnas_to_degrade = p._calculate_total_n_to_degrade(
            state["timestep"], mrna_specificity, total_kcat_endoRNase
        )
        n_total_trnas_to_degrade = p._calculate_total_n_to_degrade(
            state["timestep"], trna_specificity, total_kcat_endoRNase
        )
        n_total_rrnas_to_degrade = p._calculate_total_n_to_degrade(
            state["timestep"], rrna_specificity, total_kcat_endoRNase
        )

        # Compute RNAse specificity
        rna_specificity = frac_endoRNase_saturated / np.sum(frac_endoRNase_saturated)

        # Boolean variable that tracks existence of each RNA
        rna_exists = (total_RNA_counts > 0).astype(np.int64)

        # Compute degradation probabilities of each RNA: for mRNAs and rRNAs, this
        # is based on the specificity of each mRNA. For tRNAs and rRNAs,
        # this is distributed evenly.
        if p.degrade_misc:
            mrna_deg_probs = (
                1.0
                / np.dot(rna_specificity, is_transient_rna * rna_exists)
                * rna_specificity
                * is_transient_rna
                * rna_exists
            )
        else:
            mrna_deg_probs = (
                1.0
                / np.dot(rna_specificity, p.is_mRNA * rna_exists)
                * rna_specificity
                * p.is_mRNA
                * rna_exists
            )
        rrna_deg_probs = (
            1.0
            / np.dot(rna_specificity, p.is_rRNA * rna_exists)
            * rna_specificity
            * p.is_rRNA
            * rna_exists
        )
        trna_deg_probs = (
            1.0 / np.dot(p.is_tRNA, rna_exists) * p.is_tRNA * rna_exists
        )

        # Mask RNA counts into each class of RNAs
        if p.degrade_misc:
            mrna_counts = total_RNA_counts * is_transient_rna
        else:
            mrna_counts = total_RNA_counts * p.is_mRNA
        trna_counts = total_RNA_counts * p.is_tRNA
        rrna_counts = total_RNA_counts * p.is_rRNA

        # Determine number of individual RNAs to be degraded for each class
        # of RNA.
        n_mrnas_to_degrade = p._get_rnas_to_degrade(
            n_total_mrnas_to_degrade, mrna_deg_probs, mrna_counts
        )
        n_trnas_to_degrade = p._get_rnas_to_degrade(
            n_total_trnas_to_degrade, trna_deg_probs, trna_counts
        )
        n_rrnas_to_degrade = p._get_rnas_to_degrade(
            n_total_rrnas_to_degrade, rrna_deg_probs, rrna_counts
        )
        n_RNAs_to_degrade = n_mrnas_to_degrade + n_trnas_to_degrade + n_rrnas_to_degrade

        # Bulk RNAs (tRNAs and rRNAs) are degraded immediately. Unique RNAs
        # (mRNAs) are immediately deactivated (becomes unable to bind
        # ribosomes), but not degraded until transcription is finished and the
        # mRNA becomes a full transcript to simplify the transcript elongation
        # process.
        n_bulk_RNAs_to_degrade = n_RNAs_to_degrade.copy()
        n_bulk_RNAs_to_degrade[p.is_mRNA.astype(bool)] = 0
        p.n_unique_RNAs_to_deactivate = n_RNAs_to_degrade.copy()
        p.n_unique_RNAs_to_deactivate[np.logical_not(p.is_mRNA.astype(bool))] = 0

        request.setdefault("bulk", []).extend(
            [
                (p.bulk_rnas_idx, n_bulk_RNAs_to_degrade),
                (
                    p.fragment_bases_idx,
                    counts(state["bulk"], p.fragment_bases_idx),
                ),
            ]
        )

        # Calculate the amount of water required for total RNA hydrolysis by
        # endo and exonucleases. We first calculate the number of unique RNAs
        # that should be degraded at this timestep.
        p.unique_mRNAs_to_degrade = np.logical_and(
            np.logical_not(can_translate), is_full_transcript
        )
        p.n_unique_RNAs_to_degrade = np.bincount(
            TU_index[p.unique_mRNAs_to_degrade], minlength=p.n_total_RNAs
        )

        # Assuming complete hydrolysis for now. Note that one additional water
        # molecule is needed for each RNA to hydrolyze the 5' diphosphate.
        water_for_degraded_rnas = np.dot(
            n_bulk_RNAs_to_degrade + p.n_unique_RNAs_to_degrade, p.rna_lengths
        )
        water_for_fragments = counts(state["bulk"], p.fragment_bases_idx).sum()
        request["bulk"].append(
            (p.water_idx, water_for_degraded_rnas + water_for_fragments)
        )
        # --- end inlined ---
        p.request_set = True

        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request

        listeners = request.pop('listeners', None)
        if listeners is not None:
            result['listeners'] = listeners

        return result


class RnaDegradationEvolver(_SafeInvokeMixin, Step):
    """Reads allocation, writes bulk/unique/listener updates."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'RNAs': UniqueNumpyUpdate(),
            'active_ribosome': UniqueNumpyUpdate(),
            'listeners': ListenerStore(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'RNAs': UniqueNumpyUpdate(),
            'active_ribosome': UniqueNumpyUpdate(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)

        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)

        if not self.process.request_set:
            return {}

        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from evolve_state ---
        # Get vector of numbers of RNAs to degrade for each RNA species
        n_degraded_bulk_RNA = counts(state["bulk"], p.bulk_rnas_idx)
        n_degraded_unique_RNA = p.n_unique_RNAs_to_degrade
        n_degraded_RNA = n_degraded_bulk_RNA + n_degraded_unique_RNA

        # Deactivate and degrade unique RNAs
        TU_index, can_translate = attrs(state["RNAs"], ["TU_index", "can_translate"])
        can_translate = can_translate.copy()
        n_deactivated_unique_RNA = p.n_unique_RNAs_to_deactivate

        # Deactive unique RNAs
        non_zero_deactivation = n_deactivated_unique_RNA > 0

        for index, n_degraded in zip(
            np.arange(n_deactivated_unique_RNA.size)[non_zero_deactivation],
            n_deactivated_unique_RNA[non_zero_deactivation],
        ):
            # Get mask for translatable mRNAs belonging to the degraded species
            mask = np.logical_and(TU_index == index, can_translate)

            # Choose n_degraded indexes randomly to deactivate
            can_translate[
                p.random_state.choice(
                    size=n_degraded, a=np.where(mask)[0], replace=False
                )
            ] = False

        count_RNA_degraded_per_cistron = p.cistron_tu_mapping_matrix.dot(
            n_degraded_RNA[: p.n_transcribed_rnas]
        )
        # Add degraded counts from mature RNAs
        if p.mature_rna_exists:
            count_RNA_degraded_per_cistron[p.mature_rna_cistron_indexes] += (
                n_degraded_RNA[p.n_transcribed_rnas :]
            )

        update = {
            "listeners": {
                "rna_degradation_listener": {
                    "count_rna_degraded": n_degraded_RNA,
                    "nucleotides_from_degradation": np.dot(
                        n_degraded_RNA, p.rna_lengths
                    ),
                    "count_RNA_degraded_per_cistron": count_RNA_degraded_per_cistron,
                }
            },
            # Degrade bulk RNAs
            "bulk": [(p.bulk_rnas_idx, -n_degraded_bulk_RNA)],
            "RNAs": {
                "set": {"can_translate": can_translate},
                # Degrade full mRNAs that are inactive
                "delete": np.where(p.unique_mRNAs_to_degrade)[0],
            },
        }

        # Modeling assumption: Once a RNA is cleaved by an endonuclease its
        # resulting nucleotides are lumped together as "polymerized fragments".
        # These fragments can carry over from previous timesteps. We are also
        # assuming that during endonucleolytic cleavage the 5'terminal
        # phosphate is removed. This is modeled as all of the fragments being
        # one long linear chain of "fragment bases".

        # Example:
        # PPi-Base-PO4(-)-Base-PO4(-)-Base-OH
        # ==>
        # Pi-FragmentBase-PO4(-)-FragmentBase-PO4(-)-FragmentBase + PPi
        # Note: Lack of -OH on 3' end of chain
        metabolites_endo_cleavage = np.dot(
            p.endo_degradation_stoich_matrix, n_degraded_RNA
        )

        # Increase polymerized fragment counts
        update["bulk"].append(
            (p.fragment_metabolites_idx, metabolites_endo_cleavage)
        )
        # fragment_metabolites overlaps with fragment_bases
        bulk_count_copy = state["bulk"].copy()
        if len(bulk_count_copy.dtype) > 1:
            bulk_count_copy = bulk_count_copy["count"]
        bulk_count_copy[p.fragment_metabolites_idx] += metabolites_endo_cleavage
        fragment_bases = bulk_count_copy[p.fragment_bases_idx]

        # Check if exonucleolytic digestion can happen
        if fragment_bases.sum() != 0:
            # Calculate exolytic cleavage events

            # Modeling assumption: We model fragments as one long fragment chain of
            # polymerized nucleotides. We are also assuming that there is no
            # sequence specificity or bias towards which nucleotides are
            # hydrolyzed.

            # Example:
            # Pi-FragmentBase-PO4(-)-FragmentBase-PO4(-)-FragmentBase + 3 H2O
            # ==>
            # 3 NMP + 3 H(+)
            # Note: Lack of -OH on 3' end of chain

            n_exoRNases = counts(state["bulk"], p.exoRNase_idx)
            n_fragment_bases = fragment_bases
            n_fragment_bases_sum = n_fragment_bases.sum()

            exornase_capacity = (
                n_exoRNases.sum() * p.kcat_exoRNase * (units.s * state["timestep"])
            )

            if exornase_capacity >= n_fragment_bases_sum:
                update["bulk"].extend(
                    [
                        (p.nmps_idx, n_fragment_bases),
                        (p.water_idx, -n_fragment_bases_sum),
                        (p.proton_idx, n_fragment_bases_sum),
                        (p.fragment_bases_idx, -n_fragment_bases),
                    ]
                )
                total_fragment_bases_digested = n_fragment_bases_sum

            else:
                fragment_specificity = n_fragment_bases / n_fragment_bases_sum
                possible_bases_to_digest = p.random_state.multinomial(
                    exornase_capacity, fragment_specificity
                )
                n_fragment_bases_digested = n_fragment_bases - np.fmax(
                    n_fragment_bases - possible_bases_to_digest, 0
                )

                total_fragment_bases_digested = n_fragment_bases_digested.sum()

                update["bulk"].extend(
                    [
                        (p.nmps_idx, n_fragment_bases_digested),
                        (p.water_idx, -total_fragment_bases_digested),
                        (p.proton_idx, total_fragment_bases_digested),
                        (p.fragment_bases_idx, -n_fragment_bases_digested),
                    ]
                )

            update["listeners"]["rna_degradation_listener"]["fragment_bases_digested"] = (
                total_fragment_bases_digested
            )

        # Note that once mRNAs have been degraded,
        # chromosome_structure.py will handle deleting the active
        # ribosomes that were translating those mRNAs.
        # --- end inlined ---
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update


# ======================================================================
# transcript_initiation
# ======================================================================

class TranscriptInitiationLogic:
    """Transcript Initiation — shared state container for Requester/Evolver.

    **Defaults:**

    - **fracActiveRnapDict** (``dict``): Dictionary with keys corresponding to
      media, values being the fraction of active RNA Polymerase (RNAP)
      for that media.
    - **rnaLengths** (``numpy.ndarray[int]``): lengths of RNAs for each transcription
      unit (TU), in nucleotides
    - **rnaPolymeraseElongationRateDict** (``dict``): Dictionary with keys
      corresponding to media, values being RNAP's elongation rate in
      that media, in nucleotides/s
    - **variable_elongation** (``bool``): Whether to add amplified elongation rates
      for rRNAs. False by default.
    - **make_elongation_rates** (``func``): Function for making elongation rates
      (see :py:meth:`~reconstruction.ecoli.dataclasses.process.transcription.Transcription.make_elongation_rates`).
      Takes PRNG, basal elongation rate, timestep, and ``variable_elongation``.
      Returns an array of elongation rates for all genes.
    - **active_rnap_footprint_size** (``int``): Maximum physical footprint of RNAP
      in nucleotides to cap initiation probabilities
    - **basal_prob** (``numpy.ndarray[float]``): Baseline probability of synthesis for
      every TU.
    - **delta_prob** (``dict``): Dictionary with four keys, used to create a matrix
      encoding the effect of transcription factors (TFs) on transcription
      probabilities::

        {'deltaV' (np.ndarray[float]): deltas associated with the effects of
            TFs on TUs,
        'deltaI' (np.ndarray[int]): index of the affected TU for each delta,
        'deltaJ' (np.ndarray[int]): index of the acting TF for each delta,
        'shape' (tuple): (m, n) = (# of TUs, # of TFs)}

    - **perturbations** (``dict``): Dictionary of genetic perturbations (optional,
      can be empty)
    - **rna_data** (``numpy.ndarray``): Structured array with an entry for each TU,
      where entries look like::

            (id, deg_rate, length (nucleotides), counts_AGCU, mw
            (molecular weight), is_mRNA, is_miscRNA, is_rRNA, is_tRNA,
            is_23S_rRNA, is_16S_rRNA, is_5S_rRNA, is_ribosomal_protein,
            is_RNAP, gene_id, Km_endoRNase, replication_coordinate,
            direction)

    - **idx_rRNA** (``numpy.ndarray[int]``): indexes of TUs corresponding to rRNAs
    - **idx_mRNA** (``numpy.ndarray[int]``): indexes of TUs corresponding to mRNAs
    - **idx_tRNA** (``numpy.ndarray[int]``): indexes of TUs corresponding to tRNAs
    - **idx_rprotein** (``numpy.ndarray[int]``): indexes of TUs corresponding ribosomal
      proteins
    - **idx_rnap** (``numpy.ndarray[int]``): indexes of TU corresponding to RNAP
    - **rnaSynthProbFractions** (``dict``): Dictionary where keys are media types,
      values are sub-dictionaries with keys 'mRna', 'tRna', 'rRna', and
      values being probabilities of synthesis for each RNA type
    - **rnaSynthProbRProtein** (``dict``): Dictionary where keys are media types,
      values are arrays storing the (fixed) probability of synthesis for
      each rProtein TU, under that media condition.
    - **rnaSynthProbRnaPolymerase** (``dict``): Dictionary where keys are media
      types, values are arrays storing the (fixed) probability of
      synthesis for each RNAP TU, under that media condition.
    - **replication_coordinate** (``numpy.ndarray[int]``): Array with chromosome
      coordinates for each TU
    - **transcription_direction** (``numpy.ndarray[bool]``): Array of transcription
      directions for each TU
    - **n_avogadro** (``unum.Unum``): Avogadro's number (constant)
    - **cell_density** (``unum.Unum``): Density of cell (constant)
    - **ppgpp** (``str``): id of ppGpp
    - **inactive_RNAP** (``str``): id of inactive RNAP
    - **synth_prob** (``Callable[[Unum, int], numpy.ndarrray[float]]``):
      Function used in model of ppGpp regulation
      (see :py:func:`~reconstruction.ecoli.dataclasses.process.transcription.Transcription.synth_prob_from_ppgpp`).
      Takes ppGpp concentration (mol/volume) and copy number, returns
      normalized synthesis probability for each gene
    - **copy_number** (``Callable[[Unum, int], numpy.ndarrray[float]]``):
      see :py:func:`~reconstruction.ecoli.dataclasses.process.replication.Replication.get_average_copy_number`.
      Takes expected doubling time in minutes and chromosome coordinates of genes,
      returns average copy number of each gene expected at doubling time
    - **ppgpp_regulation** (``bool``): Whether to include model of ppGpp regulation
    - **get_rnap_active_fraction_from_ppGpp** (``Callable[[Unum], float]``):
      Returns elongation rate for a given ppGpp concentration
    - **seed** (``int``): random seed to initialize PRNG
    """

    name = "ecoli-transcript-initiation"
    topology = {
    "environment": ("environment",),
    "full_chromosomes": ("unique", "full_chromosome"),
    "RNAs": ("unique", "RNA"),
    "active_RNAPs": ("unique", "active_RNAP"),
    "promoters": ("unique", "promoter"),
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
}
    defaults = {
        "fracActiveRnapDict": {},
        "rnaLengths": np.array([]),
        "rnaPolymeraseElongationRateDict": {},
        "variable_elongation": False,
        "make_elongation_rates": (
            lambda random, rate, timestep, variable: np.array([])
        ),
        "active_rnap_foorprint_size": 1,
        "basal_prob": np.array([]),
        "delta_prob": {"deltaI": [], "deltaJ": [], "deltaV": [], "shape": tuple()},
        "get_delta_prob_matrix": None,
        "perturbations": {},
        "rna_data": {},
        "active_rnap_footprint_size": 24 * units.nt,
        "get_rnap_active_fraction_from_ppGpp": lambda x: 0.1,
        "idx_rRNA": np.array([]),
        "idx_mRNA": np.array([]),
        "idx_tRNA": np.array([]),
        "idx_rprotein": np.array([]),
        "idx_rnap": np.array([]),
        "rnaSynthProbFractions": {},
        "rnaSynthProbRProtein": {},
        "rnaSynthProbRnaPolymerase": {},
        "replication_coordinate": np.array([]),
        "transcription_direction": np.array([]),
        "n_avogadro": 6.02214076e23 / units.mol,
        "cell_density": 1100 * units.g / units.L,
        "ppgpp": "ppGpp",
        "inactive_RNAP": "APORNAP-CPLX[c]",
        "synth_prob": lambda concentration, copy: 0.0,
        "copy_number": lambda x: x,
        "ppgpp_regulation": False,
        # attenuation
        "trna_attenuation": False,
        "attenuated_rna_indices": np.array([]),
        "attenuation_adjustments": np.array([]),
        # random seed
        "seed": 0,
        "emit_unique": False,
        "time_step": 1,
    }

    # Constructor
    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

        # Load parameters
        self.fracActiveRnapDict = self.parameters["fracActiveRnapDict"]
        self.rnaLengths = self.parameters["rnaLengths"]
        self.rnaPolymeraseElongationRateDict = self.parameters[
            "rnaPolymeraseElongationRateDict"
        ]
        self.variable_elongation = self.parameters["variable_elongation"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]
        self.active_rnap_footprint_size = self.parameters["active_rnap_footprint_size"]

        # Initialize matrices used to calculate synthesis probabilities
        self.basal_prob = self.parameters["basal_prob"].copy()
        self.trna_attenuation = self.parameters["trna_attenuation"]
        if self.trna_attenuation:
            self.attenuated_rna_indices = self.parameters["attenuated_rna_indices"]
            self.attenuation_adjustments = self.parameters["attenuation_adjustments"]
            self.basal_prob[self.attenuated_rna_indices] += self.attenuation_adjustments

        self.n_TUs = len(self.basal_prob)
        self.delta_prob = self.parameters["delta_prob"]
        if self.parameters["get_delta_prob_matrix"] is not None:
            self.delta_prob_matrix = self.parameters["get_delta_prob_matrix"](
                dense=True, ppgpp=self.parameters["ppgpp_regulation"]
            )
        else:
            # make delta_prob_matrix without adjustments
            self.delta_prob_matrix = scipy.sparse.csr_matrix(
                (
                    self.delta_prob["deltaV"],
                    (self.delta_prob["deltaI"], self.delta_prob["deltaJ"]),
                ),
                shape=self.delta_prob["shape"],
            ).toarray()

        # Determine changes from genetic perturbations
        self.genetic_perturbations = {}
        self.perturbations = self.parameters["perturbations"]
        self.rna_data = self.parameters["rna_data"]

        if len(self.perturbations) > 0:
            probability_indexes = [
                (index, self.perturbations[rna_data["id"]])
                for index, rna_data in enumerate(self.rna_data)
                if rna_data["id"] in self.perturbations
            ]

            self.genetic_perturbations = {
                "fixedRnaIdxs": [pair[0] for pair in probability_indexes],
                "fixedSynthProbs": [pair[1] for pair in probability_indexes],
            }

        # ID Groups
        self.idx_rRNA = self.parameters["idx_rRNA"]
        self.idx_mRNA = self.parameters["idx_mRNA"]
        self.idx_tRNA = self.parameters["idx_tRNA"]
        self.idx_rprotein = self.parameters["idx_rprotein"]
        self.idx_rnap = self.parameters["idx_rnap"]

        # Synthesis probabilities for different categories of genes
        self.rnaSynthProbFractions = self.parameters["rnaSynthProbFractions"]
        self.rnaSynthProbRProtein = self.parameters["rnaSynthProbRProtein"]
        self.rnaSynthProbRnaPolymerase = self.parameters["rnaSynthProbRnaPolymerase"]

        # Coordinates and transcription directions of transcription units
        self.replication_coordinate = self.parameters["replication_coordinate"]
        self.transcription_direction = self.parameters["transcription_direction"]

        self.inactive_RNAP = self.parameters["inactive_RNAP"]

        # ppGpp control related
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]
        self.ppgpp = self.parameters["ppgpp"]
        self.synth_prob = self.parameters["synth_prob"]
        self.copy_number = self.parameters["copy_number"]
        self.ppgpp_regulation = self.parameters["ppgpp_regulation"]
        self.get_rnap_active_fraction_from_ppGpp = self.parameters[
            "get_rnap_active_fraction_from_ppGpp"
        ]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.ppgpp_idx = None

    def _calculateActivationProb(
        self,
        timestep,
        fracActiveRnap,
        rnaLengths,
        rnaPolymeraseElongationRates,
        synthProb,
    ):
        """
        Calculate expected RNAP termination rate based on RNAP elongation rate
        - allTranscriptionTimes: Vector of times required to transcribe each
        transcript
        - allTranscriptionTimestepCounts: Vector of numbers of timesteps
        required to transcribe each transcript
        - averageTranscriptionTimeStepCounts: Average number of timesteps
        required to transcribe a transcript, weighted by synthesis
        probabilities of each transcript
        - expectedTerminationRate: Average number of terminations in one
        timestep for one transcript
        """
        allTranscriptionTimes = 1.0 / rnaPolymeraseElongationRates * rnaLengths
        timesteps = (1.0 / (timestep * units.s) * allTranscriptionTimes).asNumber()
        allTranscriptionTimestepCounts = np.ceil(timesteps)
        averageTranscriptionTimestepCounts = np.dot(
            synthProb, allTranscriptionTimestepCounts
        )
        expectedTerminationRate = 1.0 / averageTranscriptionTimestepCounts

        """
        Modify given fraction of active RNAPs to take into account early
        terminations in between timesteps
        - allFractionTimeInactive: Vector of probabilities an "active" RNAP
        will in effect be "inactive" because it has terminated during a
        timestep
        - averageFractionTimeInactive: Average probability of an "active" RNAP
        being in effect "inactive", weighted by synthesis probabilities
        - effectiveFracActiveRnap: New higher "goal" for fraction of active
        RNAP, considering that the "effective" fraction is lower than what the
        listener sees
        """
        allFractionTimeInactive = (
            1
            - (1.0 / (timestep * units.s) * allTranscriptionTimes).asNumber()
            / allTranscriptionTimestepCounts
        )
        averageFractionTimeInactive = np.dot(allFractionTimeInactive, synthProb)
        effectiveFracActiveRnap = fracActiveRnap / (1 - averageFractionTimeInactive)

        # Return activation probability that will balance out the expected termination rate
        activation_prob = (
            effectiveFracActiveRnap
            * expectedTerminationRate
            / (1 - effectiveFracActiveRnap)
        )

        if activation_prob > 1:
            activation_prob = 1

        return activation_prob

    def _rescale_initiation_probs(self, fixed_indexes, fixed_synth_probs, TU_index):
        """
        Rescales the initiation probabilities of each promoter such that the
        total synthesis probabilities of certain types of RNAs are fixed to
        a predetermined value. For instance, if there are two copies of
        promoters for RNA A, whose synthesis probability should be fixed to
        0.1, each promoter is given an initiation probability of 0.05.
        """
        for idx, synth_prob in zip(fixed_indexes, fixed_synth_probs):
            fixed_mask = TU_index == idx
            self.promoter_init_probs[fixed_mask] = synth_prob / fixed_mask.sum()


class TranscriptInitiationRequester(_SafeInvokeMixin, Step):
    """Reads stores to compute transcript initiation request."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'full_chromosomes': UniqueNumpyUpdate(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
            'promoters': UniqueNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
            'listeners': ListenerStore(),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from calculate_request ---
        # At first update, convert all strings to indices
        if p.ppgpp_idx is None:
            bulk_ids = state["bulk"]["id"]
            p.ppgpp_idx = bulk_name_to_idx(p.ppgpp, bulk_ids)
            p.inactive_RNAP_idx = bulk_name_to_idx(p.inactive_RNAP, bulk_ids)

        # Get all inactive RNA polymerases
        request = {
            "bulk": [
                (p.inactive_RNAP_idx, counts(state["bulk"], p.inactive_RNAP_idx))
            ]
        }

        # Read current environment
        current_media_id = state["environment"]["media_id"]

        if state["full_chromosomes"]["_entryState"].sum() > 0:
            # Get attributes of promoters
            TU_index, bound_TF = attrs(state["promoters"], ["TU_index", "bound_TF"])

            if p.ppgpp_regulation:
                cell_mass = state["listeners"]["mass"]["cell_mass"] * units.fg
                cell_volume = cell_mass / p.cell_density
                counts_to_molar = 1 / (p.n_avogadro * cell_volume)
                ppgpp_conc = counts(state["bulk"], p.ppgpp_idx) * counts_to_molar
                basal_prob, _ = p.synth_prob(ppgpp_conc, p.copy_number)
                if p.trna_attenuation:
                    basal_prob[p.attenuated_rna_indices] += (
                        p.attenuation_adjustments
                    )
                p.fracActiveRnap = p.get_rnap_active_fraction_from_ppGpp(
                    ppgpp_conc
                )
                ppgpp_scale = basal_prob[TU_index]
                # Use original delta prob if no ppGpp basal
                ppgpp_scale[ppgpp_scale == 0] = 1
            else:
                basal_prob = p.basal_prob
                p.fracActiveRnap = p.fracActiveRnapDict[current_media_id]
                ppgpp_scale = 1

            # Calculate probabilities of the RNAP binding to each promoter
            p.promoter_init_probs = basal_prob[TU_index] + ppgpp_scale * np.multiply(
                p.delta_prob_matrix[TU_index, :], bound_TF
            ).sum(axis=1)

            if len(p.genetic_perturbations) > 0:
                p._rescale_initiation_probs(
                    p.genetic_perturbations["fixedRnaIdxs"],
                    p.genetic_perturbations["fixedSynthProbs"],
                    TU_index,
                )

            # Adjust probabilities to not be negative
            p.promoter_init_probs[p.promoter_init_probs < 0] = 0.0
            p.promoter_init_probs /= p.promoter_init_probs.sum()

            if not p.ppgpp_regulation:
                # Adjust synthesis probabilities depending on environment
                synthProbFractions = p.rnaSynthProbFractions[current_media_id]

                # Create masks for different types of RNAs
                is_mrna = np.isin(TU_index, p.idx_mRNA)
                is_trna = np.isin(TU_index, p.idx_tRNA)
                is_rrna = np.isin(TU_index, p.idx_rRNA)
                is_rprotein = np.isin(TU_index, p.idx_rprotein)
                is_rnap = np.isin(TU_index, p.idx_rnap)
                is_fixed = is_trna | is_rrna | is_rprotein | is_rnap

                # Rescale initiation probabilities based on type of RNA
                p.promoter_init_probs[is_mrna] *= (
                    synthProbFractions["mRna"] / p.promoter_init_probs[is_mrna].sum()
                )
                p.promoter_init_probs[is_trna] *= (
                    synthProbFractions["tRna"] / p.promoter_init_probs[is_trna].sum()
                )
                p.promoter_init_probs[is_rrna] *= (
                    synthProbFractions["rRna"] / p.promoter_init_probs[is_rrna].sum()
                )

                # Set fixed synthesis probabilities for RProteins and RNAPs
                p._rescale_initiation_probs(
                    np.concatenate((p.idx_rprotein, p.idx_rnap)),
                    np.concatenate(
                        (
                            p.rnaSynthProbRProtein[current_media_id],
                            p.rnaSynthProbRnaPolymerase[current_media_id],
                        )
                    ),
                    TU_index,
                )

                assert p.promoter_init_probs[is_fixed].sum() < 1.0

                # Scale remaining synthesis probabilities accordingly
                scaleTheRestBy = (
                    1.0 - p.promoter_init_probs[is_fixed].sum()
                ) / p.promoter_init_probs[~is_fixed].sum()
                p.promoter_init_probs[~is_fixed] *= scaleTheRestBy

        # If there are no chromosomes in the cell, set all probs to zero
        else:
            p.promoter_init_probs = np.zeros(
                state["promoters"]["_entryState"].sum()
            )

        p.rnaPolymeraseElongationRate = p.rnaPolymeraseElongationRateDict[
            current_media_id
        ]
        p.elongation_rates = p.make_elongation_rates(
            p.random_state,
            p.rnaPolymeraseElongationRate.asNumber(units.nt / units.s),
            1,  # want elongation rate, not lengths adjusted for time step
            p.variable_elongation,
        )
        # --- end inlined ---
        p.request_set = True

        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request

        listeners = request.pop('listeners', None)
        if listeners is not None:
            result['listeners'] = listeners

        return result


class TranscriptInitiationEvolver(_SafeInvokeMixin, Step):
    """Reads allocation, writes bulk/unique/listener updates."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'full_chromosomes': UniqueNumpyUpdate(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
            'promoters': UniqueNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)

        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)

        if not self.process.request_set:
            return {}

        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from evolve_state ---
        update = {
            "listeners": {
                "rna_synth_prob": {
                    "target_rna_synth_prob": np.zeros(p.n_TUs),
                    "actual_rna_synth_prob": np.zeros(p.n_TUs),
                    "tu_is_overcrowded": np.zeros(p.n_TUs, dtype=np.bool_),
                    "total_rna_init": 0,
                    "max_p": 0.0,
                },
                "ribosome_data": {"total_rna_init": 0},
                "rnap_data": {
                    "did_initialize": 0,
                    "rna_init_event": np.zeros(p.n_TUs, dtype=np.int64),
                },
            },
            "active_RNAPs": {},
            "full_chromosomes": {},
            "promoters": {},
            "RNAs": {},
        }

        # no synthesis if no chromosome
        if len(state["full_chromosomes"]) != 0:
            # Get attributes of promoters
            TU_index, domain_index_promoters = attrs(
                state["promoters"], ["TU_index", "domain_index"]
            )

            n_promoters = state["promoters"]["_entryState"].sum()
            # Construct matrix that maps promoters to transcription units
            TU_to_promoter = scipy.sparse.csr_matrix(
                (np.ones(n_promoters), (TU_index, np.arange(n_promoters))),
                shape=(p.n_TUs, n_promoters),
                dtype=np.int8,
            )

            # Compute target synthesis probabilities of each transcription unit
            target_TU_synth_probs = TU_to_promoter.dot(p.promoter_init_probs)
            update["listeners"]["rna_synth_prob"]["target_rna_synth_prob"] = (
                target_TU_synth_probs
            )

            # Calculate RNA polymerases to activate based on probabilities
            p.activationProb = p._calculateActivationProb(
                state["timestep"],
                p.fracActiveRnap,
                p.rnaLengths,
                (units.nt / units.s) * p.elongation_rates,
                target_TU_synth_probs,
            )

            n_RNAPs_to_activate = np.int64(
                p.activationProb * counts(state["bulk"], p.inactive_RNAP_idx)
            )

            if n_RNAPs_to_activate != 0:
                # Cap the initiation probabilities at the maximum level physically
                # allowed from the known RNAP footprint sizes
                max_p = (
                    p.rnaPolymeraseElongationRate
                    / p.active_rnap_footprint_size
                    * (units.s)
                    * state["timestep"]
                    / n_RNAPs_to_activate
                ).asNumber()
                update["listeners"]["rna_synth_prob"]["max_p"] = max_p
                is_overcrowded = p.promoter_init_probs > max_p

                while np.any(p.promoter_init_probs > max_p):
                    p.promoter_init_probs[is_overcrowded] = max_p
                    scale_the_rest_by = (
                        1.0 - p.promoter_init_probs[is_overcrowded].sum()
                    ) / p.promoter_init_probs[~is_overcrowded].sum()
                    p.promoter_init_probs[~is_overcrowded] *= scale_the_rest_by
                    is_overcrowded |= p.promoter_init_probs > max_p

                # Compute actual synthesis probabilities of each transcription unit
                actual_TU_synth_probs = TU_to_promoter.dot(p.promoter_init_probs)
                tu_is_overcrowded = TU_to_promoter.dot(is_overcrowded).astype(bool)
                update["listeners"]["rna_synth_prob"]["actual_rna_synth_prob"] = (
                    actual_TU_synth_probs
                )
                update["listeners"]["rna_synth_prob"]["tu_is_overcrowded"] = tu_is_overcrowded

                # Sample a multinomial distribution of initiation probabilities to
                # determine what promoters are initialized
                n_initiations = p.random_state.multinomial(
                    n_RNAPs_to_activate, p.promoter_init_probs
                )

                # Build array of transcription unit indexes for partially transcribed
                # RNAs and domain indexes for RNAPs
                TU_index_partial_RNAs = np.repeat(TU_index, n_initiations)
                domain_index_rnap = np.repeat(domain_index_promoters, n_initiations)

                # Build arrays of starting coordinates and transcription directions
                coordinates = p.replication_coordinate[TU_index_partial_RNAs]
                is_forward = p.transcription_direction[TU_index_partial_RNAs]

                # new RNAPs
                RNAP_indexes = create_unique_indices(n_RNAPs_to_activate, state["RNAs"])
                update["active_RNAPs"].update(
                    {
                        "add": {
                            "unique_index": RNAP_indexes,
                            "domain_index": domain_index_rnap,
                            "coordinates": coordinates,
                            "is_forward": is_forward,
                        }
                    }
                )

                # Decrement counts of inactive RNAPs
                update["bulk"] = [(p.inactive_RNAP_idx, -n_initiations.sum())]

                # Add partially transcribed RNAs
                is_mRNA = np.isin(TU_index_partial_RNAs, p.idx_mRNA)
                update["RNAs"].update(
                    {
                        "add": {
                            "TU_index": TU_index_partial_RNAs,
                            "transcript_length": np.zeros(cast(int, n_RNAPs_to_activate)),
                            "is_mRNA": is_mRNA,
                            "is_full_transcript": np.zeros(
                                cast(int, n_RNAPs_to_activate), dtype=bool
                            ),
                            "can_translate": is_mRNA,
                            "RNAP_index": RNAP_indexes,
                        }
                    }
                )

                rna_init_event = TU_to_promoter.dot(n_initiations)
                rRNA_initiations = rna_init_event[p.idx_rRNA]

                # Write outputs to listeners
                update["listeners"]["ribosome_data"] = {
                    "rRNA_initiated_TU": rRNA_initiations.astype(int),
                    "rRNA_init_prob_TU": rRNA_initiations / float(n_RNAPs_to_activate),
                    "total_rna_init": n_RNAPs_to_activate,
                }

                update["listeners"]["rnap_data"] = {
                    "did_initialize": n_RNAPs_to_activate,
                    "rna_init_event": rna_init_event.astype(np.int64),
                }

                update["listeners"]["rna_synth_prob"]["total_rna_init"] = n_RNAPs_to_activate
        # --- end inlined ---
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update


# ======================================================================
# transcript_elongation
# ======================================================================

def make_elongation_rates(random, rates, timestep, variable):
    return rates


def get_attenuation_stop_probabilities(trna_conc):
    return np.array([])


class TranscriptElongationLogic:
    """Transcript Elongation — shared state container for Requester/Evolver.

    defaults:
        - rnaPolymeraseElongationRateDict (dict): Array with elongation rate
            set points for different media environments.
        - rnaIds (array[str]) : array of names for each TU
        - rnaLengths (array[int]) : array of lengths for each TU
            (in nucleotides?)
        - rnaSequences (2D array[int]) : Array with the nucleotide sequences
            of each TU. This is in the form of a 2D array where each row is a
            TU, and each column is a position in the TU's sequence. Nucleotides
            are stored as an index {0, 1, 2, 3}, and the row is padded with
            -1's on the right to indicate where the sequence ends.
        - ntWeights (array[float]): Array of nucleotide weights
        - endWeight (array[float]): ???,
        - replichore_lengths (array[int]): lengths of replichores
            (in nucleotides?),
        - is_mRNA (array[bool]): Mask for mRNAs
        - ppi (str): ID of PPI
        - inactive_RNAP (str): ID of inactive RNAP
        - ntp_ids list[str]: IDs of ntp's (A, C, G, U)
        - variable_elongation (bool): Whether to use variable elongation.
                                      False by default.
        - make_elongation_rates: Function to make elongation rates, of the
            form: lambda random, rates, timestep, variable: rates
    """

    name = "ecoli-transcript-elongation"
    topology = {
    "environment": ("environment",),
    "RNAs": ("unique", "RNA"),
    "active_RNAPs": ("unique", "active_RNAP"),
    "bulk": ("bulk",),
    "bulk_total": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
}
    defaults = {
        # Parameters
        "rnaPolymeraseElongationRateDict": {},
        "rnaIds": [],
        "rnaLengths": np.array([]),
        "rnaSequences": np.array([[]]),
        "ntWeights": np.array([]),
        "endWeight": np.array([]),
        "replichore_lengths": np.array([]),
        "n_fragment_bases": 0,
        "recycle_stalled_elongation": False,
        "submass_indices": {},
        # mask for mRNAs
        "is_mRNA": np.array([]),
        # Bulk molecules
        "inactive_RNAP": "",
        "ppi": "",
        "ntp_ids": [],
        "variable_elongation": False,
        "make_elongation_rates": make_elongation_rates,
        "fragmentBases": [],
        "polymerized_ntps": [],
        "charged_trnas": [],
        # Attenuation
        "trna_attenuation": False,
        "cell_density": 1100 * units.g / units.L,
        "n_avogadro": 6.02214076e23 / units.mol,
        "get_attenuation_stop_probabilities": (get_attenuation_stop_probabilities),
        "attenuated_rna_indices": np.array([], dtype=int),
        "location_lookup": {},
        "seed": 0,
        "emit_unique": False,
        "time_step": 1,
    }

    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

        # Load parameters
        self.rnaPolymeraseElongationRateDict = self.parameters[
            "rnaPolymeraseElongationRateDict"
        ]
        self.rnaIds = self.parameters["rnaIds"]
        self.rnaLengths = self.parameters["rnaLengths"]
        self.rnaSequences = self.parameters["rnaSequences"]
        self.ppi = self.parameters["ppi"]
        self.inactive_RNAP = self.parameters["inactive_RNAP"]
        self.fragmentBases = self.parameters["fragmentBases"]
        self.charged_trnas = self.parameters["charged_trnas"]
        self.ntp_ids = self.parameters["ntp_ids"]
        self.ntWeights = self.parameters["ntWeights"]
        self.endWeight = self.parameters["endWeight"]
        self.replichore_lengths = self.parameters["replichore_lengths"]
        self.chromosome_length = self.replichore_lengths.sum()
        self.n_fragment_bases = self.parameters["n_fragment_bases"]
        self.recycle_stalled_elongation = self.parameters["recycle_stalled_elongation"]

        # Mask for mRNAs
        self.is_mRNA = self.parameters["is_mRNA"]

        self.variable_elongation = self.parameters["variable_elongation"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]

        self.polymerized_ntps = self.parameters["polymerized_ntps"]
        self.charged_trna_names = self.parameters["charged_trnas"]

        # Attenuation
        self.trna_attenuation = self.parameters["trna_attenuation"]
        self.cell_density = self.parameters["cell_density"]
        self.n_avogadro = self.parameters["n_avogadro"]
        self.stop_probabilities = self.parameters["get_attenuation_stop_probabilities"]
        self.attenuated_rna_indices = self.parameters["attenuated_rna_indices"]
        self.attenuated_rna_indices_lookup = {
            idx: i for i, idx in enumerate(self.attenuated_rna_indices)
        }
        self.attenuated_rnas = self.rnaIds[self.attenuated_rna_indices]
        self.location_lookup = self.parameters["location_lookup"]

        # random seed
        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.bulk_RNA_idx = None



class TranscriptElongationRequester(_SafeInvokeMixin, Step):
    """Reads stores to compute transcript elongation request."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
            'listeners': ListenerStore(),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from calculate_request ---
        # At first update, convert all strings to indices
        if p.bulk_RNA_idx is None:
            bulk_ids = state["bulk"]["id"]
            p.bulk_RNA_idx = bulk_name_to_idx(p.rnaIds, bulk_ids)
            p.ntps_idx = bulk_name_to_idx(p.ntp_ids, bulk_ids)
            p.ppi_idx = bulk_name_to_idx(p.ppi, bulk_ids)
            p.inactive_RNAP_idx = bulk_name_to_idx(p.inactive_RNAP, bulk_ids)
            p.fragmentBases_idx = bulk_name_to_idx(p.fragmentBases, bulk_ids)
            p.charged_trnas_idx = bulk_name_to_idx(p.charged_trnas, bulk_ids)

        # Calculate elongation rate based on the current media
        current_media_id = state["environment"]["media_id"]

        p.rnapElongationRate = p.rnaPolymeraseElongationRateDict[
            current_media_id
        ].asNumber(units.nt / units.s)

        p.elongation_rates = p.make_elongation_rates(
            p.random_state,
            p.rnapElongationRate,
            state["timestep"],
            p.variable_elongation,
        )

        # If there are no active RNA polymerases, return immediately
        if state["active_RNAPs"]["_entryState"].sum() == 0:
            request = {}
        else:
            # Determine total possible sequences of nucleotides that can be
            # transcribed in this time step for each partial transcript
            TU_indexes, transcript_lengths, is_full_transcript = attrs(
                state["RNAs"], ["TU_index", "transcript_length", "is_full_transcript"]
            )
            is_partial_transcript = np.logical_not(is_full_transcript)
            TU_indexes_partial = TU_indexes[is_partial_transcript]
            transcript_lengths_partial = transcript_lengths[is_partial_transcript]

            sequences = buildSequences(
                p.rnaSequences,
                TU_indexes_partial,
                transcript_lengths_partial,
                p.elongation_rates,
            )

            sequenceComposition = np.bincount(
                sequences[sequences != polymerize.PAD_VALUE], minlength=4
            )

            # Calculate if any nucleotides are limited and request up to the number
            # in the sequences or number available
            ntpsTotal = counts(state["bulk"], p.ntps_idx)
            maxFractionalReactionLimit = np.fmin(1, ntpsTotal / sequenceComposition)

            request = {
                "bulk": [
                    (
                        p.ntps_idx,
                        (maxFractionalReactionLimit * sequenceComposition).astype(int),
                    )
                ]
            }

            request["listeners"] = {
                "growth_limits": {
                    "ntp_pool_size": counts(state["bulk"], p.ntps_idx),
                    "ntp_request_size": (
                        maxFractionalReactionLimit * sequenceComposition
                    ).astype(int),
                }
            }
        # --- end inlined ---
        p.request_set = True

        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request

        listeners = request.pop('listeners', None)
        if listeners is not None:
            result['listeners'] = listeners

        return result


class TranscriptElongationEvolver(_SafeInvokeMixin, Step):
    """Reads allocation, writes bulk/unique/listener updates."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)

        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)

        if not self.process.request_set:
            return {}

        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from evolve_state ---
        ntpCounts = counts(state["bulk"], p.ntps_idx)

        # If there are no active RNA polymerases, return immediately
        if state["active_RNAPs"]["_entryState"].sum() == 0:
            update = {
                "listeners": {
                    "transcript_elongation_listener": {
                        "count_NTPs_used": 0,
                        "count_rna_synthesized": np.zeros(len(p.rnaIds), dtype=int),
                    },
                    "growth_limits": {
                        "ntp_used": np.zeros(len(p.ntp_ids), dtype=int),
                        "ntp_allocated": ntpCounts,
                    },
                    "rnap_data": {
                        "actual_elongations": 0,
                        "did_terminate": 0,
                        "termination_loss": 0,
                    },
                },
                "active_RNAPs": {},
                "RNAs": {},
            }
        else:
            # Get attributes from existing RNAs
            (
                TU_index_all_RNAs,
                length_all_RNAs,
                is_full_transcript,
                is_mRNA_all_RNAs,
                RNAP_index_all_RNAs,
            ) = attrs(
                state["RNAs"],
                [
                    "TU_index",
                    "transcript_length",
                    "is_full_transcript",
                    "is_mRNA",
                    "RNAP_index",
                ],
            )
            length_all_RNAs = length_all_RNAs.copy()

            update = {"listeners": {"growth_limits": {}}}

            # Determine sequences of RNAs that should be elongated
            is_partial_transcript = np.logical_not(is_full_transcript)
            partial_transcript_indexes = np.where(is_partial_transcript)[0]
            TU_index_partial_RNAs = TU_index_all_RNAs[is_partial_transcript]
            length_partial_RNAs = length_all_RNAs[is_partial_transcript]
            is_mRNA_partial_RNAs = is_mRNA_all_RNAs[is_partial_transcript]
            RNAP_index_partial_RNAs = RNAP_index_all_RNAs[is_partial_transcript]

            if p.trna_attenuation:
                cell_mass = state["listeners"]["mass"]["cell_mass"]
                cellVolume = cell_mass * units.fg / p.cell_density
                counts_to_molar = 1 / (p.n_avogadro * cellVolume)
                attenuation_probability = p.stop_probabilities(
                    counts_to_molar * counts(state["bulk_total"], p.charged_trnas_idx)
                )
                prob_lookup = {
                    tu: prob
                    for tu, prob in zip(
                        p.attenuated_rna_indices, attenuation_probability
                    )
                }
                tu_stop_probability = np.array(
                    [
                        prob_lookup.get(idx, 0)
                        * (length < p.location_lookup.get(idx, 0))
                        for idx, length in zip(TU_index_partial_RNAs, length_partial_RNAs)
                    ]
                )
                rna_to_attenuate = stochasticRound(
                    p.random_state, tu_stop_probability
                ).astype(bool)
            else:
                attenuation_probability = np.zeros(len(p.attenuated_rna_indices))
                rna_to_attenuate = np.zeros(len(TU_index_partial_RNAs), bool)
            rna_to_elongate = ~rna_to_attenuate

            sequences = buildSequences(
                p.rnaSequences,
                TU_index_partial_RNAs,
                length_partial_RNAs,
                p.elongation_rates,
            )

            # Polymerize transcripts based on sequences and available nucleotides
            reactionLimit = ntpCounts.sum()
            result = polymerize(
                sequences[rna_to_elongate],
                ntpCounts,
                reactionLimit,
                p.random_state,
                p.elongation_rates[TU_index_partial_RNAs][rna_to_elongate],
                p.variable_elongation,
            )

            sequence_elongations = np.zeros_like(length_partial_RNAs)
            sequence_elongations[rna_to_elongate] = result.sequenceElongation
            ntps_used = result.monomerUsages
            did_stall_mask = result.sequences_limited_elongation

            # Calculate changes in mass associated with polymerization
            added_mass = computeMassIncrease(
                sequences, sequence_elongations, p.ntWeights
            )
            did_initialize = (length_partial_RNAs == 0) & (sequence_elongations > 0)
            added_mass[did_initialize] += p.endWeight

            # Calculate updated transcript lengths
            updated_transcript_lengths = length_partial_RNAs + sequence_elongations

            # Get attributes of active RNAPs
            coordinates, is_forward, RNAP_unique_index = attrs(
                state["active_RNAPs"], ["coordinates", "is_forward", "unique_index"]
            )

            # Active RNAP count should equal partial transcript count
            assert len(RNAP_unique_index) == len(RNAP_index_partial_RNAs)

            # All partial RNAs must be linked to an RNAP
            assert np.count_nonzero(RNAP_index_partial_RNAs == -1) == 0

            # Get mapping indexes between partial RNAs to RNAPs
            partial_RNA_to_RNAP_mapping, _ = get_mapping_arrays(
                RNAP_index_partial_RNAs, RNAP_unique_index
            )

            # Rescale boolean array of directions to an array of 1's and -1's.
            # True is converted to 1, False is converted to -1.
            direction_rescaled = (2 * (is_forward - 0.5)).astype(np.int64)

            # Compute the updated coordinates of RNAPs. Coordinates of RNAPs
            # moving in the positive direction are increased, whereas coordinates
            # of RNAPs moving in the negative direction are decreased.
            updated_coordinates = coordinates + np.multiply(
                direction_rescaled, sequence_elongations[partial_RNA_to_RNAP_mapping]
            )

            # Reset coordinates of RNAPs that cross the boundaries between right
            # and left replichores
            updated_coordinates[updated_coordinates > p.replichore_lengths[0]] -= (
                p.chromosome_length
            )
            updated_coordinates[updated_coordinates < -p.replichore_lengths[1]] += (
                p.chromosome_length
            )

            # Update transcript lengths of RNAs and coordinates of RNAPs
            length_all_RNAs[is_partial_transcript] = updated_transcript_lengths

            # Update added submasses of RNAs. Masses of partial mRNAs are counted
            # as mRNA mass as they are already functional, but the masses of other
            # types of partial RNAs are counted as nonspecific RNA mass.
            added_nsRNA_mass_all_RNAs = np.zeros_like(TU_index_all_RNAs, dtype=np.float64)
            added_mRNA_mass_all_RNAs = np.zeros_like(TU_index_all_RNAs, dtype=np.float64)

            added_nsRNA_mass_all_RNAs[is_partial_transcript] = np.multiply(
                added_mass, np.logical_not(is_mRNA_partial_RNAs)
            )
            added_mRNA_mass_all_RNAs[is_partial_transcript] = np.multiply(
                added_mass, is_mRNA_partial_RNAs
            )

            # Determine if transcript has reached the end of the sequence
            terminal_lengths = p.rnaLengths[TU_index_partial_RNAs]
            did_terminate_mask = updated_transcript_lengths == terminal_lengths
            terminated_RNAs = np.bincount(
                TU_index_partial_RNAs[did_terminate_mask],
                minlength=p.rnaSequences.shape[0],
            )

            # Update is_full_transcript attribute of RNAs
            is_full_transcript_updated = is_full_transcript.copy()
            is_full_transcript_updated[partial_transcript_indexes[did_terminate_mask]] = (
                True
            )

            n_terminated = did_terminate_mask.sum()
            n_initialized = did_initialize.sum()
            n_elongations = ntps_used.sum()

            # Get counts of new bulk RNAs
            n_new_bulk_RNAs = terminated_RNAs.copy()
            n_new_bulk_RNAs[p.is_mRNA] = 0

            update["RNAs"] = {
                "set": {
                    "transcript_length": length_all_RNAs,
                    "is_full_transcript": is_full_transcript_updated,
                    "massDiff_nonspecific_RNA": attrs(
                        state["RNAs"], ["massDiff_nonspecific_RNA"]
                    )[0]
                    + added_nsRNA_mass_all_RNAs,
                    "massDiff_mRNA": attrs(state["RNAs"], ["massDiff_mRNA"])[0]
                    + added_mRNA_mass_all_RNAs,
                },
                "delete": partial_transcript_indexes[
                    np.logical_and(did_terminate_mask, np.logical_not(is_mRNA_partial_RNAs))
                ],
            }
            update["active_RNAPs"] = {
                "set": {"coordinates": updated_coordinates},
                "delete": np.where(did_terminate_mask[partial_RNA_to_RNAP_mapping])[0],
            }

            # Attenuation removes RNAs and RNAPs
            counts_attenuated = np.zeros(len(p.attenuated_rna_indices), dtype=int)
            if np.any(rna_to_attenuate):
                for idx in TU_index_partial_RNAs[rna_to_attenuate]:
                    counts_attenuated[p.attenuated_rna_indices_lookup[idx]] += 1
                update["RNAs"]["delete"] = np.append(
                    update["RNAs"]["delete"], partial_transcript_indexes[rna_to_attenuate]
                )
                update["active_RNAPs"]["delete"] = np.append(
                    update["active_RNAPs"]["delete"],
                    np.where(rna_to_attenuate[partial_RNA_to_RNAP_mapping])[0],
                )
            n_attenuated = rna_to_attenuate.sum()

            # Handle stalled elongation
            n_total_stalled = did_stall_mask.sum()
            if p.recycle_stalled_elongation and (n_total_stalled > 0):
                # Remove RNAPs that were bound to stalled elongation transcripts
                # and increment counts of inactive RNAPs
                update["active_RNAPs"]["delete"] = np.append(
                    update["active_RNAPs"]["delete"],
                    np.where(did_stall_mask[partial_RNA_to_RNAP_mapping])[0],
                )
                update["bulk"].append((p.inactive_RNAP_idx, n_total_stalled))

                # Remove partial transcripts from stalled elongation
                update["RNAs"]["delete"] = np.append(
                    update["RNAs"]["delete"], partial_transcript_indexes[did_stall_mask]
                )
                stalled_sequence_lengths = updated_transcript_lengths[did_stall_mask]
                n_initiated_sequences = np.count_nonzero(stalled_sequence_lengths)

                if n_initiated_sequences > 0:
                    # Get the full sequence of stalled transcripts
                    stalled_sequences = buildSequences(
                        p.rnaSequences,
                        TU_index_partial_RNAs[did_stall_mask],
                        np.zeros(n_total_stalled, dtype=np.int64),
                        np.full(n_total_stalled, updated_transcript_lengths.max()),
                    )

                    # Count the number of fragment bases in these transcripts up
                    # until the stalled length
                    base_counts = np.zeros(p.n_fragment_bases, dtype=np.int64)
                    for sl, seq in zip(stalled_sequence_lengths, stalled_sequences):
                        base_counts += np.bincount(
                            seq[:sl], minlength=p.n_fragment_bases
                        )

                    # Increment counts of fragment NTPs and phosphates
                    update["bulk"].append((p.fragmentBases_idx, base_counts))
                    update["bulk"].append((p.ppi_idx, n_initiated_sequences))

            update.setdefault("bulk", []).append((p.ntps_idx, -ntps_used))
            update["bulk"].append((p.bulk_RNA_idx, n_new_bulk_RNAs))
            update["bulk"].append((p.inactive_RNAP_idx, n_terminated + n_attenuated))
            update["bulk"].append((p.ppi_idx, n_elongations - n_initialized))

            # Write outputs to listeners
            update["listeners"]["transcript_elongation_listener"] = {
                "count_rna_synthesized": terminated_RNAs,
                "count_NTPs_used": n_elongations,
                "attenuation_probability": attenuation_probability,
                "counts_attenuated": counts_attenuated,
            }
            update["listeners"]["growth_limits"] = {"ntp_used": ntps_used}
            update["listeners"]["rnap_data"] = {
                "actual_elongations": sequence_elongations.sum(),
                "did_terminate": did_terminate_mask.sum(),
                "termination_loss": (terminal_lengths - length_partial_RNAs)[
                    did_terminate_mask
                ].sum(),
                "did_stall": n_total_stalled,
            }
        # --- end inlined ---
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update


def get_mapping_arrays(x, y):
    """
    Returns the array of indexes of each element of array x in array y, and
    vice versa. Assumes that the elements of x and y are unique, and
    set(x) == set(y).
    """

    def argsort_unique(idx):
        """
        Quicker argsort for arrays that are permutations of np.arange(n).
        """
        n = idx.size
        argsort_idx = np.empty(n, dtype=np.int64)
        argsort_idx[idx] = np.arange(n)
        return argsort_idx

    x_argsort = np.argsort(x)
    y_argsort = np.argsort(y)

    x_to_y = x_argsort[argsort_unique(y_argsort)]
    y_to_x = y_argsort[argsort_unique(x_argsort)]

    return x_to_y, y_to_x


def format_data(data, bulk_ids, rna_dtypes, rnap_dtypes, submass_dtypes):
    # Format unique and bulk data for assertions
    data["unique"]["RNA"] = [
        np.array(list(map(tuple, zip(*val))), dtype=rna_dtypes + submass_dtypes)
        for val in data["unique"]["RNA"]
    ]
    data["unique"]["active_RNAP"] = [
        np.array(list(map(tuple, zip(*val))), dtype=rnap_dtypes + submass_dtypes)
        for val in data["unique"]["active_RNAP"]
    ]
    bulk_timeseries = np.array(data["bulk"])
    data["bulk"] = {
        bulk_id: bulk_timeseries[:, i] for i, bulk_id in enumerate(bulk_ids)
    }
    return data


# ======================================================================
# polypeptide_initiation
# ======================================================================

class PolypeptideInitiationLogic:
    """Polypeptide Initiation — shared state container for Requester/Evolver."""

    name = "ecoli-polypeptide-initiation"
    topology = {
    "environment": ("environment",),
    "listeners": ("listeners",),
    "active_ribosome": ("unique", "active_ribosome"),
    "RNA": ("unique", "RNA"),
    "bulk": ("bulk",),
    "timestep": ("timestep",),
}
    defaults = {
        "protein_lengths": [],
        "translation_efficiencies": [],
        "active_ribosome_fraction": {},
        "elongation_rates": {},
        "variable_elongation": False,
        "make_elongation_rates": lambda x: [],
        "rna_id_to_cistron_indexes": {},
        "cistron_start_end_pos_in_tu": {},
        "tu_ids": [],
        "active_ribosome_footprint_size": 24 * units.nt,
        "cistron_to_monomer_mapping": {},
        "cistron_tu_mapping_matrix": {},
        "monomer_index_to_cistron_index": {},
        "monomer_index_to_tu_indexes": {},
        "ribosome30S": "ribosome30S",
        "ribosome50S": "ribosome50S",
        "seed": 0,
        "monomer_ids": [],
        "emit_unique": False,
        "time_step": 1,
    }

    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

        # Load parameters
        self.protein_lengths = self.parameters["protein_lengths"]
        self.translation_efficiencies = self.parameters["translation_efficiencies"]
        self.active_ribosome_fraction = self.parameters["active_ribosome_fraction"]
        self.ribosome_elongation_rates_dict = self.parameters["elongation_rates"]
        self.variable_elongation = self.parameters["variable_elongation"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]

        self.rna_id_to_cistron_indexes = self.parameters["rna_id_to_cistron_indexes"]
        self.cistron_start_end_pos_in_tu = self.parameters[
            "cistron_start_end_pos_in_tu"
        ]
        self.tu_ids = self.parameters["tu_ids"]
        self.n_TUs = len(self.tu_ids)
        # Convert ribosome footprint size from nucleotides to amino acids
        self.active_ribosome_footprint_size = (
            self.parameters["active_ribosome_footprint_size"] / 3
        )

        # Get mapping from cistrons to protein monomers and TUs
        self.cistron_to_monomer_mapping = self.parameters["cistron_to_monomer_mapping"]
        self.cistron_tu_mapping_matrix = self.parameters["cistron_tu_mapping_matrix"]
        self.monomer_index_to_cistron_index = self.parameters[
            "monomer_index_to_cistron_index"
        ]
        self.monomer_index_to_tu_indexes = self.parameters[
            "monomer_index_to_tu_indexes"
        ]

        self.ribosome30S = self.parameters["ribosome30S"]
        self.ribosome50S = self.parameters["ribosome50S"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.empty_update = {
            "listeners": {
                "ribosome_data": {
                    "ribosomes_initialized": 0,
                    "prob_translation_per_transcript": 0.0,
                }
            }
        }

        self.monomer_ids = self.parameters["monomer_ids"]

        # Helper indices for Numpy indexing
        self.ribosome30S_idx = None

    def calculate_activation_prob(
        self,
        fracActiveRibosome,
        proteinLengths,
        ribosomeElongationRates,
        proteinInitProb,
        timeStepSec,
    ):
        """
        Calculates the expected ribosome termination rate based on the ribosome
        elongation rate

        Args:
            allTranslationTimes: Vector of times required to translate each
                protein
            allTranslationTimestepCounts: Vector of numbers of timesteps
                required to translate each protein
            averageTranslationTimeStepCounts: Average number of timesteps
                required to translate a protein, weighted by initiation
                probabilities
            expectedTerminationRate: Average number of terminations in one
                timestep for one protein
        """
        allTranslationTimes = 1.0 / ribosomeElongationRates * proteinLengths
        allTranslationTimestepCounts = np.ceil(allTranslationTimes / timeStepSec)
        averageTranslationTimestepCounts = np.dot(
            allTranslationTimestepCounts, proteinInitProb
        )
        expectedTerminationRate = 1.0 / averageTranslationTimestepCounts

        # Modify given fraction of active ribosomes to take into account early
        # terminations in between timesteps
        # allFractionTimeInactive: Vector of probabilities an "active" ribosome
        #   will in effect be "inactive" because it has terminated during a
        #   timestep
        # averageFractionTimeInactive: Average probability of an "active"
        #   ribosome being in effect "inactive", weighted by initiation
        #   probabilities
        # effectiveFracActiveRnap: New higher "goal" for fraction of active
        #   ribosomes, considering that the "effective" fraction is lower than
        #   what the listener sees
        allFractionTimeInactive = (
            1 - allTranslationTimes / timeStepSec / allTranslationTimestepCounts
        )
        averageFractionTimeInactive = np.dot(allFractionTimeInactive, proteinInitProb)
        effectiveFracActiveRibosome = (
            fracActiveRibosome * 1 / (1 - averageFractionTimeInactive)
        )

        # Return activation probability that will balance out the expected
        # termination rate
        activationProb = (
            effectiveFracActiveRibosome
            * expectedTerminationRate
            / (1 - effectiveFracActiveRibosome)
        )

        # The upper bound for the activation probability is temporarily set to
        # 1.0 to prevent negative molecule counts. This will lower the fraction
        # of active ribosomes for timesteps longer than roughly 1.8s.
        if activationProb >= 1.0:
            activationProb = 1

        return activationProb


class PolypeptideInitiationRequester(_SafeInvokeMixin, Step):
    """Reads stores to compute initiation request. Writes to request store."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'RNA': UniqueNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
            'listeners': ListenerStore(),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from calculate_request ---
        if p.ribosome30S_idx is None:
            bulk_ids = state["bulk"]["id"]
            p.ribosome30S_idx = bulk_name_to_idx(p.ribosome30S, bulk_ids)
            p.ribosome50S_idx = bulk_name_to_idx(p.ribosome50S, bulk_ids)

        current_media_id = state["environment"]["media_id"]

        request = {
            "bulk": [
                (p.ribosome30S_idx, counts(state["bulk"], p.ribosome30S_idx)),
                (p.ribosome50S_idx, counts(state["bulk"], p.ribosome50S_idx)),
            ]
        }

        p.fracActiveRibosome = p.active_ribosome_fraction[current_media_id]

        # Read ribosome elongation rate from last timestep
        p.ribosomeElongationRate = state["listeners"]["ribosome_data"][
            "effective_elongation_rate"
        ]
        # If the ribosome elongation rate is zero (which is always the case for
        # the first timestep), set ribosome elongation rate to the one in
        # dictionary
        if p.ribosomeElongationRate == 0:
            p.ribosomeElongationRate = p.ribosome_elongation_rates_dict[
                current_media_id
            ].asNumber(units.aa / units.s)
        p.elongation_rates = p.make_elongation_rates(
            p.random_state,
            p.ribosomeElongationRate,
            1,  # want elongation rate, not lengths adjusted for time step
            p.variable_elongation,
        )

        # Ensure rates are never zero
        p.elongation_rates = np.fmax(p.elongation_rates, 1)
        # --- end inlined ---
        p.request_set = True

        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request

        listeners = request.pop('listeners', None)
        if listeners is not None:
            result['listeners'] = listeners

        return result


class PolypeptideInitiationEvolver(_SafeInvokeMixin, Step):
    """Reads allocation, writes bulk/unique/listener updates."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'RNA': UniqueNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'RNA': UniqueNumpyUpdate(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)

        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)

        if not self.process.request_set:
            return {}

        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from evolve_state ---
        # Calculate number of ribosomes that could potentially be initialized
        # based on counts of free 30S and 50S subunits
        inactive_ribosome_count = np.min(
            [
                counts(state["bulk"], p.ribosome30S_idx),
                counts(state["bulk"], p.ribosome50S_idx),
            ]
        )

        # Calculate actual number of ribosomes that should be activated based
        # on probabilities
        (
            TU_index_RNAs,
            transcript_lengths,
            can_translate,
            is_full_transcript,
            unique_index_RNAs,
        ) = attrs(
            state["RNA"],
            [
                "TU_index",
                "transcript_length",
                "can_translate",
                "is_full_transcript",
                "unique_index",
            ],
        )
        TU_index_mRNAs = TU_index_RNAs[can_translate]
        length_mRNAs = transcript_lengths[can_translate]
        unique_index_mRNAs = unique_index_RNAs[can_translate]
        is_full_transcript_mRNAs = is_full_transcript[can_translate]
        is_incomplete_transcript_mRNAs = np.logical_not(is_full_transcript_mRNAs)

        # Calculate counts of each mRNA cistron from fully transcribed
        # transcription units
        TU_index_full_mRNAs = TU_index_mRNAs[is_full_transcript_mRNAs]
        TU_counts_full_mRNAs = np.bincount(TU_index_full_mRNAs, minlength=p.n_TUs)
        cistron_counts = p.cistron_tu_mapping_matrix.dot(TU_counts_full_mRNAs)

        # Calculate counts of each mRNA cistron from partially transcribed
        # transcription units
        TU_index_incomplete_mRNAs = TU_index_mRNAs[is_incomplete_transcript_mRNAs]
        length_incomplete_mRNAs = length_mRNAs[is_incomplete_transcript_mRNAs]

        for TU_index, length in zip(TU_index_incomplete_mRNAs, length_incomplete_mRNAs):
            cistron_indexes = p.rna_id_to_cistron_indexes(p.tu_ids[TU_index])
            cistron_start_positions = np.array(
                [
                    p.cistron_start_end_pos_in_tu[(cistron_index, TU_index)][0]
                    for cistron_index in cistron_indexes
                ]
            )

            cistron_counts[cistron_indexes] += length > cistron_start_positions

        # Calculate initiation probabilities for ribosomes based on mRNA counts
        # and associated mRNA translational efficiencies
        protein_init_prob = normalize(
            cistron_counts[p.cistron_to_monomer_mapping]
            * p.translation_efficiencies
        )
        target_protein_init_prob = protein_init_prob.copy()

        # Calculate actual number of ribosomes that should be activated based
        # on probabilities
        activation_prob = p.calculate_activation_prob(
            p.fracActiveRibosome,
            p.protein_lengths,
            p.elongation_rates,
            target_protein_init_prob,
            state["timestep"],
        )

        n_ribosomes_to_activate = np.int64(activation_prob * inactive_ribosome_count)

        if n_ribosomes_to_activate == 0:
            update = {
                "listeners": {
                    "ribosome_data": zero_listener(state["listeners"]["ribosome_data"])
                }
            }
        else:
            # Cap the initiation probabilities at the maximum level physically
            # allowed from the known ribosome footprint sizes based on the
            # number of mRNAs
            max_p = (
                p.ribosomeElongationRate
                / p.active_ribosome_footprint_size
                * (units.s)
                * state["timestep"]
                / n_ribosomes_to_activate
            ).asNumber()
            max_p_per_protein = max_p * cistron_counts[p.cistron_to_monomer_mapping]
            is_overcrowded = protein_init_prob > max_p_per_protein

            # Initialize flag to record if the number of ribosomes activated at this
            # time step needed to be reduced to prevent overcrowding
            is_n_ribosomes_to_activate_reduced = False

            # If needed, resolve overcrowding
            while np.any(protein_init_prob > max_p_per_protein):
                if protein_init_prob[~is_overcrowded].sum() != 0:
                    # Resolve overcrowding through rescaling (preferred)
                    protein_init_prob[is_overcrowded] = max_p_per_protein[is_overcrowded]
                    scale_the_rest_by = (
                        1.0 - protein_init_prob[is_overcrowded].sum()
                    ) / protein_init_prob[~is_overcrowded].sum()
                    protein_init_prob[~is_overcrowded] *= scale_the_rest_by
                    is_overcrowded |= protein_init_prob > max_p_per_protein
                else:
                    # If we cannot resolve the overcrowding through rescaling,
                    # we need to activate fewer ribosomes. Set the number of
                    # ribosomes to activate so that there will be no overcrowding.
                    is_n_ribosomes_to_activate_reduced = True
                    max_index = np.argmax(
                        protein_init_prob[is_overcrowded]
                        / max_p_per_protein[is_overcrowded]
                    )
                    max_init_prob = protein_init_prob[is_overcrowded][max_index]
                    associated_cistron_counts = cistron_counts[
                        p.cistron_to_monomer_mapping
                    ][is_overcrowded][max_index]
                    n_ribosomes_to_activate = np.int64(
                        (
                            p.ribosomeElongationRate
                            / p.active_ribosome_footprint_size
                            * (units.s)
                            * state["timestep"]
                            / max_init_prob
                            * associated_cistron_counts
                        ).asNumber()
                    )

                    # Update maximum probabilities based on new number of activated
                    # ribosomes.
                    max_p = (
                        p.ribosomeElongationRate
                        / p.active_ribosome_footprint_size
                        * (units.s)
                        * state["timestep"]
                        / n_ribosomes_to_activate
                    ).asNumber()
                    max_p_per_protein = (
                        max_p * cistron_counts[p.cistron_to_monomer_mapping]
                    )
                    is_overcrowded = protein_init_prob > max_p_per_protein
                    assert is_overcrowded.sum() == 0  # We expect no overcrowding

            # Compute actual transcription probabilities of each transcript
            actual_protein_init_prob = protein_init_prob.copy()

            # Sample multinomial distribution to determine which mRNAs have full
            # 70S ribosomes initialized on them
            n_new_proteins = p.random_state.multinomial(
                n_ribosomes_to_activate, protein_init_prob
            )

            # Build attributes for active ribosomes.
            # Each ribosome is assigned a protein index for the protein that
            # corresponds to the polypeptide it will polymerize. This is done in
            # blocks of protein ids for efficiency.
            protein_indexes = np.empty(n_ribosomes_to_activate, np.int64)
            mRNA_indexes = np.empty(n_ribosomes_to_activate, np.int64)
            positions_on_mRNA = np.empty(n_ribosomes_to_activate, np.int64)
            nonzero_count = n_new_proteins > 0
            start_index = 0

            for protein_index, protein_counts in zip(
                np.arange(n_new_proteins.size)[nonzero_count], n_new_proteins[nonzero_count]
            ):
                # Set protein index
                protein_indexes[start_index : start_index + protein_counts] = protein_index

                cistron_index = p.monomer_index_to_cistron_index[protein_index]

                attribute_indexes = []
                cistron_start_positions = []

                for TU_index in p.monomer_index_to_tu_indexes[protein_index]:
                    attribute_indexes_this_TU = np.where(TU_index_mRNAs == TU_index)[0]
                    cistron_start_position = p.cistron_start_end_pos_in_tu[
                        (cistron_index, TU_index)
                    ][0]
                    is_transcript_long_enough = (
                        length_mRNAs[attribute_indexes_this_TU] >= cistron_start_position
                    )

                    attribute_indexes.extend(
                        attribute_indexes_this_TU[is_transcript_long_enough]
                    )
                    cistron_start_positions.extend(
                        [cistron_start_position]
                        * len(attribute_indexes_this_TU[is_transcript_long_enough])
                    )

                n_mRNAs = len(attribute_indexes)

                # Distribute ribosomes among these mRNAs
                n_ribosomes_per_RNA = p.random_state.multinomial(
                    protein_counts, np.full(n_mRNAs, 1.0 / n_mRNAs)
                )

                # Get unique indexes of each mRNA
                mRNA_indexes[start_index : start_index + protein_counts] = np.repeat(
                    unique_index_mRNAs[attribute_indexes], n_ribosomes_per_RNA
                )

                positions_on_mRNA[start_index : start_index + protein_counts] = np.repeat(
                    cistron_start_positions, n_ribosomes_per_RNA
                )

                start_index += protein_counts

            # Create active 70S ribosomes and assign their attributes
            update = {
                "bulk": [
                    (p.ribosome30S_idx, -n_new_proteins.sum()),
                    (p.ribosome50S_idx, -n_new_proteins.sum()),
                ],
                "active_ribosome": {
                    "add": {
                        "protein_index": protein_indexes,
                        "peptide_length": np.zeros(n_ribosomes_to_activate, dtype=np.int64),
                        "mRNA_index": mRNA_indexes,
                        "pos_on_mRNA": positions_on_mRNA,
                    },
                },
                "listeners": {
                    "ribosome_data": {
                        "did_initialize": n_new_proteins.sum(),
                        "ribosome_init_event_per_monomer": n_new_proteins,
                        "target_prob_translation_per_transcript": target_protein_init_prob,
                        "actual_prob_translation_per_transcript": actual_protein_init_prob,
                        "mRNA_is_overcrowded": is_overcrowded,
                        "max_p": max_p,
                        "max_p_per_protein": max_p_per_protein,
                        "is_n_ribosomes_to_activate_reduced": is_n_ribosomes_to_activate_reduced,
                    }
                },
            }
        # --- end inlined ---
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update


# ======================================================================
# polypeptide_elongation
# ======================================================================

MICROMOLAR_UNITS = units.umol / units.L
"""Units used for all concentrations."""
REMOVED_FROM_CHARGING = {"L-SELENOCYSTEINE[c]"}
"""Amino acids to remove from charging when running with 
``steady_state_trna_charging``"""


DEFAULT_AA_NAMES = [
    "L-ALPHA-ALANINE[c]",
    "ARG[c]",
    "ASN[c]",
    "L-ASPARTATE[c]",
    "CYS[c]",
    "GLT[c]",
    "GLN[c]",
    "GLY[c]",
    "HIS[c]",
    "ILE[c]",
    "LEU[c]",
    "LYS[c]",
    "MET[c]",
    "PHE[c]",
    "PRO[c]",
    "SER[c]",
    "THR[c]",
    "TRP[c]",
    "TYR[c]",
    "L-SELENOCYSTEINE[c]",
    "VAL[c]",
]


class PolypeptideElongationLogic:
    """Polypeptide Elongation — shared state container for Requester/Evolver.

    defaults:
        proteinIds: array length n of protein names
    """

    name = "ecoli-polypeptide-elongation"
    topology = {
    "environment": ("environment",),
    "boundary": ("boundary",),
    "listeners": ("listeners",),
    "active_ribosome": ("unique", "active_ribosome"),
    "bulk": ("bulk",),
    "polypeptide_elongation": ("process_state", "polypeptide_elongation"),
    # Non-partitioned counts
    "bulk_total": ("bulk",),
    "timestep": ("timestep",),
}
    defaults = {
        "time_step": 1,
        "n_avogadro": 6.02214076e23 / units.mol,
        "proteinIds": np.array([]),
        "proteinLengths": np.array([]),
        "proteinSequences": np.array([[]]),
        "aaWeightsIncorporated": np.array([]),
        "endWeight": np.array([2.99146113e-08]),
        "variable_elongation": False,
        "make_elongation_rates": (
            lambda random, rate, timestep, variable: np.array([])
        ),
        "next_aa_pad": 1,
        "ribosomeElongationRate": 17.388824902723737,
        "translation_aa_supply": {"minimal": np.array([])},
        "import_threshold": 1e-05,
        "aa_from_trna": np.zeros(21),
        "gtpPerElongation": 4.2,
        "aa_supply_in_charging": False,
        "mechanistic_translation_supply": False,
        "mechanistic_aa_transport": False,
        "ppgpp_regulation": False,
        "disable_ppgpp_elongation_inhibition": False,
        "trna_charging": False,
        "translation_supply": False,
        "mechanistic_supply": False,
        "ribosome30S": "ribosome30S",
        "ribosome50S": "ribosome50S",
        "amino_acids": DEFAULT_AA_NAMES,
        "aa_exchange_names": DEFAULT_AA_NAMES,
        "basal_elongation_rate": 22.0,
        "ribosomeElongationRateDict": {
            "minimal": 17.388824902723737 * units.aa / units.s
        },
        "uncharged_trna_names": np.array([]),
        "aaNames": DEFAULT_AA_NAMES,
        "aa_enzymes": [],
        "proton": "PROTON",
        "water": "H2O",
        "cellDensity": 1100 * units.g / units.L,
        "elongation_max": 22 * units.aa / units.s,
        "aa_from_synthetase": np.array([[]]),
        "charging_stoich_matrix": np.array([[]]),
        "charged_trna_names": [],
        "charging_molecule_names": [],
        "synthetase_names": [],
        "ppgpp_reaction_names": [],
        "ppgpp_reaction_metabolites": [],
        "ppgpp_reaction_stoich": np.array([[]]),
        "ppgpp_synthesis_reaction": "GDPPYPHOSKIN-RXN",
        "ppgpp_degradation_reaction": "PPGPPSYN-RXN",
        "aa_importers": [],
        "amino_acid_export": None,
        "synthesis_index": 0,
        "aa_exporters": [],
        "get_pathway_enzyme_counts_per_aa": None,
        "import_constraint_threshold": 0,
        "unit_conversion": 0,
        "elong_rate_by_ppgpp": 0,
        "amino_acid_import": None,
        "degradation_index": 1,
        "amino_acid_synthesis": None,
        "rela": "RELA",
        "spot": "SPOT",
        "ppgpp": "ppGpp",
        "kS": 100.0,
        "KMtf": 1.0,
        "KMaa": 100.0,
        "krta": 1.0,
        "krtf": 500.0,
        "KD_RelA": 0.26,
        "k_RelA": 75.0,
        "k_SpoT_syn": 2.6,
        "k_SpoT_deg": 0.23,
        "KI_SpoT": 20.0,
        "aa_supply_scaling": lambda aa_conc, aa_in_media: 0,
        "seed": 0,
        "emit_unique": False,
    }

    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

        # Simulation options
        self.aa_supply_in_charging = self.parameters["aa_supply_in_charging"]
        self.mechanistic_translation_supply = self.parameters[
            "mechanistic_translation_supply"
        ]
        self.mechanistic_aa_transport = self.parameters["mechanistic_aa_transport"]
        self.ppgpp_regulation = self.parameters["ppgpp_regulation"]
        self.disable_ppgpp_elongation_inhibition = self.parameters[
            "disable_ppgpp_elongation_inhibition"
        ]
        self.variable_elongation = self.parameters["variable_elongation"]
        self.variable_polymerize = self.ppgpp_regulation or self.variable_elongation
        translation_supply = self.parameters["translation_supply"]
        trna_charging = self.parameters["trna_charging"]

        # Load parameters
        self.n_avogadro = self.parameters["n_avogadro"]
        self.proteinIds = self.parameters["proteinIds"]
        self.protein_lengths = self.parameters["proteinLengths"]
        self.proteinSequences = self.parameters["proteinSequences"]
        self.aaWeightsIncorporated = self.parameters["aaWeightsIncorporated"]
        self.endWeight = self.parameters["endWeight"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]
        self.next_aa_pad = self.parameters["next_aa_pad"]

        self.ribosome30S = self.parameters["ribosome30S"]
        self.ribosome50S = self.parameters["ribosome50S"]
        self.amino_acids = self.parameters["amino_acids"]
        self.aa_exchange_names = self.parameters["aa_exchange_names"]
        self.aa_environment_names = [aa[:-3] for aa in self.aa_exchange_names]
        self.aa_enzymes = self.parameters["aa_enzymes"]

        self.ribosomeElongationRate = self.parameters["ribosomeElongationRate"]

        # Amino acid supply calculations
        self.translation_aa_supply = self.parameters["translation_aa_supply"]
        self.import_threshold = self.parameters["import_threshold"]

        # Used for figure in publication
        self.trpAIndex = np.where(self.proteinIds == "TRYPSYN-APROTEIN[c]")[0][0]

        self.elngRateFactor = 1.0

        # Data structures for charging
        self.aa_from_trna = self.parameters["aa_from_trna"]

        # Set modeling method
        # TODO: Test that these models all work properly
        if trna_charging:
            self.elongation_model = SteadyStateElongationModel(self.parameters, self)
        elif translation_supply:
            self.elongation_model = TranslationSupplyElongationModel(
                self.parameters, self
            )
        else:
            self.elongation_model = BaseElongationModel(self.parameters, self)

        # Growth associated maintenance energy requirements for elongations
        self.gtpPerElongation = self.parameters["gtpPerElongation"]
        # Need to account for ATP hydrolysis for charging that has been
        # removed from measured GAM (ATP -> AMP is 2 hydrolysis reactions)
        # if charging reactions are not explicitly modeled
        if not trna_charging:
            self.gtpPerElongation += 2

        # basic molecule names
        self.proton = self.parameters["proton"]
        self.water = self.parameters["water"]
        self.rela = self.parameters["rela"]
        self.spot = self.parameters["spot"]
        self.ppgpp = self.parameters["ppgpp"]
        self.aa_importers = self.parameters["aa_importers"]
        self.aa_exporters = self.parameters["aa_exporters"]
        # Numpy index for bulk molecule
        self.proton_idx = None

        # Names of molecules associated with tRNA charging
        self.ppgpp_reaction_metabolites = self.parameters["ppgpp_reaction_metabolites"]
        self.uncharged_trna_names = self.parameters["uncharged_trna_names"]
        self.charged_trna_names = self.parameters["charged_trna_names"]
        self.charging_molecule_names = self.parameters["charging_molecule_names"]
        self.synthetase_names = self.parameters["synthetase_names"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)



class PolypeptideElongationRequester(_SafeInvokeMixin, Step):
    """Reads stores to compute polypeptide elongation request."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'boundary': InPlaceDict(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'polypeptide_elongation': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
            'listeners': ListenerStore(),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from calculate_request ---
        if p.proton_idx is None:
            bulk_ids = state["bulk"]["id"]
            p.proton_idx = bulk_name_to_idx(p.proton, bulk_ids)
            p.water_idx = bulk_name_to_idx(p.water, bulk_ids)
            p.rela_idx = bulk_name_to_idx(p.rela, bulk_ids)
            p.spot_idx = bulk_name_to_idx(p.spot, bulk_ids)
            p.ppgpp_idx = bulk_name_to_idx(p.ppgpp, bulk_ids)
            p.monomer_idx = bulk_name_to_idx(p.proteinIds, bulk_ids)
            p.amino_acid_idx = bulk_name_to_idx(p.amino_acids, bulk_ids)
            p.aa_enzyme_idx = bulk_name_to_idx(p.aa_enzymes, bulk_ids)
            p.ppgpp_rxn_metabolites_idx = bulk_name_to_idx(
                p.ppgpp_reaction_metabolites, bulk_ids
            )
            p.uncharged_trna_idx = bulk_name_to_idx(
                p.uncharged_trna_names, bulk_ids
            )
            p.charged_trna_idx = bulk_name_to_idx(p.charged_trna_names, bulk_ids)
            p.charging_molecule_idx = bulk_name_to_idx(
                p.charging_molecule_names, bulk_ids
            )
            p.synthetase_idx = bulk_name_to_idx(p.synthetase_names, bulk_ids)
            p.ribosome30S_idx = bulk_name_to_idx(p.ribosome30S, bulk_ids)
            p.ribosome50S_idx = bulk_name_to_idx(p.ribosome50S, bulk_ids)
            p.aa_importer_idx = bulk_name_to_idx(p.aa_importers, bulk_ids)
            p.aa_exporter_idx = bulk_name_to_idx(p.aa_exporters, bulk_ids)

        # MODEL SPECIFIC: get ribosome elongation rate
        p.ribosomeElongationRate = p.elongation_model.elongation_rate(state)

        # If there are no active ribosomes, return immediately
        if state["active_ribosome"]["_entryState"].sum() == 0:
            request = {"listeners": {"ribosome_data": {}, "growth_limits": {}}}
        else:
            # Build sequences to request appropriate amount of amino acids to
            # polymerize for next timestep
            (
                proteinIndexes,
                peptideLengths,
            ) = attrs(state["active_ribosome"], ["protein_index", "peptide_length"])

            p.elongation_rates = p.make_elongation_rates(
                p.random_state,
                p.ribosomeElongationRate,
                state["timestep"],
                p.variable_elongation,
            )

            sequences = buildSequences(
                p.proteinSequences, proteinIndexes, peptideLengths, p.elongation_rates
            )

            sequenceHasAA = sequences != polymerize.PAD_VALUE
            aasInSequences = np.bincount(sequences[sequenceHasAA], minlength=21)

            # Calculate AA supply for expected doubling of protein
            dryMass = state["listeners"]["mass"]["dry_mass"] * units.fg
            current_media_id = state["environment"]["media_id"]
            translation_supply_rate = (
                p.translation_aa_supply[current_media_id] * p.elngRateFactor
            )
            mol_aas_supplied = (
                translation_supply_rate * dryMass * state["timestep"] * units.s
            )
            p.aa_supply = units.strip_empty_units(mol_aas_supplied * p.n_avogadro)

            # MODEL SPECIFIC: Calculate AA request
            fraction_charged, aa_counts_for_translation, request = (
                p.elongation_model.request(state, aasInSequences)
            )

            # Write to listeners
            listeners = request.setdefault("listeners", {})
            ribosome_data_listener = listeners.setdefault("ribosome_data", {})
            ribosome_data_listener["translation_supply"] = (
                translation_supply_rate.asNumber()
            )
            growth_limits_listener = request["listeners"].setdefault("growth_limits", {})
            growth_limits_listener["fraction_trna_charged"] = np.dot(
                fraction_charged, p.aa_from_trna
            )
            growth_limits_listener["aa_pool_size"] = counts(
                state["bulk_total"], p.amino_acid_idx
            )
            growth_limits_listener["aa_request_size"] = aa_counts_for_translation
            # Simulations without mechanistic translation supply need this to be
            # manually zeroed after division
            proc_data = request.setdefault("polypeptide_elongation", {})
            proc_data.setdefault("aa_exchange_rates", np.zeros(len(p.amino_acids)))
        # --- end inlined ---
        p.request_set = True

        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request

        listeners = request.pop('listeners', None)
        if listeners is not None:
            result['listeners'] = listeners

        return result


class PolypeptideElongationEvolver(_SafeInvokeMixin, Step):
    """Reads allocation, writes bulk/unique/listener/boundary updates."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'boundary': InPlaceDict(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'polypeptide_elongation': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'polypeptide_elongation': InPlaceDict(),
            'boundary': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)

        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)

        if not self.process.request_set:
            return {}

        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from evolve_state ---
        update = {
            "listeners": {"ribosome_data": {}, "growth_limits": {}},
            "polypeptide_elongation": {},
            "active_ribosome": {},
            "bulk": [],
        }

        # Begin wcEcoli evolveState()
        # Set values for metabolism in case of early return
        update["polypeptide_elongation"]["gtp_to_hydrolyze"] = 0
        update["polypeptide_elongation"]["aa_count_diff"] = np.zeros(
            len(p.amino_acids), dtype=np.float64
        )

        # Get number of active ribosomes
        n_active_ribosomes = state["active_ribosome"]["_entryState"].sum()
        update["listeners"]["growth_limits"]["active_ribosome_allocated"] = (
            n_active_ribosomes
        )
        update["listeners"]["growth_limits"]["aa_allocated"] = counts(
            state["bulk"], p.amino_acid_idx
        )

        # If there are no active ribosomes, return immediately
        if n_active_ribosomes != 0:
            # Polypeptide elongation requires counts to be updated in real-time
            # so make a writeable copy of bulk counts to do so
            state["bulk"] = counts(state["bulk"], range(len(state["bulk"])))

            # Build amino acids sequences for each ribosome to polymerize
            protein_indexes, peptide_lengths, positions_on_mRNA = attrs(
                state["active_ribosome"],
                ["protein_index", "peptide_length", "pos_on_mRNA"],
            )

            all_sequences = buildSequences(
                p.proteinSequences,
                protein_indexes,
                peptide_lengths,
                p.elongation_rates + p.next_aa_pad,
            )
            sequences = all_sequences[:, : -p.next_aa_pad].copy()

            if sequences.size != 0:
                # Calculate elongation resource capacity
                aaCountInSequence = np.bincount(sequences[(sequences != polymerize.PAD_VALUE)])
                total_aa_counts = counts(state["bulk"], p.amino_acid_idx)
                charged_trna_counts = counts(state["bulk"], p.charged_trna_idx)

                # MODEL SPECIFIC: Get amino acid counts
                aa_counts_for_translation = p.elongation_model.final_amino_acids(
                    total_aa_counts, charged_trna_counts
                )

                # Using polymerization algorithm elongate each ribosome up to the limits
                # of amino acids, sequence, and GTP
                result = polymerize(
                    sequences,
                    aa_counts_for_translation,
                    10000000,  # Set to a large number, the limit is now taken care of in metabolism
                    p.random_state,
                    p.elongation_rates[protein_indexes],
                    variable_elongation=p.variable_polymerize,
                )

                sequence_elongations = result.sequenceElongation
                aas_used = result.monomerUsages
                nElongations = result.nReactions

                next_amino_acid = all_sequences[
                    np.arange(len(sequence_elongations)), sequence_elongations
                ]
                next_amino_acid_count = np.bincount(
                    next_amino_acid[next_amino_acid != polymerize.PAD_VALUE], minlength=21
                )

                # Update masses of ribosomes attached to polymerizing polypeptides
                added_protein_mass = computeMassIncrease(
                    sequences, sequence_elongations, p.aaWeightsIncorporated
                )

                updated_lengths = peptide_lengths + sequence_elongations
                updated_positions_on_mRNA = positions_on_mRNA + 3 * sequence_elongations

                didInitialize = (sequence_elongations > 0) & (peptide_lengths == 0)

                added_protein_mass[didInitialize] += p.endWeight

                # Write current average elongation to listener
                currElongRate = (sequence_elongations.sum() / n_active_ribosomes) / state[
                    "timestep"
                ]

                # Ribosomes that reach the end of their sequences are terminated and
                # dissociated into 30S and 50S subunits. The polypeptide that they are
                # polymerizing is converted into a protein in BulkMolecules
                terminalLengths = p.protein_lengths[protein_indexes]

                didTerminate = updated_lengths == terminalLengths

                terminatedProteins = np.bincount(
                    protein_indexes[didTerminate], minlength=p.proteinSequences.shape[0]
                )

                (protein_mass,) = attrs(state["active_ribosome"], ["massDiff_protein"])
                update["active_ribosome"].update(
                    {
                        "delete": np.where(didTerminate)[0],
                        "set": {
                            "massDiff_protein": protein_mass + added_protein_mass,
                            "peptide_length": updated_lengths,
                            "pos_on_mRNA": updated_positions_on_mRNA,
                        },
                    }
                )

                update["bulk"].append((p.monomer_idx, terminatedProteins))
                state["bulk"][p.monomer_idx] += terminatedProteins

                nTerminated = didTerminate.sum()
                nInitialized = didInitialize.sum()

                update["bulk"].append((p.ribosome30S_idx, nTerminated))
                update["bulk"].append((p.ribosome50S_idx, nTerminated))
                state["bulk"][p.ribosome30S_idx] += nTerminated
                state["bulk"][p.ribosome50S_idx] += nTerminated

                # MODEL SPECIFIC: evolve
                net_charged, aa_count_diff, evolve_update = p.elongation_model.evolve(
                    state,
                    total_aa_counts,
                    aas_used,
                    next_amino_acid_count,
                    nElongations,
                    nInitialized,
                )

                evolve_bulk_update = evolve_update.pop("bulk")
                update = deep_merge(update, evolve_update)
                update["bulk"].extend(evolve_bulk_update)

                update["polypeptide_elongation"]["aa_count_diff"] = aa_count_diff
                # GTP hydrolysis is carried out in Metabolism process for growth
                # associated maintenance. This is passed to metabolism.
                update["polypeptide_elongation"]["gtp_to_hydrolyze"] = (
                    p.gtpPerElongation * nElongations
                )

                # Write data to listeners
                update["listeners"]["growth_limits"]["net_charged"] = net_charged
                update["listeners"]["growth_limits"]["aas_used"] = aas_used
                update["listeners"]["growth_limits"]["aa_count_diff"] = aa_count_diff

                ribosome_data_listener = update["listeners"].setdefault("ribosome_data", {})
                ribosome_data_listener["effective_elongation_rate"] = currElongRate
                ribosome_data_listener["aa_count_in_sequence"] = aaCountInSequence
                ribosome_data_listener["aa_counts"] = aa_counts_for_translation
                ribosome_data_listener["actual_elongations"] = sequence_elongations.sum()
                ribosome_data_listener["actual_elongation_hist"] = np.histogram(
                    sequence_elongations, bins=np.arange(0, 23)
                )[0]
                ribosome_data_listener["elongations_non_terminating_hist"] = np.histogram(
                    sequence_elongations[~didTerminate], bins=np.arange(0, 23)
                )[0]
                ribosome_data_listener["did_terminate"] = didTerminate.sum()
                ribosome_data_listener["termination_loss"] = (
                    terminalLengths - peptide_lengths
                )[didTerminate].sum()
                ribosome_data_listener["num_trpA_terminated"] = terminatedProteins[
                    p.trpAIndex
                ]
                ribosome_data_listener["process_elongation_rate"] = (
                    p.ribosomeElongationRate / state["timestep"]
                )
        # --- end inlined ---
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update


class BaseElongationModel(object):
    """
    Base Model: Request amino acids according to upcoming sequence, assuming
    max ribosome elongation.
    """

    def __init__(self, parameters, process):
        self.parameters = parameters
        self.process = process
        self.basal_elongation_rate = self.parameters["basal_elongation_rate"]
        self.ribosomeElongationRateDict = self.parameters["ribosomeElongationRateDict"]

    def elongation_rate(self, states):
        """
        Sets ribosome elongation rate accordint to the media; returns
        max value of 22 amino acids/second.
        """
        current_media_id = states["environment"]["media_id"]
        rate = self.process.elngRateFactor * self.ribosomeElongationRateDict[
            current_media_id
        ].asNumber(units.aa / units.s)
        return np.min([self.basal_elongation_rate, rate])

    def amino_acid_counts(self, aasInSequences):
        return aasInSequences

    def request(
        self, states: dict, aasInSequences: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict]:
        aa_counts_for_translation = self.amino_acid_counts(aasInSequences)

        # Bulk requests have to be integers (wcEcoli implicitly casts floats to ints)
        requests = {
            "bulk": [
                (
                    self.process.amino_acid_idx,
                    aa_counts_for_translation.astype(np.int64),
                )
            ]
        }

        # Not modeling charging so set fraction charged to 0 for all tRNA
        fraction_charged = np.zeros(len(self.process.amino_acid_idx))

        return fraction_charged, aa_counts_for_translation.astype(float), requests

    def final_amino_acids(self, total_aa_counts, charged_trna_counts):
        return total_aa_counts

    def evolve(
        self,
        states,
        total_aa_counts,
        aas_used,
        next_amino_acid_count,
        nElongations,
        nInitialized,
    ):
        # Update counts of amino acids and water to reflect polymerization
        # reactions
        net_charged = np.zeros(
            len(self.parameters["uncharged_trna_names"]), dtype=np.int64
        )
        return (
            net_charged,
            np.zeros(len(self.process.amino_acids), dtype=np.float64),
            {
                "bulk": [
                    (self.process.amino_acid_idx, -aas_used),
                    (self.process.water_idx, nElongations - nInitialized),
                ]
            },
        )


class TranslationSupplyElongationModel(BaseElongationModel):
    """
    Translation Supply Model: Requests minimum of 1) upcoming amino acid
    sequence assuming max ribosome elongation (ie. Base Model) and 2)
    estimation based on doubling the proteome in one cell cycle (does not
    use ribosome elongation, computed in Parca).
    """

    def __init__(self, parameters, process):
        super().__init__(parameters, process)

    def elongation_rate(self, states):
        """
        Sets ribosome elongation rate accordint to the media; returns
        max value of 22 amino acids/second.
        """
        return self.basal_elongation_rate

    def amino_acid_counts(self, aasInSequences):
        # Check if this is required. It is a better request but there may be
        # fewer elongations.
        return np.fmin(self.process.aa_supply, aasInSequences)


class SteadyStateElongationModel(TranslationSupplyElongationModel):
    """
    Steady State Charging Model: Requests amino acids based on the
    Michaelis-Menten competitive inhibition model.
    """

    def __init__(self, parameters, process):
        super().__init__(parameters, process)

        # Cell parameters
        self.cellDensity = self.parameters["cellDensity"]

        # Names of molecules associated with tRNA charging
        self.charged_trna_names = self.parameters["charged_trna_names"]
        self.charging_molecule_names = self.parameters["charging_molecule_names"]
        self.synthetase_names = self.parameters["synthetase_names"]

        # Data structures for charging
        self.aa_from_synthetase = self.parameters["aa_from_synthetase"]
        self.charging_stoich_matrix = self.parameters["charging_stoich_matrix"]
        self.charging_molecules_not_aa = np.array(
            [
                mol not in set(self.parameters["amino_acids"])
                for mol in self.charging_molecule_names
            ]
        )

        # ppGpp synthesis
        self.ppgpp_reaction_metabolites = self.parameters["ppgpp_reaction_metabolites"]
        self.elong_rate_by_ppgpp = self.parameters["elong_rate_by_ppgpp"]

        # Parameters for tRNA charging, ribosome elongation and ppGpp reactions
        self.charging_params = {
            "kS": self.parameters["kS"],
            "KMaa": self.parameters["KMaa"],
            "KMtf": self.parameters["KMtf"],
            "krta": self.parameters["krta"],
            "krtf": self.parameters["krtf"],
            "max_elong_rate": float(
                self.parameters["elongation_max"].asNumber(units.aa / units.s)
            ),
            "charging_mask": np.array(
                [
                    aa not in REMOVED_FROM_CHARGING
                    for aa in self.parameters["amino_acids"]
                ]
            ),
            "unit_conversion": self.parameters["unit_conversion"],
        }
        self.ppgpp_params = {
            "KD_RelA": self.parameters["KD_RelA"],
            "k_RelA": self.parameters["k_RelA"],
            "k_SpoT_syn": self.parameters["k_SpoT_syn"],
            "k_SpoT_deg": self.parameters["k_SpoT_deg"],
            "KI_SpoT": self.parameters["KI_SpoT"],
            "ppgpp_reaction_stoich": self.parameters["ppgpp_reaction_stoich"],
            "synthesis_index": self.parameters["synthesis_index"],
            "degradation_index": self.parameters["degradation_index"],
        }

        # Amino acid supply calculations
        self.aa_supply_scaling = self.parameters["aa_supply_scaling"]

        self.amino_acid_synthesis = self.parameters["amino_acid_synthesis"]
        self.amino_acid_import = self.parameters["amino_acid_import"]
        self.amino_acid_export = self.parameters["amino_acid_export"]
        self.get_pathway_enzyme_counts_per_aa = self.parameters[
            "get_pathway_enzyme_counts_per_aa"
        ]

        # Comparing two values with units is faster than converting units
        # and comparing magnitudes
        self.import_constraint_threshold = (
            self.parameters["import_constraint_threshold"] * vivunits.mM
        )

    def elongation_rate(self, states):
        if (
            self.process.ppgpp_regulation
            and not self.process.disable_ppgpp_elongation_inhibition
        ):
            cell_mass = states["listeners"]["mass"]["cell_mass"] * units.fg
            cell_volume = cell_mass / self.cellDensity
            counts_to_molar = 1 / (self.process.n_avogadro * cell_volume)
            ppgpp_count = counts(states["bulk"], self.process.ppgpp_idx)
            ppgpp_conc = ppgpp_count * counts_to_molar
            rate = self.elong_rate_by_ppgpp(
                ppgpp_conc, self.basal_elongation_rate
            ).asNumber(units.aa / units.s)
        else:
            rate = super().elongation_rate(states)
        return rate

    def request(
        self, states: dict, aasInSequences: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict]:
        # Conversion from counts to molarity
        cell_mass = states["listeners"]["mass"]["cell_mass"] * units.fg
        dry_mass = states["listeners"]["mass"]["dry_mass"] * units.fg
        cell_volume = cell_mass / self.cellDensity
        self.counts_to_molar = 1 / (self.process.n_avogadro * cell_volume)

        # ppGpp related concentrations
        ppgpp_conc = self.counts_to_molar * counts(
            states["bulk_total"], self.process.ppgpp_idx
        )
        rela_conc = self.counts_to_molar * counts(
            states["bulk_total"], self.process.rela_idx
        )
        spot_conc = self.counts_to_molar * counts(
            states["bulk_total"], self.process.spot_idx
        )

        # Get counts and convert synthetase and tRNA to a per AA basis
        synthetase_counts = np.dot(
            self.aa_from_synthetase,
            counts(states["bulk_total"], self.process.synthetase_idx),
        )
        aa_counts = counts(states["bulk_total"], self.process.amino_acid_idx)
        uncharged_trna_array = counts(
            states["bulk_total"], self.process.uncharged_trna_idx
        )
        charged_trna_array = counts(states["bulk_total"], self.process.charged_trna_idx)
        uncharged_trna_counts = np.dot(self.process.aa_from_trna, uncharged_trna_array)
        charged_trna_counts = np.dot(self.process.aa_from_trna, charged_trna_array)
        ribosome_counts = states["active_ribosome"]["_entryState"].sum()

        # Get concentration
        f = aasInSequences / aasInSequences.sum()
        synthetase_conc = self.counts_to_molar * synthetase_counts
        aa_conc = self.counts_to_molar * aa_counts
        uncharged_trna_conc = self.counts_to_molar * uncharged_trna_counts
        charged_trna_conc = self.counts_to_molar * charged_trna_counts
        ribosome_conc = self.counts_to_molar * ribosome_counts

        # Calculate amino acid supply
        aa_in_media = np.array(
            [
                states["boundary"]["external"][aa] > self.import_constraint_threshold
                for aa in self.process.aa_environment_names
            ]
        )
        fwd_enzyme_counts, rev_enzyme_counts = self.get_pathway_enzyme_counts_per_aa(
            counts(states["bulk_total"], self.process.aa_enzyme_idx)
        )
        importer_counts = counts(states["bulk_total"], self.process.aa_importer_idx)
        exporter_counts = counts(states["bulk_total"], self.process.aa_exporter_idx)
        synthesis, fwd_saturation, rev_saturation = self.amino_acid_synthesis(
            fwd_enzyme_counts, rev_enzyme_counts, aa_conc
        )
        import_rates = self.amino_acid_import(
            aa_in_media,
            dry_mass,
            aa_conc,
            importer_counts,
            self.process.mechanistic_aa_transport,
        )
        export_rates = self.amino_acid_export(
            exporter_counts, aa_conc, self.process.mechanistic_aa_transport
        )
        exchange_rates = import_rates - export_rates

        supply_function = get_charging_supply_function(
            self.process.aa_supply_in_charging,
            self.process.mechanistic_translation_supply,
            self.process.mechanistic_aa_transport,
            self.amino_acid_synthesis,
            self.amino_acid_import,
            self.amino_acid_export,
            self.aa_supply_scaling,
            self.counts_to_molar,
            self.process.aa_supply,
            fwd_enzyme_counts,
            rev_enzyme_counts,
            dry_mass,
            importer_counts,
            exporter_counts,
            aa_in_media,
        )

        # Calculate steady state tRNA levels and resulting elongation rate
        self.charging_params["max_elong_rate"] = self.elongation_rate(states)
        (
            fraction_charged,
            v_rib,
            synthesis_in_charging,
            import_in_charging,
            export_in_charging,
        ) = calculate_trna_charging(
            synthetase_conc,
            uncharged_trna_conc,
            charged_trna_conc,
            aa_conc,
            ribosome_conc,
            f,
            self.charging_params,
            supply=supply_function,
            limit_v_rib=True,
            time_limit=states["timestep"],
        )

        # Use the supply calculated from each sub timestep while solving the charging steady state
        if self.process.aa_supply_in_charging:
            conversion = (
                1 / self.counts_to_molar.asNumber(MICROMOLAR_UNITS) / states["timestep"]
            )
            synthesis = conversion * synthesis_in_charging
            import_rates = conversion * import_in_charging
            export_rates = conversion * export_in_charging
            self.process.aa_supply = synthesis + import_rates - export_rates
        # Use the supply calculated from the starting amino acid concentrations only
        elif self.process.mechanistic_translation_supply:
            # Set supply based on mechanistic synthesis and supply
            self.process.aa_supply = states["timestep"] * (synthesis + exchange_rates)
        else:
            # Adjust aa_supply higher if amino acid concentrations are low
            # Improves stability of charging and mimics amino acid synthesis
            # inhibition and export
            # Polypeptide elongation operates using concentration units of CONC_UNITS (uM)
            # but aa_supply_scaling uses M units, so convert using unit_conversion (1e-6)
            self.process.aa_supply *= self.aa_supply_scaling(
                self.charging_params["unit_conversion"] * aa_conc.asNumber(CONC_UNITS),
                aa_in_media,
            )

        aa_counts_for_translation = (
            v_rib
            * f
            * states["timestep"]
            / self.counts_to_molar.asNumber(MICROMOLAR_UNITS)
        )

        total_trna = charged_trna_array + uncharged_trna_array
        final_charged_trna = stochasticRound(
            self.process.random_state,
            np.dot(fraction_charged, self.process.aa_from_trna * total_trna),
        )

        # Request charged tRNA that will become uncharged
        charged_trna_request = charged_trna_array - final_charged_trna
        charged_trna_request[charged_trna_request < 0] = 0
        uncharged_trna_request = final_charged_trna - charged_trna_array
        uncharged_trna_request[uncharged_trna_request < 0] = 0
        self.uncharged_trna_to_charge = uncharged_trna_request

        self.aa_counts_for_translation = np.array(aa_counts_for_translation)

        fraction_trna_per_aa = total_trna / np.dot(
            np.dot(self.process.aa_from_trna, total_trna), self.process.aa_from_trna
        )
        total_charging_reactions = stochasticRound(
            self.process.random_state,
            np.dot(aa_counts_for_translation, self.process.aa_from_trna)
            * fraction_trna_per_aa
            + uncharged_trna_request,
        )

        # Only request molecules that will be consumed in the charging reactions
        aa_from_uncharging = -self.charging_stoich_matrix @ charged_trna_request
        aa_from_uncharging[self.charging_molecules_not_aa] = 0
        requested_molecules = (
            -np.dot(self.charging_stoich_matrix, total_charging_reactions)
            - aa_from_uncharging
        )
        requested_molecules[requested_molecules < 0] = 0
        self.uncharged_trna_to_charge = uncharged_trna_request

        # ppGpp reactions based on charged tRNA
        bulk_request = [
            (
                self.process.charging_molecule_idx,
                requested_molecules.astype(int),
            ),
            (self.process.charged_trna_idx, charged_trna_request.astype(int)),
            # Request water for transfer of AA from tRNA for initial polypeptide.
            # This is severe overestimate assuming the worst case that every
            # elongation is initializing a polypeptide. This excess of water
            # shouldn't matter though.
            (self.process.water_idx, int(aa_counts_for_translation.sum())),
        ]
        if self.process.ppgpp_regulation:
            total_trna_conc = self.counts_to_molar * (
                uncharged_trna_counts + charged_trna_counts
            )
            updated_charged_trna_conc = total_trna_conc * fraction_charged
            updated_uncharged_trna_conc = total_trna_conc - updated_charged_trna_conc
            delta_metabolites, *_ = ppgpp_metabolite_changes(
                updated_uncharged_trna_conc,
                updated_charged_trna_conc,
                ribosome_conc,
                f,
                rela_conc,
                spot_conc,
                ppgpp_conc,
                self.counts_to_molar,
                v_rib,
                self.charging_params,
                self.ppgpp_params,
                states["timestep"],
                request=True,
                random_state=self.process.random_state,
            )

            request_ppgpp_metabolites = -delta_metabolites.astype(int)
            ppgpp_request = counts(states["bulk"], self.process.ppgpp_idx)
            bulk_request.append((self.process.ppgpp_idx, ppgpp_request))
            bulk_request.append(
                (
                    self.process.ppgpp_rxn_metabolites_idx,
                    request_ppgpp_metabolites,
                )
            )

        return (
            fraction_charged,
            aa_counts_for_translation,
            {
                "bulk": bulk_request,
                "listeners": {
                    "growth_limits": {
                        "original_aa_supply": self.process.aa_supply,
                        "aa_in_media": aa_in_media,
                        "synthetase_conc": synthetase_conc.asNumber(MICROMOLAR_UNITS),
                        "uncharged_trna_conc": uncharged_trna_conc.asNumber(
                            MICROMOLAR_UNITS
                        ),
                        "charged_trna_conc": charged_trna_conc.asNumber(
                            MICROMOLAR_UNITS
                        ),
                        "aa_conc": aa_conc.asNumber(MICROMOLAR_UNITS),
                        "ribosome_conc": ribosome_conc.asNumber(MICROMOLAR_UNITS),
                        "fraction_aa_to_elongate": f,
                        "aa_supply": self.process.aa_supply,
                        "aa_synthesis": synthesis * states["timestep"],
                        "aa_import": import_rates * states["timestep"],
                        "aa_export": export_rates * states["timestep"],
                        "aa_supply_enzymes_fwd": fwd_enzyme_counts,
                        "aa_supply_enzymes_rev": rev_enzyme_counts,
                        "aa_importers": importer_counts,
                        "aa_exporters": exporter_counts,
                        "aa_supply_aa_conc": aa_conc.asNumber(units.mmol / units.L),
                        "aa_supply_fraction_fwd": fwd_saturation,
                        "aa_supply_fraction_rev": rev_saturation,
                        "ppgpp_conc": ppgpp_conc.asNumber(MICROMOLAR_UNITS),
                        "rela_conc": rela_conc.asNumber(MICROMOLAR_UNITS),
                        "spot_conc": spot_conc.asNumber(MICROMOLAR_UNITS),
                    }
                },
                "polypeptide_elongation": {
                    "aa_exchange_rates": (
                        self.counts_to_molar / units.s * (import_rates - export_rates)
                    ).asNumber(CONC_UNITS / TIME_UNITS)
                },
            },
        )

    def final_amino_acids(self, total_aa_counts, charged_trna_counts):
        charged_counts_to_uncharge = self.process.aa_from_trna @ charged_trna_counts
        return np.fmin(
            total_aa_counts + charged_counts_to_uncharge, self.aa_counts_for_translation
        )

    def evolve(
        self,
        states,
        total_aa_counts,
        aas_used,
        next_amino_acid_count,
        nElongations,
        nInitialized,
    ):
        update = {
            "bulk": [],
            "listeners": {"growth_limits": {}},
        }

        # Get tRNA counts
        uncharged_trna = counts(states["bulk"], self.process.uncharged_trna_idx)
        charged_trna = counts(states["bulk"], self.process.charged_trna_idx)
        total_trna = uncharged_trna + charged_trna

        # Adjust molecules for number of charging reactions that occurred
        ## Determine limitations for charging and uncharging reactions
        charged_and_elongated_per_aa = np.fmax(
            0, (aas_used - self.process.aa_from_trna @ charged_trna)
        )
        aa_for_charging = total_aa_counts - charged_and_elongated_per_aa
        n_aa_charged = np.fmin(
            aa_for_charging,
            np.dot(
                self.process.aa_from_trna,
                np.fmin(self.uncharged_trna_to_charge, uncharged_trna),
            ),
        )
        n_uncharged_per_aa = aas_used - charged_and_elongated_per_aa

        ## Calculate changes in tRNA based on limitations
        n_trna_charged = self.distribution_from_aa(n_aa_charged, uncharged_trna, True)
        n_trna_uncharged = self.distribution_from_aa(
            n_uncharged_per_aa, charged_trna, True
        )

        ## Determine reactions that are charged and elongated in same time step without changing
        ## charged or uncharged counts
        charged_and_elongated = self.distribution_from_aa(
            charged_and_elongated_per_aa, total_trna
        )

        ## Determine total number of reactions that occur
        total_uncharging_reactions = charged_and_elongated + n_trna_uncharged
        total_charging_reactions = charged_and_elongated + n_trna_charged
        net_charged = total_charging_reactions - total_uncharging_reactions
        charging_mol_delta = np.dot(
            self.charging_stoich_matrix, total_charging_reactions
        ).astype(int)
        update["bulk"].append((self.process.charging_molecule_idx, charging_mol_delta))
        states["bulk"][self.process.charging_molecule_idx] += charging_mol_delta

        ## Account for uncharging of tRNA during elongation
        update["bulk"].append(
            (self.process.charged_trna_idx, -total_uncharging_reactions)
        )
        update["bulk"].append(
            (self.process.uncharged_trna_idx, total_uncharging_reactions)
        )
        states["bulk"][self.process.charged_trna_idx] += -total_uncharging_reactions
        states["bulk"][self.process.uncharged_trna_idx] += total_uncharging_reactions

        # Update proton counts to reflect polymerization reactions and transfer of AA from tRNA
        # Peptide bond formation releases a water but transferring AA from tRNA consumes a OH-
        # Net production of H+ for each elongation, consume extra water for each initialization
        # since a peptide bond doesn't form
        update["bulk"].append((self.process.proton_idx, nElongations))
        update["bulk"].append((self.process.water_idx, -nInitialized))
        states["bulk"][self.process.proton_idx] += nElongations
        states["bulk"][self.process.water_idx] += -nInitialized

        # Create or degrade ppGpp
        # This should come after all countInc/countDec calls since it shares some molecules with
        # other views and those counts should be updated to get the proper limits on ppGpp reactions
        if self.process.ppgpp_regulation:
            v_rib = (nElongations * self.counts_to_molar).asNumber(
                MICROMOLAR_UNITS
            ) / states["timestep"]
            ribosome_conc = (
                self.counts_to_molar * states["active_ribosome"]["_entryState"].sum()
            )
            updated_uncharged_trna_counts = (
                counts(states["bulk_total"], self.process.uncharged_trna_idx)
                - net_charged
            )
            updated_charged_trna_counts = (
                counts(states["bulk_total"], self.process.charged_trna_idx)
                + net_charged
            )
            uncharged_trna_conc = self.counts_to_molar * np.dot(
                self.process.aa_from_trna, updated_uncharged_trna_counts
            )
            charged_trna_conc = self.counts_to_molar * np.dot(
                self.process.aa_from_trna, updated_charged_trna_counts
            )
            ppgpp_conc = self.counts_to_molar * counts(
                states["bulk_total"], self.process.ppgpp_idx
            )
            rela_conc = self.counts_to_molar * counts(
                states["bulk_total"], self.process.rela_idx
            )
            spot_conc = self.counts_to_molar * counts(
                states["bulk_total"], self.process.spot_idx
            )

            # Need to include the next amino acid the ribosome sees for certain
            # cases where elongation does not occur, otherwise f will be NaN
            aa_at_ribosome = aas_used + next_amino_acid_count
            f = aa_at_ribosome / aa_at_ribosome.sum()
            limits = counts(states["bulk"], self.process.ppgpp_rxn_metabolites_idx)
            (
                delta_metabolites,
                ppgpp_syn,
                ppgpp_deg,
                rela_syn,
                spot_syn,
                spot_deg,
                spot_deg_inhibited,
            ) = ppgpp_metabolite_changes(
                uncharged_trna_conc,
                charged_trna_conc,
                ribosome_conc,
                f,
                rela_conc,
                spot_conc,
                ppgpp_conc,
                self.counts_to_molar,
                v_rib,
                self.charging_params,
                self.ppgpp_params,
                states["timestep"],
                random_state=self.process.random_state,
                limits=limits,
            )

            update["listeners"]["growth_limits"] = {
                "rela_syn": rela_syn,
                "spot_syn": spot_syn,
                "spot_deg": spot_deg,
                "spot_deg_inhibited": spot_deg_inhibited,
            }

            update["bulk"].append(
                (self.process.ppgpp_rxn_metabolites_idx, delta_metabolites.astype(int))
            )
            states["bulk"][self.process.ppgpp_rxn_metabolites_idx] += (
                delta_metabolites.astype(int)
            )

        # Use the difference between (expected AA supply based on expected
        # doubling time and current DCW) and AA used to charge tRNA to update
        # the concentration target in metabolism during the next time step
        aa_used_trna = np.dot(self.process.aa_from_trna, total_charging_reactions)
        aa_diff = self.process.aa_supply - aa_used_trna

        update["listeners"]["growth_limits"]["trna_charged"] = aa_used_trna.astype(int)

        return (
            net_charged,
            aa_diff,
            update,
        )

    def distribution_from_aa(
        self,
        n_aa: npt.NDArray[np.int64],
        n_trna: npt.NDArray[np.int64],
        limited: bool = False,
    ) -> npt.NDArray[np.int64]:
        """
        Distributes counts of amino acids to tRNAs that are associated with
        each amino acid. Uses self.process.aa_from_trna mapping to distribute
        from amino acids to tRNA based on the fraction that each tRNA species
        makes up for all tRNA species that code for the same amino acid.

        Args:
            n_aa: counts of each amino acid to distribute to each tRNA
            n_trna: counts of each tRNA to determine the distribution
            limited: optional, if True, limits the amino acids
                distributed to each tRNA to the number of tRNA that are
                available (n_trna)

        Returns:
            Distributed counts for each tRNA
        """

        # Determine the fraction each tRNA species makes up out of all tRNA of
        # the associated amino acid
        with np.errstate(invalid="ignore"):
            f_trna = n_trna / np.dot(
                np.dot(self.process.aa_from_trna, n_trna), self.process.aa_from_trna
            )
        f_trna[~np.isfinite(f_trna)] = 0

        trna_counts = np.zeros(f_trna.shape, np.int64)
        for count, row in zip(n_aa, self.process.aa_from_trna):
            idx = row == 1
            frac = f_trna[idx]

            counts = np.floor(frac * count)
            diff = int(count - counts.sum())

            # Add additional counts to get up to counts to distribute
            # Prevent adding over the number of tRNA available if limited
            if diff > 0:
                if limited:
                    for _ in range(diff):
                        frac[(n_trna[idx] - counts) == 0] = 0
                        # normalize for multinomial distribution
                        frac /= frac.sum()
                        adjustment = self.process.random_state.multinomial(1, frac)
                        counts += adjustment
                else:
                    adjustment = self.process.random_state.multinomial(diff, frac)
                    counts += adjustment

            trna_counts[idx] = counts

        return trna_counts


def ppgpp_metabolite_changes(
    uncharged_trna_conc: Unum,
    charged_trna_conc: Unum,
    ribosome_conc: Unum,
    f: npt.NDArray[np.float64],
    rela_conc: Unum,
    spot_conc: Unum,
    ppgpp_conc: Unum,
    counts_to_molar: Unum,
    v_rib: Unum,
    charging_params: dict[str, Any],
    ppgpp_params: dict[str, Any],
    time_step: float,
    request: bool = False,
    limits: Optional[npt.NDArray[np.float64]] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> tuple[npt.NDArray[np.int64], int, int, Unum, Unum, Unum, Unum]:
    """
    Calculates the changes in metabolite counts based on ppGpp synthesis and
    degradation reactions.

    Args:
        uncharged_trna_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
            of uncharged tRNA associated with each amino acid
        charged_trna_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
            of charged tRNA associated with each amino acid
        ribosome_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
            of active ribosomes
        f: fraction of each amino acid to be incorporated
            to total amino acids incorporated
        rela_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`) of RelA
        spot_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`) of SpoT
        ppgpp_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`) of ppGpp
        counts_to_molar: conversion factor
            from counts to molarity (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
        v_rib: rate of amino acid incorporation at the ribosome (units of uM/s)
        charging_params: parameters used in charging equations
        ppgpp_params: parameters used in ppGpp reactions
        time_step: length of the current time step
        request: if True, only considers reactant stoichiometry,
            otherwise considers reactants and products. For use in
            calculateRequest. GDP appears as both a reactant and product
            and the request can be off the actual use if not handled in this
            manner.
        limits: counts of molecules that are available to prevent
            negative total counts as a result of delta_metabolites.
            If None, no limits are placed on molecule changes.
        random_state: random state for the process
    Returns:
        7-element tuple containing

        - **delta_metabolites**: the change in counts of each metabolite
          involved in ppGpp reactions
        - **n_syn_reactions**: the number of ppGpp synthesis reactions
        - **n_deg_reactions**: the number of ppGpp degradation reactions
        - **v_rela_syn**: rate of synthesis from RelA per amino
          acid tRNA species
        - **v_spot_syn**: rate of synthesis from SpoT
        - **v_deg**: rate of degradation from SpoT
        - **v_deg_inhibited**: rate of degradation from SpoT per
          amino acid tRNA species
    """

    if random_state is None:
        random_state = np.random.RandomState()

    uncharged_trna_conc = uncharged_trna_conc.asNumber(MICROMOLAR_UNITS)
    charged_trna_conc = charged_trna_conc.asNumber(MICROMOLAR_UNITS)
    ribosome_conc = ribosome_conc.asNumber(MICROMOLAR_UNITS)
    rela_conc = rela_conc.asNumber(MICROMOLAR_UNITS)
    spot_conc = spot_conc.asNumber(MICROMOLAR_UNITS)
    ppgpp_conc = ppgpp_conc.asNumber(MICROMOLAR_UNITS)
    counts_to_micromolar = counts_to_molar.asNumber(MICROMOLAR_UNITS)

    numerator = (
        1
        + charged_trna_conc / charging_params["krta"]
        + uncharged_trna_conc / charging_params["krtf"]
    )
    saturated_charged = charged_trna_conc / charging_params["krta"] / numerator
    saturated_uncharged = uncharged_trna_conc / charging_params["krtf"] / numerator
    if v_rib == 0:
        ribosome_conc_a_site = f * ribosome_conc
    else:
        ribosome_conc_a_site = (
            f * v_rib / (saturated_charged * charging_params["max_elong_rate"])
        )
    ribosomes_bound_to_uncharged = ribosome_conc_a_site * saturated_uncharged

    # Handle rare cases when tRNA concentrations are 0
    # Can result in inf and nan so assume a fraction of ribosomes
    # bind to the uncharged tRNA if any tRNA are present or 0 if not
    mask = ~np.isfinite(ribosomes_bound_to_uncharged)
    ribosomes_bound_to_uncharged[mask] = (
        ribosome_conc
        * f[mask]
        * np.array(uncharged_trna_conc[mask] + charged_trna_conc[mask] > 0)
    )

    # Calculate active fraction of RelA
    competitive_inhibition = 1 + ribosomes_bound_to_uncharged / ppgpp_params["KD_RelA"]
    inhibition_product = np.prod(competitive_inhibition)
    with np.errstate(divide="ignore"):
        frac_rela = 1 / (
            ppgpp_params["KD_RelA"]
            / ribosomes_bound_to_uncharged
            * inhibition_product
            / competitive_inhibition
            + 1
        )

    # Calculate rates for synthesis and degradation
    v_rela_syn = ppgpp_params["k_RelA"] * rela_conc * frac_rela
    v_spot_syn = ppgpp_params["k_SpoT_syn"] * spot_conc
    v_syn = v_rela_syn.sum() + v_spot_syn
    max_deg = ppgpp_params["k_SpoT_deg"] * spot_conc * ppgpp_conc
    fractions = uncharged_trna_conc / ppgpp_params["KI_SpoT"]
    v_deg = max_deg / (1 + fractions.sum())
    v_deg_inhibited = (max_deg - v_deg) * fractions / fractions.sum()

    # Convert to discrete reactions
    n_syn_reactions = stochasticRound(
        random_state, v_syn * time_step / counts_to_micromolar
    )[0]
    n_deg_reactions = stochasticRound(
        random_state, v_deg * time_step / counts_to_micromolar
    )[0]

    # Only look at reactant stoichiometry if requesting molecules to use
    if request:
        ppgpp_reaction_stoich = np.zeros_like(ppgpp_params["ppgpp_reaction_stoich"])
        reactants = ppgpp_params["ppgpp_reaction_stoich"] < 0
        ppgpp_reaction_stoich[reactants] = ppgpp_params["ppgpp_reaction_stoich"][
            reactants
        ]
    else:
        ppgpp_reaction_stoich = ppgpp_params["ppgpp_reaction_stoich"]

    # Calculate the change in metabolites and adjust to limits if provided
    # Possible reactions are adjusted down to limits if the change in any
    # metabolites would result in negative counts
    max_iterations = int(n_deg_reactions + n_syn_reactions + 1)
    old_counts = None
    for it in range(max_iterations):
        delta_metabolites = (
            ppgpp_reaction_stoich[:, ppgpp_params["synthesis_index"]] * n_syn_reactions
            + ppgpp_reaction_stoich[:, ppgpp_params["degradation_index"]]
            * n_deg_reactions
        )

        if limits is None:
            break
        else:
            final_counts = delta_metabolites + limits

            if np.all(final_counts >= 0) or (
                old_counts is not None and np.all(final_counts == old_counts)
            ):
                break

            limited_index = np.argmin(final_counts)
            if (
                ppgpp_reaction_stoich[limited_index, ppgpp_params["synthesis_index"]]
                < 0
            ):
                limited = np.ceil(
                    final_counts[limited_index]
                    / ppgpp_reaction_stoich[
                        limited_index, ppgpp_params["synthesis_index"]
                    ]
                )
                n_syn_reactions -= min(limited, n_syn_reactions)
            if (
                ppgpp_reaction_stoich[limited_index, ppgpp_params["degradation_index"]]
                < 0
            ):
                limited = np.ceil(
                    final_counts[limited_index]
                    / ppgpp_reaction_stoich[
                        limited_index, ppgpp_params["degradation_index"]
                    ]
                )
                n_deg_reactions -= min(limited, n_deg_reactions)

            old_counts = final_counts
    else:
        raise ValueError("Failed to meet molecule limits with ppGpp reactions.")

    return (
        delta_metabolites,
        n_syn_reactions,
        n_deg_reactions,
        v_rela_syn,
        v_spot_syn,
        v_deg,
        v_deg_inhibited,
    )


def calculate_trna_charging(
    synthetase_conc: Unum,
    uncharged_trna_conc: Unum,
    charged_trna_conc: Unum,
    aa_conc: Unum,
    ribosome_conc: Unum,
    f: Unum,
    params: dict[str, Any],
    supply: Optional[Callable] = None,
    time_limit: float = 1000,
    limit_v_rib: bool = False,
    use_disabled_aas: bool = False,
) -> tuple[Unum, float, Unum, Unum, Unum]:
    """
    Calculates the steady state value of tRNA based on charging and
    incorporation through polypeptide elongation. The fraction of
    charged/uncharged is also used to determine how quickly the
    ribosome is elongating. All concentrations are given in units of
    :py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`.

    Args:
        synthetase_conc: concentration of synthetases associated with
            each amino acid
        uncharged_trna_conc: concentration of uncharged tRNA associated
            with each amino acid
        charged_trna_conc: concentration of charged tRNA associated with
            each amino acid
        aa_conc: concentration of each amino acid
        ribosome_conc: concentration of active ribosomes
        f: fraction of each amino acid to be incorporated to total amino
            acids incorporated
        params: parameters used in charging equations
        supply: function to get the rate of amino acid supply (synthesis
            and import) based on amino acid concentrations. If None, amino
            acid concentrations remain constant during charging
        time_limit: time limit to reach steady state
        limit_v_rib: if True, v_rib is limited to the number of amino acids
            that are available
        use_disabled_aas: if False, amino acids in
            :py:data:`~ecoli.processes.polypeptide_elongation.REMOVED_FROM_CHARGING`
            are excluded from charging

    Returns:
        5-element tuple containing

        - **new_fraction_charged**: fraction of total tRNA that is charged for each
          amino acid species
        - **v_rib**: ribosomal elongation rate in units of uM/s
        - **total_synthesis**: the total amount of amino acids synthesized during charging
          in units of MICROMOLAR_UNITS. Will be zeros if supply function is not given.
        - **total_import**: the total amount of amino acids imported during charging
          in units of MICROMOLAR_UNITS. Will be zeros if supply function is not given.
        - **total_export**: the total amount of amino acids exported during charging
          in units of MICROMOLAR_UNITS. Will be zeros if supply function is not given.
    """

    def negative_check(trna1: npt.NDArray[np.float64], trna2: npt.NDArray[np.float64]):
        """
        Check for floating point precision issues that can lead to small
        negative numbers instead of 0. Adjusts both species of tRNA to
        bring concentration of trna1 to 0 and keep the same total concentration.

        Args:
            trna1: concentration of one tRNA species (charged or uncharged)
            trna2: concentration of another tRNA species (charged or uncharged)
        """

        mask = trna1 < 0
        trna2[mask] = trna1[mask] + trna2[mask]
        trna1[mask] = 0

    def dcdt(t: float, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Function for solve_ivp to integrate

        Args:
            c: 1D array of concentrations of uncharged and charged tRNAs
                dims: 2 * number of amino acids (uncharged tRNA come first, then charged)
            t: time of integration step

        Returns:
            Array of dc/dt for tRNA concentrations
                dims: 2 * number of amino acids (uncharged tRNA come first, then charged)
        """

        v_charging, dtrna, daa = dcdt_jit(
            t,
            c,
            n_aas_masked,
            n_aas,
            mask,
            params["kS"],
            synthetase_conc,
            params["KMaa"],
            params["KMtf"],
            f,
            params["krta"],
            params["krtf"],
            params["max_elong_rate"],
            ribosome_conc,
            limit_v_rib,
            aa_rate_limit,
            v_rib_max,
        )

        if supply is None:
            v_synthesis = np.zeros(n_aas)
            v_import = np.zeros(n_aas)
            v_export = np.zeros(n_aas)
        else:
            aa_conc = c[2 * n_aas_masked : 2 * n_aas_masked + n_aas]
            v_synthesis, v_import, v_export = supply(unit_conversion * aa_conc)
            v_supply = v_synthesis + v_import - v_export
            daa[mask] = v_supply[mask] - v_charging

        return np.hstack((-dtrna, dtrna, daa, v_synthesis, v_import, v_export))

    # Convert inputs for integration
    synthetase_conc = synthetase_conc.asNumber(MICROMOLAR_UNITS)
    uncharged_trna_conc = uncharged_trna_conc.asNumber(MICROMOLAR_UNITS)
    charged_trna_conc = charged_trna_conc.asNumber(MICROMOLAR_UNITS)
    aa_conc = aa_conc.asNumber(MICROMOLAR_UNITS)
    ribosome_conc = ribosome_conc.asNumber(MICROMOLAR_UNITS)
    unit_conversion = params["unit_conversion"]

    # Remove disabled amino acids from calculations
    n_total_aas = len(aa_conc)
    if use_disabled_aas:
        mask = np.ones(n_total_aas, bool)
    else:
        mask = params["charging_mask"]
    synthetase_conc = synthetase_conc[mask]
    original_uncharged_trna_conc = uncharged_trna_conc[mask]
    original_charged_trna_conc = charged_trna_conc[mask]
    original_aa_conc = aa_conc[mask]
    f = f[mask]

    n_aas = len(aa_conc)
    n_aas_masked = len(original_aa_conc)

    # Limits for integration
    aa_rate_limit = original_aa_conc / time_limit
    trna_rate_limit = original_charged_trna_conc / time_limit
    v_rib_max = max(0, ((aa_rate_limit + trna_rate_limit) / f).min())

    # Integrate rates of charging and elongation
    c_init = np.hstack(
        (
            original_uncharged_trna_conc,
            original_charged_trna_conc,
            aa_conc,
            np.zeros(n_aas),
            np.zeros(n_aas),
            np.zeros(n_aas),
        )
    )
    sol = solve_ivp(dcdt, [0, time_limit], c_init, method="BDF")
    c_sol = sol.y.T

    # Determine new values from integration results
    final_uncharged_trna_conc = c_sol[-1, :n_aas_masked]
    final_charged_trna_conc = c_sol[-1, n_aas_masked : 2 * n_aas_masked]
    total_synthesis = c_sol[-1, 2 * n_aas_masked + n_aas : 2 * n_aas_masked + 2 * n_aas]
    total_import = c_sol[
        -1, 2 * n_aas_masked + 2 * n_aas : 2 * n_aas_masked + 3 * n_aas
    ]
    total_export = c_sol[
        -1, 2 * n_aas_masked + 3 * n_aas : 2 * n_aas_masked + 4 * n_aas
    ]

    negative_check(final_uncharged_trna_conc, final_charged_trna_conc)
    negative_check(final_charged_trna_conc, final_uncharged_trna_conc)

    fraction_charged = final_charged_trna_conc / (
        final_uncharged_trna_conc + final_charged_trna_conc
    )
    numerator_ribosome = 1 + np.sum(
        f
        * (
            params["krta"] / final_charged_trna_conc
            + final_uncharged_trna_conc
            / final_charged_trna_conc
            * params["krta"]
            / params["krtf"]
        )
    )
    v_rib = params["max_elong_rate"] * ribosome_conc / numerator_ribosome
    if limit_v_rib:
        v_rib_max = max(
            0,
            (
                (
                    original_aa_conc
                    + (original_charged_trna_conc - final_charged_trna_conc)
                )
                / time_limit
                / f
            ).min(),
        )
        v_rib = min(v_rib, v_rib_max)

    # Replace SEL fraction charged with average
    new_fraction_charged = np.zeros(n_total_aas)
    new_fraction_charged[mask] = fraction_charged
    new_fraction_charged[~mask] = fraction_charged.mean()

    return new_fraction_charged, v_rib, total_synthesis, total_import, total_export


@njit(error_model="numpy")
def dcdt_jit(
    t,
    c,
    n_aas_masked,
    n_aas,
    mask,
    kS,
    synthetase_conc,
    KMaa,
    KMtf,
    f,
    krta,
    krtf,
    max_elong_rate,
    ribosome_conc,
    limit_v_rib,
    aa_rate_limit,
    v_rib_max,
):
    uncharged_trna_conc = c[:n_aas_masked]
    charged_trna_conc = c[n_aas_masked : 2 * n_aas_masked]
    aa_conc = c[2 * n_aas_masked : 2 * n_aas_masked + n_aas]
    masked_aa_conc = aa_conc[mask]

    v_charging = (
        kS
        * synthetase_conc
        * uncharged_trna_conc
        * masked_aa_conc
        / (KMaa[mask] * KMtf[mask])
        / (
            1
            + uncharged_trna_conc / KMtf[mask]
            + masked_aa_conc / KMaa[mask]
            + uncharged_trna_conc * masked_aa_conc / KMtf[mask] / KMaa[mask]
        )
    )
    numerator_ribosome = 1 + np.sum(
        f
        * (
            krta / charged_trna_conc
            + uncharged_trna_conc / charged_trna_conc * krta / krtf
        )
    )
    v_rib = max_elong_rate * ribosome_conc / numerator_ribosome

    # Handle case when f is 0 and charged_trna_conc is 0
    if not np.isfinite(v_rib):
        v_rib = 0

    # Limit v_rib and v_charging to the amount of available amino acids
    if limit_v_rib:
        v_charging = np.fmin(v_charging, aa_rate_limit)
        v_rib = min(v_rib, v_rib_max)

    dtrna = v_charging - v_rib * f
    daa = np.zeros(n_aas)

    return v_charging, dtrna, daa


def get_charging_supply_function(
    supply_in_charging: bool,
    mechanistic_supply: bool,
    mechanistic_aa_transport: bool,
    amino_acid_synthesis: Callable,
    amino_acid_import: Callable,
    amino_acid_export: Callable,
    aa_supply_scaling: Callable,
    counts_to_molar: Unum,
    aa_supply: npt.NDArray[np.float64],
    fwd_enzyme_counts: npt.NDArray[np.int64],
    rev_enzyme_counts: npt.NDArray[np.int64],
    dry_mass: Unum,
    importer_counts: npt.NDArray[np.int64],
    exporter_counts: npt.NDArray[np.int64],
    aa_in_media: npt.NDArray[np.bool_],
) -> Optional[Callable[[npt.NDArray[np.float64]], Tuple[Unum, Unum, Unum]]]:
    """
    Get a function mapping internal amino acid concentrations to the amount of
    amino acid supply expected.

    Args:
        supply_in_charging: True if using aa_supply_in_charging option
        mechanistic_supply: True if using mechanistic_translation_supply option
        mechanistic_aa_transport: True if using mechanistic_aa_transport option
        amino_acid_synthesis: function to provide rates of synthesis for amino
            acids based on the internal state
        amino_acid_import: function to provide import rates for amino
            acids based on the internal and external state
        amino_acid_export: function to provide export rates for amino
            acids based on the internal state
        aa_supply_scaling: function to scale the amino acid supply based
            on the internal state
        counts_to_molar: conversion factor for counts to molar
            (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
        aa_supply: rate of amino acid supply expected
        fwd_enzyme_counts: enzyme counts in forward reactions for each amino acid
        rev_enzyme_counts: enzyme counts in loss reactions for each amino acid
        dry_mass: dry mass of the cell with mass units
        importer_counts: counts for amino acid importers
        exporter_counts: counts for amino acid exporters
        aa_in_media: True for each amino acid that is present in the media
    Returns:
        Function that provides the amount of supply (synthesis, import, export)
        for each amino acid based on the internal state of the cell
    """

    # Create functions that are only dependent on amino acid concentrations for more stable
    # charging and amino acid concentrations.  If supply_in_charging is not set, then
    # setting None will maintain constant amino acid concentrations throughout charging.
    supply_function = None
    if supply_in_charging:
        counts_to_molar = counts_to_molar.asNumber(MICROMOLAR_UNITS)
        zeros = counts_to_molar * np.zeros_like(aa_supply)
        if mechanistic_supply:
            if mechanistic_aa_transport:

                def supply_function(aa_conc):
                    return (
                        counts_to_molar
                        * amino_acid_synthesis(
                            fwd_enzyme_counts, rev_enzyme_counts, aa_conc
                        )[0],
                        counts_to_molar
                        * amino_acid_import(
                            aa_in_media,
                            dry_mass,
                            aa_conc,
                            importer_counts,
                            mechanistic_aa_transport,
                        ),
                        counts_to_molar
                        * amino_acid_export(
                            exporter_counts, aa_conc, mechanistic_aa_transport
                        ),
                    )
            else:

                def supply_function(aa_conc):
                    return (
                        counts_to_molar
                        * amino_acid_synthesis(
                            fwd_enzyme_counts, rev_enzyme_counts, aa_conc
                        )[0],
                        counts_to_molar
                        * amino_acid_import(
                            aa_in_media,
                            dry_mass,
                            aa_conc,
                            importer_counts,
                            mechanistic_aa_transport,
                        ),
                        zeros,
                    )
        else:

            def supply_function(aa_conc):
                return (
                    counts_to_molar
                    * aa_supply
                    * aa_supply_scaling(aa_conc, aa_in_media),
                    zeros,
                    zeros,
                )

    return supply_function


# ======================================================================
# chromosome_replication
# ======================================================================

class ChromosomeReplicationLogic:
    """Chromosome Replication — shared state container for Requester/Evolver."""

    name = "ecoli-chromosome-replication"
    topology = {
    "bulk": ("bulk",),
    "active_replisomes": ("unique", "active_replisome"),
    "oriCs": ("unique", "oriC"),
    "chromosome_domains": ("unique", "chromosome_domain"),
    "full_chromosomes": ("unique", "full_chromosome"),
    "listeners": ("listeners",),
    "environment": ("environment",),
    "timestep": ("timestep",),
}
    defaults = {
        "get_dna_critical_mass": lambda doubling_time: units.Unum,
        "criticalInitiationMass": 975 * units.fg,
        "nutrientToDoublingTime": {},
        "replichore_lengths": np.array([]),
        "sequences": np.array([]),
        "polymerized_dntp_weights": [],
        "replication_coordinate": np.array([]),
        "D_period": np.array([]),
        "replisome_protein_mass": 0,
        "no_child_place_holder": -1,
        "basal_elongation_rate": 967,
        "make_elongation_rates": (
            lambda random, replisomes, base, time_step: units.Unum
        ),
        "mechanistic_replisome": True,
        # molecules
        "replisome_trimers_subunits": [],
        "replisome_monomers_subunits": [],
        "dntps": [],
        "ppi": [],
        # random seed
        "seed": 0,
        "emit_unique": False,
        "time_step": 1,
    }

    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

        # Load parameters
        self.get_dna_critical_mass = self.parameters["get_dna_critical_mass"]
        self.criticalInitiationMass = self.parameters["criticalInitiationMass"]
        self.nutrientToDoublingTime = self.parameters["nutrientToDoublingTime"]
        self.replichore_lengths = self.parameters["replichore_lengths"]
        self.sequences = self.parameters["sequences"]
        self.polymerized_dntp_weights = self.parameters["polymerized_dntp_weights"]
        self.replication_coordinate = self.parameters["replication_coordinate"]
        self.D_period = self.parameters["D_period"]
        self.replisome_protein_mass = self.parameters["replisome_protein_mass"]
        self.no_child_place_holder = self.parameters["no_child_place_holder"]
        self.basal_elongation_rate = self.parameters["basal_elongation_rate"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]

        # Sim options
        self.mechanistic_replisome = self.parameters["mechanistic_replisome"]

        # random state
        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.emit_unique = self.parameters.get("emit_unique", True)

        # Bulk molecule names
        self.replisome_trimers_subunits = self.parameters["replisome_trimers_subunits"]
        self.replisome_monomers_subunits = self.parameters[
            "replisome_monomers_subunits"
        ]
        self.dntps = self.parameters["dntps"]
        self.ppi = self.parameters["ppi"]

        self.ppi_idx = None



class ChromosomeReplicationRequester(_SafeInvokeMixin, Step):
    """Reads stores to compute chromosome replication request."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'active_replisomes': UniqueNumpyUpdate(),
            'oriCs': UniqueNumpyUpdate(),
            'chromosome_domains': UniqueNumpyUpdate(),
            'full_chromosomes': UniqueNumpyUpdate(),
            'listeners': ListenerStore(),
            'environment': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
            'listeners': ListenerStore(),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from calculate_request ---
        if p.ppi_idx is None:
            p.ppi_idx = bulk_name_to_idx(p.ppi, state["bulk"]["id"])
            p.replisome_trimers_idx = bulk_name_to_idx(
                p.replisome_trimers_subunits, state["bulk"]["id"]
            )
            p.replisome_monomers_idx = bulk_name_to_idx(
                p.replisome_monomers_subunits, state["bulk"]["id"]
            )
            p.dntps_idx = bulk_name_to_idx(p.dntps, state["bulk"]["id"])
        request = {}
        # Get total count of existing oriC's
        n_oriC = state["oriCs"]["_entryState"].sum()
        # If there are no origins, return immediately
        if n_oriC != 0:
            # Get current cell mass
            cellMass = state["listeners"]["mass"]["cell_mass"] * units.fg

            # Get critical initiation mass for current simulation environment
            current_media_id = state["environment"]["media_id"]
            p.criticalInitiationMass = p.get_dna_critical_mass(
                p.nutrientToDoublingTime[current_media_id]
            )

            # Calculate mass per origin of replication, and compare to critical
            # initiation mass. If the cell mass has reached this critical mass,
            # the process will initiate a round of chromosome replication for each
            # origin of replication.
            massPerOrigin = cellMass / n_oriC
            p.criticalMassPerOriC = massPerOrigin / p.criticalInitiationMass

            # If replication should be initiated, request subunits required for
            # building two replisomes per one origin of replication, and edit
            # access to oriC and chromosome domain attributes
            request["bulk"] = []
            if p.criticalMassPerOriC >= 1.0:
                request["bulk"].append((p.replisome_trimers_idx, 6 * n_oriC))
                request["bulk"].append((p.replisome_monomers_idx, 2 * n_oriC))

            # If there are no active forks return
            n_active_replisomes = state["active_replisomes"]["_entryState"].sum()
            if n_active_replisomes != 0:
                # Get current locations of all replication forks
                (fork_coordinates,) = attrs(state["active_replisomes"], ["coordinates"])
                sequence_length = np.abs(np.repeat(fork_coordinates, 2))

                p.elongation_rates = p.make_elongation_rates(
                    p.random_state,
                    len(p.sequences),
                    p.basal_elongation_rate,
                    state["timestep"],
                )

                sequences = buildSequences(
                    p.sequences,
                    np.tile(np.arange(4), n_active_replisomes // 2),
                    sequence_length,
                    p.elongation_rates,
                )

                # Count number of each dNTP in sequences for the next timestep
                sequenceComposition = np.bincount(
                    sequences[sequences != polymerize.PAD_VALUE], minlength=4
                )

                # If one dNTP is limiting then limit the request for the other three by
                # the same ratio
                dNtpsTotal = counts(state["bulk"], p.dntps_idx)
                maxFractionalReactionLimit = (
                    np.fmin(1, dNtpsTotal / sequenceComposition)
                ).min()

                # Request dNTPs
                request["bulk"].append(
                    (
                        p.dntps_idx,
                        (maxFractionalReactionLimit * sequenceComposition).astype(int),
                    )
                )
        # --- end inlined ---
        p.request_set = True

        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request

        listeners = request.pop('listeners', None)
        if listeners is not None:
            result['listeners'] = listeners

        return result


class ChromosomeReplicationEvolver(_SafeInvokeMixin, Step):
    """Reads allocation, writes bulk/unique/listener updates."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'active_replisomes': UniqueNumpyUpdate(),
            'oriCs': UniqueNumpyUpdate(),
            'chromosome_domains': UniqueNumpyUpdate(),
            'full_chromosomes': UniqueNumpyUpdate(),
            'listeners': ListenerStore(),
            'environment': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'active_replisomes': UniqueNumpyUpdate(),
            'oriCs': UniqueNumpyUpdate(),
            'chromosome_domains': UniqueNumpyUpdate(),
            'full_chromosomes': UniqueNumpyUpdate(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)

        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)

        if not self.process.request_set:
            return {}

        timestep = state.get('timestep', 1.0)
        p = self.process
        # --- inlined from evolve_state ---
        # Initialize the update dictionary
        update = {
            "bulk": [],
            "active_replisomes": {},
            "oriCs": {},
            "chromosome_domains": {},
            "full_chromosomes": {},
            "listeners": {"replication_data": {}},
        }

        # Module 1: Replication initiation
        # Get number of existing replisomes and oriCs
        n_active_replisomes = state["active_replisomes"]["_entryState"].sum()
        n_oriC = state["oriCs"]["_entryState"].sum()

        # If there are no origins, return immediately
        if n_oriC != 0:
            # Get attributes of existing chromosome domains
            domain_index_existing_domain, child_domains = attrs(
                state["chromosome_domains"], ["domain_index", "child_domains"]
            )

            initiate_replication = False
            if p.criticalMassPerOriC >= 1.0:
                # Get number of available replisome subunits
                n_replisome_trimers = counts(state["bulk"], p.replisome_trimers_idx)
                n_replisome_monomers = counts(state["bulk"], p.replisome_monomers_idx)
                # Initiate replication only when
                # 1) The cell has reached the critical mass per oriC
                # 2) If mechanistic replisome option is on, there are enough
                # replisome subunits to assemble two replisomes per existing OriC.
                # Note that we assume asynchronous initiation does not happen.
                initiate_replication = not p.mechanistic_replisome or (
                    np.all(n_replisome_trimers == 6 * n_oriC)
                    and np.all(n_replisome_monomers == 2 * n_oriC)
                )

            # If all conditions are met, initiate a round of replication on every
            # origin of replication
            if initiate_replication:
                # Get attributes of existing oriCs and domains
                (domain_index_existing_oric,) = attrs(state["oriCs"], ["domain_index"])

                # Get indexes of the domains that would be getting child domains
                # (domains that contain an origin)
                new_parent_domains = np.where(
                    np.isin(domain_index_existing_domain, domain_index_existing_oric)
                )[0]

                # Calculate counts of new replisomes and domains to add
                n_new_replisome = 2 * n_oriC
                n_new_domain = 2 * n_oriC

                # Calculate the domain indexes of new domains and oriC's
                max_domain_index = domain_index_existing_domain.max()
                domain_index_new = np.arange(
                    max_domain_index + 1, max_domain_index + 2 * n_oriC + 1, dtype=np.int32
                )

                # Add new oriC's, and reset attributes of existing oriC's
                # All oriC's must be assigned new domain indexes
                update["oriCs"]["set"] = {"domain_index": domain_index_new[:n_oriC]}
                update["oriCs"]["add"] = {
                    "domain_index": domain_index_new[n_oriC:],
                }

                # Add and set attributes of newly created replisomes.
                # New replisomes inherit the domain indexes of the oriC's they
                # were initiated from. Two replisomes are formed per oriC, one on
                # the right replichore, and one on the left.
                coordinates_replisome = np.zeros(n_new_replisome, dtype=np.int64)
                right_replichore = np.tile(np.array([True, False], dtype=np.bool_), n_oriC)
                right_replichore = right_replichore.tolist()
                domain_index_new_replisome = np.repeat(domain_index_existing_oric, 2)
                massDiff_protein_new_replisome = np.full(
                    n_new_replisome,
                    p.replisome_protein_mass if p.mechanistic_replisome else 0.0,
                )
                update["active_replisomes"]["add"] = {
                    "coordinates": coordinates_replisome,
                    "right_replichore": right_replichore,
                    "domain_index": domain_index_new_replisome,
                    "massDiff_protein": massDiff_protein_new_replisome,
                }

                # Add and set attributes of new chromosome domains. All new domains
                # should have have no children domains.
                new_child_domains = np.full(
                    (n_new_domain, 2), p.no_child_place_holder, dtype=np.int32
                )
                new_domains_update = {
                    "add": {
                        "domain_index": domain_index_new,
                        "child_domains": new_child_domains,
                    }
                }

                # Add new domains as children of existing domains
                child_domains[new_parent_domains] = domain_index_new.reshape(-1, 2)
                existing_domains_update = {"set": {"child_domains": child_domains}}
                update["chromosome_domains"].update(
                    {**new_domains_update, **existing_domains_update}
                )

                # Decrement counts of replisome subunits
                if p.mechanistic_replisome:
                    update["bulk"].append((p.replisome_trimers_idx, -6 * n_oriC))
                    update["bulk"].append((p.replisome_monomers_idx, -2 * n_oriC))

            # Write data from this module to a listener
            update["listeners"]["replication_data"]["critical_mass_per_oriC"] = (
                p.criticalMassPerOriC.asNumber()
            )
            update["listeners"]["replication_data"]["critical_initiation_mass"] = (
                p.criticalInitiationMass.asNumber(units.fg)
            )

            # Module 2: replication elongation
            # If no active replisomes are present, return immediately
            # Note: the new replication forks added in the previous module are not
            # elongated until the next timestep.
            if n_active_replisomes != 0:
                # Get allocated counts of dNTPs
                dNtpCounts = counts(state["bulk"], p.dntps_idx)

                # Get attributes of existing replisomes
                (
                    domain_index_replisome,
                    right_replichore,
                    coordinates_replisome,
                ) = attrs(
                    state["active_replisomes"],
                    ["domain_index", "right_replichore", "coordinates"],
                )

                # Build sequences to polymerize
                sequence_length = np.abs(np.repeat(coordinates_replisome, 2))
                sequence_indexes = np.tile(np.arange(4), n_active_replisomes // 2)

                sequences = buildSequences(
                    p.sequences, sequence_indexes, sequence_length, p.elongation_rates
                )

                # Use polymerize algorithm to quickly calculate the number of
                # elongations each fork catalyzes
                reactionLimit = dNtpCounts.sum()

                active_elongation_rates = p.elongation_rates[sequence_indexes]

                result = polymerize(
                    sequences,
                    dNtpCounts,
                    reactionLimit,
                    p.random_state,
                    active_elongation_rates,
                )

                sequenceElongations = result.sequenceElongation
                dNtpsUsed = result.monomerUsages

                # Compute mass increase for each elongated sequence
                mass_increase_dna = computeMassIncrease(
                    sequences,
                    sequenceElongations,
                    p.polymerized_dntp_weights.asNumber(units.fg),
                )

                # Compute masses that should be added to each replisome
                added_dna_mass = mass_increase_dna[0::2] + mass_increase_dna[1::2]

                # Update positions of each fork
                updated_length = sequence_length + sequenceElongations
                updated_coordinates = updated_length[0::2]

                # Reverse signs of fork coordinates on left replichore
                updated_coordinates[~right_replichore] = -updated_coordinates[~right_replichore]

                # Update attributes and submasses of replisomes
                (current_dna_mass,) = attrs(state["active_replisomes"], ["massDiff_DNA"])
                update["active_replisomes"].update(
                    {
                        "set": {
                            "coordinates": updated_coordinates,
                            "massDiff_DNA": current_dna_mass + added_dna_mass,
                        }
                    }
                )

                # Update counts of polymerized metabolites
                update["bulk"].append((p.dntps_idx, -dNtpsUsed))
                update["bulk"].append((p.ppi_idx, dNtpsUsed.sum()))

                # Module 3: replication termination
                # Determine if any forks have reached the end of their sequences. If
                # so, delete the replisomes and domains that were terminated.
                terminal_lengths = p.replichore_lengths[
                    np.logical_not(right_replichore).astype(np.int64)
                ]
                terminated_replisomes = np.abs(updated_coordinates) == terminal_lengths

                # If any forks were terminated,
                if terminated_replisomes.sum() > 0:
                    # Get domain indexes of terminated forks
                    terminated_domains = np.unique(
                        domain_index_replisome[terminated_replisomes]
                    )

                    # Get attributes of existing domains and full chromosomes
                    (
                        domain_index_domains,
                        child_domains,
                    ) = attrs(state["chromosome_domains"], ["domain_index", "child_domains"])
                    (domain_index_full_chroms,) = attrs(
                        state["full_chromosomes"], ["domain_index"]
                    )

                    # Initialize array of replisomes that should be deleted
                    replisomes_to_delete = np.zeros_like(domain_index_replisome, dtype=np.bool_)

                    # Count number of new full chromosomes that should be created
                    n_new_chromosomes = 0

                    # Initialize array for domain indexes of new full chromosomes
                    domain_index_new_full_chroms = []

                    for terminated_domain_index in terminated_domains:
                        # Get all terminated replisomes in the terminated domain
                        terminated_domain_matching_replisomes = np.logical_and(
                            domain_index_replisome == terminated_domain_index,
                            terminated_replisomes,
                        )

                        # If both replisomes in the domain have terminated, we are
                        # ready to split the chromosome and update the attributes.
                        if terminated_domain_matching_replisomes.sum() == 2:
                            # Tag replisomes and domains with the given domain index
                            # for deletion
                            replisomes_to_delete = np.logical_or(
                                replisomes_to_delete, terminated_domain_matching_replisomes
                            )

                            domain_mask = domain_index_domains == terminated_domain_index

                            # Get child domains of deleted domain
                            child_domains_this_domain = child_domains[
                                np.where(domain_mask)[0][0], :
                            ]

                            # Modify domain index of one existing full chromosome to
                            # index of first child domain
                            domain_index_full_chroms = domain_index_full_chroms.copy()
                            domain_index_full_chroms[
                                np.where(domain_index_full_chroms == terminated_domain_index)[0]
                            ] = child_domains_this_domain[0]

                            # Increment count of new full chromosome
                            n_new_chromosomes += 1

                            # Append chromosome index of new full chromosome
                            domain_index_new_full_chroms.append(child_domains_this_domain[1])

                    # Delete terminated replisomes
                    update["active_replisomes"]["delete"] = np.where(replisomes_to_delete)[0]

                    # Generate new full chromosome molecules
                    if n_new_chromosomes > 0:
                        chromosome_add_update = {
                            "add": {
                                "domain_index": domain_index_new_full_chroms,
                                "division_time": state["global_time"] + p.D_period,
                                "has_triggered_division": False,
                            }
                        }

                        # Reset domain index of existing chromosomes that have finished
                        # replication
                        chromosome_existing_update = {
                            "set": {"domain_index": domain_index_full_chroms}
                        }

                        update["full_chromosomes"].update(
                            {**chromosome_add_update, **chromosome_existing_update}
                        )

                    # Increment counts of replisome subunits
                    if p.mechanistic_replisome:
                        update["bulk"].append(
                            (p.replisome_trimers_idx, 3 * replisomes_to_delete.sum())
                        )
                        update["bulk"].append(
                            (p.replisome_monomers_idx, replisomes_to_delete.sum())
                        )
        # --- end inlined ---
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update

