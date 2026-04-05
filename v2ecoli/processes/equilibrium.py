"""
===========
Equilibrium
===========

This process models how ligands are bound to or unbound
from their transcription factor binding partners in a fashion
that maintains equilibrium.
"""

import numpy as np

from process_bigraph import Step
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Float, Overwrite, Node

from v2ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts, listener_schema
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.steps.partition import _protect_state
from v2ecoli.library.units import units


class EquilibriumLogic:
    """Biological logic for equilibrium reactions.

    Extracted from the original PartitionedProcess so that
    Requester and Evolver each get their own instance.
    """

    def __init__(self, parameters):
        self.parameters = {**Equilibrium.defaults, **(parameters or {})}
        parameters = self.parameters

        # Simulation options
        self.jit = parameters.get("jit", False)

        # Get constants
        self.n_avogadro = parameters.get("n_avogadro", 0.0)
        self.cell_density = parameters.get("cell_density", 0.0)

        # Create matrix and method
        self.stoichMatrix = parameters.get("stoichMatrix", [[]])
        self.fluxesAndMoleculesToSS = parameters.get(
            "fluxesAndMoleculesToSS",
            lambda counts, volume, avogadro, random, jit: ([], []),
        )

        self.product_indices = [
            idx for idx in np.where(np.any(np.array(self.stoichMatrix) > 0, axis=1))[0]
        ]

        # Build views
        self.moleculeNames = parameters.get("moleculeNames", [])
        self.molecule_idx = None

        self.seed = parameters.get("seed", 0)
        self.random_state = np.random.RandomState(seed=self.seed)

        self.complex_ids = parameters.get("complex_ids", [])
        self.reaction_ids = parameters.get("reaction_ids", [])

    def _init_indices(self, bulk_ids):
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(self.moleculeNames, bulk_ids)


class EquilibriumRequester(Step):
    """Requester step for equilibrium.

    Solves ODEs to steady state, writes bulk molecule requests
    to the request store.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop('_logic', None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = EquilibriumLogic(config)
        self.process_name = 'ecoli-equilibrium'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(_default=1.0),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def invoke(self, state, interval=None):
        try:
            update = self.update(state)
        except Exception:
            update = {}
        return SyncUpdate(update)

    def update(self, state, interval=None):
        if state.get('next_update_time', 0) > state.get('global_time', 0):
            return {}

        state = _protect_state(state)
        proc = self.process
        proc._init_indices(state['bulk']['id'])

        # Get molecule counts
        moleculeCounts = counts(state['bulk'], proc.molecule_idx)

        # Get cell mass and volume
        cellMass = (state['listeners']['mass']['cell_mass'] * units.fg).asNumber(
            units.g
        )
        cellVolume = cellMass / proc.cell_density

        # Solve ODEs to steady state
        rxnFluxes, req = proc.fluxesAndMoleculesToSS(
            moleculeCounts,
            cellVolume,
            proc.n_avogadro,
            proc.random_state,
            jit=proc.jit,
        )

        # Cache fluxes for Evolver (shared Logic instance)
        proc.rxnFluxes = rxnFluxes.copy()

        # Request counts of molecules needed
        bulk_request = [(proc.molecule_idx, req.astype(int))]

        return {
            'request': {self.process_name: {'bulk': bulk_request}},
        }


class EquilibriumEvolver(Step):
    """Evolver step for equilibrium.

    Reads allocated bulk molecules from the allocate store,
    recomputes fluxes with allocated counts, and applies delta.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop('_logic', None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = EquilibriumLogic(config)
        self.process_name = 'ecoli-equilibrium'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'allocate': InPlaceDict(),
            'listeners': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(_default=1.0),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def invoke(self, state, interval=None):
        try:
            update = self.update(state)
        except Exception:
            update = {}
        return SyncUpdate(update)

    def update(self, state, interval=None):
        if state.get('next_update_time', 0) > state.get('global_time', 0):
            return {}

        state = _protect_state(state)
        proc = self.process
        proc._init_indices(state['bulk']['id'])

        # Apply allocation: replace bulk counts with allocated amounts
        allocation = state.pop('allocate', {})
        bulk_alloc = allocation.get('bulk')
        if bulk_alloc is not None and hasattr(bulk_alloc, '__len__') and len(bulk_alloc) > 0 and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk

        # Get molecule counts (allocated)
        moleculeCounts = counts(state['bulk'], proc.molecule_idx)

        # Use cached rxnFluxes from Requester (shared Logic instance)
        if not hasattr(proc, 'rxnFluxes') or proc.rxnFluxes is None:
            return {'next_update_time': state.get('global_time', 0) + state.get('timestep', 1.0)}
        rxnFluxes = proc.rxnFluxes.copy()
        proc.rxnFluxes = None  # Consume: don't reuse stale values
        max_iterations = int(np.abs(rxnFluxes).sum()) + 1
        for it in range(max_iterations):
            negative_metabolite_idxs = np.where(
                np.dot(proc.stoichMatrix, rxnFluxes) + moleculeCounts < 0
            )[0]
            if len(negative_metabolite_idxs) == 0:
                break

            limited_rxn_stoich = proc.stoichMatrix[negative_metabolite_idxs, :]
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
        deltaMolecules = np.dot(proc.stoichMatrix, rxnFluxes).astype(int)

        timestep = state.get('timestep', 1.0)
        update = {
            'bulk': [(proc.molecule_idx, deltaMolecules)],
            'listeners': {
                'equilibrium_listener': {
                    'reaction_rates': deltaMolecules[proc.product_indices]
                    / timestep
                }
            },
            'next_update_time': state.get('global_time', 0) + timestep,
        }

        return update


# Keep legacy class for backward compatibility during migration
from v2ecoli.steps.partition import PartitionedProcess

class Equilibrium(PartitionedProcess):
    """Legacy PartitionedProcess wrapper — will be removed after migration."""

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

    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)
        self._logic = EquilibriumLogic(self.parameters)

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "listeners": {
                "mass": listener_schema({"cell_mass": 0}),
                "equilibrium_listener": {
                    **listener_schema(
                        {
                            "reaction_rates": (
                                [0.0] * len(self._logic.reaction_ids),
                                self._logic.reaction_ids,
                            )
                        }
                    )
                },
            },
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        proc = self._logic
        proc._init_indices(states["bulk"]["id"])

        moleculeCounts = counts(states["bulk"], proc.molecule_idx)
        cellMass = (states["listeners"]["mass"]["cell_mass"] * units.fg).asNumber(
            units.g
        )
        cellVolume = cellMass / proc.cell_density

        self.rxnFluxes, self.req = proc.fluxesAndMoleculesToSS(
            moleculeCounts,
            cellVolume,
            proc.n_avogadro,
            proc.random_state,
            jit=proc.jit,
        )

        requests = {"bulk": [(proc.molecule_idx, self.req.astype(int))]}
        return requests

    def evolve_state(self, timestep, states):
        proc = self._logic
        proc._init_indices(states["bulk"]["id"])

        moleculeCounts = counts(states["bulk"], proc.molecule_idx)
        rxnFluxes = self.rxnFluxes.copy()

        max_iterations = int(np.abs(rxnFluxes).sum()) + 1
        for it in range(max_iterations):
            negative_metabolite_idxs = np.where(
                np.dot(proc.stoichMatrix, rxnFluxes) + moleculeCounts < 0
            )[0]
            if len(negative_metabolite_idxs) == 0:
                break

            limited_rxn_stoich = proc.stoichMatrix[negative_metabolite_idxs, :]
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

        deltaMolecules = np.dot(proc.stoichMatrix, rxnFluxes).astype(int)

        update = {
            "bulk": [(proc.molecule_idx, deltaMolecules)],
            "listeners": {
                "equilibrium_listener": {
                    "reaction_rates": deltaMolecules[proc.product_indices]
                    / states["timestep"]
                }
            },
        }

        return update
