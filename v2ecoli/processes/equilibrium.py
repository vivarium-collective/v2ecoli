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
from bigraph_schema.schema import Float, Overwrite

from v2ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts, listener_schema
from v2ecoli.steps.partition import PartitionedProcess, _protect_state, deep_merge, _SafeInvokeMixin
from v2ecoli.library.units import units
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore


class Equilibrium(PartitionedProcess):
    """Equilibrium PartitionedProcess

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
    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)

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

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "listeners": {
                "mass": listener_schema({"cell_mass": 0}),
                "equilibrium_listener": {
                    **listener_schema(
                        {
                            "reaction_rates": (
                                [0.0] * len(self.reaction_ids),
                                self.reaction_ids,
                            )
                        }
                    )
                },
            },
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        # At t=0, convert all strings to indices
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(
                self.moleculeNames, states["bulk"]["id"]
            )

        # Get molecule counts
        moleculeCounts = counts(states["bulk"], self.molecule_idx)

        # Get cell mass and volume
        cellMass = (states["listeners"]["mass"]["cell_mass"] * units.fg).asNumber(
            units.g
        )
        cellVolume = cellMass / self.cell_density

        # Solve ODEs to steady state
        self.rxnFluxes, self.req = self.fluxesAndMoleculesToSS(
            moleculeCounts,
            cellVolume,
            self.n_avogadro,
            self.random_state,
            jit=self.jit,
        )

        # Request counts of molecules needed
        requests = {"bulk": [(self.molecule_idx, self.req.astype(int))]}
        return requests

    def evolve_state(self, timestep, states):
        # Get molecule counts
        moleculeCounts = counts(states["bulk"], self.molecule_idx)

        # Get counts of molecules allocated to this process
        rxnFluxes = self.rxnFluxes.copy()

        # If we didn't get allocated all the molecules we need, make do with
        # what we have (decrease reaction fluxes so that they make use of what
        # we have, but not more). Reduces at least one reaction every iteration
        # so the max number of iterations is the number of reactions that were
        # originally expected to occur + 1 to reach the break statement.
        max_iterations = int(np.abs(rxnFluxes).sum()) + 1
        for it in range(max_iterations):
            # Check if any metabolites will have negative counts with current reactions
            negative_metabolite_idxs = np.where(
                np.dot(self.stoichMatrix, rxnFluxes) + moleculeCounts < 0
            )[0]
            if len(negative_metabolite_idxs) == 0:
                break

            # Reduce reactions that consume metabolites with negative counts
            limited_rxn_stoich = self.stoichMatrix[negative_metabolite_idxs, :]
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
        deltaMolecules = np.dot(self.stoichMatrix, rxnFluxes).astype(int)

        update = {
            "bulk": [(self.molecule_idx, deltaMolecules)],
            "listeners": {
                "equilibrium_listener": {
                    "reaction_rates": deltaMolecules[self.product_indices]
                    / states["timestep"]
                }
            },
        }

        return update


class EquilibriumRequester(_SafeInvokeMixin, Step):
    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
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
        request = self.process.calculate_request(timestep, state)
        self.process.request_set = True
        bulk_request = request.pop('bulk', None)
        result = {'request': {self.process_name: {}}}
        if bulk_request is not None:
            result['request'][self.process_name]['bulk'] = bulk_request
        return result


class EquilibriumEvolver(_SafeInvokeMixin, Step):
    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
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
        update = self.process.evolve_state(timestep, state)
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update
