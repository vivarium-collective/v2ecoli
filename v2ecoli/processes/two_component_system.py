"""
====================
Two Component System
====================

This process models the phosphotransfer reactions of signal transduction pathways.

Specifically, phosphate groups are transferred from histidine kinases to response regulators
and back in response to counts of ligand stimulants.
"""

import numpy as np

from process_bigraph import Step
from bigraph_schema.schema import Float, Overwrite

from v2ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts
from v2ecoli.library.units import units
from v2ecoli.steps.partition import PartitionedProcess, _protect_state, deep_merge, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
class TwoComponentSystem(PartitionedProcess):
    """Two Component System PartitionedProcess"""

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
    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)

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

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "listeners": {"mass": {"cell_mass": {"_default": 0}}},
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
        self.cellVolume = cellMass / self.cell_density

        # Solve ODEs to next time step using the BDF solver through solve_ivp.
        # Note: the BDF solver has been empirically tested to be the fastest
        # solver for this setting among the list of solvers that can be used
        # by the scipy ODE suite.
        self.molecules_required, self.all_molecule_changes = (
            self.moleculesToNextTimeStep(
                moleculeCounts,
                self.cellVolume,
                self.n_avogadro,
                states["timestep"],
                self.random_state,
                method="BDF",
                jit=self.jit,
            )
        )
        requests = {"bulk": [(self.molecule_idx, self.molecules_required.astype(int))]}
        return requests

    def evolve_state(self, timestep, states):
        moleculeCounts = counts(states["bulk"], self.molecule_idx)
        # Check if any molecules were allocated fewer counts than requested
        if (self.molecules_required > moleculeCounts).any():
            _, self.all_molecule_changes = self.moleculesToNextTimeStep(
                moleculeCounts,
                self.cellVolume,
                self.n_avogadro,
                10000,
                self.random_state,
                method="BDF",
                min_time_step=states["timestep"],
                jit=self.jit,
            )
        # Increment changes in molecule counts
        update = {"bulk": [(self.molecule_idx, self.all_molecule_changes.astype(int))]}

        return update


class TwoComponentSystemRequester(_SafeInvokeMixin, Step):
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
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request
        return result


class TwoComponentSystemEvolver(_SafeInvokeMixin, Step):
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
