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
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Float, Overwrite, Node

from v2ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.steps.partition import _protect_state
from v2ecoli.library.units import units


class TwoComponentSystemLogic:
    """Biological logic for two-component system phosphotransfer.

    Extracted from the original PartitionedProcess so that
    Requester and Evolver each get their own instance.
    """

    def __init__(self, parameters):
        self.parameters = {**TwoComponentSystem.defaults, **(parameters or {})}
        parameters = self.parameters

        # Simulation options
        self.jit = parameters.get("jit", False)

        # Get constants
        self.n_avogadro = parameters.get("n_avogadro", 0.0)
        self.cell_density = parameters.get("cell_density", 0.0)

        # Create method
        self.moleculesToNextTimeStep = parameters.get(
            "moleculesToNextTimeStep",
            lambda counts, volume, avogadro, timestep, random, method, min_step, jit: ([], []),
        )

        # Build views
        self.moleculeNames = parameters.get("moleculeNames", [])

        self.seed = parameters.get("seed", 0)
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.molecule_idx = None

    def _init_indices(self, bulk_ids):
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(self.moleculeNames, bulk_ids)


class TwoComponentSystemRequester(Step):
    """Requester step for two-component system.

    Solves ODEs to next time step, writes bulk molecule requests
    to the request store.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop('_logic', None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = TwoComponentSystemLogic(config)
        self.process_name = 'ecoli-two-component-system'

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

        state = _protect_state(state, cell_state=getattr(self, "_cell_state", None))
        proc = self.process
        proc._init_indices(state['bulk']['id'])

        # Get molecule counts
        moleculeCounts = counts(state['bulk'], proc.molecule_idx)

        # Get cell mass and volume
        cellMass = (state['listeners']['mass']['cell_mass'] * units.fg).asNumber(
            units.g
        )
        cellVolume = cellMass / proc.cell_density

        timestep = state.get('timestep', 1.0)

        # Solve ODEs to next time step
        molecules_required, all_molecule_changes = proc.moleculesToNextTimeStep(
            moleculeCounts,
            cellVolume,
            proc.n_avogadro,
            timestep,
            proc.random_state,
            method="BDF",
            jit=proc.jit,
        )

        # Cache for Evolver (shared Logic instance)
        proc.all_molecule_changes = all_molecule_changes

        bulk_request = [(proc.molecule_idx, molecules_required.astype(int))]

        return {
            'request': {self.process_name: {'bulk': bulk_request}},
        }


class TwoComponentSystemEvolver(Step):
    """Evolver step for two-component system.

    Reads allocated bulk molecules from the allocate store,
    recomputes molecule changes with allocated counts, and applies delta.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop('_logic', None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = TwoComponentSystemLogic(config)
        self.process_name = 'ecoli-two-component-system'

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

        state = _protect_state(state, cell_state=getattr(self, "_cell_state", None))
        proc = self.process
        proc._init_indices(state['bulk']['id'])

        # Apply allocation: replace bulk counts with allocated amounts
        allocation = state.pop('allocate', {})
        bulk_alloc = allocation.get('bulk')
        if bulk_alloc is not None and hasattr(bulk_alloc, '__len__') and len(bulk_alloc) > 0 and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk

        # Get allocated molecule counts
        moleculeCounts = counts(state['bulk'], proc.molecule_idx)

        # Get cell mass and volume
        cellMass = (state['listeners']['mass']['cell_mass'] * units.fg).asNumber(
            units.g
        )
        cellVolume = cellMass / proc.cell_density

        timestep = state.get('timestep', 1.0)

        # Use cached all_molecule_changes from Requester (shared Logic)
        if not hasattr(proc, 'all_molecule_changes') or proc.all_molecule_changes is None:
            return {'next_update_time': state.get('global_time', 0) + timestep}
        all_molecule_changes = proc.all_molecule_changes
        proc.all_molecule_changes = None  # Consume: don't reuse stale values

        return {
            'bulk': [(proc.molecule_idx, all_molecule_changes.astype(int))],
            'next_update_time': state.get('global_time', 0) + timestep,
        }


# Keep legacy class for backward compatibility during migration
from v2ecoli.steps.partition import PartitionedProcess

class TwoComponentSystem(PartitionedProcess):
    """Legacy PartitionedProcess wrapper — will be removed after migration."""

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

    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)
        self._logic = TwoComponentSystemLogic(self.parameters)

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "listeners": {"mass": {"cell_mass": {"_default": 0}}},
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        proc = self._logic
        proc._init_indices(states["bulk"]["id"])

        moleculeCounts = counts(states["bulk"], proc.molecule_idx)

        cellMass = (states["listeners"]["mass"]["cell_mass"] * units.fg).asNumber(
            units.g
        )
        self.cellVolume = cellMass / proc.cell_density

        self.molecules_required, self.all_molecule_changes = (
            proc.moleculesToNextTimeStep(
                moleculeCounts,
                self.cellVolume,
                proc.n_avogadro,
                states["timestep"],
                proc.random_state,
                method="BDF",
                jit=proc.jit,
            )
        )
        requests = {"bulk": [(proc.molecule_idx, self.molecules_required.astype(int))]}
        return requests

    def evolve_state(self, timestep, states):
        proc = self._logic
        moleculeCounts = counts(states["bulk"], proc.molecule_idx)
        if (self.molecules_required > moleculeCounts).any():
            _, self.all_molecule_changes = proc.moleculesToNextTimeStep(
                moleculeCounts,
                self.cellVolume,
                proc.n_avogadro,
                10000,
                proc.random_state,
                method="BDF",
                min_time_step=states["timestep"],
                jit=proc.jit,
            )
        update = {"bulk": [(proc.molecule_idx, self.all_molecule_changes.astype(int))]}
        return update
