"""Process-bigraph partitioned process: two_component_system."""

from typing import Any, Callable, Optional, Tuple, cast

from numba import njit
import numpy as np
import numpy.typing as npt
import scipy.sparse
import warnings
from scipy.integrate import solve_ivp

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
from v2ecoli.library.unit_defs import units
from v2ecoli.steps.partition import _protect_state, deep_merge, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore, SetStore


def _apply_config_defaults(config_schema, parameters):
    """Merge config_schema defaults with provided parameters."""
    merged = {}
    for key, spec in config_schema.items():
        if isinstance(spec, dict) and "_default" in spec:
            merged[key] = spec["_default"]
    merged.update(parameters or {})
    return merged

class TwoComponentSystemLogic:
    """Two Component System — shared state container for Requester/Evolver."""

    name = "ecoli-two-component-system"
    topology = {
    "listeners": ("listeners",),
    "bulk": ("bulk",),
    "timestep": ("timestep",),
}
    config_schema = {
        "jit": {"_default": False},
        "n_avogadro": {"_default": 0.0},
        "cell_density": {"_default": 0.0},
        "moleculesToNextTimeStep": {"_default": None},
        "moleculeNames": {"_default": []},
        "seed": {"_default": 0},
    }

    # Constructor
    def __init__(self, parameters=None):
        self.parameters = _apply_config_defaults(self.config_schema, parameters)
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
            'allocate': SetStore(),
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
