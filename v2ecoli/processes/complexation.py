"""Process-bigraph partitioned process: complexation."""

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

from stochastic_arrow import StochasticSystem


def _apply_config_defaults(config_schema, parameters):
    """Merge config_schema defaults with provided parameters."""
    merged = {}
    for key, spec in config_schema.items():
        if isinstance(spec, dict) and "_default" in spec:
            merged[key] = spec["_default"]
    merged.update(parameters or {})
    return merged

class ComplexationLogic:
    """Complexation — shared state container for Requester/Evolver."""

    name = "ecoli-complexation"
    topology = {"bulk": ("bulk",), "listeners": ("listeners",), "timestep": ("timestep",)}
    config_schema = {
        "stoichiometry": {"_default": None},
        "rates": {"_default": None},
        "molecule_names": {"_default": []},
        "seed": {"_default": 0},
        "reaction_ids": {"_default": []},
        "complex_ids": {"_default": []},
        "time_step": {"_default": 1},
    }

    def __init__(self, parameters=None):
        self.parameters = _apply_config_defaults(self.config_schema, parameters)
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
