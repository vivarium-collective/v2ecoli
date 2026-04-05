"""
============
Complexation
============

This process encodes molecular simulation of macromolecular complexation,
in which monomers are assembled into complexes. Macromolecular complexation
is done by identifying complexation reactions that are possible (which are
reactions that have sufﬁcient counts of all sub-components), performing one
randomly chosen possible reaction, and re-identifying all possible complexation
reactions. This process assumes that macromolecular complexes form spontaneously,
and that complexation reactions are fast and complete within the time step of the
simulation.
"""

# TODO(wcEcoli):
# - allow for shuffling when appropriate (maybe in another process)
# - handle protein complex dissociation

import numpy as np
from stochastic_arrow import StochasticSystem

from process_bigraph import Step
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Float, Overwrite, Node

from v2ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts, listener_schema
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.steps.partition import _protect_state


class ComplexationLogic:
    """Biological logic for macromolecular complexation.

    Extracted from the original PartitionedProcess so that
    Requester and Evolver each get their own instance.
    """

    def __init__(self, parameters):
        self.parameters = {**Complexation.defaults, **(parameters or {})}
        parameters = self.parameters

        self.stoichiometry = parameters.get("stoichiometry", np.array([[]]))
        self.rates = parameters.get("rates", np.array([]))
        self.molecule_names = parameters.get("molecule_names", [])
        self.molecule_idx = None
        self.reaction_ids = parameters.get("reaction_ids", [])
        self.complex_ids = parameters.get("complex_ids", [])

        self.randomState = np.random.RandomState(seed=parameters.get("seed", 0))
        self.seed = self.randomState.randint(2**31)
        self.system = StochasticSystem(self.stoichiometry, random_seed=self.seed)

    def _init_indices(self, bulk_ids):
        if self.molecule_idx is None:
            self.molecule_idx = bulk_name_to_idx(self.molecule_names, bulk_ids)


class ComplexationRequester(Step):
    """Requester step for complexation.

    Runs StochasticSystem.evolve() to determine molecule surplus,
    writes bulk molecule requests to the request store.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop('_logic', None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = ComplexationLogic(config)
        self.process_name = 'ecoli-complexation'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
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

        timestep = state.get('timestep', 1.0)
        moleculeCounts = counts(state['bulk'], proc.molecule_idx)

        result = proc.system.evolve(timestep, moleculeCounts, proc.rates)
        updatedMoleculeCounts = result["outcome"]

        # Cache for Evolver (shared Logic instance)
        proc._cached_result = result
        proc._cached_initial_counts = moleculeCounts.copy()

        bulk_request = [
            (proc.molecule_idx, np.fmax(moleculeCounts - updatedMoleculeCounts, 0))
        ]

        return {
            'request': {self.process_name: {'bulk': bulk_request}},
        }


class ComplexationEvolver(Step):
    """Evolver step for complexation.

    Reads allocated bulk molecules from the allocate store,
    reruns StochasticSystem.evolve() with allocated counts,
    and applies the delta.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop('_logic', None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = ComplexationLogic(config)
        self.process_name = 'ecoli-complexation'

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

        timestep = state.get('timestep', 1.0)

        # Use cached result from Requester (shared Logic instance)
        if not hasattr(proc, '_cached_result') or proc._cached_result is None:
            return {'next_update_time': state.get('global_time', 0) + state.get('timestep', 1.0)}
        result = proc._cached_result
        initial_counts = proc._cached_initial_counts
        proc._cached_result = None  # Consume
        complexationEvents = result["occurrences"]
        outcome = result["outcome"] - initial_counts

        update = {
            'bulk': [(proc.molecule_idx, outcome)],
            'listeners': {
                'complexation_listener': {
                    'complexation_events': complexationEvents.astype(int)
                }
            },
            'next_update_time': state.get('global_time', 0) + timestep,
        }

        return update


# Keep legacy class for backward compatibility during migration
from v2ecoli.steps.partition import PartitionedProcess

class Complexation(PartitionedProcess):
    """Legacy PartitionedProcess wrapper — will be removed after migration."""

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

    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)
        self._logic = ComplexationLogic(self.parameters)

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "listeners": {
                "complexation_listener": {
                    **listener_schema(
                        {
                            "complexation_events": (
                                [0] * len(self._logic.reaction_ids),
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
        timestep = states["timestep"]
        proc._init_indices(states["bulk"]["id"])

        moleculeCounts = counts(states["bulk"], proc.molecule_idx)

        result = proc.system.evolve(timestep, moleculeCounts, proc.rates)
        updatedMoleculeCounts = result["outcome"]
        requests = {}
        requests["bulk"] = [
            (proc.molecule_idx, np.fmax(moleculeCounts - updatedMoleculeCounts, 0))
        ]
        return requests

    def evolve_state(self, timestep, states):
        proc = self._logic
        timestep = states["timestep"]
        substrate = counts(states["bulk"], proc.molecule_idx)

        result = proc.system.evolve(timestep, substrate, proc.rates)
        complexationEvents = result["occurrences"]
        outcome = result["outcome"] - substrate

        update = {
            "bulk": [(proc.molecule_idx, outcome)],
            "listeners": {
                "complexation_listener": {
                    "complexation_events": complexationEvents.astype(int)
                }
            },
        }

        return update
