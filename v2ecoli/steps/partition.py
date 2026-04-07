"""
Partition utilities for v2ecoli.

Provides shared helpers used by per-process Requester/Evolver Steps:
- _protect_state(): copies bulk/unique arrays before process execution
- _SafeInvokeMixin: catches errors in update() to prevent cascade crashes
- deep_merge(): recursive dict merge
"""

import numpy as np
from process_bigraph import Step
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Float, Overwrite

from v2ecoli.types.stores import InPlaceDict


def _protect_state(state):
    """Return a shallow copy of state with bulk/unique arrays copied.

    Processes from v1 mutate their input arrays in place. Since core.view
    returns the live state object, we must copy arrays that processes
    might modify to prevent corruption of the simulation state.
    """
    protected = dict(state)
    if 'bulk' in protected and hasattr(protected['bulk'], 'copy'):
        protected['bulk'] = protected['bulk'].copy()
        protected['bulk'].flags.writeable = True
    if 'unique' in protected and isinstance(protected['unique'], dict):
        protected['unique'] = {
            k: v.copy() if hasattr(v, 'copy') else v
            for k, v in protected['unique'].items()
        }
        for arr in protected['unique'].values():
            if hasattr(arr, 'flags'):
                arr.flags.writeable = True
    return protected


class _SafeInvokeMixin:
    """Mixin that catches and LOGS errors in update() to prevent cascade crashes."""
    def invoke(self, state, interval=None):
        try:
            update = self.update(state)
        except Exception as e:
            import warnings
            step_name = getattr(self, 'name', type(self).__name__)
            warnings.warn(
                f"Step {step_name} raised {type(e).__name__}: {e}",
                RuntimeWarning, stacklevel=2)
            update = {}
        return SyncUpdate(update)


def deep_merge(base, override):
    """Recursively merge override into base (modifies base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class RequesterBase(_SafeInvokeMixin, Step):
    """Base class for partitioned process Requesters.

    Subclasses must implement:
        inputs() -> dict
        outputs() -> dict  (optional, default returns request + next_update_time)
        compute_request(state, timestep) -> dict with 'bulk' and/or 'listeners'
    """

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

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
        request = self.compute_request(state, timestep)
        self.process.request_set = True

        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request

        listeners = request.pop('listeners', None)
        if listeners is not None:
            result['listeners'] = listeners

        return result

    def compute_request(self, state, timestep):
        raise NotImplementedError


class EvolverBase(_SafeInvokeMixin, Step):
    """Base class for partitioned process Evolvers.

    Subclasses must implement:
        inputs() -> dict
        outputs() -> dict
        compute_evolve(state, timestep) -> dict with updates
    """

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

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
        update = self.compute_evolve(state, timestep)
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update

    def compute_evolve(self, state, timestep):
        raise NotImplementedError

