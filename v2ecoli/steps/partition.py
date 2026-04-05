"""
Partitioning system for v2ecoli.

All classes are process-bigraph Steps with proper inputs()/outputs()/update().
PartitionedProcess is the biological base class that also extends Step.
Requester and Evolver wrap a PartitionedProcess for the partition cycle.
"""

import abc
import warnings

from process_bigraph import Step
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Node, Float, Overwrite

from v2ecoli.steps.base import _translate_schema
from v2ecoli.types.stores import SetStore, InPlaceDict, ListenerStore


_timestep_snapshot = {
    'bulk': None,
    'unique': None,
    'last_time': -1.0,
}


def set_timestep_snapshot(cell_state):
    """Capture bulk/unique state at the start of a timestep.

    All steps in the timestep will see this snapshot, matching v1's
    behavior where all processes in a layer receive the same state.
    Called once at the start of each timestep cycle.
    """
    import numpy as np
    if 'bulk' in cell_state and hasattr(cell_state['bulk'], 'copy'):
        _timestep_snapshot['bulk'] = cell_state['bulk'].copy()
    if 'unique' in cell_state and isinstance(cell_state['unique'], dict):
        _timestep_snapshot['unique'] = {
            k: v.copy() if hasattr(v, 'copy') else v
            for k, v in cell_state['unique'].items()
        }


def _protect_state(state, use_snapshot=False, cell_state=None):
    """Return state with bulk/unique copied to prevent mutation.

    If cell_state is provided and 'listeners' is missing from state,
    injects listeners from cell_state. This allows steps to access
    listener data without declaring it as a dependency (which would
    create cross-layer cycles with listener steps).
    """
    import numpy as np
    protected = dict(state)

    # Inject stores from cell_state if not in core.view.
    # This provides bulk/unique/listeners to steps that don't declare
    # them as input dependencies (to avoid R/W cycles).
    if cell_state is not None:
        for store in ('listeners', 'bulk', 'unique', 'environment', 'boundary',
                      'process_state'):
            if store not in protected:
                val = cell_state.get(store)
                if val is not None:
                    protected[store] = val

    # Auto-take snapshot on first use_snapshot call of each timestep
    if use_snapshot:
        gt = state.get('global_time', 0.0)
        if gt != _timestep_snapshot['last_time']:
            _timestep_snapshot['last_time'] = gt
            if 'bulk' in state and hasattr(state['bulk'], 'copy'):
                _timestep_snapshot['bulk'] = state['bulk'].copy()
            if 'unique' in state and isinstance(state['unique'], dict):
                _timestep_snapshot['unique'] = {
                    k: v.copy() if hasattr(v, 'copy') else v
                    for k, v in state['unique'].items()
                }

    # Use timestep snapshot if requested and available
    if use_snapshot and _timestep_snapshot['bulk'] is not None and 'bulk' in protected:
        protected['bulk'] = _timestep_snapshot['bulk'].copy()
        protected['bulk'].flags.writeable = True
    elif 'bulk' in protected and hasattr(protected['bulk'], 'copy'):
        protected['bulk'] = protected['bulk'].copy()
        protected['bulk'].flags.writeable = True

    if _timestep_snapshot['unique'] is not None and 'unique' in protected:
        protected['unique'] = {
            k: v.copy() if hasattr(v, 'copy') else v
            for k, v in _timestep_snapshot['unique'].items()
        }
    elif 'unique' in protected and isinstance(protected['unique'], dict):
        protected['unique'] = {
            k: v.copy() if hasattr(v, 'copy') else v
            for k, v in protected['unique'].items()
        }

    if 'unique' in protected and isinstance(protected['unique'], dict):
        for arr in protected['unique'].values():
            if hasattr(arr, 'flags'):
                arr.flags.writeable = True

    return protected


class _SafeInvokeMixin:
    """Mixin that catches errors in update() to prevent cascade crashes."""
    def invoke(self, state, interval=None):
        try:
            update = self.update(state)
        except Exception:
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


class PartitionedProcess(_SafeInvokeMixin, Step):
    """Base class for biological processes that need resource allocation.

    Subclasses implement calculate_request() and evolve_state().
    When run standalone (not through Requester/Evolver), it executes
    both request and evolve in sequence.

    The Requester stores its request on the process instance directly
    (self._last_request), and the Evolver reads it. This bypasses the
    request/allocate stores which have schema routing issues with
    unmodified bigraph-schema.
    """

    name = ''
    topology = {}
    defaults = {}
    config_schema = {}

    def __init__(self, config=None, core=None, parameters=None):
        # Accept both config= and parameters= for compatibility
        if parameters is not None and config is None:
            config = parameters
        self.parameters = {**self.defaults, **(config or {})}
        self.request_set = False
        super().__init__(config=config or {}, core=core)

        # Infer config_schema from actual parameters (like genEcoli)
        if core is not None and self.parameters:
            try:
                self._inferred_config_schema = core.infer(self.parameters)
            except Exception:
                self._inferred_config_schema = None

    @abc.abstractmethod
    def ports_schema(self):
        """Return the v1-style ports schema dict."""
        return {}

    def inputs(self):
        return _translate_schema(self.ports_schema())

    def outputs(self):
        return _translate_schema(self.ports_schema())

    def initial_state(self, config=None):
        return {}

    @abc.abstractmethod
    def calculate_request(self, timestep, states):
        return {}

    @abc.abstractmethod
    def evolve_state(self, timestep, states):
        return {}

    def next_update(self, timestep, states):
        """Default: run calculate_request then evolve_state."""
        requests = self.calculate_request(timestep, states)
        bulk_requests = requests.pop("bulk", [])
        if bulk_requests:
            bulk_copy = states["bulk"].copy()
            for bulk_idx, request in bulk_requests:
                bulk_copy[bulk_idx] = request
            states["bulk"] = bulk_copy
        states = deep_merge(states, requests)
        update = self.evolve_state(timestep, states)
        if "listeners" in requests:
            update["listeners"] = deep_merge(
                update.get("listeners", {}), requests["listeners"])
        return update

    def update(self, state, interval=None):
        """Run standalone via next_update."""
        state = _protect_state(state)
        timestep = state.get('timestep', self.parameters.get('timestep', 1.0))
        return self.next_update(timestep, state)


class Requester(_SafeInvokeMixin, Step):
    """Runs calculate_request() on a PartitionedProcess.

    Writes request data to the request store keyed by process name.
    The output is wired to ('request',) — the entire request dict —
    so that core.apply uses InPlaceDict semantics for the merge.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)
        self.name = f"{self.process.name}_requester"

    def inputs(self):
        ports = _translate_schema(self.process.ports_schema())
        ports['global_time'] = Float(_default=0.0)
        ports['timestep'] = Float(_default=self.process.parameters.get('timestep', 1.0))
        ports['next_update_time'] = Float(
            _default=self.process.parameters.get('timestep', 1.0))
        return ports

    def outputs(self):
        return {
            'request': InPlaceDict(),
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
        process = self.process
        timestep = state.get('timestep', 1.0)
        request = process.calculate_request(timestep, state)
        process.request_set = True

        bulk_request = request.pop('bulk', None)

        # Write request keyed by process name into the request store.
        # The output is wired to ('request',) so InPlaceDict apply merges.
        result = {'request': {self.process_name: {}}}
        if bulk_request is not None:
            result['request'][self.process_name]['bulk'] = bulk_request

        listeners = request.pop('listeners', None)
        if listeners is not None:
            result['listeners'] = listeners

        return result


class Evolver(_SafeInvokeMixin, Step):
    """Runs evolve_state() on a PartitionedProcess."""

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.process = config['process']
        self.name = f"{self.process.name}_evolver"

    def inputs(self):
        ports = _translate_schema(self.process.ports_schema())
        ports['allocate'] = InPlaceDict()
        ports['global_time'] = Float(_default=0.0)
        ports['timestep'] = Float(_default=self.process.parameters.get('timestep', 1.0))
        ports['next_update_time'] = Float(
            _default=self.process.parameters.get('timestep', 1.0))
        return ports

    def outputs(self):
        ports = _translate_schema(self.process.ports_schema())
        ports['next_update_time'] = Overwrite(_value=Float())
        return ports

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)

        process = self.process
        if not process.request_set:
            return {}

        # Read allocation from allocate store.
        # Replace bulk counts with allocated amounts so evolve_state
        # only sees what the allocator granted (v1 partition semantics).
        import numpy as np
        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk

        state = deep_merge(state, allocations)

        timestep = state.get('timestep', 1.0)
        update = process.evolve_state(timestep, state)
        update['next_update_time'] = global_time + timestep
        return update
