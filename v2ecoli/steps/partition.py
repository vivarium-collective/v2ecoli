"""
Partitioning system for v2ecoli.

ProcessLogic is the base class for biological processes — a plain class
(not a Step) providing calculate_request() and evolve_state().

ExplicitRequester and ExplicitEvolver are process-bigraph Steps that
wrap a ProcessLogic instance for the partition cycle.

StandaloneStep wraps a ProcessLogic for non-partitioned processes
(those that run request + evolve in one step).

PartitionedProcess is a backwards-compatible alias for ProcessLogic
that also extends Step, for processes not yet migrated.
"""

import abc
import warnings

import numpy as np
from process_bigraph import Step
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Node, Float, Overwrite

from v2ecoli.steps.base import _translate_schema
from v2ecoli.types.stores import SetStore, InPlaceDict, ListenerStore


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


# ---------------------------------------------------------------------------
# ProcessLogic — plain class, no Step inheritance
# ---------------------------------------------------------------------------

class ProcessLogic:
    """Base class for biological process logic.

    Provides the contract for the partition cycle:
    - calculate_request(timestep, states) → request dict
    - evolve_state(timestep, states) → update dict
    - ports_schema() → v1-style port schema dict (legacy, being removed)
    - topology, defaults, parameters, name

    This is NOT a process-bigraph Step.
    """

    name = ''
    topology = {}
    defaults = {}

    def __init__(self, config=None, core=None, parameters=None):
        if parameters is not None and config is None:
            config = parameters
        self.parameters = {**self.defaults, **(config or {})}
        self.request_set = False

    def ports_schema(self):
        return {}

    def calculate_request(self, timestep, states):
        return {}

    def evolve_state(self, timestep, states):
        return {}

    def next_update(self, timestep, states):
        """Run calculate_request then evolve_state in sequence."""
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


# Backwards-compatible alias — processes can inherit from either.
# PartitionedProcess adds Step so it can still be used directly as a step
# (for standalone processes not yet wrapped in StandaloneStep).
class PartitionedProcess(_SafeInvokeMixin, Step, ProcessLogic):
    """Backwards-compatible base class (ProcessLogic + Step).

    New processes should inherit from ProcessLogic directly.
    """
    config_schema = {}

    def __init__(self, config=None, core=None, parameters=None):
        if parameters is not None and config is None:
            config = parameters
        self.parameters = {**self.defaults, **(config or {})}
        self.request_set = False
        Step.__init__(self, config=config or {}, core=core)

    def inputs(self):
        return _translate_schema(self.ports_schema())

    def outputs(self):
        return _translate_schema(self.ports_schema())

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        state = _protect_state(state)
        timestep = state.get('timestep', self.parameters.get('timestep', 1.0))
        return self.next_update(timestep, state)


# ---------------------------------------------------------------------------
# StandaloneStep — wraps a ProcessLogic for non-partitioned execution
# ---------------------------------------------------------------------------

class StandaloneStep(_SafeInvokeMixin, Step):
    """Wraps a ProcessLogic instance as a process-bigraph Step.

    Runs calculate_request → self-allocate → evolve_state in one update().
    Used for processes that don't go through the partition cycle
    (TfBinding, TfUnbinding, ChromosomeStructure, Metabolism).
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config or {}, core=core)
        self.process = config['process']

    def inputs(self):
        return _translate_schema(self.process.ports_schema())

    def outputs(self):
        return _translate_schema(self.process.ports_schema())

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        state = _protect_state(state)
        timestep = state.get('timestep', self.process.parameters.get('timestep', 1.0))
        return self.process.next_update(timestep, state)


# ---------------------------------------------------------------------------
# Explicit Requester/Evolver with proper input/output separation
# ---------------------------------------------------------------------------

class ExplicitRequester(_SafeInvokeMixin, Step):
    """Requester with proper input/output topology separation.

    inputs(): all process ports + timing (reads everything)
    outputs(): only request + next_update_time + listeners (writes only)

    Request output is flat (no process_name nesting) — the topology
    routes to the per-process request store.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)
        self._writes_listeners = config.get('writes_listeners', False)

    def inputs(self):
        ports = _translate_schema(self.process.ports_schema())
        ports['global_time'] = Float(_default=0.0)
        ports.setdefault('timestep', Float(
            _default=self.process.parameters.get('timestep', 1.0)))
        ports['next_update_time'] = Float(
            _default=self.process.parameters.get('timestep', 1.0))
        return ports

    def outputs(self):
        result = {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
        }
        if self._writes_listeners:
            result['listeners'] = ListenerStore()
        return result

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

        listeners = request.pop('listeners', None)
        if listeners is not None:
            result['listeners'] = listeners

        return result


class ExplicitEvolver(_SafeInvokeMixin, Step):
    """Evolver with proper input/output topology separation.

    inputs(): all process ports + allocate + timing
    outputs(): all process ports + next_update_time (excluding allocate, timestep)
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.process = config['process']
        # Ports that the evolver writes to (derived from evolve_state returns)
        self._output_ports = config.get('output_ports', None)

    def inputs(self):
        ports = _translate_schema(self.process.ports_schema())
        ports['allocate'] = InPlaceDict()
        ports['global_time'] = Float(_default=0.0)
        ports.setdefault('timestep', Float(
            _default=self.process.parameters.get('timestep', 1.0)))
        ports['next_update_time'] = Float(
            _default=self.process.parameters.get('timestep', 1.0))
        return ports

    def outputs(self):
        if self._output_ports is not None:
            # Use explicitly specified output ports
            all_ports = _translate_schema(self.process.ports_schema())
            result = {}
            for port in self._output_ports:
                if port in all_ports:
                    result[port] = all_ports[port]
            result['next_update_time'] = Overwrite(_value=Float())
            return result
        # Default: all ports except read-only ones
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
        update['next_update_time'] = global_time + timestep
        return update
