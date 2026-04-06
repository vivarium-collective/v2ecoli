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


def _protect_state(state):
    """Return a shallow copy of state with bulk/unique arrays copied.

    Processes from v1 mutate their input arrays in place. Since core.view
    returns the live state object, we must copy arrays that processes
    might modify to prevent corruption of the simulation state.
    """
    import numpy as np
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


class PartitionedProcess(_SafeInvokeMixin, Step):
    """Base class for biological processes that need resource allocation.

    Subclasses implement calculate_request() and evolve_state().
    When run standalone (not through Requester/Evolver), it executes
    both request and evolve in sequence.
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
    """Runs calculate_request() on a PartitionedProcess."""

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

        # Write request keyed by process name into the request store
        bulk_request = request.pop('bulk', None)
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

        # Apply allocation: replace bulk counts with allocated amounts
        import numpy as np
        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)

        process = self.process
        if not process.request_set:
            return {}

        timestep = state.get('timestep', 1.0)
        update = process.evolve_state(timestep, state)
        update['next_update_time'] = global_time + timestep
        return update


# ---------------------------------------------------------------------------
# Explicit Requester/Evolver with proper input/output separation
# ---------------------------------------------------------------------------

class ExplicitRequester(_SafeInvokeMixin, Step):
    """Requester with proper input/output topology separation.

    inputs(): all process ports + timing (reads everything)
    outputs(): only request + next_update_time + listeners (writes only)
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)
        # Which output ports the requester writes (besides request + next_update_time)
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
        result = {'request': {self.process_name: {}}}
        if bulk_request is not None:
            result['request'][self.process_name]['bulk'] = bulk_request

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

        import numpy as np
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
