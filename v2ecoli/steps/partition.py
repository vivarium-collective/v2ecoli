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


class _SafeInvokeMixin:
    """Mixin that catches errors in update() to prevent cascade crashes."""
    def invoke(self, state, interval=None):
        try:
            update = self.update(state)
        except (KeyError, TypeError, AttributeError, ValueError, AssertionError, RuntimeError):
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

    def update(self, state, interval=None):
        """Run both request and evolve (standalone execution)."""
        timestep = state.get('timestep', self.parameters.get('timestep', 1.0))

        requests = self.calculate_request(timestep, state)
        bulk_requests = requests.pop("bulk", [])
        if bulk_requests:
            bulk_copy = state["bulk"].copy()
            for bulk_idx, request in bulk_requests:
                bulk_copy[bulk_idx] = request
            state["bulk"] = bulk_copy
        state = deep_merge(state, requests)
        update = self.evolve_state(timestep, state)
        if "listeners" in requests:
            update["listeners"] = deep_merge(
                update.get("listeners", {}), requests["listeners"])
        return update


class Requester(_SafeInvokeMixin, Step):
    """Runs calculate_request() on a PartitionedProcess."""

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.process = config['process']
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
            'request': Node(),
            'listeners': Node(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        process = self.process
        timestep = state.get('timestep', 1.0)
        request = process.calculate_request(timestep, state)
        process.request_set = True

        result = {'request': {}}
        bulk_request = request.pop('bulk', None)
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request

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
        ports['allocate'] = Node()
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

        allocations = state.pop('allocate', {})
        state = deep_merge(state, allocations)

        process = self.process
        if not process.request_set:
            return {}

        timestep = state.get('timestep', 1.0)
        update = process.evolve_state(timestep, state)
        update['next_update_time'] = global_time + timestep
        return update
