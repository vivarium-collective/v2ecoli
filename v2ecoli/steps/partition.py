"""
Partitioning system for v2ecoli.

PartitionedProcess is a base class (not a bigraph Step/Process) that
defines calculate_request() and evolve_state() for biological processes
that need resource allocation.

Requester and Evolver are v2-native BigraphSteps that call into a
PartitionedProcess and coordinate with an Allocator.
"""

import abc
import warnings

from process_bigraph import Step


def deep_merge(base, override):
    """Recursively merge override into base (modifies base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class PartitionedProcess:
    """Base class for processes whose updates are partitioned.

    Subclasses implement calculate_request() and evolve_state().
    This is NOT a bigraph Step — it holds biological logic only.
    The Requester and Evolver steps call into it.
    """

    name = ''
    topology = {}
    defaults = {}

    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

    @abc.abstractmethod
    def ports_schema(self):
        return {}

    def get_schema(self):
        """Get the ports schema (alias for compatibility)."""
        return self.ports_schema()

    @abc.abstractmethod
    def calculate_request(self, timestep, states):
        return {}

    @abc.abstractmethod
    def evolve_state(self, timestep, states):
        return {}


class Requester(Step):
    """Runs calculate_request() on a PartitionedProcess.

    Reads bulk/unique molecule state, writes resource requests to
    the 'request' store for the Allocator to read.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.process = config['process']
        assert isinstance(self.process, PartitionedProcess)
        self.name = f"{self.process.name}_requester"

    def inputs(self):
        ports = self.process.get_schema()
        # Additional inputs for coordination
        from bigraph_schema.schema import Node, Float, Overwrite
        ports['global_time'] = Float(_default=0.0)
        ports['timestep'] = Float(_default=self.process.parameters.get('timestep', 1.0))
        ports['next_update_time'] = Float(
            _default=self.process.parameters.get('timestep', 1.0))
        ports['process'] = Node()
        return ports

    def outputs(self):
        from bigraph_schema.schema import Node, Float, Overwrite
        ports = {}
        ports['request'] = Node()
        ports['process'] = Overwrite(_value=Node())
        ports['listeners'] = Node()
        ports['next_update_time'] = Overwrite(_value=Float())
        return ports

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        # Variable timestepping: check if it's time to run
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        process = self.process
        request = process.calculate_request(state.get('timestep', 1.0), state)
        process.request_set = True

        # Separate bulk requests into the request port
        result = {}
        result['request'] = {}
        bulk_request = request.pop('bulk', None)
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request

        # Pass through listener updates
        listeners = request.pop('listeners', None)
        if listeners is not None:
            result['listeners'] = listeners

        # Update shared process instance
        result['process'] = (process,)
        return result


class Evolver(Step):
    """Runs evolve_state() on a PartitionedProcess.

    Reads allocations from the Allocator and applies them to state,
    then calls evolve_state() to compute the biological update.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.process = config['process']
        assert isinstance(self.process, PartitionedProcess)
        self.name = f"{self.process.name}_evolver"

    def inputs(self):
        ports = self.process.get_schema()
        from bigraph_schema.schema import Node, Float
        ports['allocate'] = Node()
        ports['global_time'] = Float(_default=0.0)
        ports['timestep'] = Float(_default=self.process.parameters.get('timestep', 1.0))
        ports['next_update_time'] = Float(
            _default=self.process.parameters.get('timestep', 1.0))
        ports['process'] = Node()
        return ports

    def outputs(self):
        from bigraph_schema.schema import Node, Float, Overwrite
        ports = self.process.get_schema()
        ports['process'] = Overwrite(_value=Node())
        ports['next_update_time'] = Overwrite(_value=Float())
        return ports

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        # Variable timestepping
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        # Merge allocations into state
        allocations = state.pop('allocate', {})
        state = deep_merge(state, allocations)

        process = self.process

        # Skip if Requester hasn't run yet (e.g. after division)
        if not process.request_set:
            return {}

        update = process.evolve_state(state.get('timestep', 1.0), state)
        update['process'] = (process,)
        update['next_update_time'] = global_time + state.get('timestep', 1.0)
        return update
