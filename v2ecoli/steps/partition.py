"""
Partitioning system for v2ecoli.

PartitionedProcess is the base class for biological processes that need
resource allocation. Requester and Evolver wrap a PartitionedProcess
and coordinate with an Allocator.

All classes expose ports_schema() and next_update() for compatibility
with the v2ecoli simulation engine.
"""

import abc
import warnings

from v2ecoli.library.schema import numpy_schema


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
        return self.ports_schema()

    @abc.abstractmethod
    def calculate_request(self, timestep, states):
        return {}

    @abc.abstractmethod
    def evolve_state(self, timestep, states):
        return {}

    def next_update(self, timestep, states):
        """Run both request and evolve (for standalone execution)."""
        if getattr(self, 'request_only', False):
            return self.calculate_request(timestep, states)
        if getattr(self, 'evolve_only', False):
            return self.evolve_state(timestep, states)

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


class Requester:
    """Runs calculate_request() on a PartitionedProcess.

    Reads bulk/unique molecule state, writes resource requests to
    the 'request' store for the Allocator to read.
    """

    def __init__(self, parameters=None, config=None, **kwargs):
        params = parameters or config or {}
        self.parameters = params
        assert isinstance(params['process'], PartitionedProcess)
        self.process = params['process']
        self.name = f"{self.process.name}_requester"
        self.cached_bulk_ports = None

    def ports_schema(self):
        process = self.process
        ports = process.get_schema()
        ports['request'] = {
            'bulk': {
                '_updater': 'set',
                '_emit': False,
            }
        }
        ports['process'] = {
            '_default': tuple(),
            '_updater': 'set',
            '_emit': False,
        }
        ports['global_time'] = {'_default': 0.0}
        ports['timestep'] = {'_default': process.parameters.get('timestep', 1.0)}
        ports['next_update_time'] = {
            '_default': process.parameters.get('timestep', 1.0),
            '_updater': 'set',
        }
        if self.cached_bulk_ports is None:
            self.cached_bulk_ports = list(ports['request'].keys())
        return ports

    def next_update(self, timestep, states):
        # Variable timestepping
        if states.get('next_update_time', 0.0) > states.get('global_time', 0.0):
            return {}

        process = self.process
        request = process.calculate_request(states.get('timestep', 1.0), states)
        process.request_set = True

        request['request'] = {}
        if self.cached_bulk_ports is None:
            self.ports_schema()  # Initialize cached_bulk_ports
        for bulk_port in self.cached_bulk_ports:
            bulk_request = request.pop(bulk_port, None)
            if bulk_request is not None:
                request['request'][bulk_port] = bulk_request

        listeners = request.pop('listeners', None)
        if listeners is not None:
            request['listeners'] = listeners

        request['process'] = (process,)
        return request


class Evolver:
    """Runs evolve_state() on a PartitionedProcess.

    Reads allocations from the Allocator and applies them to state,
    then calls evolve_state() to compute the biological update.
    """

    def __init__(self, parameters=None, config=None, **kwargs):
        params = parameters or config or {}
        self.parameters = params
        assert isinstance(params['process'], PartitionedProcess)
        self.process = params['process']
        self.name = f"{self.process.name}_evolver"

    def ports_schema(self):
        process = self.process
        ports = process.get_schema()
        ports['allocate'] = {
            'bulk': {
                '_updater': 'set',
                '_emit': False,
            }
        }
        ports['process'] = {
            '_default': tuple(),
            '_updater': 'set',
            '_emit': False,
        }
        ports['global_time'] = {'_default': 0.0}
        ports['timestep'] = {'_default': process.parameters.get('timestep', 1.0)}
        ports['next_update_time'] = {
            '_default': process.parameters.get('timestep', 1.0),
            '_updater': 'set',
        }
        return ports

    def next_update(self, timestep, states):
        # Variable timestepping
        if states.get('next_update_time', 0.0) > states.get('global_time', 0.0):
            return {}

        allocations = states.pop('allocate', {})
        states = deep_merge(states, allocations)
        process = self.process

        if not process.request_set:
            return {}

        update = process.evolve_state(states.get('timestep', 1.0), states)
        update['process'] = (process,)
        update['next_update_time'] = states.get('global_time', 0.0) + states.get('timestep', 1.0)
        return update
