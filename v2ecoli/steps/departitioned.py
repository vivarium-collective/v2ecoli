"""
==========================
Departitioned Step Wrapper
==========================

Wraps a PartitionedProcess so it runs as a single Step — no
Requester/Allocator/Evolver machinery. The request is computed and
immediately applied as the allocation.

Uses the existing PartitionedProcess._do_update() method which already
implements the combined request+evolve logic.
"""

import warnings

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.steps.partition import PartitionedProcess


class DepartitionedStep(Step):
    """Single-step wrapper that delegates to PartitionedProcess._do_update().

    Given a PartitionedProcess, this Step:
    1. Checks the time-gating condition (next_update_time vs global_time)
    2. Delegates to process._do_update() which calls calculate_request()
       then evolve_state() in sequence
    3. Advances next_update_time

    This eliminates the allocator overhead at the cost of "unfair" allocation:
    each process gets exactly what it requests.
    """

    config_schema = {
        'process': 'node',
    }

    def inputs(self):
        process = self.parameters.get('process')
        ports = process.inputs()
        ports['global_time'] = 'float'
        ports['timestep'] = 'integer'
        ports['next_update_time'] = 'float'
        return ports

    def outputs(self):
        process = self.parameters.get('process')
        ports = process.outputs()
        ports['next_update_time'] = 'overwrite[float]'
        for k in ('global_time', 'timestep'):
            ports.pop(k, None)
        return ports

    def initialize(self, config):
        assert isinstance(self.parameters['process'], PartitionedProcess)
        if getattr(self.parameters['process'], 'parallel', False):
            raise RuntimeError(
                'PartitionedProcess objects cannot be parallelized.')
        self.parameters['name'] = f"{self.parameters['process'].name}_departitioned"

    def update_condition(self, timestep, states):
        if states['next_update_time'] <= states['global_time']:
            if states['next_update_time'] < states['global_time']:
                warnings.warn(
                    f"{self.name} updated at t="
                    f"{states['global_time']} instead of t="
                    f"{states['next_update_time']}.")
            return True
        return False

    def update(self, states, interval=None):
        process = self.parameters.get('process')
        timestep = states.get('timestep', 1)

        # _do_update handles: calculate_request -> apply bulk -> evolve_state
        update = process._do_update(timestep, states)

        # Advance next_update_time
        update['next_update_time'] = (
            states.get('global_time', 0.0) + timestep)

        return update
