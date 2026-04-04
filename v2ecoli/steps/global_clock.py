"""
Global clock process for v2ecoli.

Tracks global_time for steps that use variable timestepping.
"""

from process_bigraph import Process
from bigraph_schema.schema import Float, Node


class GlobalClock(Process):
    """Advances global_time by the minimum of next_update_times."""

    name = "global_clock"
    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)

    def inputs(self):
        return {
            'global_time': Float(_default=0.0),
            'next_update_time': Node(),
        }

    def outputs(self):
        return {
            'global_time': Float(_default=0.0),
        }

    def initial_state(self, config=None):
        return {}

    def calculate_timestep(self, interval, state):
        next_times = state.get('next_update_time', {})
        if next_times:
            return min(
                t - state['global_time']
                for t in next_times.values())
        return interval

    def update(self, state, interval):
        return {'global_time': interval}
