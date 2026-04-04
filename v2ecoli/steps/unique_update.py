"""
UniqueUpdate step for v2ecoli.

Placed after all steps of each execution layer to signal
UniqueNumpyUpdaters to flush their accumulated updates.
"""

from process_bigraph import Step
from bigraph_schema.schema import Node


class UniqueUpdate(Step):
    """Signals unique molecule updaters to apply accumulated changes."""

    name = "unique-update"
    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.unique_names = config.get('unique_names', [])

    def inputs(self):
        return {name: Node() for name in self.unique_names}

    def outputs(self):
        return {name: Node() for name in self.unique_names}

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        return {name: {"update": True} for name in self.unique_names}
