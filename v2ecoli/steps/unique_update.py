"""
UniqueUpdate step for v2ecoli.

Placed after all steps of each execution layer to signal
UniqueNumpyUpdaters to flush their accumulated updates.
Proper process-bigraph Step.
"""

from process_bigraph import Step

from v2ecoli.types.unique_numpy import UniqueNumpyUpdate


class UniqueUpdate(Step):
    """Signals unique molecule updaters to apply accumulated changes."""

    name = "unique-update"
    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        params = config or {}
        self.parameters = params
        self.unique_topo = params.get('unique_topo', {})

    def inputs(self):
        return {name: UniqueNumpyUpdate() for name in self.unique_topo}

    def outputs(self):
        return {name: UniqueNumpyUpdate() for name in self.unique_topo}

    def initial_state(self, config=None):
        return {}

    def update(self, state, interval=None):
        return {name: {"update": True} for name in self.unique_topo}
