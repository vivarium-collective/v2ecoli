"""
UniqueUpdate step for v2ecoli.

Placed after all steps of each execution layer to signal
UniqueNumpyUpdaters to flush their accumulated updates.
"""

from v2ecoli.library.schema import numpy_schema


class UniqueUpdate:
    """Signals unique molecule updaters to apply accumulated changes."""

    name = "unique-update"

    def __init__(self, parameters=None, config=None, **kwargs):
        params = parameters or config or {}
        self.parameters = params
        self.unique_topo = params.get('unique_topo', {})

    def ports_schema(self):
        return {
            unique_mol: numpy_schema(unique_mol, emit=self.parameters.get('emit_unique', False))
            for unique_mol in self.unique_topo
        }

    def next_update(self, timestep, states):
        return {unique_mol: {"update": True} for unique_mol in self.unique_topo.keys()}
