"""
===============================
Unique Molecule Counts Listener
===============================

Counts unique molecules
"""

from v2ecoli.steps.base import V2Step as Step
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from bigraph_schema.schema import Float


class UniqueMoleculeCounts(Step):
    """UniqueMoleculeCounts"""

    name = "unique_molecule_counts"
    config_schema = {
        "time_step": {"_default": 1},
        "emit_unique": {"_default": False},
    }
    topology = {
        "unique": ("unique",),
        "listeners": ("listeners",),
        "global_time": ("global_time",),
        "timestep": ("timestep",),
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}
        self.unique_ids = self.parameters["unique_ids"]

    def inputs(self):
        return {
            "unique": ListenerStore(),
            "listeners": ListenerStore(),
            "global_time": Float(_default=0.0),
            "timestep": Float(_default=1.0),
        }

    def outputs(self):
        return self.inputs()

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def next_update(self, timestep, states):
        return {
            "listeners": {
                "unique_molecule_counts": {
                    str(unique_id): states["unique"][unique_id]["_entryState"].sum()
                    for unique_id in self.unique_ids
                }
            }
        }

    def update(self, state, interval=None):
        return self.next_update(state.get('timestep', 1.0), state)
