"""
===============================
Unique Molecule Counts Listener
===============================

Counts unique molecules
"""

from process_bigraph import Step
from v2ecoli.library.schema import numpy_schema, listener_schema


class UniqueMoleculeCounts(Step):
    """UniqueMoleculeCounts"""

    name = "unique_molecule_counts"
    config_schema = {}
    topology = {
        "unique": ("unique",),
        "listeners": ("listeners",),
        "global_time": ("global_time",),
        "timestep": ("timestep",),
    }

    defaults = {
        "time_step": 1,
        "emit_unique": False,
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}
        self.unique_ids = self.parameters["unique_ids"]

    def ports_schema(self):
        ports = {
            "unique": {
                str(mol_id): numpy_schema(
                    mol_id + "s", emit=self.parameters.get("emit_unique", False)
                )
                for mol_id in self.unique_ids
                if mol_id not in ["DnaA_box", "active_ribosome"]
            },
            "listeners": {
                "unique_molecule_counts": listener_schema(
                    {str(mol_id): 0 for mol_id in self.unique_ids}
                )
            },
            "global_time": {"_default": 0.0},
            "timestep": {"_default": self.parameters.get("time_step", 1)},
        }
        ports["unique"].update(
            {
                "active_ribosome": numpy_schema(
                    "active_ribosome", emit=self.parameters.get("emit_unique", False)
                ),
                "DnaA_box": numpy_schema(
                    "DnaA_boxes", emit=self.parameters.get("emit_unique", False)
                ),
            }
        )
        return ports

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
