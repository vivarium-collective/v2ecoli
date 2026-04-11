"""
===============================
Unique Molecule Counts Listener
===============================

Counts unique molecules
"""

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema import numpy_schema, listener_schema
# topology_registry removed — topology defined as class attribute

# Register default topology for this process, associating it with process name
NAME = "unique_molecule_counts"
TOPOLOGY = {
    "unique": ("unique",),
    "listeners": ("listeners",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


class UniqueMoleculeCounts(Step):
    """UniqueMoleculeCounts"""

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'time_step': 'float{1.0}',
        'emit_unique': 'boolean{false}',
        'unique_ids': 'list[string]',
    }


    def inputs(self):
        return {
            'unique': {'_type': 'map[node]', '_default': {}},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1.0},
        }

    def outputs(self):
        return {
            'listeners': {
                'unique_molecule_counts': 'map[integer]',
            },
        }


    def initialize(self, config):
        self.unique_ids = self.parameters["unique_ids"]


    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def update(self, states, interval=None):
        # Guard: return empty on first tick if data not yet populated
        return {
            "listeners": {
                "unique_molecule_counts": {
                    str(unique_id): states["unique"][unique_id]["_entryState"].sum()
                    for unique_id in self.unique_ids
                }
            }
        }
