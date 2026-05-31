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


    def initialize(self, config):
        self.unique_ids = self.parameters["unique_ids"]

    def inputs(self):
        return {
            'unique': {'_type': 'map[node]', '_default': {}},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1.0},
        }

    def outputs(self):
        # One concrete overwrite[integer] field per unique-molecule type.
        # The bare 'map[integer]' that lived here from commit 967f638 had
        # broken apply semantics: bigraph Map.apply only updates keys
        # already in state (so first-tick adds of fresh keys were dropped),
        # and Integer.apply is additive (so repeated updates would
        # accumulate instead of overwrite). Mirrors the per-field
        # overwrite[...] pattern already used by mass_listener.
        return {
            'listeners': {
                'unique_molecule_counts': {
                    str(uid): {'_type': 'overwrite[integer]', '_default': 0}
                    for uid in self.unique_ids
                },
            },
        }


    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def update(self, states, interval=None):
        # self.unique_ids comes from the cache, which lists every
        # unique-molecule type in the ParCa fixture (incl. plasmid molecules
        # baked in by --mode full). State from a non-plasmid sim may omit
        # those keys — report 0 instead of raising KeyError.
        unique = states["unique"]
        return {
            "listeners": {
                "unique_molecule_counts": {
                    str(unique_id): (
                        unique[unique_id]["_entryState"].sum()
                        if unique_id in unique else 0
                    )
                    for unique_id in self.unique_ids
                }
            }
        }
