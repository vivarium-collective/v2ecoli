"""
=========================
Replication Data Listener
=========================
"""

import numpy as np
from v2ecoli.steps.base import V2Step as Step
from v2ecoli.library.schema import attrs
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from bigraph_schema.schema import Float


class ReplicationData(Step):
    """
    Listener for replication data.
    """

    name = "replication_data_listener"
    config_schema = {
        "time_step": {"_default": 1},
        "emit_unique": {"_default": False},
    }
    topology = {
        "listeners": ("listeners",),
        "oriCs": ("unique", "oriC"),
        "DnaA_boxes": ("unique", "DnaA_box"),
        "active_replisomes": ("unique", "active_replisome"),
        "global_time": ("global_time",),
        "timestep": ("timestep",),
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}

    def inputs(self):
        return {
            "listeners": ListenerStore(),
            "oriCs": UniqueNumpyUpdate(),
            "active_replisomes": UniqueNumpyUpdate(),
            "DnaA_boxes": UniqueNumpyUpdate(),
            "global_time": Float(_default=0.0),
            "timestep": Float(_default=1.0),
        }

    def outputs(self):
        return self.inputs()

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def next_update(self, timestep, states):
        fork_coordinates, fork_domains, fork_unique_index = attrs(
            states["active_replisomes"], ["coordinates", "domain_index", "unique_index"]
        )

        (DnaA_box_bound,) = attrs(states["DnaA_boxes"], ["DnaA_bound"])

        update = {
            "listeners": {
                "replication_data": {
                    "fork_coordinates": fork_coordinates,
                    "fork_domains": fork_domains,
                    "fork_unique_index": fork_unique_index,
                    "number_of_oric": states["oriCs"]["_entryState"].sum(),
                    "total_DnaA_boxes": len(DnaA_box_bound),
                    "free_DnaA_boxes": np.count_nonzero(np.logical_not(DnaA_box_bound)),
                }
            }
        }
        return update

    def update(self, state, interval=None):
        return self.next_update(state.get('timestep', 1.0), state)
