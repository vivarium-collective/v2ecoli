"""
=========================
Replication Data Listener
=========================
"""

import numpy as np
from process_bigraph import Step
from v2ecoli.library.schema import numpy_schema, listener_schema, attrs


class ReplicationData(Step):
    """
    Listener for replication data.
    """

    name = "replication_data_listener"
    config_schema = {}

    defaults = {"time_step": 1, "emit_unique": False}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}

    def ports_schema(self):
        return {
            "listeners": {
                "replication_data": listener_schema(
                    {
                        "fork_coordinates": [],
                        "fork_domains": [],
                        "fork_unique_index": [],
                        "number_of_oric": [],
                        "free_DnaA_boxes": [],
                        "total_DnaA_boxes": [],
                    }
                )
            },
            "oriCs": numpy_schema("oriCs", emit=self.parameters.get("emit_unique", False)),
            "active_replisomes": numpy_schema(
                "active_replisomes", emit=self.parameters.get("emit_unique", False)
            ),
            "DnaA_boxes": numpy_schema(
                "DnaA_boxes", emit=self.parameters.get("emit_unique", False)
            ),
            "global_time": {"_default": 0.0},
            "timestep": {"_default": self.parameters.get("time_step", 1)},
        }

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
