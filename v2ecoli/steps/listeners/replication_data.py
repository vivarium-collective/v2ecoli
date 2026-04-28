"""
=========================
Replication Data Listener
=========================

Emits replication-related signals each tick: fork coordinates / domains,
oriC count, DnaA-box totals, and the DnaA nucleotide-pool partitioning
(apo-DnaA, DnaA-ATP, DnaA-ADP). The pool-partition signals come from the
bulk array — both nucleotide-bound forms are equilibrium-coupled to
apo-DnaA + ATP/ADP via reactions wired into the equilibrium process
(``MONOMER0-160_RXN`` and ``MONOMER0-4565_RXN`` in
``flat/equilibrium_reactions.tsv``).
"""

import numpy as np

from v2ecoli.data.replication_initiation import (
    DNAA_ADP_BULK_ID, DNAA_APO_BULK_ID, DNAA_ATP_BULK_ID,
)
from v2ecoli.library.schema import (
    attrs, bulk_name_to_idx, counts, listener_schema, numpy_schema,
)
from v2ecoli.library.schema_types import (
    ACTIVE_REPLISOME_ARRAY, DNAA_BOX_ARRAY, ORIC_ARRAY,
)
from v2ecoli.library.ecoli_step import EcoliStep as Step

# topology_registry removed — topology defined as class attribute


NAME = "replication_data_listener"
TOPOLOGY = {
    "listeners": ("listeners",),
    "oriCs": ("unique", "oriC"),
    "DnaA_boxes": ("unique", "DnaA_box"),
    "active_replisomes": ("unique", "active_replisome"),
    "bulk": ("bulk",),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


class ReplicationData(Step):
    """
    Listener for replication data.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'time_step': 'float{1.0}',
        'emit_unique': 'boolean{false}',
    }


    def inputs(self):
        return {
            'oriCs': {'_type': ORIC_ARRAY, '_default': []},
            'DnaA_boxes': {'_type': DNAA_BOX_ARRAY, '_default': []},
            'active_replisomes': {'_type': ACTIVE_REPLISOME_ARRAY, '_default': []},
            'bulk': {'_type': 'bulk_array', '_default': []},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1},
        }

    def outputs(self):
        return {
            'listeners': {
                'replication_data': {
                    'fork_coordinates': {'_type': 'array[integer]', '_default': []},
                    'fork_domains': {'_type': 'array[integer]', '_default': []},
                    'fork_unique_index': {'_type': 'array[integer]', '_default': []},
                    'number_of_oric': {'_type': 'overwrite[integer]', '_default': []},
                    'free_DnaA_boxes': {'_type': 'overwrite[integer]', '_default': []},
                    'total_DnaA_boxes': {'_type': 'overwrite[integer]', '_default': []},
                    'dnaA_apo_count': {'_type': 'overwrite[integer]', '_default': []},
                    'dnaA_atp_count': {'_type': 'overwrite[integer]', '_default': []},
                    'dnaA_adp_count': {'_type': 'overwrite[integer]', '_default': []},
                },
            },
        }


    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def update(self, states, interval=None):
        # Guard: return empty on first tick if data not yet populated
        fork_coordinates, fork_domains, fork_unique_index = attrs(
            states["active_replisomes"], ["coordinates", "domain_index", "unique_index"]
        )

        (DnaA_box_bound,) = attrs(states["DnaA_boxes"], ["DnaA_bound"])

        # Cache bulk indices on first call. The bulk array's id ordering is
        # stable across a run, so this lookup is one-shot.
        bulk = states["bulk"]
        if not hasattr(self, "_bulk_idx"):
            self._bulk_idx = bulk_name_to_idx(
                [DNAA_APO_BULK_ID, DNAA_ATP_BULK_ID, DNAA_ADP_BULK_ID],
                bulk["id"],
            )
        apo_count, atp_count, adp_count = counts(bulk, self._bulk_idx)

        update = {
            "listeners": {
                "replication_data": {
                    "fork_coordinates": fork_coordinates,
                    "fork_domains": fork_domains,
                    "fork_unique_index": fork_unique_index,
                    "number_of_oric": states["oriCs"]["_entryState"].sum(),
                    "total_DnaA_boxes": len(DnaA_box_bound),
                    "free_DnaA_boxes": np.count_nonzero(np.logical_not(DnaA_box_bound)),
                    "dnaA_apo_count": int(apo_count),
                    "dnaA_atp_count": int(atp_count),
                    "dnaA_adp_count": int(adp_count),
                }
            }
        }
        return update
