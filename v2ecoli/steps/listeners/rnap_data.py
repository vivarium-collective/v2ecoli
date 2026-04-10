"""
====================
RNAP Data Listener
====================
"""

import numpy as np
import warnings
from bigraph_schema.schema import Overwrite, Float
from v2ecoli.steps.base import V2Step as Step
from v2ecoli.library.schema import attrs
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate


def get_mapping_arrays(x, y):
    """
    Returns the array of indexes of each element of array x in array y, and
    vice versa. Assumes that the elements of x and y are unique, and
    set(x) == set(y).
    """

    def argsort_unique(idx):
        """
        Quicker argsort for arrays that are permutations of np.arange(n).
        """
        n = idx.size
        argsort_idx = np.empty(n, dtype=np.int64)
        argsort_idx[idx] = np.arange(n)
        return argsort_idx

    x_argsort = np.argsort(x)
    y_argsort = np.argsort(y)

    x_to_y = x_argsort[argsort_unique(y_argsort)]
    y_to_x = y_argsort[argsort_unique(x_argsort)]

    return x_to_y, y_to_x


class RnapData(Step):
    """
    Listener for RNAP data.
    """

    name = "rnap_data_listener"
    config_schema = {
        "stable_RNA_indexes": {"_default": []},
        "cistron_ids": {"_default": []},
        "cistron_tu_mapping_matrix": {"_default": []},
        "time_step": {"_default": 1},
        "emit_unique": {"_default": False},
    }
    topology = {
        "listeners": ("listeners",),
        "active_RNAPs": ("unique", "active_RNAP"),
        "RNAs": ("unique", "RNA"),
        "active_ribosomes": ("unique", "active_ribosome"),
        "global_time": ("global_time",),
        "timestep": ("timestep",),
        "next_update_time": ("next_update_time", "rnap_data_listener"),
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}
        self.stable_RNA_indexes = self.parameters.get("stable_RNA_indexes", [])
        self.cistron_ids = self.parameters.get("cistron_ids", [])
        self.cistron_tu_mapping_matrix = self.parameters.get("cistron_tu_mapping_matrix", [])

    def inputs(self):
        return {
            "listeners": ListenerStore(),
            "active_RNAPs": UniqueNumpyUpdate(),
            "RNAs": UniqueNumpyUpdate(),
            "active_ribosomes": UniqueNumpyUpdate(),
            "global_time": Float(_default=0.0),
            "timestep": Float(_default=1.0),
            "next_update_time": Overwrite(),
        }

    def outputs(self):
        return self.inputs()

    def update_condition(self, timestep, states):
        """
        See :py:meth:`~ecoli.processes.partition.Requester.update_condition`.
        """
        if states["next_update_time"] <= states["global_time"]:
            if states["next_update_time"] < states["global_time"]:
                warnings.warn(
                    f"{self.name} updated at t="
                    f"{states['global_time']} instead of t="
                    f"{states['next_update_time']}. Decrease the "
                    "timestep for the global clock process for more "
                    "accurate timekeeping."
                )
            return True
        return False

    def next_update(self, timestep, states):
        # Read coordinates of all active RNAPs
        coordinates, domain_indexes, RNAP_unique_indexes = attrs(
            states["active_RNAPs"], ["coordinates", "domain_index", "unique_index"]
        )

        (RNA_RNAP_index, is_full_transcript, RNA_unique_indexes, TU_indexes) = attrs(
            states["RNAs"],
            ["RNAP_index", "is_full_transcript", "unique_index", "TU_index"],
        )

        is_partial_transcript = np.logical_not(is_full_transcript)
        is_stable_RNA = np.isin(TU_indexes, self.stable_RNA_indexes)
        partial_RNA_RNAP_indexes = RNA_RNAP_index[is_partial_transcript]
        partial_RNA_unique_indexes = RNA_unique_indexes[is_partial_transcript]

        (ribosome_RNA_index,) = attrs(states["active_ribosomes"], ["mRNA_index"])

        RNA_index_counts = dict(zip(*np.unique(ribosome_RNA_index, return_counts=True)))

        partial_RNA_to_RNAP_mapping, _ = get_mapping_arrays(
            partial_RNA_RNAP_indexes, RNAP_unique_indexes
        )

        update = {
            "listeners": {
                "rnap_data": {
                    "active_rnap_coordinates": coordinates,
                    "active_rnap_domain_indexes": domain_indexes,
                    "active_rnap_unique_indexes": RNAP_unique_indexes,
                    "active_rnap_on_stable_RNA_indexes": RNA_RNAP_index[
                        np.logical_and(is_stable_RNA, is_partial_transcript)
                    ],
                    "active_rnap_n_bound_ribosomes": np.array(
                        [
                            RNA_index_counts.get(partial_RNA_unique_indexes[i], 0)
                            for i in partial_RNA_to_RNAP_mapping
                        ]
                    ),
                    # Calculate hypothetical RNA initiation events per cistron
                    "rna_init_event_per_cistron": self.cistron_tu_mapping_matrix.dot(
                        states["listeners"]["rnap_data"]["rna_init_event"]
                    ),
                }
            },
            "next_update_time": states["global_time"] + states["timestep"],
        }
        return update

    def update(self, state, interval=None):
        return self.next_update(state.get('timestep', 1.0), state)
