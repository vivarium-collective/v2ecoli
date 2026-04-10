"""
=====================
RnaSynthProb Listener
=====================
"""

import numpy as np
from v2ecoli.steps.base import V2Step as Step
from v2ecoli.library.schema import attrs
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from bigraph_schema.schema import Float


class RnaSynthProb(Step):
    """
    Listener for additional RNA synthesis data.
    """

    name = "rna_synth_prob_listener"
    config_schema = {
        "time_step": {"_default": 1},
        "emit_unique": {"_default": False},
    }
    topology = {
        "rna_synth_prob": ("listeners", "rna_synth_prob"),
        "promoters": ("unique", "promoter"),
        "genes": ("unique", "gene"),
        "global_time": ("global_time",),
        "timestep": ("timestep",),
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self.parameters = config or {}
        self.rna_ids = self.parameters["rna_ids"]
        self.gene_ids = self.parameters["gene_ids"]
        self.tf_ids = self.parameters["tf_ids"]
        self.cistron_ids = self.parameters["cistron_ids"]
        self.n_TU = len(self.rna_ids)
        self.n_TF = len(self.tf_ids)
        self.n_cistron = len(self.cistron_ids)
        self.cistron_tu_mapping_matrix = self.parameters["cistron_tu_mapping_matrix"]

    def inputs(self):
        return {
            "rna_synth_prob": ListenerStore(),
            "promoters": UniqueNumpyUpdate(),
            "genes": UniqueNumpyUpdate(),
            "global_time": Float(_default=0.0),
            "timestep": Float(_default=1.0),
        }

    def outputs(self):
        return self.inputs()

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def next_update(self, timestep, states):
        TU_indexes, all_coordinates, all_domains, bound_TFs = attrs(
            states["promoters"], ["TU_index", "coordinates", "domain_index", "bound_TF"]
        )
        bound_promoter_indexes, TF_indexes = np.where(bound_TFs)
        (cistron_indexes,) = attrs(states["genes"], ["cistron_index"])

        actual_rna_synth_prob_per_cistron = self.cistron_tu_mapping_matrix.dot(
            states["rna_synth_prob"]["actual_rna_synth_prob"]
        )
        # The expected value of rna initiations per cistron. Realized values
        # during simulation will be different, because they will be integers
        # drawn from a multinomial distribution
        expected_rna_init_per_cistron = (
            actual_rna_synth_prob_per_cistron
            * states["rna_synth_prob"]["total_rna_init"]
        )

        if actual_rna_synth_prob_per_cistron.sum() != 0:
            actual_rna_synth_prob_per_cistron = (
                actual_rna_synth_prob_per_cistron
                / actual_rna_synth_prob_per_cistron.sum()
            )
        target_rna_synth_prob_per_cistron = self.cistron_tu_mapping_matrix.dot(
            states["rna_synth_prob"]["target_rna_synth_prob"]
        )
        if target_rna_synth_prob_per_cistron.sum() != 0:
            target_rna_synth_prob_per_cistron = (
                target_rna_synth_prob_per_cistron
                / target_rna_synth_prob_per_cistron.sum()
            )

        return {
            "rna_synth_prob": {
                "promoter_copy_number": np.bincount(TU_indexes, minlength=self.n_TU),
                "gene_copy_number": np.bincount(
                    cistron_indexes, minlength=self.n_cistron
                ),
                "bound_TF_indexes": TF_indexes,
                "bound_TF_coordinates": all_coordinates[bound_promoter_indexes],
                "bound_TF_domains": all_domains[bound_promoter_indexes],
                "expected_rna_init_per_cistron": expected_rna_init_per_cistron,
                "actual_rna_synth_prob_per_cistron": actual_rna_synth_prob_per_cistron,
                "target_rna_synth_prob_per_cistron": target_rna_synth_prob_per_cistron,
                "n_bound_TF_per_cistron": self.cistron_tu_mapping_matrix.dot(
                    states["rna_synth_prob"]["n_bound_TF_per_TU"]
                )
                .astype(np.int16)
                .T,
            }
        }

    def update(self, state, interval=None):
        return self.next_update(state.get('timestep', 1.0), state)
