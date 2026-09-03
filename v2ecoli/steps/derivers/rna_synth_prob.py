"""
=====================
RnaSynthProb Listener
=====================
"""

import numpy as np
from v2ecoli.library.schema import numpy_schema, listener_schema, attrs
from v2ecoli.library.schema_types import PROMOTER_ARRAY, GENE_ARRAY
from v2ecoli.library.ecoli_step import EcoliStep as Step

# topology_registry removed — topology defined as class attribute


NAME = "rna_synth_prob_listener"
TOPOLOGY = {
    "rna_synth_prob": ("listeners", "rna_synth_prob"),
    "promoters": ("unique", "promoter"),
    "genes": ("unique", "gene"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


class RnaSynthProb(Step):
    """
    Listener for additional RNA synthesis data.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'time_step': 'float{1.0}',
        'emit_unique': 'boolean{false}',
        'rna_ids': 'list[string]',
        'gene_ids': 'list[string]',
        'tf_ids': 'list[string]',
        'cistron_ids': 'list[string]',
        'cistron_tu_mapping_matrix': 'csr_matrix',
    }


    def inputs(self):
        return {
            'rna_synth_prob': {
                'actual_rna_synth_prob': {'_type': f'overwrite[array[{self.n_TU},float]]', '_default': []},
                'target_rna_synth_prob': {'_type': f'overwrite[array[{self.n_TU},float]]', '_default': []},
                'n_bound_TF_per_TU': {'_type': f'overwrite[array[({self.n_TU}|{self.n_TF}),integer]]', '_default': []},
                'total_rna_init': {'_type': 'integer', '_default': 0},
            },
            'promoters': {'_type': PROMOTER_ARRAY, '_default': []},
            'genes': {'_type': GENE_ARRAY, '_default': []},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1.0},
        }

    def outputs(self):
        return {
            'rna_synth_prob': {
                'promoter_copy_number': {'_type': f'overwrite[array[{self.n_TU},integer]]', '_default': []},
                'gene_copy_number': {'_type': f'overwrite[array[{self.n_cistron},integer]]', '_default': []},
                'bound_TF_indexes': {'_type': 'overwrite[array[integer]]', '_default': []},
                'bound_TF_coordinates': {'_type': 'overwrite[array[integer]]', '_default': []},
                'bound_TF_domains': {'_type': 'overwrite[array[integer]]', '_default': []},
                # Probabilities — dimensionless floats in [0, 1]
                'expected_rna_init_per_cistron': {'_type': f'overwrite[array[{self.n_cistron},float]]', '_default': []},
                'actual_rna_synth_prob_per_cistron': {'_type': f'overwrite[array[{self.n_cistron},float]]', '_default': []},
                'target_rna_synth_prob_per_cistron': {'_type': f'overwrite[array[{self.n_cistron},float]]', '_default': []},
                'n_bound_TF_per_cistron': {'_type': f'overwrite[array[{self.n_cistron},integer]]', '_default': []},
            },
        }


    def initialize(self, config):
        self.rna_ids = self.parameters["rna_ids"]
        self.gene_ids = self.parameters["gene_ids"]
        self.tf_ids = self.parameters["tf_ids"]
        self.cistron_ids = self.parameters["cistron_ids"]
        self.n_TU = len(self.rna_ids)
        self.n_TF = len(self.tf_ids)
        self.n_cistron = len(self.cistron_ids)
        self.cistron_tu_mapping_matrix = self.parameters["cistron_tu_mapping_matrix"]

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def update(self, states, interval=None):
        # Guard: return empty on the first tick, before TranscriptInitiation has
        # written the probability arrays this step derives from.
        #
        # n_bound_TF_per_TU is deliberately NOT part of this guard. It is an
        # OPTIONAL input: TfBinding emits it only when its `emit_n_bound_TF_per_TU`
        # parameter is set, which defaults to False because the matrix is
        # n_TU x n_TF (~900 KB/tick). Requiring it here meant the guard was never
        # satisfied on a default build, so this step returned {} on every tick and
        # ALL TEN of its outputs sat at their `[]` defaults for the life of every
        # simulation -- promoter_copy_number, gene_copy_number, the bound_TF_*
        # arrays and the four per-cistron arrays included. One performance toggle
        # for a single heavy debug listener was silently disabling nine unrelated
        # observables, and downstream studies read the resulting empty arrays as
        # real absences. Only n_bound_TF_per_cistron genuinely needs the matrix, so
        # only it is gated on availability now.
        if (len(states["rna_synth_prob"]["actual_rna_synth_prob"]) != self.n_TU
                or len(states["rna_synth_prob"]["target_rna_synth_prob"]) != self.n_TU):
            return {}
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

        update = {
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
            }
        }
        # Only this field needs the optional n_bound_TF_per_TU matrix; emit it
        # when TfBinding was asked for it, and leave it absent otherwise rather
        # than blocking the other nine.
        n_bound = states["rna_synth_prob"]["n_bound_TF_per_TU"]
        if len(n_bound) == self.n_TU:
            update["rna_synth_prob"]["n_bound_TF_per_cistron"] = (
                self.cistron_tu_mapping_matrix.dot(n_bound).astype(np.int16).T
            )
        return update
