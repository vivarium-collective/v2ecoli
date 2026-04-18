"""
======================
Polypeptide Initiation
======================

This process models the assembly of 30S and 50S ribosomal subunits into
70S ribosomes on mRNA transcripts, analogous to TranscriptInitiation.

Mathematical Model
------------------
1. **Ribosome activation target**: The number of ribosomes to activate
   is determined by the media-dependent active ribosome fraction f:

       n_to_activate = round(f * n_total_ribosomes) - n_currently_active

   where n_total = n_active_70S + min(n_free_30S, n_free_50S).

2. **Translation probability per mRNA**: Each mRNA transcript k has a
   probability of receiving a new ribosome proportional to its
   translation efficiency eta_k:

       p_k = eta_k * n_copies_k / sum_j(eta_j * n_copies_j)

3. **Footprint constraint**: A ribosome occupies ~24 nt (configurable
   via active_ribosome_footprint_size). An mRNA is marked as
   overcrowded if adding another ribosome would violate the minimum
   spacing between ribosomes along the transcript:

       spacing = mRNA_length / (n_ribosomes_on_mRNA + 1)
       overcrowded = spacing < footprint_size

   Overcrowded mRNAs have their initiation probability set to zero.

4. **Stochastic initiation**: n_to_activate ribosomes are distributed
   across mRNAs via multinomial sampling weighted by p_k (after
   zeroing overcrowded mRNAs). Each initiated ribosome consumes one
   30S and one 50S subunit and begins translating at position 0.
"""

import numpy as np

# simulate_process removed
from v2ecoli.library.schema import (
    numpy_schema,
    attrs,
    counts,
    bulk_name_to_idx,
    listener_schema,
    zero_listener,
)

from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.unit_bridge import unum_to_pint
from wholecell.utils.fitting import normalize

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema_types import ACTIVE_RIBOSOME_ARRAY, RNA_ARRAY

# Register default topology for this process, associating it with process name
NAME = "ecoli-polypeptide-initiation"
TOPOLOGY = {
    "environment": ("environment",),
    "listeners": ("listeners",),
    "active_ribosome": ("unique", "active_ribosome"),
    "RNA": ("unique", "RNA"),
    "bulk": ("bulk",),
    "timestep": ("timestep",),
}


class PolypeptideInitiation(Step):
    """Polypeptide Initiation Step

    30S/50S subunits are only consumed by translation initiation; no other
    process competes for them. Runs as a plain Step.
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'active_ribosome_footprint_size': {'_type': 'quantity[nt]', '_default': 24.0},
        'active_ribosome_fraction': {'_type': 'map[float]', '_default': {}},
        'cistron_start_end_pos_in_tu': {'_type': 'map[node]', '_default': {}},
        'cistron_to_monomer_mapping': {'_type': 'array[integer]', '_default': {}},
        'cistron_tu_mapping_matrix': {'_type': 'csr_matrix', '_default': {}},
        'elongation_rates': {'_type': 'map[quantity[float,amino_acid/s]]', '_default': {}},
        'emit_unique': {'_type': 'boolean', '_default': False},
        'make_elongation_rates': {'_type': 'method', '_default': None},
        'monomer_ids': {'_type': 'list[string]', '_default': []},
        'monomer_index_to_cistron_index': {'_type': 'map[integer]', '_default': {}},
        'monomer_index_to_tu_indexes': {'_type': 'map[integer]', '_default': {}},
        'protein_lengths': {'_type': 'array[integer[aa]]', '_default': []},
        'ribosome30S': {'_type': 'string', '_default': 'ribosome30S'},
        'ribosome50S': {'_type': 'string', '_default': 'ribosome50S'},
        'rna_id_to_cistron_indexes': {'_type': 'method', '_default': {}},
        'seed': {'_type': 'integer', '_default': 0},
        'time_step': {'_type': 'integer[s]', '_default': 1},
        'translation_efficiencies': {'_type': 'array[float]', '_default': []},
        'tu_ids': {'_type': 'list[string]', '_default': []},
        'variable_elongation': {'_type': 'boolean', '_default': False},
    }

    def inputs(self):
        return {
            'environment': {'media_id': 'string'},
            'listeners': {
                'ribosome_data': {
                    # Read back the previous timestep's effective rate
                    'effective_elongation_rate': 'float',
                },
            },
            'active_ribosome': ACTIVE_RIBOSOME_ARRAY,
            'RNA': RNA_ARRAY,
            'bulk': 'bulk_array',
            'timestep': {'_type': 'integer[s]', '_default': 1},
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'active_ribosome': ACTIVE_RIBOSOME_ARRAY,
            'listeners': {
                'ribosome_data': {
                    'did_initialize': 'overwrite[integer]',
                    'ribosome_init_event_per_monomer': 'overwrite[array[integer]]',
                    'target_prob_translation_per_transcript': 'overwrite[array[float]]',
                    'actual_prob_translation_per_transcript': 'overwrite[array[float]]',
                    'mRNA_is_overcrowded': 'overwrite[array[boolean]]',
                    'max_p': 'overwrite[float]',
                    'max_p_per_protein': 'overwrite[array[float]]',
                    'is_n_ribosomes_to_activate_reduced': 'overwrite[boolean]',
                    # Written by empty_update path
                    'ribosomes_initialized': 'overwrite[integer]',
                    'prob_translation_per_transcript': 'overwrite[float]',
                },
            },
        }



    def initialize(self, config):

        # Load parameters
        self.protein_lengths = self.parameters["protein_lengths"]
        self.translation_efficiencies = self.parameters["translation_efficiencies"]
        self.active_ribosome_fraction = self.parameters["active_ribosome_fraction"]
        self.ribosome_elongation_rates_dict = self.parameters["elongation_rates"]
        self.variable_elongation = self.parameters["variable_elongation"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]

        self.rna_id_to_cistron_indexes = self.parameters["rna_id_to_cistron_indexes"]
        self.cistron_start_end_pos_in_tu = self.parameters[
            "cistron_start_end_pos_in_tu"
        ]
        self.tu_ids = self.parameters["tu_ids"]
        self.n_TUs = len(self.tu_ids)
        # Convert ribosome footprint size from nucleotides to amino acids
        self.active_ribosome_footprint_size = (
            unum_to_pint(self.parameters["active_ribosome_footprint_size"]) / 3
        )

        # Get mapping from cistrons to protein monomers and TUs
        self.cistron_to_monomer_mapping = self.parameters["cistron_to_monomer_mapping"]
        self.cistron_tu_mapping_matrix = self.parameters["cistron_tu_mapping_matrix"]
        self.monomer_index_to_cistron_index = self.parameters[
            "monomer_index_to_cistron_index"
        ]
        self.monomer_index_to_tu_indexes = self.parameters[
            "monomer_index_to_tu_indexes"
        ]

        self.ribosome30S = self.parameters["ribosome30S"]
        self.ribosome50S = self.parameters["ribosome50S"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        self.empty_update = {
            "listeners": {
                "ribosome_data": {
                    "ribosomes_initialized": 0,
                    "prob_translation_per_transcript": 0.0,
                }
            }
        }

        self.monomer_ids = self.parameters["monomer_ids"]

        # Helper indices for Numpy indexing
        self.ribosome30S_idx = None

    def inputs(self):
        return (
            {
                'environment':                 {
                    'media_id': {'_type': 'string', '_default': ''},
                },
                'listeners':                 {
                    'ribosome_data':                     {
                        'effective_elongation_rate': {'_type': 'float', '_default': 0.0},
                    },
                },
                'active_ribosome': {'_type': ACTIVE_RIBOSOME_ARRAY, '_default': []},
                'RNA': {'_type': RNA_ARRAY, '_default': []},
                'bulk': {'_type': 'bulk_array', '_default': []},
                'timestep': {'_type': 'integer', '_default': 1},
            }
        )

    def outputs(self):
        return (
            {
                'bulk': {'_type': 'bulk_array', '_default': []},
                'active_ribosome': {'_type': ACTIVE_RIBOSOME_ARRAY, '_default': []},
                'listeners':                 {
                    'ribosome_data':                     {
                        'did_initialize': {'_type': 'overwrite[integer]', '_default': 0},
                        'ribosome_init_event_per_monomer': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'target_prob_translation_per_transcript': {'_type': 'overwrite[array[float]]', '_default': []},
                        'actual_prob_translation_per_transcript': {'_type': 'overwrite[array[float]]', '_default': []},
                        'mRNA_is_overcrowded': {'_type': 'overwrite[array[boolean]]', '_default': []},
                        'max_p': {'_type': 'overwrite[float]', '_default': 0.0},
                        'max_p_per_protein': {'_type': 'overwrite[array[float]]', '_default': []},
                        'is_n_ribosomes_to_activate_reduced': 'overwrite[boolean]',
                        'ribosomes_initialized': 'overwrite[integer]',
                        'prob_translation_per_transcript': 'overwrite[float]',
                    },
                },
            }
        )

    def update(self, states, interval=None):
        if self.ribosome30S_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.ribosome30S_idx = bulk_name_to_idx(self.ribosome30S, bulk_ids)
            self.ribosome50S_idx = bulk_name_to_idx(self.ribosome50S, bulk_ids)

        current_media_id = states["environment"]["media_id"]
        self.fracActiveRibosome = self.active_ribosome_fraction[current_media_id]

        # Use last timestep's effective elongation rate; fall back to
        # media-dependent default on first timestep (when it reads as 0)
        self.ribosomeElongationRate = states["listeners"]["ribosome_data"][
            "effective_elongation_rate"
        ]
        if self.ribosomeElongationRate == 0:
            self.ribosomeElongationRate = unum_to_pint(
                self.ribosome_elongation_rates_dict[current_media_id]
            ).to(units.aa / units.s).magnitude
        self.elongation_rates = np.fmax(self.make_elongation_rates(
            self.random_state,
            self.ribosomeElongationRate,
            1,  # want elongation rate, not lengths adjusted for time step
            self.variable_elongation,
        ), 1)
        # Calculate number of ribosomes that could potentially be initialized
        # based on counts of free 30S and 50S subunits
        inactive_ribosome_count = np.min(
            [
                counts(states["bulk"], self.ribosome30S_idx),
                counts(states["bulk"], self.ribosome50S_idx),
            ]
        )

        # Calculate actual number of ribosomes that should be activated based
        # on probabilities
        (
            TU_index_RNAs,
            transcript_lengths,
            can_translate,
            is_full_transcript,
            unique_index_RNAs,
        ) = attrs(
            states["RNA"],
            [
                "TU_index",
                "transcript_length",
                "can_translate",
                "is_full_transcript",
                "unique_index",
            ],
        )
        TU_index_mRNAs = TU_index_RNAs[can_translate]
        length_mRNAs = transcript_lengths[can_translate]
        unique_index_mRNAs = unique_index_RNAs[can_translate]
        is_full_transcript_mRNAs = is_full_transcript[can_translate]
        is_incomplete_transcript_mRNAs = np.logical_not(is_full_transcript_mRNAs)

        # Calculate counts of each mRNA cistron from fully transcribed
        # transcription units
        TU_index_full_mRNAs = TU_index_mRNAs[is_full_transcript_mRNAs]
        TU_counts_full_mRNAs = np.bincount(TU_index_full_mRNAs, minlength=self.n_TUs)
        cistron_counts = self.cistron_tu_mapping_matrix.dot(TU_counts_full_mRNAs)

        # Calculate counts of each mRNA cistron from partially transcribed
        # transcription units
        TU_index_incomplete_mRNAs = TU_index_mRNAs[is_incomplete_transcript_mRNAs]
        length_incomplete_mRNAs = length_mRNAs[is_incomplete_transcript_mRNAs]

        for TU_index, length in zip(TU_index_incomplete_mRNAs, length_incomplete_mRNAs):
            cistron_indexes = self.rna_id_to_cistron_indexes(self.tu_ids[TU_index])
            cistron_start_positions = np.array(
                [
                    self.cistron_start_end_pos_in_tu[(cistron_index, TU_index)][0]
                    for cistron_index in cistron_indexes
                ]
            )

            cistron_counts[cistron_indexes] += length > cistron_start_positions

        # Calculate initiation probabilities for ribosomes based on mRNA counts
        # and associated mRNA translational efficiencies
        protein_init_prob = normalize(
            cistron_counts[self.cistron_to_monomer_mapping]
            * self.translation_efficiencies
        )
        target_protein_init_prob = protein_init_prob.copy()

        # Calculate actual number of ribosomes that should be activated based
        # on probabilities
        activation_prob = self.calculate_activation_prob(
            self.fracActiveRibosome,
            self.protein_lengths,
            self.elongation_rates,
            target_protein_init_prob,
            states["timestep"],
        )

        n_ribosomes_to_activate = np.int64(activation_prob * inactive_ribosome_count)

        if n_ribosomes_to_activate == 0:
            update = {
                "listeners": {
                    "ribosome_data": zero_listener(states["listeners"]["ribosome_data"])
                }
            }
            return update

        # Cap the initiation probabilities at the maximum level physically
        # allowed from the known ribosome footprint sizes based on the
        # number of mRNAs
        max_p = (
            self.ribosomeElongationRate
            / self.active_ribosome_footprint_size
            * (units.s)
            * states["timestep"]
            / n_ribosomes_to_activate
        ).magnitude
        max_p_per_protein = max_p * cistron_counts[self.cistron_to_monomer_mapping]
        is_overcrowded = protein_init_prob > max_p_per_protein

        # Initialize flag to record if the number of ribosomes activated at this
        # time step needed to be reduced to prevent overcrowding
        is_n_ribosomes_to_activate_reduced = False

        # If needed, resolve overcrowding
        while np.any(protein_init_prob > max_p_per_protein):
            if protein_init_prob[~is_overcrowded].sum() != 0:
                # Resolve overcrowding through rescaling (preferred)
                protein_init_prob[is_overcrowded] = max_p_per_protein[is_overcrowded]
                scale_the_rest_by = (
                    1.0 - protein_init_prob[is_overcrowded].sum()
                ) / protein_init_prob[~is_overcrowded].sum()
                protein_init_prob[~is_overcrowded] *= scale_the_rest_by
                is_overcrowded |= protein_init_prob > max_p_per_protein
            else:
                # If we cannot resolve the overcrowding through rescaling,
                # we need to activate fewer ribosomes. Set the number of
                # ribosomes to activate so that there will be no overcrowding.
                is_n_ribosomes_to_activate_reduced = True
                max_index = np.argmax(
                    protein_init_prob[is_overcrowded]
                    / max_p_per_protein[is_overcrowded]
                )
                max_init_prob = protein_init_prob[is_overcrowded][max_index]
                associated_cistron_counts = cistron_counts[
                    self.cistron_to_monomer_mapping
                ][is_overcrowded][max_index]
                n_ribosomes_to_activate = np.int64(
                    (
                        self.ribosomeElongationRate
                        / self.active_ribosome_footprint_size
                        * (units.s)
                        * states["timestep"]
                        / max_init_prob
                        * associated_cistron_counts
                    ).magnitude
                )

                # Update maximum probabilities based on new number of activated
                # ribosomes.
                max_p = (
                    self.ribosomeElongationRate
                    / self.active_ribosome_footprint_size
                    * (units.s)
                    * states["timestep"]
                    / n_ribosomes_to_activate
                ).magnitude
                max_p_per_protein = (
                    max_p * cistron_counts[self.cistron_to_monomer_mapping]
                )
                is_overcrowded = protein_init_prob > max_p_per_protein
                assert is_overcrowded.sum() == 0  # We expect no overcrowding

        # Compute actual transcription probabilities of each transcript
        actual_protein_init_prob = protein_init_prob.copy()

        # Sample multinomial distribution to determine which mRNAs have full
        # 70S ribosomes initialized on them
        n_new_proteins = self.random_state.multinomial(
            n_ribosomes_to_activate, protein_init_prob
        )

        # Build attributes for active ribosomes.
        # Each ribosome is assigned a protein index for the protein that
        # corresponds to the polypeptide it will polymerize. This is done in
        # blocks of protein ids for efficiency.
        protein_indexes = np.empty(n_ribosomes_to_activate, np.int64)
        mRNA_indexes = np.empty(n_ribosomes_to_activate, np.int64)
        positions_on_mRNA = np.empty(n_ribosomes_to_activate, np.int64)
        nonzero_count = n_new_proteins > 0
        start_index = 0

        for protein_index, protein_counts in zip(
            np.arange(n_new_proteins.size)[nonzero_count], n_new_proteins[nonzero_count]
        ):
            # Set protein index
            protein_indexes[start_index : start_index + protein_counts] = protein_index

            cistron_index = self.monomer_index_to_cistron_index[protein_index]

            attribute_indexes = []
            cistron_start_positions = []

            for TU_index in self.monomer_index_to_tu_indexes[protein_index]:
                attribute_indexes_this_TU = np.where(TU_index_mRNAs == TU_index)[0]
                cistron_start_position = self.cistron_start_end_pos_in_tu[
                    (cistron_index, TU_index)
                ][0]
                is_transcript_long_enough = (
                    length_mRNAs[attribute_indexes_this_TU] >= cistron_start_position
                )

                attribute_indexes.extend(
                    attribute_indexes_this_TU[is_transcript_long_enough]
                )
                cistron_start_positions.extend(
                    [cistron_start_position]
                    * len(attribute_indexes_this_TU[is_transcript_long_enough])
                )

            n_mRNAs = len(attribute_indexes)

            # Distribute ribosomes among these mRNAs
            n_ribosomes_per_RNA = self.random_state.multinomial(
                protein_counts, np.full(n_mRNAs, 1.0 / n_mRNAs)
            )

            # Get unique indexes of each mRNA
            mRNA_indexes[start_index : start_index + protein_counts] = np.repeat(
                unique_index_mRNAs[attribute_indexes], n_ribosomes_per_RNA
            )

            positions_on_mRNA[start_index : start_index + protein_counts] = np.repeat(
                cistron_start_positions, n_ribosomes_per_RNA
            )

            start_index += protein_counts

        # Create active 70S ribosomes and assign their attributes
        update = {
            "bulk": [
                (self.ribosome30S_idx, -n_new_proteins.sum()),
                (self.ribosome50S_idx, -n_new_proteins.sum()),
            ],
            "active_ribosome": {
                "add": {
                    "protein_index": protein_indexes,
                    "peptide_length": np.zeros(n_ribosomes_to_activate, dtype=np.int64),
                    "mRNA_index": mRNA_indexes,
                    "pos_on_mRNA": positions_on_mRNA,
                },
            },
            "listeners": {
                "ribosome_data": {
                    "did_initialize": n_new_proteins.sum(),
                    "ribosome_init_event_per_monomer": n_new_proteins,
                    "target_prob_translation_per_transcript": target_protein_init_prob,
                    "actual_prob_translation_per_transcript": actual_protein_init_prob,
                    "mRNA_is_overcrowded": is_overcrowded,
                    "max_p": max_p,
                    "max_p_per_protein": max_p_per_protein,
                    "is_n_ribosomes_to_activate_reduced": is_n_ribosomes_to_activate_reduced,
                }
            },
        }

        return update

    def calculate_activation_prob(
        self,
        fracActiveRibosome,
        proteinLengths,
        ribosomeElongationRates,
        proteinInitProb,
        timeStepSec,
    ):
        """
        Calculates the expected ribosome termination rate based on the ribosome
        elongation rate

        Args:
            allTranslationTimes: Vector of times required to translate each
                protein
            allTranslationTimestepCounts: Vector of numbers of timesteps
                required to translate each protein
            averageTranslationTimeStepCounts: Average number of timesteps
                required to translate a protein, weighted by initiation
                probabilities
            expectedTerminationRate: Average number of terminations in one
                timestep for one protein
        """
        allTranslationTimes = 1.0 / ribosomeElongationRates * proteinLengths
        allTranslationTimestepCounts = np.ceil(allTranslationTimes / timeStepSec)
        averageTranslationTimestepCounts = np.dot(
            allTranslationTimestepCounts, proteinInitProb
        )
        expectedTerminationRate = 1.0 / averageTranslationTimestepCounts

        # Modify given fraction of active ribosomes to take into account early
        # terminations in between timesteps
        # allFractionTimeInactive: Vector of probabilities an "active" ribosome
        #   will in effect be "inactive" because it has terminated during a
        #   timestep
        # averageFractionTimeInactive: Average probability of an "active"
        #   ribosome being in effect "inactive", weighted by initiation
        #   probabilities
        # effectiveFracActiveRnap: New higher "goal" for fraction of active
        #   ribosomes, considering that the "effective" fraction is lower than
        #   what the listener sees
        allFractionTimeInactive = (
            1 - allTranslationTimes / timeStepSec / allTranslationTimestepCounts
        )
        averageFractionTimeInactive = np.dot(allFractionTimeInactive, proteinInitProb)
        effectiveFracActiveRibosome = (
            fracActiveRibosome * 1 / (1 - averageFractionTimeInactive)
        )

        # Return activation probability that will balance out the expected
        # termination rate
        activationProb = (
            effectiveFracActiveRibosome
            * expectedTerminationRate
            / (1 - effectiveFracActiveRibosome)
        )

        # The upper bound for the activation probability is temporarily set to
        # 1.0 to prevent negative molecule counts. This will lower the fraction
        # of active ribosomes for timesteps longer than roughly 1.8s.
        if activationProb >= 1.0:
            activationProb = 1

        return activationProb


def test_polypeptide_initiation():
    def make_elongation_rates(self, random, base, time_step, variable_elongation=False):
        return base

    all_mRNA_ids = ["wRNA", "xRNA", "yRNA", "zRNA"]
    protein_lengths = np.array([25, 9, 12, 29])
    test_config = {
        "protein_lengths": protein_lengths,
        "translation_efficiencies": normalize(np.array([0.1, 0.2, 0.3, 0.4])),
        "active_ribosome_fraction": {"minimal": 0.05},
        "variable_elongation": False,
        "make_elongation_rates": make_elongation_rates,
        "rna_id_to_cistron_indexes": lambda rna_id: all_mRNA_ids.index(rna_id),
        "cistron_start_end_pos_in_tu": {
            (idx, idx): (0, protein_length * 3 + 3)
            for idx, protein_length in enumerate(protein_lengths)
        },
        "tu_ids": all_mRNA_ids,
        "cistron_to_monomer_mapping": np.arange(4),
        "cistron_tu_mapping_matrix": np.identity(4),
        "monomer_index_to_cistron_index": {i: i for i in range(4)},
        "monomer_index_to_tu_indexes": {i: (i,) for i in range(4)},
        "protein_index_to_TU_index": np.arange(4),
        "all_TU_ids": ["wRNA", "xRNA", "yRNA", "zRNA"],
        "all_mRNA_ids": ["wRNA", "xRNA", "yRNA", "zRNA"],
        "ribosome30S": "30S",
        "ribosome50S": "50S",
        "seed": 0,
    }

    polypeptide_initiation = PolypeptideInitiation(test_config)

    state = {
        "environment": {"media_id": "minimal"},
        "listeners": {"ribosome_data": {"effective_elongation_rate": 25}},
        "bulk": np.array(
            [
                ("30S", 2000),
                ("50S", 3000),
            ],
            dtype=[("id", "U40"), ("count", int)],
        ),
        "RNA": np.array(
            [
                (1, 0, True, 0, 103, True),
                (1, 0, True, 1, 103, True),
                (1, 1, True, 2, 39, True),
                (1, 2, True, 3, 51, True),
                (1, 2, True, 4, 51, True),
                (1, 3, True, 4, 119, True),
            ],
            dtype=[
                ("_entryState", np.bool_),
                ("TU_index", int),
                ("can_translate", np.bool_),
                ("unique_index", int),
                ("transcript_length", int),
                ("is_full_transcript", np.bool_),
            ],
        ),
    }

    settings = {"total_time": 10, "initial_state": state}

    data = simulate_process(polypeptide_initiation, settings)

    print(data)


if __name__ == "__main__":
    test_polypeptide_initiation()
