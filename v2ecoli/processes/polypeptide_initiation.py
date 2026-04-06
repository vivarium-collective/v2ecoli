"""
======================
Polypeptide Initiation
======================

This process models the complementation of 30S and 50S ribosomal subunits
into 70S ribosomes on mRNA transcripts. This process is in many ways
analogous to the TranscriptInitiation process - the number of initiation
events per transcript is determined in a probabilistic manner and dependent
on the number of free ribosomal subunits, each mRNA transcript's translation
efficiency, and the counts of each type of transcript.
"""

import numpy as np

from process_bigraph import Step
from bigraph_schema.schema import Float, Overwrite

from v2ecoli.library.schema import (
    numpy_schema,
    attrs,
    counts,
    bulk_name_to_idx,
    listener_schema,
    zero_listener,
)

from v2ecoli.library.units import units
from v2ecoli.library.fitting import normalize

from v2ecoli.steps.partition import _protect_state, deep_merge, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore


class PolypeptideInitiationLogic:
    """Polypeptide Initiation logic — pure computation, no Step inheritance."""

    name = "ecoli-polypeptide-initiation"
    topology = {
    "environment": ("environment",),
    "listeners": ("listeners",),
    "active_ribosome": ("unique", "active_ribosome"),
    "RNA": ("unique", "RNA"),
    "bulk": ("bulk",),
    "timestep": ("timestep",),
}
    defaults = {
        "protein_lengths": [],
        "translation_efficiencies": [],
        "active_ribosome_fraction": {},
        "elongation_rates": {},
        "variable_elongation": False,
        "make_elongation_rates": lambda x: [],
        "rna_id_to_cistron_indexes": {},
        "cistron_start_end_pos_in_tu": {},
        "tu_ids": [],
        "active_ribosome_footprint_size": 24 * units.nt,
        "cistron_to_monomer_mapping": {},
        "cistron_tu_mapping_matrix": {},
        "monomer_index_to_cistron_index": {},
        "monomer_index_to_tu_indexes": {},
        "ribosome30S": "ribosome30S",
        "ribosome50S": "ribosome50S",
        "seed": 0,
        "monomer_ids": [],
        "emit_unique": False,
        "time_step": 1,
    }

    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

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
            self.parameters["active_ribosome_footprint_size"] / 3
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

    def ports_schema(self):
        return {
            "environment": {"media_id": {"_default": "", "_updater": "set"}},
            "listeners": {
                "ribosome_data": listener_schema(
                    {
                        "did_initialize": 0,
                        "target_prob_translation_per_transcript": (
                            [0.0] * len(self.monomer_ids),
                            self.monomer_ids,
                        ),
                        "actual_prob_translation_per_transcript": (
                            [0.0] * len(self.monomer_ids),
                            self.monomer_ids,
                        ),
                        "mRNA_is_overcrowded": (
                            [False] * len(self.monomer_ids),
                            self.monomer_ids,
                        ),
                        "ribosome_init_event_per_monomer": (
                            [0] * len(self.monomer_ids),
                            self.monomer_ids,
                        ),
                        "effective_elongation_rate": 0.0,
                        "max_p": 0.0,
                        "max_p_per_protein": (
                            np.zeros(len(self.monomer_ids), np.float64),
                            self.monomer_ids,
                        ),
                    }
                ),
            },
            "active_ribosome": numpy_schema(
                "active_ribosome", emit=self.parameters["emit_unique"]
            ),
            "RNA": numpy_schema("RNAs", emit=self.parameters["emit_unique"]),
            "bulk": numpy_schema("bulk"),
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        if self.ribosome30S_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.ribosome30S_idx = bulk_name_to_idx(self.ribosome30S, bulk_ids)
            self.ribosome50S_idx = bulk_name_to_idx(self.ribosome50S, bulk_ids)

        current_media_id = states["environment"]["media_id"]

        # requests = {'subunits': states['subunits']}
        requests = {
            "bulk": [
                (self.ribosome30S_idx, counts(states["bulk"], self.ribosome30S_idx)),
                (self.ribosome50S_idx, counts(states["bulk"], self.ribosome50S_idx)),
            ]
        }

        self.fracActiveRibosome = self.active_ribosome_fraction[current_media_id]

        # Read ribosome elongation rate from last timestep
        self.ribosomeElongationRate = states["listeners"]["ribosome_data"][
            "effective_elongation_rate"
        ]
        # If the ribosome elongation rate is zero (which is always the case for
        # the first timestep), set ribosome elongation rate to the one in
        # dictionary
        if self.ribosomeElongationRate == 0:
            self.ribosomeElongationRate = self.ribosome_elongation_rates_dict[
                current_media_id
            ].asNumber(units.aa / units.s)
        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.ribosomeElongationRate,
            1,  # want elongation rate, not lengths adjusted for time step
            self.variable_elongation,
        )

        # Ensure rates are never zero
        self.elongation_rates = np.fmax(self.elongation_rates, 1)
        return requests

    def evolve_state(self, timestep, states):
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
        ).asNumber()
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
                    ).asNumber()
                )

                # Update maximum probabilities based on new number of activated
                # ribosomes.
                max_p = (
                    self.ribosomeElongationRate
                    / self.active_ribosome_footprint_size
                    * (units.s)
                    * states["timestep"]
                    / n_ribosomes_to_activate
                ).asNumber()
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

    def next_update(self, timestep, states):
        requests = self.calculate_request(timestep, states)
        bulk_requests = requests.pop("bulk", [])
        if bulk_requests:
            bulk_copy = states["bulk"].copy()
            for bulk_idx, request in bulk_requests:
                bulk_copy[bulk_idx] = request
            states["bulk"] = bulk_copy
        states = deep_merge(states, requests)
        update = self.evolve_state(timestep, states)
        if "listeners" in requests:
            update["listeners"] = deep_merge(
                update.get("listeners", {}), requests["listeners"])
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


class PolypeptideInitiationRequester(_SafeInvokeMixin, Step):
    """Reads stores to compute initiation request. Writes to request store."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'RNA': UniqueNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
            'listeners': ListenerStore(),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state.get('timestep', 1.0)
        request = self.process.calculate_request(timestep, state)
        self.process.request_set = True

        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request

        listeners = request.pop('listeners', None)
        if listeners is not None:
            result['listeners'] = listeners

        return result


class PolypeptideInitiationEvolver(_SafeInvokeMixin, Step):
    """Reads allocation, writes bulk/unique/listener updates."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'RNA': UniqueNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'RNA': UniqueNumpyUpdate(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)

        allocations = state.pop('allocate', {})
        bulk_alloc = allocations.get('bulk')
        if bulk_alloc is not None and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk
        state = deep_merge(state, allocations)

        if not self.process.request_set:
            return {}

        timestep = state.get('timestep', 1.0)
        update = self.process.evolve_state(timestep, state)
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update
