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
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Float, Overwrite, Node

from v2ecoli.library.schema import (
    numpy_schema,
    attrs,
    counts,
    bulk_name_to_idx,
    listener_schema,
    zero_listener,
)
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.steps.partition import _protect_state

from v2ecoli.library.units import units
from v2ecoli.library.fitting import normalize


class PolypeptideInitiationLogic:
    """Biological logic for polypeptide initiation.

    Extracted from the original PartitionedProcess so that
    Requester and Evolver each get their own instance.
    """

    def __init__(self, parameters):
        self.parameters = {**PolypeptideInitiation.defaults, **(parameters or {})}
        parameters = self.parameters

        # Load parameters
        self.protein_lengths = parameters.get("protein_lengths", [])
        self.translation_efficiencies = parameters.get("translation_efficiencies", [])
        self.active_ribosome_fraction = parameters.get("active_ribosome_fraction", {})
        self.ribosome_elongation_rates_dict = parameters.get("elongation_rates", {})
        self.variable_elongation = parameters.get("variable_elongation", False)
        self.make_elongation_rates = parameters.get("make_elongation_rates", lambda x: [])

        self.rna_id_to_cistron_indexes = parameters.get("rna_id_to_cistron_indexes", {})
        self.cistron_start_end_pos_in_tu = parameters.get("cistron_start_end_pos_in_tu", {})
        self.tu_ids = parameters.get("tu_ids", [])
        self.n_TUs = len(self.tu_ids)
        # Convert ribosome footprint size from nucleotides to amino acids
        self.active_ribosome_footprint_size = (
            parameters.get("active_ribosome_footprint_size", 24 * units.nt) / 3
        )

        # Get mapping from cistrons to protein monomers and TUs
        self.cistron_to_monomer_mapping = parameters.get("cistron_to_monomer_mapping", {})
        self.cistron_tu_mapping_matrix = parameters.get("cistron_tu_mapping_matrix", {})
        self.monomer_index_to_cistron_index = parameters.get("monomer_index_to_cistron_index", {})
        self.monomer_index_to_tu_indexes = parameters.get("monomer_index_to_tu_indexes", {})

        self.ribosome30S = parameters.get("ribosome30S", "ribosome30S")
        self.ribosome50S = parameters.get("ribosome50S", "ribosome50S")

        self.seed = parameters.get("seed", 0)
        self.random_state = np.random.RandomState(seed=self.seed)

        self.monomer_ids = parameters.get("monomer_ids", [])
        self.emit_unique = parameters.get("emit_unique", False)

        # Helper indices for Numpy indexing (lazy init)
        self.ribosome30S_idx = None

    def _init_indices(self, bulk_ids):
        if self.ribosome30S_idx is None:
            self.ribosome30S_idx = bulk_name_to_idx(self.ribosome30S, bulk_ids)
            self.ribosome50S_idx = bulk_name_to_idx(self.ribosome50S, bulk_ids)

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
        elongation rate.
        """
        allTranslationTimes = 1.0 / ribosomeElongationRates * proteinLengths
        allTranslationTimestepCounts = np.ceil(allTranslationTimes / timeStepSec)
        averageTranslationTimestepCounts = np.dot(
            allTranslationTimestepCounts, proteinInitProb
        )
        expectedTerminationRate = 1.0 / averageTranslationTimestepCounts

        allFractionTimeInactive = (
            1 - allTranslationTimes / timeStepSec / allTranslationTimestepCounts
        )
        averageFractionTimeInactive = np.dot(allFractionTimeInactive, proteinInitProb)
        effectiveFracActiveRibosome = (
            fracActiveRibosome * 1 / (1 - averageFractionTimeInactive)
        )

        activationProb = (
            effectiveFracActiveRibosome
            * expectedTerminationRate
            / (1 - effectiveFracActiveRibosome)
        )

        if activationProb >= 1.0:
            activationProb = 1

        return activationProb


class PolypeptideInitiationRequester(Step):
    """Requester step for polypeptide initiation.

    Requests ribosome subunits (30S and 50S) and reads
    environment/listener state for elongation rates.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop('_logic', None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = PolypeptideInitiationLogic(config)
        self.process_name = 'ecoli-polypeptide-initiation'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(_default=1.0),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def invoke(self, state, interval=None):
        try:
            update = self.update(state)
        except Exception:
            update = {}
        return SyncUpdate(update)

    def update(self, state, interval=None):
        if state.get('next_update_time', 0) > state.get('global_time', 0):
            return {}

        state = _protect_state(state)
        proc = self.process
        proc._init_indices(state['bulk']['id'])

        bulk_request = [
            (proc.ribosome30S_idx, counts(state['bulk'], proc.ribosome30S_idx)),
            (proc.ribosome50S_idx, counts(state['bulk'], proc.ribosome50S_idx)),
        ]

        return {
            'request': {self.process_name: {'bulk': bulk_request}},
        }


class PolypeptideInitiationEvolver(Step):
    """Evolver step for polypeptide initiation.

    Reads allocated ribosome subunits, determines which mRNAs get
    new ribosomes, creates active ribosome unique molecules, and
    consumes ribosome subunits from bulk.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop('_logic', None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = PolypeptideInitiationLogic(config)
        self.process_name = 'ecoli-polypeptide-initiation'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'allocate': InPlaceDict(),
            'environment': InPlaceDict(),
            'listeners': InPlaceDict(),
            'active_ribosome': InPlaceDict(),
            'RNA': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(_default=1.0),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'active_ribosome': InPlaceDict(),
            'listeners': ListenerStore(),
            'next_update_time': Overwrite(_value=Float()),
        }

    def invoke(self, state, interval=None):
        try:
            update = self.update(state)
        except Exception:
            update = {}
        return SyncUpdate(update)

    def update(self, state, interval=None):
        if state.get('next_update_time', 0) > state.get('global_time', 0):
            return {}

        state = _protect_state(state)
        proc = self.process
        proc._init_indices(state['bulk']['id'])

        # Apply allocation: replace bulk counts with allocated amounts
        allocation = state.pop('allocate', {})
        bulk_alloc = allocation.get('bulk')
        if bulk_alloc is not None and hasattr(bulk_alloc, '__len__') and len(bulk_alloc) > 0 and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk

        timestep = state.get('timestep', 1.0)

        # Read environment for media-dependent params
        current_media_id = state['environment']['media_id']
        fracActiveRibosome = proc.active_ribosome_fraction[current_media_id]

        # Read ribosome elongation rate from last timestep
        ribosomeElongationRate = state['listeners']['ribosome_data'][
            'effective_elongation_rate'
        ]
        if ribosomeElongationRate == 0:
            ribosomeElongationRate = proc.ribosome_elongation_rates_dict[
                current_media_id
            ].asNumber(units.aa / units.s)
        elongation_rates = proc.make_elongation_rates(
            proc.random_state,
            ribosomeElongationRate,
            1,
            proc.variable_elongation,
        )
        elongation_rates = np.fmax(elongation_rates, 1)

        # Calculate number of ribosomes that could potentially be initialized
        inactive_ribosome_count = np.min(
            [
                counts(state['bulk'], proc.ribosome30S_idx),
                counts(state['bulk'], proc.ribosome50S_idx),
            ]
        )

        # Calculate actual number of ribosomes that should be activated
        (
            TU_index_RNAs,
            transcript_lengths,
            can_translate,
            is_full_transcript,
            unique_index_RNAs,
        ) = attrs(
            state['RNA'],
            [
                'TU_index',
                'transcript_length',
                'can_translate',
                'is_full_transcript',
                'unique_index',
            ],
        )
        TU_index_mRNAs = TU_index_RNAs[can_translate]
        length_mRNAs = transcript_lengths[can_translate]
        unique_index_mRNAs = unique_index_RNAs[can_translate]
        is_full_transcript_mRNAs = is_full_transcript[can_translate]
        is_incomplete_transcript_mRNAs = np.logical_not(is_full_transcript_mRNAs)

        # Calculate counts of each mRNA cistron from fully transcribed TUs
        TU_index_full_mRNAs = TU_index_mRNAs[is_full_transcript_mRNAs]
        TU_counts_full_mRNAs = np.bincount(TU_index_full_mRNAs, minlength=proc.n_TUs)
        cistron_counts = proc.cistron_tu_mapping_matrix.dot(TU_counts_full_mRNAs)

        # Calculate counts of each mRNA cistron from partially transcribed TUs
        TU_index_incomplete_mRNAs = TU_index_mRNAs[is_incomplete_transcript_mRNAs]
        length_incomplete_mRNAs = length_mRNAs[is_incomplete_transcript_mRNAs]

        for TU_index, length in zip(TU_index_incomplete_mRNAs, length_incomplete_mRNAs):
            cistron_indexes = proc.rna_id_to_cistron_indexes(proc.tu_ids[TU_index])
            cistron_start_positions = np.array(
                [
                    proc.cistron_start_end_pos_in_tu[(cistron_index, TU_index)][0]
                    for cistron_index in cistron_indexes
                ]
            )
            cistron_counts[cistron_indexes] += length > cistron_start_positions

        # Calculate initiation probabilities
        protein_init_prob = normalize(
            cistron_counts[proc.cistron_to_monomer_mapping]
            * proc.translation_efficiencies
        )
        target_protein_init_prob = protein_init_prob.copy()

        # Calculate activation probability
        activation_prob = proc.calculate_activation_prob(
            fracActiveRibosome,
            proc.protein_lengths,
            elongation_rates,
            target_protein_init_prob,
            timestep,
        )

        n_ribosomes_to_activate = np.int64(activation_prob * inactive_ribosome_count)

        if n_ribosomes_to_activate == 0:
            update = {
                'listeners': {
                    'ribosome_data': zero_listener(state['listeners']['ribosome_data'])
                },
                'next_update_time': state.get('global_time', 0) + timestep,
            }
            return update

        # Cap initiation probabilities at maximum physically allowed
        max_p = (
            ribosomeElongationRate
            / proc.active_ribosome_footprint_size
            * (units.s)
            * timestep
            / n_ribosomes_to_activate
        ).asNumber()
        max_p_per_protein = max_p * cistron_counts[proc.cistron_to_monomer_mapping]
        is_overcrowded = protein_init_prob > max_p_per_protein

        is_n_ribosomes_to_activate_reduced = False

        # Resolve overcrowding
        while np.any(protein_init_prob > max_p_per_protein):
            if protein_init_prob[~is_overcrowded].sum() != 0:
                protein_init_prob[is_overcrowded] = max_p_per_protein[is_overcrowded]
                scale_the_rest_by = (
                    1.0 - protein_init_prob[is_overcrowded].sum()
                ) / protein_init_prob[~is_overcrowded].sum()
                protein_init_prob[~is_overcrowded] *= scale_the_rest_by
                is_overcrowded |= protein_init_prob > max_p_per_protein
            else:
                is_n_ribosomes_to_activate_reduced = True
                max_index = np.argmax(
                    protein_init_prob[is_overcrowded]
                    / max_p_per_protein[is_overcrowded]
                )
                max_init_prob = protein_init_prob[is_overcrowded][max_index]
                associated_cistron_counts = cistron_counts[
                    proc.cistron_to_monomer_mapping
                ][is_overcrowded][max_index]
                n_ribosomes_to_activate = np.int64(
                    (
                        ribosomeElongationRate
                        / proc.active_ribosome_footprint_size
                        * (units.s)
                        * timestep
                        / max_init_prob
                        * associated_cistron_counts
                    ).asNumber()
                )

                max_p = (
                    ribosomeElongationRate
                    / proc.active_ribosome_footprint_size
                    * (units.s)
                    * timestep
                    / n_ribosomes_to_activate
                ).asNumber()
                max_p_per_protein = (
                    max_p * cistron_counts[proc.cistron_to_monomer_mapping]
                )
                is_overcrowded = protein_init_prob > max_p_per_protein
                assert is_overcrowded.sum() == 0

        actual_protein_init_prob = protein_init_prob.copy()

        # Sample multinomial distribution
        n_new_proteins = proc.random_state.multinomial(
            n_ribosomes_to_activate, protein_init_prob
        )

        # Build attributes for active ribosomes
        protein_indexes = np.empty(n_ribosomes_to_activate, np.int64)
        mRNA_indexes = np.empty(n_ribosomes_to_activate, np.int64)
        positions_on_mRNA = np.empty(n_ribosomes_to_activate, np.int64)
        nonzero_count = n_new_proteins > 0
        start_index = 0

        for protein_index, protein_counts in zip(
            np.arange(n_new_proteins.size)[nonzero_count], n_new_proteins[nonzero_count]
        ):
            protein_indexes[start_index : start_index + protein_counts] = protein_index

            cistron_index = proc.monomer_index_to_cistron_index[protein_index]

            attribute_indexes = []
            cistron_start_positions = []

            for TU_index in proc.monomer_index_to_tu_indexes[protein_index]:
                attribute_indexes_this_TU = np.where(TU_index_mRNAs == TU_index)[0]
                cistron_start_position = proc.cistron_start_end_pos_in_tu[
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

            n_ribosomes_per_RNA = proc.random_state.multinomial(
                protein_counts, np.full(n_mRNAs, 1.0 / n_mRNAs)
            )

            mRNA_indexes[start_index : start_index + protein_counts] = np.repeat(
                unique_index_mRNAs[attribute_indexes], n_ribosomes_per_RNA
            )

            positions_on_mRNA[start_index : start_index + protein_counts] = np.repeat(
                cistron_start_positions, n_ribosomes_per_RNA
            )

            start_index += protein_counts

        update = {
            'bulk': [
                (proc.ribosome30S_idx, -n_new_proteins.sum()),
                (proc.ribosome50S_idx, -n_new_proteins.sum()),
            ],
            'active_ribosome': {
                'add': {
                    'protein_index': protein_indexes,
                    'peptide_length': np.zeros(n_ribosomes_to_activate, dtype=np.int64),
                    'mRNA_index': mRNA_indexes,
                    'pos_on_mRNA': positions_on_mRNA,
                },
            },
            'listeners': {
                'ribosome_data': {
                    'did_initialize': n_new_proteins.sum(),
                    'ribosome_init_event_per_monomer': n_new_proteins,
                    'target_prob_translation_per_transcript': target_protein_init_prob,
                    'actual_prob_translation_per_transcript': actual_protein_init_prob,
                    'mRNA_is_overcrowded': is_overcrowded,
                    'max_p': max_p,
                    'max_p_per_protein': max_p_per_protein,
                    'is_n_ribosomes_to_activate_reduced': is_n_ribosomes_to_activate_reduced,
                }
            },
            'next_update_time': state.get('global_time', 0) + timestep,
        }

        return update


# Keep legacy class for backward compatibility during migration
from v2ecoli.steps.partition import PartitionedProcess

class PolypeptideInitiation(PartitionedProcess):
    """Legacy PartitionedProcess wrapper — will be removed after migration."""

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

    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)
        self._logic = PolypeptideInitiationLogic(self.parameters)

    def ports_schema(self):
        proc = self._logic
        return {
            "environment": {"media_id": {"_default": "", "_updater": "set"}},
            "listeners": {
                "ribosome_data": listener_schema(
                    {
                        "did_initialize": 0,
                        "target_prob_translation_per_transcript": (
                            [0.0] * len(proc.monomer_ids),
                            proc.monomer_ids,
                        ),
                        "actual_prob_translation_per_transcript": (
                            [0.0] * len(proc.monomer_ids),
                            proc.monomer_ids,
                        ),
                        "mRNA_is_overcrowded": (
                            [False] * len(proc.monomer_ids),
                            proc.monomer_ids,
                        ),
                        "ribosome_init_event_per_monomer": (
                            [0] * len(proc.monomer_ids),
                            proc.monomer_ids,
                        ),
                        "effective_elongation_rate": 0.0,
                        "max_p": 0.0,
                        "max_p_per_protein": (
                            np.zeros(len(proc.monomer_ids), np.float64),
                            proc.monomer_ids,
                        ),
                    }
                ),
            },
            "active_ribosome": numpy_schema(
                "active_ribosome", emit=proc.emit_unique
            ),
            "RNA": numpy_schema("RNAs", emit=proc.emit_unique),
            "bulk": numpy_schema("bulk"),
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        proc = self._logic
        proc._init_indices(states["bulk"]["id"])

        current_media_id = states["environment"]["media_id"]

        requests = {
            "bulk": [
                (proc.ribosome30S_idx, counts(states["bulk"], proc.ribosome30S_idx)),
                (proc.ribosome50S_idx, counts(states["bulk"], proc.ribosome50S_idx)),
            ]
        }

        self.fracActiveRibosome = proc.active_ribosome_fraction[current_media_id]

        self.ribosomeElongationRate = states["listeners"]["ribosome_data"][
            "effective_elongation_rate"
        ]
        if self.ribosomeElongationRate == 0:
            self.ribosomeElongationRate = proc.ribosome_elongation_rates_dict[
                current_media_id
            ].asNumber(units.aa / units.s)
        self.elongation_rates = proc.make_elongation_rates(
            proc.random_state,
            self.ribosomeElongationRate,
            1,
            proc.variable_elongation,
        )
        self.elongation_rates = np.fmax(self.elongation_rates, 1)
        return requests

    def evolve_state(self, timestep, states):
        proc = self._logic

        inactive_ribosome_count = np.min(
            [
                counts(states["bulk"], proc.ribosome30S_idx),
                counts(states["bulk"], proc.ribosome50S_idx),
            ]
        )

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

        TU_index_full_mRNAs = TU_index_mRNAs[is_full_transcript_mRNAs]
        TU_counts_full_mRNAs = np.bincount(TU_index_full_mRNAs, minlength=proc.n_TUs)
        cistron_counts = proc.cistron_tu_mapping_matrix.dot(TU_counts_full_mRNAs)

        TU_index_incomplete_mRNAs = TU_index_mRNAs[is_incomplete_transcript_mRNAs]
        length_incomplete_mRNAs = length_mRNAs[is_incomplete_transcript_mRNAs]

        for TU_index, length in zip(TU_index_incomplete_mRNAs, length_incomplete_mRNAs):
            cistron_indexes = proc.rna_id_to_cistron_indexes(proc.tu_ids[TU_index])
            cistron_start_positions = np.array(
                [
                    proc.cistron_start_end_pos_in_tu[(cistron_index, TU_index)][0]
                    for cistron_index in cistron_indexes
                ]
            )
            cistron_counts[cistron_indexes] += length > cistron_start_positions

        protein_init_prob = normalize(
            cistron_counts[proc.cistron_to_monomer_mapping]
            * proc.translation_efficiencies
        )
        target_protein_init_prob = protein_init_prob.copy()

        activation_prob = proc.calculate_activation_prob(
            self.fracActiveRibosome,
            proc.protein_lengths,
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

        max_p = (
            self.ribosomeElongationRate
            / proc.active_ribosome_footprint_size
            * (units.s)
            * states["timestep"]
            / n_ribosomes_to_activate
        ).asNumber()
        max_p_per_protein = max_p * cistron_counts[proc.cistron_to_monomer_mapping]
        is_overcrowded = protein_init_prob > max_p_per_protein

        is_n_ribosomes_to_activate_reduced = False

        while np.any(protein_init_prob > max_p_per_protein):
            if protein_init_prob[~is_overcrowded].sum() != 0:
                protein_init_prob[is_overcrowded] = max_p_per_protein[is_overcrowded]
                scale_the_rest_by = (
                    1.0 - protein_init_prob[is_overcrowded].sum()
                ) / protein_init_prob[~is_overcrowded].sum()
                protein_init_prob[~is_overcrowded] *= scale_the_rest_by
                is_overcrowded |= protein_init_prob > max_p_per_protein
            else:
                is_n_ribosomes_to_activate_reduced = True
                max_index = np.argmax(
                    protein_init_prob[is_overcrowded]
                    / max_p_per_protein[is_overcrowded]
                )
                max_init_prob = protein_init_prob[is_overcrowded][max_index]
                associated_cistron_counts = cistron_counts[
                    proc.cistron_to_monomer_mapping
                ][is_overcrowded][max_index]
                n_ribosomes_to_activate = np.int64(
                    (
                        self.ribosomeElongationRate
                        / proc.active_ribosome_footprint_size
                        * (units.s)
                        * states["timestep"]
                        / max_init_prob
                        * associated_cistron_counts
                    ).asNumber()
                )

                max_p = (
                    self.ribosomeElongationRate
                    / proc.active_ribosome_footprint_size
                    * (units.s)
                    * states["timestep"]
                    / n_ribosomes_to_activate
                ).asNumber()
                max_p_per_protein = (
                    max_p * cistron_counts[proc.cistron_to_monomer_mapping]
                )
                is_overcrowded = protein_init_prob > max_p_per_protein
                assert is_overcrowded.sum() == 0

        actual_protein_init_prob = protein_init_prob.copy()

        n_new_proteins = proc.random_state.multinomial(
            n_ribosomes_to_activate, protein_init_prob
        )

        protein_indexes = np.empty(n_ribosomes_to_activate, np.int64)
        mRNA_indexes = np.empty(n_ribosomes_to_activate, np.int64)
        positions_on_mRNA = np.empty(n_ribosomes_to_activate, np.int64)
        nonzero_count = n_new_proteins > 0
        start_index = 0

        for protein_index, protein_counts in zip(
            np.arange(n_new_proteins.size)[nonzero_count], n_new_proteins[nonzero_count]
        ):
            protein_indexes[start_index : start_index + protein_counts] = protein_index

            cistron_index = proc.monomer_index_to_cistron_index[protein_index]

            attribute_indexes = []
            cistron_start_positions = []

            for TU_index in proc.monomer_index_to_tu_indexes[protein_index]:
                attribute_indexes_this_TU = np.where(TU_index_mRNAs == TU_index)[0]
                cistron_start_position = proc.cistron_start_end_pos_in_tu[
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

            n_ribosomes_per_RNA = proc.random_state.multinomial(
                protein_counts, np.full(n_mRNAs, 1.0 / n_mRNAs)
            )

            mRNA_indexes[start_index : start_index + protein_counts] = np.repeat(
                unique_index_mRNAs[attribute_indexes], n_ribosomes_per_RNA
            )

            positions_on_mRNA[start_index : start_index + protein_counts] = np.repeat(
                cistron_start_positions, n_ribosomes_per_RNA
            )

            start_index += protein_counts

        update = {
            "bulk": [
                (proc.ribosome30S_idx, -n_new_proteins.sum()),
                (proc.ribosome50S_idx, -n_new_proteins.sum()),
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
