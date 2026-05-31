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
from scipy.stats import poisson as scipy_poisson

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
        # Phase 2 — same dispatch as TranscriptInitiation's
        # ``pdmp_initiation_mode``. ``"discrete"`` (default) keeps the
        # legacy multinomial sampling; ``"poisson"`` treats each protein-
        # coding-monomer slot as an independent Poisson jump process
        # with rate n_ribosomes_to_activate · p_i and tau-leaps the tick.
        'pdmp_initiation_mode': {'_type': 'string', '_default': 'discrete'},
        # Phase-3 sprint-10 ABC-SMC knob, mirror of
        # transcript_init_prob_scale on TranscriptInitiation. In poisson
        # mode, multiplies the per-protein initiation rate by this
        # scalar before Poisson sampling. Default 1.0 = unperturbed.
        'polypeptide_init_prob_scale': {'_type': 'float', '_default': 1.0},
        'time_step': {'_type': 'integer[s]', '_default': 1},
        'translation_efficiencies': {'_type': 'array[float]', '_default': []},
        'tu_ids': {'_type': 'list[string]', '_default': []},
        'variable_elongation': {'_type': 'boolean', '_default': False},
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
        # The ribosome footprint is stored as a quantity[nt] in the config
        # (default 24 nt) and divided by 3 to get the "amino-acid-coded
        # length" the model actually uses inside max-p calculations. Strip
        # to a plain float here — the per-tick max_p computation that
        # consumes this value treats it as a dimensionless divisor and
        # always calls .magnitude on the result, so carrying pint metadata
        # just buys per-tick Quantity arithmetic for no semantic gain.
        self.active_ribosome_footprint_size = float(
            (self.parameters["active_ribosome_footprint_size"] / 3).magnitude
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

        self.polypeptide_init_prob_scale = float(
            self.parameters.get("polypeptide_init_prob_scale", 1.0))
        if self.polypeptide_init_prob_scale <= 0:
            raise ValueError(
                f"polypeptide_init_prob_scale must be > 0; got "
                f"{self.polypeptide_init_prob_scale!r}")
        self.pdmp_initiation_mode = str(
            self.parameters.get("pdmp_initiation_mode", "discrete"))
        if self.pdmp_initiation_mode not in ("discrete", "poisson"):
            raise ValueError(
                f"pdmp_initiation_mode must be 'discrete' or 'poisson'; "
                f"got {self.pdmp_initiation_mode!r}"
            )

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
                        'effective_elongation_rate': {'_type': 'quantity[float,amino_acid/s]', '_default': 0.0},
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
                        'log_likelihood': {'_type': 'overwrite[float]', '_default': 0.0},
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

        # effective_elongation_rate is a pint Quantity[amino_acid/s] carried
        # from the previous timestep's polypeptide_elongation listener. On the
        # first tick (before elongation has run) it reads as the default 0, so
        # fall back to the media-dependent rate. Strip to a bare aa/s magnitude
        # for make_elongation_rates, which operates on plain floats.
        eff_rate = states["listeners"]["ribosome_data"]["effective_elongation_rate"]
        if eff_rate == 0:
            eff_rate = self.ribosome_elongation_rates_dict[current_media_id]
        self.ribosomeElongationRate = eff_rate.to(units.aa / units.s).magnitude
        self.elongation_rates = np.fmax(self.make_elongation_rates(
            self.random_state,
            self.ribosomeElongationRate,
            1,  # want elongation rate, not lengths adjusted for time step
            self.variable_elongation,
        ), 1)
        # Calculate number of ribosomes that could potentially be initialized
        # based on counts of free 30S and 50S subunits. The `counts(...)` calls
        # each return a 0-d ndarray for a scalar index; Python's `min` over
        # two scalars is faster than np.min over a 2-element list (which
        # builds the list + converts to ndarray).
        inactive_ribosome_count = min(
            counts(states["bulk"], self.ribosome30S_idx),
            counts(states["bulk"], self.ribosome50S_idx),
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
        # number of mRNAs. With the elongation rate and footprint both held
        # as plain floats (aa/s and aa, respectively) and timestep in s,
        # the expression is dimensionless count, so do the math directly in
        # float form — the previous Quantity chain (units.s * timestep) re-
        # attached and then stripped a unit per tick for the same result.
        max_p = (
            self.ribosomeElongationRate
            * states["timestep"]
            / (self.active_ribosome_footprint_size * n_ribosomes_to_activate)
        )
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
                # Same float-only arithmetic as the initial max_p, with
                # an extra associated_cistron_counts factor in the numerator.
                n_ribosomes_to_activate = np.int64(
                    self.ribosomeElongationRate
                    * states["timestep"]
                    * associated_cistron_counts
                    / (self.active_ribosome_footprint_size * max_init_prob)
                )

                # Update maximum probabilities based on new number of activated
                # ribosomes.
                max_p = (
                    self.ribosomeElongationRate
                    * states["timestep"]
                    / (self.active_ribosome_footprint_size * n_ribosomes_to_activate)
                )
                max_p_per_protein = (
                    max_p * cistron_counts[self.cistron_to_monomer_mapping]
                )
                is_overcrowded = protein_init_prob > max_p_per_protein
                assert is_overcrowded.sum() == 0  # We expect no overcrowding

        # Compute actual transcription probabilities of each transcript
        actual_protein_init_prob = protein_init_prob.copy()

        # Sample per-protein new-ribosome counts. Same dispatch as
        # ``TranscriptInitiation``:
        #
        # - ``"discrete"`` (legacy): multinomial(n_target, protein_init_prob).
        #   Σ N_i = n_ribosomes_to_activate exactly, but each per-protein
        #   marginal is coupled to all the others through that constraint.
        # - ``"poisson"`` (Phase 2 PDMP): per-protein Poisson(n_target · p_i)
        #   independent draws — each protein slot is its own jump process
        #   in continuous time, with the per-tick marginal Phase 3's
        #   likelihood machinery can integrate against. Resource cap is
        #   the actual ``inactive_ribosome_count`` pool (NOT
        #   n_ribosomes_to_activate — see the equivalent fix in
        #   v2ecoli/processes/transcript_initiation.py for the asymmetric-
        #   truncation undercount rationale).
        if self.pdmp_initiation_mode == "poisson":
            poisson_means = (self.polypeptide_init_prob_scale
                             * n_ribosomes_to_activate * protein_init_prob)
            n_new_proteins = self.random_state.poisson(poisson_means).astype(np.int64)
            total_drawn = int(n_new_proteins.sum())
            if total_drawn > inactive_ribosome_count:
                # Resource cap: subsample inactive_ribosome_count events
                # from the inflated draw pool, weighted by per-protein
                # counts.
                event_proteins = np.repeat(
                    np.arange(n_new_proteins.size), n_new_proteins,
                )
                keep_idx = self.random_state.choice(
                    event_proteins.size,
                    size=int(inactive_ribosome_count),
                    replace=False,
                )
                n_new_proteins = np.bincount(
                    event_proteins[keep_idx], minlength=n_new_proteins.size,
                ).astype(np.int64)
        else:
            n_new_proteins = self.random_state.multinomial(
                n_ribosomes_to_activate, protein_init_prob
            )

        # After sampling, the actual count of events may differ from
        # n_ribosomes_to_activate (Poisson tau-leap fluctuates around the
        # target; the multinomial path always equals the target). Rebind
        # to the observed sum so downstream array allocations + the
        # ribosome-id range match what the sampler actually emitted.
        n_ribosomes_to_activate = int(n_new_proteins.sum())

        # Build attributes for active ribosomes.
        # Each ribosome is assigned a protein index for the protein that
        # corresponds to the polypeptide it will polymerize. This is done in
        # blocks of protein ids for efficiency.
        protein_indexes = np.empty(n_ribosomes_to_activate, np.int64)
        mRNA_indexes = np.empty(n_ribosomes_to_activate, np.int64)
        positions_on_mRNA = np.empty(n_ribosomes_to_activate, np.int64)
        nonzero_count = n_new_proteins > 0
        start_index = 0

        # Pre-bucket mRNA attribute indices by TU_index so the inner loop is an
        # O(1) dict fetch instead of an O(N) np.where scan over TU_index_mRNAs
        # per (protein × TU) pair (the previous hot path: ~580 np.where calls
        # per tick, ~1k list.extend per tick).
        TU_index_to_attribute_indices: dict[int, np.ndarray] = {}
        if TU_index_mRNAs.size:
            sort_order = np.argsort(TU_index_mRNAs, kind="stable")
            sorted_TUs = TU_index_mRNAs[sort_order]
            # Group boundaries: where TU changes.
            unique_TUs, group_starts = np.unique(sorted_TUs, return_index=True)
            group_ends = np.append(group_starts[1:], sorted_TUs.size)
            for tu, s, e in zip(unique_TUs, group_starts, group_ends):
                TU_index_to_attribute_indices[int(tu)] = sort_order[s:e]

        for protein_index, protein_counts in zip(
            np.arange(n_new_proteins.size)[nonzero_count], n_new_proteins[nonzero_count]
        ):
            # Set protein index
            protein_indexes[start_index : start_index + protein_counts] = protein_index

            cistron_index = self.monomer_index_to_cistron_index[protein_index]

            # Collect per-TU contributions; pre-allocated as numpy arrays once
            # we know the per-TU sizes (one append loop, no Python-list extend).
            tu_indices_iter = self.monomer_index_to_tu_indexes[protein_index]
            per_TU_attr_arrays: list[np.ndarray] = []
            per_TU_starts: list[int] = []
            per_TU_sizes: list[int] = []
            for TU_index in tu_indices_iter:
                attribute_indexes_this_TU = TU_index_to_attribute_indices.get(
                    int(TU_index)
                )
                if attribute_indexes_this_TU is None or attribute_indexes_this_TU.size == 0:
                    continue
                cistron_start_position = self.cistron_start_end_pos_in_tu[
                    (cistron_index, TU_index)
                ][0]
                is_long_enough = (
                    length_mRNAs[attribute_indexes_this_TU] >= cistron_start_position
                )
                kept = attribute_indexes_this_TU[is_long_enough]
                if kept.size == 0:
                    continue
                per_TU_attr_arrays.append(kept)
                per_TU_starts.append(cistron_start_position)
                per_TU_sizes.append(kept.size)

            if per_TU_attr_arrays:
                attribute_indexes = np.concatenate(per_TU_attr_arrays)
                cistron_start_positions = np.repeat(per_TU_starts, per_TU_sizes)
            else:
                attribute_indexes = np.empty(0, dtype=np.int64)
                cistron_start_positions = np.empty(0, dtype=np.int64)

            n_mRNAs = attribute_indexes.size

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

        # Phase-3 sprint-2: per-tick log-likelihood of the observed
        # n_new_proteins under the Poisson rates that drove the
        # sampler. Mirrors TranscriptInitiation's pattern.
        if self.pdmp_initiation_mode == "poisson":
            log_lik = float(
                scipy_poisson.logpmf(n_new_proteins, poisson_means).sum())
        else:
            log_lik = 0.0
        update["listeners"]["ribosome_data"]["log_likelihood"] = log_lik

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
