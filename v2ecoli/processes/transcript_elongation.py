"""
=====================
Transcript Elongation
=====================

This process models nucleotide polymerization into RNA molecules
by RNA polymerases. Polymerization occurs across all polymerases
simultaneously and resources are allocated to maximize the progress
of all polymerases up to the limit of the expected polymerase elongation
rate and available nucleotides. The termination of RNA elongation occurs
once a RNA polymerase has reached the end of the annotated gene.
"""

import numpy as np
import warnings

from process_bigraph import Step

from v2ecoli.library.random import stochasticRound
from v2ecoli.library.polymerize import buildSequences, polymerize, computeMassIncrease
from v2ecoli.library.units import units

from v2ecoli.library.schema import (
    counts,
    attrs,
    bulk_name_to_idx,
)
from v2ecoli.steps.partition import _protect_state, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore


def make_elongation_rates(random, rates, timestep, variable):
    return rates


def get_attenuation_stop_probabilities(trna_conc):
    return np.array([])


class TranscriptElongationStep(_SafeInvokeMixin, Step):
    """Transcript Elongation — merged single Step."""

    config_schema = {}

    topology = {
        "environment": ("environment",),
        "RNAs": ("unique", "RNA"),
        "active_RNAPs": ("unique", "active_RNAP"),
        "bulk": ("bulk",),
        "bulk_total": ("bulk",),
        "listeners": ("listeners",),
        "timestep": ("timestep",),
    }

    def initialize(self, config):
        defaults = {
            # Parameters
            "rnaPolymeraseElongationRateDict": {},
            "rnaIds": [],
            "rnaLengths": np.array([]),
            "rnaSequences": np.array([[]]),
            "ntWeights": np.array([]),
            "endWeight": np.array([]),
            "replichore_lengths": np.array([]),
            "n_fragment_bases": 0,
            "recycle_stalled_elongation": False,
            "submass_indices": {},
            # mask for mRNAs
            "is_mRNA": np.array([]),
            # Bulk molecules
            "inactive_RNAP": "",
            "ppi": "",
            "ntp_ids": [],
            "variable_elongation": False,
            "make_elongation_rates": make_elongation_rates,
            "fragmentBases": [],
            "polymerized_ntps": [],
            "charged_trnas": [],
            # Attenuation
            "trna_attenuation": False,
            "cell_density": 1100 * units.g / units.L,
            "n_avogadro": 6.02214076e23 / units.mol,
            "get_attenuation_stop_probabilities": (get_attenuation_stop_probabilities),
            "attenuated_rna_indices": np.array([], dtype=int),
            "location_lookup": {},
            "seed": 0,
            "emit_unique": False,
            "time_step": 1,
        }
        params = {**defaults, **config}

        # Load parameters
        self.rnaPolymeraseElongationRateDict = params["rnaPolymeraseElongationRateDict"]
        self.rnaIds = params["rnaIds"]
        self.rnaLengths = params["rnaLengths"]
        self.rnaSequences = params["rnaSequences"]
        self.ppi = params["ppi"]
        self.inactive_RNAP = params["inactive_RNAP"]
        self.fragmentBases = params["fragmentBases"]
        self.charged_trnas = params["charged_trnas"]
        self.ntp_ids = params["ntp_ids"]
        self.ntWeights = params["ntWeights"]
        self.endWeight = params["endWeight"]
        self.replichore_lengths = params["replichore_lengths"]
        self.chromosome_length = self.replichore_lengths.sum()
        self.n_fragment_bases = params["n_fragment_bases"]
        self.recycle_stalled_elongation = params["recycle_stalled_elongation"]

        # Mask for mRNAs
        self.is_mRNA = params["is_mRNA"]

        self.variable_elongation = params["variable_elongation"]
        self.make_elongation_rates = params["make_elongation_rates"]

        self.polymerized_ntps = params["polymerized_ntps"]
        self.charged_trna_names = params["charged_trnas"]

        # Attenuation
        self.trna_attenuation = params["trna_attenuation"]
        self.cell_density = params["cell_density"]
        self.n_avogadro = params["n_avogadro"]
        self.stop_probabilities = params["get_attenuation_stop_probabilities"]
        self.attenuated_rna_indices = params["attenuated_rna_indices"]
        self.attenuated_rna_indices_lookup = {
            idx: i for i, idx in enumerate(self.attenuated_rna_indices)
        }
        self.attenuated_rnas = self.rnaIds[self.attenuated_rna_indices] if len(self.attenuated_rna_indices) > 0 else np.array([])
        self.location_lookup = params["location_lookup"]

        # random seed
        self.seed = params["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.bulk_RNA_idx = None

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
            'timestep': InPlaceDict(),
            'global_time': InPlaceDict(),
            'next_update_time': InPlaceDict(),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
            'next_update_time': InPlaceDict(),
        }

    def update(self, state, interval=None):
        # Time-gating check
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state["timestep"]

        # Lazy index initialization
        if self.bulk_RNA_idx is None:
            bulk_ids = state["bulk"]["id"]
            self.bulk_RNA_idx = bulk_name_to_idx(self.rnaIds, bulk_ids)
            self.ntps_idx = bulk_name_to_idx(self.ntp_ids, bulk_ids)
            self.ppi_idx = bulk_name_to_idx(self.ppi, bulk_ids)
            self.inactive_RNAP_idx = bulk_name_to_idx(self.inactive_RNAP, bulk_ids)
            self.fragmentBases_idx = bulk_name_to_idx(self.fragmentBases, bulk_ids)
            self.charged_trnas_idx = bulk_name_to_idx(self.charged_trnas, bulk_ids)

        # Calculate elongation rate based on the current media
        current_media_id = state["environment"]["media_id"]
        rnapElongationRate = self.rnaPolymeraseElongationRateDict[
            current_media_id
        ].asNumber(units.nt / units.s)

        elongation_rates = self.make_elongation_rates(
            self.random_state,
            rnapElongationRate,
            timestep,
            self.variable_elongation,
        )

        # If there are no active RNA polymerases, return immediately
        if state["active_RNAPs"]["_entryState"].sum() == 0:
            ntpCounts = counts(state["bulk"], self.ntps_idx)
            update = {
                "listeners": {
                    "transcript_elongation_listener": {
                        "count_NTPs_used": 0,
                        "count_rna_synthesized": np.zeros(len(self.rnaIds), dtype=int),
                    },
                    "growth_limits": {
                        "ntp_used": np.zeros(len(self.ntp_ids), dtype=int),
                        "ntp_allocated": ntpCounts,
                        "ntp_pool_size": ntpCounts,
                        "ntp_request_size": np.zeros(len(self.ntp_ids), dtype=int),
                    },
                    "rnap_data": {
                        "actual_elongations": 0,
                        "did_terminate": 0,
                        "termination_loss": 0,
                    },
                },
                "next_update_time": global_time + timestep,
            }
            return update

        # Get actual NTP counts (no allocation — use live state)
        ntpCounts = counts(state["bulk"], self.ntps_idx)

        # Get attributes from existing RNAs
        (
            TU_index_all_RNAs,
            length_all_RNAs,
            is_full_transcript,
            is_mRNA_all_RNAs,
            RNAP_index_all_RNAs,
        ) = attrs(
            state["RNAs"],
            [
                "TU_index",
                "transcript_length",
                "is_full_transcript",
                "is_mRNA",
                "RNAP_index",
            ],
        )
        length_all_RNAs = length_all_RNAs.copy()

        update = {"listeners": {"growth_limits": {}}}

        # Determine sequences of RNAs that should be elongated
        is_partial_transcript = np.logical_not(is_full_transcript)
        partial_transcript_indexes = np.where(is_partial_transcript)[0]
        TU_index_partial_RNAs = TU_index_all_RNAs[is_partial_transcript]
        length_partial_RNAs = length_all_RNAs[is_partial_transcript]
        is_mRNA_partial_RNAs = is_mRNA_all_RNAs[is_partial_transcript]
        RNAP_index_partial_RNAs = RNAP_index_all_RNAs[is_partial_transcript]

        if self.trna_attenuation:
            cell_mass = state["listeners"]["mass"]["cell_mass"]
            cellVolume = cell_mass * units.fg / self.cell_density
            counts_to_molar = 1 / (self.n_avogadro * cellVolume)
            attenuation_probability = self.stop_probabilities(
                counts_to_molar * counts(state["bulk_total"], self.charged_trnas_idx)
            )
            prob_lookup = {
                tu: prob
                for tu, prob in zip(
                    self.attenuated_rna_indices, attenuation_probability
                )
            }
            tu_stop_probability = np.array(
                [
                    prob_lookup.get(idx, 0)
                    * (length < self.location_lookup.get(idx, 0))
                    for idx, length in zip(TU_index_partial_RNAs, length_partial_RNAs)
                ]
            )
            rna_to_attenuate = stochasticRound(
                self.random_state, tu_stop_probability
            ).astype(bool)
        else:
            attenuation_probability = np.zeros(len(self.attenuated_rna_indices))
            rna_to_attenuate = np.zeros(len(TU_index_partial_RNAs), bool)
        rna_to_elongate = ~rna_to_attenuate

        sequences = buildSequences(
            self.rnaSequences,
            TU_index_partial_RNAs,
            length_partial_RNAs,
            elongation_rates,
        )

        # Polymerize transcripts based on sequences and available nucleotides
        reactionLimit = ntpCounts.sum()
        result = polymerize(
            sequences[rna_to_elongate],
            ntpCounts,
            reactionLimit,
            self.random_state,
            elongation_rates[TU_index_partial_RNAs][rna_to_elongate],
            self.variable_elongation,
        )

        sequence_elongations = np.zeros_like(length_partial_RNAs)
        sequence_elongations[rna_to_elongate] = result.sequenceElongation
        ntps_used = result.monomerUsages
        did_stall_mask = result.sequences_limited_elongation

        # Calculate changes in mass associated with polymerization
        added_mass = computeMassIncrease(
            sequences, sequence_elongations, self.ntWeights
        )
        did_initialize = (length_partial_RNAs == 0) & (sequence_elongations > 0)
        added_mass[did_initialize] += self.endWeight

        # Calculate updated transcript lengths
        updated_transcript_lengths = length_partial_RNAs + sequence_elongations

        # Get attributes of active RNAPs
        coordinates, is_forward, RNAP_unique_index = attrs(
            state["active_RNAPs"], ["coordinates", "is_forward", "unique_index"]
        )

        # Active RNAP count should equal partial transcript count
        assert len(RNAP_unique_index) == len(RNAP_index_partial_RNAs)

        # All partial RNAs must be linked to an RNAP
        assert np.count_nonzero(RNAP_index_partial_RNAs == -1) == 0

        # Get mapping indexes between partial RNAs to RNAPs
        partial_RNA_to_RNAP_mapping, _ = get_mapping_arrays(
            RNAP_index_partial_RNAs, RNAP_unique_index
        )

        # Rescale boolean array of directions to an array of 1's and -1's
        direction_rescaled = (2 * (is_forward - 0.5)).astype(np.int64)

        # Compute the updated coordinates of RNAPs
        updated_coordinates = coordinates + np.multiply(
            direction_rescaled, sequence_elongations[partial_RNA_to_RNAP_mapping]
        )

        # Reset coordinates of RNAPs that cross the boundaries
        updated_coordinates[updated_coordinates > self.replichore_lengths[0]] -= (
            self.chromosome_length
        )
        updated_coordinates[updated_coordinates < -self.replichore_lengths[1]] += (
            self.chromosome_length
        )

        # Update transcript lengths of RNAs and coordinates of RNAPs
        length_all_RNAs[is_partial_transcript] = updated_transcript_lengths

        # Update added submasses of RNAs
        added_nsRNA_mass_all_RNAs = np.zeros_like(TU_index_all_RNAs, dtype=np.float64)
        added_mRNA_mass_all_RNAs = np.zeros_like(TU_index_all_RNAs, dtype=np.float64)

        added_nsRNA_mass_all_RNAs[is_partial_transcript] = np.multiply(
            added_mass, np.logical_not(is_mRNA_partial_RNAs)
        )
        added_mRNA_mass_all_RNAs[is_partial_transcript] = np.multiply(
            added_mass, is_mRNA_partial_RNAs
        )

        # Determine if transcript has reached the end of the sequence
        terminal_lengths = self.rnaLengths[TU_index_partial_RNAs]
        did_terminate_mask = updated_transcript_lengths == terminal_lengths
        terminated_RNAs = np.bincount(
            TU_index_partial_RNAs[did_terminate_mask],
            minlength=self.rnaSequences.shape[0],
        )

        # Update is_full_transcript attribute of RNAs
        is_full_transcript_updated = is_full_transcript.copy()
        is_full_transcript_updated[partial_transcript_indexes[did_terminate_mask]] = (
            True
        )

        n_terminated = did_terminate_mask.sum()
        n_initialized = did_initialize.sum()
        n_elongations = ntps_used.sum()

        # Get counts of new bulk RNAs
        n_new_bulk_RNAs = terminated_RNAs.copy()
        n_new_bulk_RNAs[self.is_mRNA] = 0

        update["RNAs"] = {
            "set": {
                "transcript_length": length_all_RNAs,
                "is_full_transcript": is_full_transcript_updated,
                "massDiff_nonspecific_RNA": attrs(
                    state["RNAs"], ["massDiff_nonspecific_RNA"]
                )[0]
                + added_nsRNA_mass_all_RNAs,
                "massDiff_mRNA": attrs(state["RNAs"], ["massDiff_mRNA"])[0]
                + added_mRNA_mass_all_RNAs,
            },
            "delete": partial_transcript_indexes[
                np.logical_and(did_terminate_mask, np.logical_not(is_mRNA_partial_RNAs))
            ],
        }
        update["active_RNAPs"] = {
            "set": {"coordinates": updated_coordinates},
            "delete": np.where(did_terminate_mask[partial_RNA_to_RNAP_mapping])[0],
        }

        # Attenuation removes RNAs and RNAPs
        counts_attenuated = np.zeros(len(self.attenuated_rna_indices), dtype=int)
        if np.any(rna_to_attenuate):
            for idx in TU_index_partial_RNAs[rna_to_attenuate]:
                counts_attenuated[self.attenuated_rna_indices_lookup[idx]] += 1
            update["RNAs"]["delete"] = np.append(
                update["RNAs"]["delete"], partial_transcript_indexes[rna_to_attenuate]
            )
            update["active_RNAPs"]["delete"] = np.append(
                update["active_RNAPs"]["delete"],
                np.where(rna_to_attenuate[partial_RNA_to_RNAP_mapping])[0],
            )
        n_attenuated = rna_to_attenuate.sum()

        # Handle stalled elongation
        n_total_stalled = did_stall_mask.sum()
        if self.recycle_stalled_elongation and (n_total_stalled > 0):
            update["active_RNAPs"]["delete"] = np.append(
                update["active_RNAPs"]["delete"],
                np.where(did_stall_mask[partial_RNA_to_RNAP_mapping])[0],
            )
            update.setdefault("bulk", []).append((self.inactive_RNAP_idx, n_total_stalled))

            update["RNAs"]["delete"] = np.append(
                update["RNAs"]["delete"], partial_transcript_indexes[did_stall_mask]
            )
            stalled_sequence_lengths = updated_transcript_lengths[did_stall_mask]
            n_initiated_sequences = np.count_nonzero(stalled_sequence_lengths)

            if n_initiated_sequences > 0:
                stalled_sequences = buildSequences(
                    self.rnaSequences,
                    TU_index_partial_RNAs[did_stall_mask],
                    np.zeros(n_total_stalled, dtype=np.int64),
                    np.full(n_total_stalled, updated_transcript_lengths.max()),
                )

                base_counts = np.zeros(self.n_fragment_bases, dtype=np.int64)
                for sl, seq in zip(stalled_sequence_lengths, stalled_sequences):
                    base_counts += np.bincount(
                        seq[:sl], minlength=self.n_fragment_bases
                    )

                update.setdefault("bulk", []).append((self.fragmentBases_idx, base_counts))
                update["bulk"].append((self.ppi_idx, n_initiated_sequences))

        update.setdefault("bulk", []).append((self.ntps_idx, -ntps_used))
        update["bulk"].append((self.bulk_RNA_idx, n_new_bulk_RNAs))
        update["bulk"].append((self.inactive_RNAP_idx, n_terminated + n_attenuated))
        update["bulk"].append((self.ppi_idx, n_elongations - n_initialized))

        # Write outputs to listeners
        update["listeners"]["transcript_elongation_listener"] = {
            "count_rna_synthesized": terminated_RNAs,
            "count_NTPs_used": n_elongations,
            "attenuation_probability": attenuation_probability,
            "counts_attenuated": counts_attenuated,
        }
        update["listeners"]["growth_limits"] = {
            "ntp_used": ntps_used,
            "ntp_pool_size": counts(state["bulk"], self.ntps_idx),
            "ntp_request_size": ntps_used,
        }
        update["listeners"]["rnap_data"] = {
            "actual_elongations": sequence_elongations.sum(),
            "did_terminate": did_terminate_mask.sum(),
            "termination_loss": (terminal_lengths - length_partial_RNAs)[
                did_terminate_mask
            ].sum(),
            "did_stall": n_total_stalled,
        }

        update["next_update_time"] = global_time + timestep
        return update


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


def format_data(data, bulk_ids, rna_dtypes, rnap_dtypes, submass_dtypes):
    # Format unique and bulk data for assertions
    data["unique"]["RNA"] = [
        np.array(list(map(tuple, zip(*val))), dtype=rna_dtypes + submass_dtypes)
        for val in data["unique"]["RNA"]
    ]
    data["unique"]["active_RNAP"] = [
        np.array(list(map(tuple, zip(*val))), dtype=rnap_dtypes + submass_dtypes)
        for val in data["unique"]["active_RNAP"]
    ]
    bulk_timeseries = np.array(data["bulk"])
    data["bulk"] = {
        bulk_id: bulk_timeseries[:, i] for i, bulk_id in enumerate(bulk_ids)
    }
    return data
