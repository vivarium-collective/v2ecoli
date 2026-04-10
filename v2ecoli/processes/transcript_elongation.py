"""Process-bigraph partitioned process: transcript_elongation."""

from typing import Any, Callable, Optional, Tuple, cast

from numba import njit
import numpy as np
import numpy.typing as npt
import scipy.sparse
import warnings
from scipy.integrate import solve_ivp

from process_bigraph import Step
from bigraph_schema.schema import Float, Overwrite

from v2ecoli.library.fitting import normalize
from v2ecoli.library.polymerize import buildSequences, polymerize, computeMassIncrease
from v2ecoli.library.random import stochasticRound
from v2ecoli.library.schema import (
    create_unique_indices,
    counts,
    attrs,
    bulk_name_to_idx,
    MetadataArray,
    zero_listener,
)
from v2ecoli.library.unit_defs import units
from v2ecoli.steps.partition import _protect_state, deep_merge, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore, SetStore


def _apply_config_defaults(config_schema, parameters):
    """Merge config_schema defaults with provided parameters."""
    merged = {}
    for key, spec in config_schema.items():
        if isinstance(spec, dict) and "_default" in spec:
            merged[key] = spec["_default"]
    merged.update(parameters or {})
    return merged

def make_elongation_rates(random, rates, timestep, variable):
    return rates


def get_attenuation_stop_probabilities(trna_conc):
    return np.array([])


class TranscriptElongationLogic:
    """Transcript Elongation — shared state container for Requester/Evolver.

    defaults:
        - rnaPolymeraseElongationRateDict (dict): Array with elongation rate
            set points for different media environments.
        - rnaIds (array[str]) : array of names for each TU
        - rnaLengths (array[int]) : array of lengths for each TU
            (in nucleotides?)
        - rnaSequences (2D array[int]) : Array with the nucleotide sequences
            of each TU. This is in the form of a 2D array where each row is a
            TU, and each column is a position in the TU's sequence. Nucleotides
            are stored as an index {0, 1, 2, 3}, and the row is padded with
            -1's on the right to indicate where the sequence ends.
        - ntWeights (array[float]): Array of nucleotide weights
        - endWeight (array[float]): ???,
        - replichore_lengths (array[int]): lengths of replichores
            (in nucleotides?),
        - is_mRNA (array[bool]): Mask for mRNAs
        - ppi (str): ID of PPI
        - inactive_RNAP (str): ID of inactive RNAP
        - ntp_ids list[str]: IDs of ntp's (A, C, G, U)
        - variable_elongation (bool): Whether to use variable elongation.
                                      False by default.
        - make_elongation_rates: Function to make elongation rates, of the
            form: lambda random, rates, timestep, variable: rates
    """

    name = "ecoli-transcript-elongation"
    topology = {
    "environment": ("environment",),
    "RNAs": ("unique", "RNA"),
    "active_RNAPs": ("unique", "active_RNAP"),
    "bulk": ("bulk",),
    "bulk_total": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
}
    config_schema = {
        # Parameters
        "rnaPolymeraseElongationRateDict": {"_default": {}},
        "rnaIds": {"_default": []},
        "rnaLengths": {"_default": None},
        "rnaSequences": {"_default": None},
        "ntWeights": {"_default": None},
        "endWeight": {"_default": None},
        "replichore_lengths": {"_default": None},
        "n_fragment_bases": {"_default": 0},
        "recycle_stalled_elongation": {"_default": False},
        "submass_indices": {"_default": {}},
        # mask for mRNAs
        "is_mRNA": {"_default": None},
        # Bulk molecules
        "inactive_RNAP": {"_default": ""},
        "ppi": {"_default": ""},
        "ntp_ids": {"_default": []},
        "variable_elongation": {"_default": False},
        "make_elongation_rates": {"_default": None},
        "fragmentBases": {"_default": []},
        "polymerized_ntps": {"_default": []},
        "charged_trnas": {"_default": []},
        # Attenuation
        "trna_attenuation": {"_default": False},
        "cell_density": {"_default": None},
        "n_avogadro": {"_default": None},
        "get_attenuation_stop_probabilities": {"_default": None},
        "attenuated_rna_indices": {"_default": None},
        "location_lookup": {"_default": {}},
        "seed": {"_default": 0},
        "emit_unique": {"_default": False},
        "time_step": {"_default": 1},
    }

    def __init__(self, parameters=None):
        self.parameters = _apply_config_defaults(self.config_schema, parameters)
        self.request_set = False

        # Load parameters
        self.rnaPolymeraseElongationRateDict = self.parameters[
            "rnaPolymeraseElongationRateDict"
        ]
        self.rnaIds = self.parameters["rnaIds"]
        self.rnaLengths = self.parameters["rnaLengths"]
        self.rnaSequences = self.parameters["rnaSequences"]
        self.ppi = self.parameters["ppi"]
        self.inactive_RNAP = self.parameters["inactive_RNAP"]
        self.fragmentBases = self.parameters["fragmentBases"]
        self.charged_trnas = self.parameters["charged_trnas"]
        self.ntp_ids = self.parameters["ntp_ids"]
        self.ntWeights = self.parameters["ntWeights"]
        self.endWeight = self.parameters["endWeight"]
        self.replichore_lengths = self.parameters["replichore_lengths"]
        self.chromosome_length = self.replichore_lengths.sum()
        self.n_fragment_bases = self.parameters["n_fragment_bases"]
        self.recycle_stalled_elongation = self.parameters["recycle_stalled_elongation"]

        # Mask for mRNAs
        self.is_mRNA = self.parameters["is_mRNA"]

        self.variable_elongation = self.parameters["variable_elongation"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]

        self.polymerized_ntps = self.parameters["polymerized_ntps"]
        self.charged_trna_names = self.parameters["charged_trnas"]

        # Attenuation
        self.trna_attenuation = self.parameters["trna_attenuation"]
        self.cell_density = self.parameters["cell_density"]
        self.n_avogadro = self.parameters["n_avogadro"]
        self.stop_probabilities = self.parameters["get_attenuation_stop_probabilities"]
        self.attenuated_rna_indices = self.parameters["attenuated_rna_indices"]
        self.attenuated_rna_indices_lookup = {
            idx: i for i, idx in enumerate(self.attenuated_rna_indices)
        }
        self.attenuated_rnas = self.rnaIds[self.attenuated_rna_indices]
        self.location_lookup = self.parameters["location_lookup"]

        # random seed
        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.bulk_RNA_idx = None



class TranscriptElongationRequester(_SafeInvokeMixin, Step):
    """Reads stores to compute transcript elongation request."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
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
        p = self.process
        # --- inlined from calculate_request ---
        # At first update, convert all strings to indices
        if p.bulk_RNA_idx is None:
            bulk_ids = state["bulk"]["id"]
            p.bulk_RNA_idx = bulk_name_to_idx(p.rnaIds, bulk_ids)
            p.ntps_idx = bulk_name_to_idx(p.ntp_ids, bulk_ids)
            p.ppi_idx = bulk_name_to_idx(p.ppi, bulk_ids)
            p.inactive_RNAP_idx = bulk_name_to_idx(p.inactive_RNAP, bulk_ids)
            p.fragmentBases_idx = bulk_name_to_idx(p.fragmentBases, bulk_ids)
            p.charged_trnas_idx = bulk_name_to_idx(p.charged_trnas, bulk_ids)

        # Calculate elongation rate based on the current media
        current_media_id = state["environment"]["media_id"]

        p.rnapElongationRate = p.rnaPolymeraseElongationRateDict[
            current_media_id
        ].asNumber(units.nt / units.s)

        p.elongation_rates = p.make_elongation_rates(
            p.random_state,
            p.rnapElongationRate,
            state["timestep"],
            p.variable_elongation,
        )

        # If there are no active RNA polymerases, return immediately
        if state["active_RNAPs"]["_entryState"].sum() == 0:
            request = {}
        else:
            # Determine total possible sequences of nucleotides that can be
            # transcribed in this time step for each partial transcript
            TU_indexes, transcript_lengths, is_full_transcript = attrs(
                state["RNAs"], ["TU_index", "transcript_length", "is_full_transcript"]
            )
            is_partial_transcript = np.logical_not(is_full_transcript)
            TU_indexes_partial = TU_indexes[is_partial_transcript]
            transcript_lengths_partial = transcript_lengths[is_partial_transcript]

            sequences = buildSequences(
                p.rnaSequences,
                TU_indexes_partial,
                transcript_lengths_partial,
                p.elongation_rates,
            )

            sequenceComposition = np.bincount(
                sequences[sequences != polymerize.PAD_VALUE], minlength=4
            )

            # Calculate if any nucleotides are limited and request up to the number
            # in the sequences or number available
            ntpsTotal = counts(state["bulk"], p.ntps_idx)
            maxFractionalReactionLimit = np.fmin(1, ntpsTotal / sequenceComposition)

            request = {
                "bulk": [
                    (
                        p.ntps_idx,
                        (maxFractionalReactionLimit * sequenceComposition).astype(int),
                    )
                ]
            }

            request["listeners"] = {
                "growth_limits": {
                    "ntp_pool_size": counts(state["bulk"], p.ntps_idx),
                    "ntp_request_size": (
                        maxFractionalReactionLimit * sequenceComposition
                    ).astype(int),
                }
            }
        # --- end inlined ---
        p.request_set = True

        bulk_request = request.pop('bulk', None)
        result = {'request': {}}
        if bulk_request is not None:
            result['request']['bulk'] = bulk_request

        listeners = request.pop('listeners', None)
        if listeners is not None:
            result['listeners'] = listeners

        return result


class TranscriptElongationEvolver(_SafeInvokeMixin, Step):
    """Reads allocation, writes bulk/unique/listener updates."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': SetStore(),
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
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
        p = self.process
        # --- inlined from evolve_state ---
        ntpCounts = counts(state["bulk"], p.ntps_idx)

        # If there are no active RNA polymerases, return immediately
        if state["active_RNAPs"]["_entryState"].sum() == 0:
            update = {
                "listeners": {
                    "transcript_elongation_listener": {
                        "count_NTPs_used": 0,
                        "count_rna_synthesized": np.zeros(len(p.rnaIds), dtype=int),
                    },
                    "growth_limits": {
                        "ntp_used": np.zeros(len(p.ntp_ids), dtype=int),
                        "ntp_allocated": ntpCounts,
                    },
                    "rnap_data": {
                        "actual_elongations": 0,
                        "did_terminate": 0,
                        "termination_loss": 0,
                    },
                },
                "active_RNAPs": {},
                "RNAs": {},
            }
        else:
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

            if p.trna_attenuation:
                cell_mass = state["listeners"]["mass"]["cell_mass"]
                cellVolume = cell_mass * units.fg / p.cell_density
                counts_to_molar = 1 / (p.n_avogadro * cellVolume)
                attenuation_probability = p.stop_probabilities(
                    counts_to_molar * counts(state["bulk_total"], p.charged_trnas_idx)
                )
                prob_lookup = {
                    tu: prob
                    for tu, prob in zip(
                        p.attenuated_rna_indices, attenuation_probability
                    )
                }
                tu_stop_probability = np.array(
                    [
                        prob_lookup.get(idx, 0)
                        * (length < p.location_lookup.get(idx, 0))
                        for idx, length in zip(TU_index_partial_RNAs, length_partial_RNAs)
                    ]
                )
                rna_to_attenuate = stochasticRound(
                    p.random_state, tu_stop_probability
                ).astype(bool)
            else:
                attenuation_probability = np.zeros(len(p.attenuated_rna_indices))
                rna_to_attenuate = np.zeros(len(TU_index_partial_RNAs), bool)
            rna_to_elongate = ~rna_to_attenuate

            sequences = buildSequences(
                p.rnaSequences,
                TU_index_partial_RNAs,
                length_partial_RNAs,
                p.elongation_rates,
            )

            # Polymerize transcripts based on sequences and available nucleotides
            reactionLimit = ntpCounts.sum()
            result = polymerize(
                sequences[rna_to_elongate],
                ntpCounts,
                reactionLimit,
                p.random_state,
                p.elongation_rates[TU_index_partial_RNAs][rna_to_elongate],
                p.variable_elongation,
            )

            sequence_elongations = np.zeros_like(length_partial_RNAs)
            sequence_elongations[rna_to_elongate] = result.sequenceElongation
            ntps_used = result.monomerUsages
            did_stall_mask = result.sequences_limited_elongation

            # Calculate changes in mass associated with polymerization
            added_mass = computeMassIncrease(
                sequences, sequence_elongations, p.ntWeights
            )
            did_initialize = (length_partial_RNAs == 0) & (sequence_elongations > 0)
            added_mass[did_initialize] += p.endWeight

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

            # Rescale boolean array of directions to an array of 1's and -1's.
            # True is converted to 1, False is converted to -1.
            direction_rescaled = (2 * (is_forward - 0.5)).astype(np.int64)

            # Compute the updated coordinates of RNAPs. Coordinates of RNAPs
            # moving in the positive direction are increased, whereas coordinates
            # of RNAPs moving in the negative direction are decreased.
            updated_coordinates = coordinates + np.multiply(
                direction_rescaled, sequence_elongations[partial_RNA_to_RNAP_mapping]
            )

            # Reset coordinates of RNAPs that cross the boundaries between right
            # and left replichores
            updated_coordinates[updated_coordinates > p.replichore_lengths[0]] -= (
                p.chromosome_length
            )
            updated_coordinates[updated_coordinates < -p.replichore_lengths[1]] += (
                p.chromosome_length
            )

            # Update transcript lengths of RNAs and coordinates of RNAPs
            length_all_RNAs[is_partial_transcript] = updated_transcript_lengths

            # Update added submasses of RNAs. Masses of partial mRNAs are counted
            # as mRNA mass as they are already functional, but the masses of other
            # types of partial RNAs are counted as nonspecific RNA mass.
            added_nsRNA_mass_all_RNAs = np.zeros_like(TU_index_all_RNAs, dtype=np.float64)
            added_mRNA_mass_all_RNAs = np.zeros_like(TU_index_all_RNAs, dtype=np.float64)

            added_nsRNA_mass_all_RNAs[is_partial_transcript] = np.multiply(
                added_mass, np.logical_not(is_mRNA_partial_RNAs)
            )
            added_mRNA_mass_all_RNAs[is_partial_transcript] = np.multiply(
                added_mass, is_mRNA_partial_RNAs
            )

            # Determine if transcript has reached the end of the sequence
            terminal_lengths = p.rnaLengths[TU_index_partial_RNAs]
            did_terminate_mask = updated_transcript_lengths == terminal_lengths
            terminated_RNAs = np.bincount(
                TU_index_partial_RNAs[did_terminate_mask],
                minlength=p.rnaSequences.shape[0],
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
            n_new_bulk_RNAs[p.is_mRNA] = 0

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
            counts_attenuated = np.zeros(len(p.attenuated_rna_indices), dtype=int)
            if np.any(rna_to_attenuate):
                for idx in TU_index_partial_RNAs[rna_to_attenuate]:
                    counts_attenuated[p.attenuated_rna_indices_lookup[idx]] += 1
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
            if p.recycle_stalled_elongation and (n_total_stalled > 0):
                # Remove RNAPs that were bound to stalled elongation transcripts
                # and increment counts of inactive RNAPs
                update["active_RNAPs"]["delete"] = np.append(
                    update["active_RNAPs"]["delete"],
                    np.where(did_stall_mask[partial_RNA_to_RNAP_mapping])[0],
                )
                update["bulk"].append((p.inactive_RNAP_idx, n_total_stalled))

                # Remove partial transcripts from stalled elongation
                update["RNAs"]["delete"] = np.append(
                    update["RNAs"]["delete"], partial_transcript_indexes[did_stall_mask]
                )
                stalled_sequence_lengths = updated_transcript_lengths[did_stall_mask]
                n_initiated_sequences = np.count_nonzero(stalled_sequence_lengths)

                if n_initiated_sequences > 0:
                    # Get the full sequence of stalled transcripts
                    stalled_sequences = buildSequences(
                        p.rnaSequences,
                        TU_index_partial_RNAs[did_stall_mask],
                        np.zeros(n_total_stalled, dtype=np.int64),
                        np.full(n_total_stalled, updated_transcript_lengths.max()),
                    )

                    # Count the number of fragment bases in these transcripts up
                    # until the stalled length
                    base_counts = np.zeros(p.n_fragment_bases, dtype=np.int64)
                    for sl, seq in zip(stalled_sequence_lengths, stalled_sequences):
                        base_counts += np.bincount(
                            seq[:sl], minlength=p.n_fragment_bases
                        )

                    # Increment counts of fragment NTPs and phosphates
                    update["bulk"].append((p.fragmentBases_idx, base_counts))
                    update["bulk"].append((p.ppi_idx, n_initiated_sequences))

            update.setdefault("bulk", []).append((p.ntps_idx, -ntps_used))
            update["bulk"].append((p.bulk_RNA_idx, n_new_bulk_RNAs))
            update["bulk"].append((p.inactive_RNAP_idx, n_terminated + n_attenuated))
            update["bulk"].append((p.ppi_idx, n_elongations - n_initialized))

            # Write outputs to listeners
            update["listeners"]["transcript_elongation_listener"] = {
                "count_rna_synthesized": terminated_RNAs,
                "count_NTPs_used": n_elongations,
                "attenuation_probability": attenuation_probability,
                "counts_attenuated": counts_attenuated,
            }
            update["listeners"]["growth_limits"] = {"ntp_used": ntps_used}
            update["listeners"]["rnap_data"] = {
                "actual_elongations": sequence_elongations.sum(),
                "did_terminate": did_terminate_mask.sum(),
                "termination_loss": (terminal_lengths - length_partial_RNAs)[
                    did_terminate_mask
                ].sum(),
                "did_stall": n_total_stalled,
            }
        # --- end inlined ---
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
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
