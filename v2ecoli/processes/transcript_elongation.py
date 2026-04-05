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
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Float, Overwrite, Node

from v2ecoli.library.random import stochasticRound
from v2ecoli.library.polymerize import buildSequences, polymerize, computeMassIncrease
from v2ecoli.library.units import units

from v2ecoli.library.schema import (
    counts,
    attrs,
    numpy_schema,
    bulk_name_to_idx,
    listener_schema,
)
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.steps.partition import _protect_state


def make_elongation_rates(random, rates, timestep, variable):
    return rates


def get_attenuation_stop_probabilities(trna_conc):
    return np.array([])


class TranscriptElongationLogic:
    """Biological logic for transcript elongation.

    Extracted from the original PartitionedProcess so that
    Requester and Evolver each get their own instance.
    """

    def __init__(self, parameters):
        self.parameters = {**TranscriptElongation.defaults, **(parameters or {})}
        parameters = self.parameters

        self.rnaPolymeraseElongationRateDict = parameters[
            "rnaPolymeraseElongationRateDict"
        ]
        self.rnaIds = parameters["rnaIds"]
        self.rnaLengths = parameters["rnaLengths"]
        self.rnaSequences = parameters["rnaSequences"]
        self.ppi = parameters["ppi"]
        self.inactive_RNAP = parameters["inactive_RNAP"]
        self.fragmentBases = parameters["fragmentBases"]
        self.charged_trnas = parameters["charged_trnas"]
        self.ntp_ids = parameters["ntp_ids"]
        self.ntWeights = parameters["ntWeights"]
        self.endWeight = parameters["endWeight"]
        self.replichore_lengths = parameters["replichore_lengths"]
        self.chromosome_length = self.replichore_lengths.sum()
        self.n_fragment_bases = parameters["n_fragment_bases"]
        self.recycle_stalled_elongation = parameters["recycle_stalled_elongation"]

        self.is_mRNA = parameters["is_mRNA"]
        self.variable_elongation = parameters["variable_elongation"]
        self.make_elongation_rates = parameters["make_elongation_rates"]

        self.polymerized_ntps = parameters["polymerized_ntps"]
        self.charged_trna_names = parameters["charged_trnas"]

        # Attenuation
        self.trna_attenuation = parameters["trna_attenuation"]
        self.cell_density = parameters["cell_density"]
        self.n_avogadro = parameters["n_avogadro"]
        self.stop_probabilities = parameters["get_attenuation_stop_probabilities"]
        self.attenuated_rna_indices = parameters["attenuated_rna_indices"]
        self.attenuated_rna_indices_lookup = {
            idx: i for i, idx in enumerate(self.attenuated_rna_indices)
        }
        self.attenuated_rnas = self.rnaIds[self.attenuated_rna_indices]
        self.location_lookup = parameters["location_lookup"]

        self.seed = parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices
        self.bulk_RNA_idx = None
        # Cached values
        self.elongation_rates = None
        self.rnapElongationRate = None

    def _init_indices(self, bulk_ids):
        if self.bulk_RNA_idx is None:
            self.bulk_RNA_idx = bulk_name_to_idx(self.rnaIds, bulk_ids)
            self.ntps_idx = bulk_name_to_idx(self.ntp_ids, bulk_ids)
            self.ppi_idx = bulk_name_to_idx(self.ppi, bulk_ids)
            self.inactive_RNAP_idx = bulk_name_to_idx(self.inactive_RNAP, bulk_ids)
            self.fragmentBases_idx = bulk_name_to_idx(self.fragmentBases, bulk_ids)
            self.charged_trnas_idx = bulk_name_to_idx(self.charged_trnas, bulk_ids)

    def calculate_request(self, timestep, states):
        self._init_indices(states["bulk"]["id"])

        current_media_id = states["environment"]["media_id"]

        self.rnapElongationRate = self.rnaPolymeraseElongationRateDict[
            current_media_id
        ].asNumber(units.nt / units.s)

        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.rnapElongationRate,
            states["timestep"],
            self.variable_elongation,
        )

        if states["active_RNAPs"]["_entryState"].sum() == 0:
            return {}

        TU_indexes, transcript_lengths, is_full_transcript = attrs(
            states["RNAs"], ["TU_index", "transcript_length", "is_full_transcript"]
        )
        is_partial_transcript = np.logical_not(is_full_transcript)
        TU_indexes_partial = TU_indexes[is_partial_transcript]
        transcript_lengths_partial = transcript_lengths[is_partial_transcript]

        sequences = buildSequences(
            self.rnaSequences,
            TU_indexes_partial,
            transcript_lengths_partial,
            self.elongation_rates,
        )

        sequenceComposition = np.bincount(
            sequences[sequences != polymerize.PAD_VALUE], minlength=4
        )

        ntpsTotal = counts(states["bulk"], self.ntps_idx)
        maxFractionalReactionLimit = np.fmin(1, ntpsTotal / sequenceComposition)

        requests = {
            "bulk": [
                (
                    self.ntps_idx,
                    (maxFractionalReactionLimit * sequenceComposition).astype(int),
                )
            ]
        }

        requests["listeners"] = {
            "growth_limits": {
                "ntp_pool_size": counts(states["bulk"], self.ntps_idx),
                "ntp_request_size": (
                    maxFractionalReactionLimit * sequenceComposition
                ).astype(int),
            }
        }

        return requests

    def evolve_state(self, timestep, states):
        self._init_indices(states["bulk"]["id"])

        ntpCounts = counts(states["bulk"], self.ntps_idx)

        if states["active_RNAPs"]["_entryState"].sum() == 0:
            return {
                "listeners": {
                    "transcript_elongation_listener": {
                        "count_NTPs_used": 0,
                        "count_rna_synthesized": np.zeros(len(self.rnaIds), dtype=int),
                    },
                    "growth_limits": {
                        "ntp_used": np.zeros(len(self.ntp_ids), dtype=int),
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

        (
            TU_index_all_RNAs,
            length_all_RNAs,
            is_full_transcript,
            is_mRNA_all_RNAs,
            RNAP_index_all_RNAs,
        ) = attrs(
            states["RNAs"],
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

        is_partial_transcript = np.logical_not(is_full_transcript)
        partial_transcript_indexes = np.where(is_partial_transcript)[0]
        TU_index_partial_RNAs = TU_index_all_RNAs[is_partial_transcript]
        length_partial_RNAs = length_all_RNAs[is_partial_transcript]
        is_mRNA_partial_RNAs = is_mRNA_all_RNAs[is_partial_transcript]
        RNAP_index_partial_RNAs = RNAP_index_all_RNAs[is_partial_transcript]

        if self.trna_attenuation:
            cell_mass = states["listeners"]["mass"]["cell_mass"]
            cellVolume = cell_mass * units.fg / self.cell_density
            counts_to_molar = 1 / (self.n_avogadro * cellVolume)
            attenuation_probability = self.stop_probabilities(
                counts_to_molar * counts(states["bulk_total"], self.charged_trnas_idx)
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
            self.elongation_rates,
        )

        reactionLimit = ntpCounts.sum()
        result = polymerize(
            sequences[rna_to_elongate],
            ntpCounts,
            reactionLimit,
            self.random_state,
            self.elongation_rates[TU_index_partial_RNAs][rna_to_elongate],
            self.variable_elongation,
        )

        sequence_elongations = np.zeros_like(length_partial_RNAs)
        sequence_elongations[rna_to_elongate] = result.sequenceElongation
        ntps_used = result.monomerUsages
        did_stall_mask = result.sequences_limited_elongation

        added_mass = computeMassIncrease(
            sequences, sequence_elongations, self.ntWeights
        )
        did_initialize = (length_partial_RNAs == 0) & (sequence_elongations > 0)
        added_mass[did_initialize] += self.endWeight

        updated_transcript_lengths = length_partial_RNAs + sequence_elongations

        coordinates, is_forward, RNAP_unique_index = attrs(
            states["active_RNAPs"], ["coordinates", "is_forward", "unique_index"]
        )

        assert len(RNAP_unique_index) == len(RNAP_index_partial_RNAs)
        assert np.count_nonzero(RNAP_index_partial_RNAs == -1) == 0

        partial_RNA_to_RNAP_mapping, _ = get_mapping_arrays(
            RNAP_index_partial_RNAs, RNAP_unique_index
        )

        direction_rescaled = (2 * (is_forward - 0.5)).astype(np.int64)

        updated_coordinates = coordinates + np.multiply(
            direction_rescaled, sequence_elongations[partial_RNA_to_RNAP_mapping]
        )

        updated_coordinates[updated_coordinates > self.replichore_lengths[0]] -= (
            self.chromosome_length
        )
        updated_coordinates[updated_coordinates < -self.replichore_lengths[1]] += (
            self.chromosome_length
        )

        length_all_RNAs[is_partial_transcript] = updated_transcript_lengths

        added_nsRNA_mass_all_RNAs = np.zeros_like(TU_index_all_RNAs, dtype=np.float64)
        added_mRNA_mass_all_RNAs = np.zeros_like(TU_index_all_RNAs, dtype=np.float64)

        added_nsRNA_mass_all_RNAs[is_partial_transcript] = np.multiply(
            added_mass, np.logical_not(is_mRNA_partial_RNAs)
        )
        added_mRNA_mass_all_RNAs[is_partial_transcript] = np.multiply(
            added_mass, is_mRNA_partial_RNAs
        )

        terminal_lengths = self.rnaLengths[TU_index_partial_RNAs]
        did_terminate_mask = updated_transcript_lengths == terminal_lengths
        terminated_RNAs = np.bincount(
            TU_index_partial_RNAs[did_terminate_mask],
            minlength=self.rnaSequences.shape[0],
        )

        is_full_transcript_updated = is_full_transcript.copy()
        is_full_transcript_updated[partial_transcript_indexes[did_terminate_mask]] = (
            True
        )

        n_terminated = did_terminate_mask.sum()
        n_initialized = did_initialize.sum()
        n_elongations = ntps_used.sum()

        n_new_bulk_RNAs = terminated_RNAs.copy()
        n_new_bulk_RNAs[self.is_mRNA] = 0

        update["RNAs"] = {
            "set": {
                "transcript_length": length_all_RNAs,
                "is_full_transcript": is_full_transcript_updated,
                "massDiff_nonspecific_RNA": attrs(
                    states["RNAs"], ["massDiff_nonspecific_RNA"]
                )[0]
                + added_nsRNA_mass_all_RNAs,
                "massDiff_mRNA": attrs(states["RNAs"], ["massDiff_mRNA"])[0]
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

                update["bulk"].append((self.fragmentBases_idx, base_counts))
                update["bulk"].append((self.ppi_idx, n_initiated_sequences))

        update.setdefault("bulk", []).append((self.ntps_idx, -ntps_used))
        update["bulk"].append((self.bulk_RNA_idx, n_new_bulk_RNAs))
        update["bulk"].append((self.inactive_RNAP_idx, n_terminated + n_attenuated))
        update["bulk"].append((self.ppi_idx, n_elongations - n_initialized))

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

        return update


class TranscriptElongationRequester(Step):
    """Requester step for transcript elongation."""

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop("_logic", None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = TranscriptElongationLogic(config)
        self.process_name = 'ecoli-transcript-elongation'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'RNAs': InPlaceDict(),
            'active_RNAPs': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(_default=1.0),
        }

    def outputs(self):
        return {
            'request': InPlaceDict(),
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

        state = _protect_state(state, cell_state=getattr(self, "_cell_state", None))
        proc = self.process
        timestep = state.get('timestep', 1.0)

        requests = proc.calculate_request(timestep, state)

        # Extract listener data from the request
        listener_data = requests.pop('listeners', {})
        result = {'request': {self.process_name: requests}}
        if listener_data:
            result['listeners'] = listener_data
        return result


class TranscriptElongationEvolver(Step):
    """Evolver step for transcript elongation.

    RECOMPUTES cached values (elongation_rates, rnapElongationRate)
    since Requester and Evolver are separate instances.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance from Requester
        self.process = config.pop("_logic", None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = TranscriptElongationLogic(config)
        self.process_name = 'ecoli-transcript-elongation'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'allocate': InPlaceDict(),
            'environment': InPlaceDict(),
            'RNAs': InPlaceDict(),
            'active_RNAPs': InPlaceDict(),
            'bulk_total': InPlaceDict(),
            'listeners': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(_default=1.0),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'RNAs': InPlaceDict(),
            'active_RNAPs': InPlaceDict(),
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

        state = _protect_state(state, cell_state=getattr(self, "_cell_state", None))
        proc = self.process
        timestep = state.get('timestep', 1.0)

        # Apply allocation
        allocation = state.pop('allocate', {})
        bulk_alloc = allocation.get('bulk')
        if bulk_alloc is not None and hasattr(bulk_alloc, '__len__') and len(bulk_alloc) > 0 and hasattr(state.get('bulk'), 'dtype'):
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk

                # Evolve
        update = proc.evolve_state(timestep, state)
        update['next_update_time'] = state.get('global_time', 0) + state.get('timestep', 1.0)
        return update


# Keep legacy class for backward compatibility during migration
from v2ecoli.steps.partition import PartitionedProcess

class TranscriptElongation(PartitionedProcess):
    """Legacy PartitionedProcess wrapper -- will be removed after migration."""

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
    defaults = {
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
        "is_mRNA": np.array([]),
        "inactive_RNAP": "",
        "ppi": "",
        "ntp_ids": [],
        "variable_elongation": False,
        "make_elongation_rates": make_elongation_rates,
        "fragmentBases": [],
        "polymerized_ntps": [],
        "charged_trnas": [],
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

    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)
        self._logic = TranscriptElongationLogic(self.parameters)

    def ports_schema(self):
        return {
            "environment": {"media_id": {"_default": ""}},
            "RNAs": numpy_schema("RNAs", emit=self.parameters["emit_unique"]),
            "active_RNAPs": numpy_schema(
                "active_RNAPs", emit=self.parameters["emit_unique"]
            ),
            "bulk": numpy_schema("bulk"),
            "bulk_total": numpy_schema("bulk"),
            "listeners": {
                "mass": listener_schema({"cell_mass": 0.0}),
                "transcript_elongation_listener": listener_schema(
                    {
                        "count_NTPs_used": 0,
                        "count_rna_synthesized": ([0] * len(self._logic.rnaIds), self._logic.rnaIds),
                        "attenuation_probability": (
                            [0.0] * len(self._logic.attenuated_rnas),
                            self._logic.attenuated_rnas,
                        ),
                        "counts_attenuated": (
                            [0] * len(self._logic.attenuated_rnas),
                            self._logic.attenuated_rnas,
                        ),
                    }
                ),
                "growth_limits": listener_schema(
                    {
                        "ntp_used": ([0] * len(self._logic.ntp_ids), self._logic.ntp_ids),
                        "ntp_pool_size": ([0] * len(self._logic.ntp_ids), self._logic.ntp_ids),
                        "ntp_request_size": ([0] * len(self._logic.ntp_ids), self._logic.ntp_ids),
                        "ntp_allocated": ([0] * len(self._logic.ntp_ids), self._logic.ntp_ids),
                    }
                ),
                "rnap_data": listener_schema(
                    {
                        "actual_elongations": 0,
                        "did_terminate": 0,
                        "termination_loss": 0,
                        "did_stall": 0,
                    }
                ),
            },
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        return self._logic.calculate_request(timestep, states)

    def evolve_state(self, timestep, states):
        return self._logic.evolve_state(timestep, states)


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
