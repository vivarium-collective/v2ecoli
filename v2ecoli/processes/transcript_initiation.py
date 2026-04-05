"""
=====================
Transcript Initiation
=====================

This process models the binding of RNA polymerase to each gene.
The number of RNA polymerases to activate in each time step is determined
such that the average fraction of RNA polymerases that are active throughout
the simulation matches measured fractions, which are dependent on the
cellular growth rate. This is done by assuming a steady state concentration
of active RNA polymerases.

TODO:
  - use transcription units instead of single genes
  - match sigma factors to promoters
"""

import numpy as np
import scipy.sparse
from typing import cast

from process_bigraph import Step
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Float, Overwrite, Node

from v2ecoli.library.schema import (
    create_unique_indices,
    listener_schema,
    numpy_schema,
    counts,
    attrs,
    bulk_name_to_idx,
    MetadataArray,
)

from v2ecoli.library.units import units
from v2ecoli.library.random import stochasticRound
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.steps.partition import _protect_state


class TranscriptInitiationLogic:
    """Biological logic for transcript initiation.

    Extracted from the original PartitionedProcess so that
    Requester and Evolver each get their own instance.
    """

    def __init__(self, parameters):
        self.parameters = {**TranscriptInitiation.defaults, **(parameters or {})}
        parameters = self.parameters

        # Load parameters
        self.fracActiveRnapDict = parameters["fracActiveRnapDict"]
        self.rnaLengths = parameters["rnaLengths"]
        self.rnaPolymeraseElongationRateDict = parameters[
            "rnaPolymeraseElongationRateDict"
        ]
        self.variable_elongation = parameters["variable_elongation"]
        self.make_elongation_rates = parameters["make_elongation_rates"]
        self.active_rnap_footprint_size = parameters["active_rnap_footprint_size"]

        # Initialize matrices used to calculate synthesis probabilities
        self.basal_prob = parameters["basal_prob"].copy()
        self.trna_attenuation = parameters["trna_attenuation"]
        if self.trna_attenuation:
            self.attenuated_rna_indices = parameters["attenuated_rna_indices"]
            self.attenuation_adjustments = parameters["attenuation_adjustments"]
            self.basal_prob[self.attenuated_rna_indices] += self.attenuation_adjustments

        self.n_TUs = len(self.basal_prob)
        self.delta_prob = parameters["delta_prob"]
        if parameters["get_delta_prob_matrix"] is not None:
            self.delta_prob_matrix = parameters["get_delta_prob_matrix"](
                dense=True, ppgpp=parameters["ppgpp_regulation"]
            )
        else:
            self.delta_prob_matrix = scipy.sparse.csr_matrix(
                (
                    self.delta_prob["deltaV"],
                    (self.delta_prob["deltaI"], self.delta_prob["deltaJ"]),
                ),
                shape=self.delta_prob["shape"],
            ).toarray()

        # Determine changes from genetic perturbations
        self.genetic_perturbations = {}
        self.perturbations = parameters["perturbations"]
        self.rna_data = parameters["rna_data"]

        if len(self.perturbations) > 0:
            probability_indexes = [
                (index, self.perturbations[rna_data["id"]])
                for index, rna_data in enumerate(self.rna_data)
                if rna_data["id"] in self.perturbations
            ]
            self.genetic_perturbations = {
                "fixedRnaIdxs": [pair[0] for pair in probability_indexes],
                "fixedSynthProbs": [pair[1] for pair in probability_indexes],
            }

        # ID Groups
        self.idx_rRNA = parameters["idx_rRNA"]
        self.idx_mRNA = parameters["idx_mRNA"]
        self.idx_tRNA = parameters["idx_tRNA"]
        self.idx_rprotein = parameters["idx_rprotein"]
        self.idx_rnap = parameters["idx_rnap"]

        # Synthesis probabilities for different categories of genes
        self.rnaSynthProbFractions = parameters["rnaSynthProbFractions"]
        self.rnaSynthProbRProtein = parameters["rnaSynthProbRProtein"]
        self.rnaSynthProbRnaPolymerase = parameters["rnaSynthProbRnaPolymerase"]

        # Coordinates and transcription directions of transcription units
        self.replication_coordinate = parameters["replication_coordinate"]
        self.transcription_direction = parameters["transcription_direction"]

        self.inactive_RNAP = parameters["inactive_RNAP"]

        # ppGpp control related
        self.n_avogadro = parameters["n_avogadro"]
        self.cell_density = parameters["cell_density"]
        self.ppgpp = parameters["ppgpp"]
        self.synth_prob = parameters["synth_prob"]
        self.copy_number = parameters["copy_number"]
        self.ppgpp_regulation = parameters["ppgpp_regulation"]
        self.get_rnap_active_fraction_from_ppGpp = parameters[
            "get_rnap_active_fraction_from_ppGpp"
        ]

        self.seed = parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices
        self.ppgpp_idx = None

    def _init_indices(self, bulk_ids):
        if self.ppgpp_idx is None:
            self.ppgpp_idx = bulk_name_to_idx(self.ppgpp, bulk_ids)
            self.inactive_RNAP_idx = bulk_name_to_idx(self.inactive_RNAP, bulk_ids)

    def calculate_request(self, timestep, states):
        """Calculate the request for inactive RNAPs and compute promoter_init_probs."""
        self._init_indices(states["bulk"]["id"])

        # Get all inactive RNA polymerases
        requests = {
            "bulk": [
                (self.inactive_RNAP_idx, counts(states["bulk"], self.inactive_RNAP_idx))
            ]
        }

        current_media_id = states["environment"]["media_id"]

        if states["full_chromosomes"]["_entryState"].sum() > 0:
            TU_index, bound_TF = attrs(states["promoters"], ["TU_index", "bound_TF"])

            if self.ppgpp_regulation:
                cell_mass = states["listeners"]["mass"]["cell_mass"] * units.fg
                cell_volume = cell_mass / self.cell_density
                counts_to_molar = 1 / (self.n_avogadro * cell_volume)
                ppgpp_conc = counts(states["bulk"], self.ppgpp_idx) * counts_to_molar
                basal_prob, _ = self.synth_prob(ppgpp_conc, self.copy_number)
                if self.trna_attenuation:
                    basal_prob[self.attenuated_rna_indices] += (
                        self.attenuation_adjustments
                    )
                self.fracActiveRnap = self.get_rnap_active_fraction_from_ppGpp(
                    ppgpp_conc
                )
                ppgpp_scale = basal_prob[TU_index]
                ppgpp_scale[ppgpp_scale == 0] = 1
            else:
                basal_prob = self.basal_prob
                self.fracActiveRnap = self.fracActiveRnapDict[current_media_id]
                ppgpp_scale = 1

            promoter_init_probs = basal_prob[TU_index] + ppgpp_scale * np.multiply(
                self.delta_prob_matrix[TU_index, :], bound_TF
            ).sum(axis=1)

            if len(self.genetic_perturbations) > 0:
                self._rescale_initiation_probs(
                    self.genetic_perturbations["fixedRnaIdxs"],
                    self.genetic_perturbations["fixedSynthProbs"],
                    TU_index,
                    promoter_init_probs,
                )

            promoter_init_probs[promoter_init_probs < 0] = 0.0
            promoter_init_probs /= promoter_init_probs.sum()

            if not self.ppgpp_regulation:
                synthProbFractions = self.rnaSynthProbFractions[current_media_id]
                is_mrna = np.isin(TU_index, self.idx_mRNA)
                is_trna = np.isin(TU_index, self.idx_tRNA)
                is_rrna = np.isin(TU_index, self.idx_rRNA)
                is_rprotein = np.isin(TU_index, self.idx_rprotein)
                is_rnap = np.isin(TU_index, self.idx_rnap)
                is_fixed = is_trna | is_rrna | is_rprotein | is_rnap

                promoter_init_probs[is_mrna] *= (
                    synthProbFractions["mRna"] / promoter_init_probs[is_mrna].sum()
                )
                promoter_init_probs[is_trna] *= (
                    synthProbFractions["tRna"] / promoter_init_probs[is_trna].sum()
                )
                promoter_init_probs[is_rrna] *= (
                    synthProbFractions["rRna"] / promoter_init_probs[is_rrna].sum()
                )

                self._rescale_initiation_probs(
                    np.concatenate((self.idx_rprotein, self.idx_rnap)),
                    np.concatenate(
                        (
                            self.rnaSynthProbRProtein[current_media_id],
                            self.rnaSynthProbRnaPolymerase[current_media_id],
                        )
                    ),
                    TU_index,
                    promoter_init_probs,
                )

                assert promoter_init_probs[is_fixed].sum() < 1.0

                scaleTheRestBy = (
                    1.0 - promoter_init_probs[is_fixed].sum()
                ) / promoter_init_probs[~is_fixed].sum()
                promoter_init_probs[~is_fixed] *= scaleTheRestBy
        else:
            promoter_init_probs = np.zeros(
                states["promoters"]["_entryState"].sum()
            )

        self.rnaPolymeraseElongationRate = self.rnaPolymeraseElongationRateDict[
            current_media_id
        ]
        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.rnaPolymeraseElongationRate.asNumber(units.nt / units.s),
            1,
            self.variable_elongation,
        )
        # Store for potential reuse
        self.promoter_init_probs = promoter_init_probs
        return requests

    def evolve_state(self, timestep, states, promoter_init_probs):
        """Evolve state given the computed promoter_init_probs."""
        self._init_indices(states["bulk"]["id"])

        update = {
            "listeners": {
                "rna_synth_prob": {
                    "target_rna_synth_prob": np.zeros(self.n_TUs),
                    "actual_rna_synth_prob": np.zeros(self.n_TUs),
                    "tu_is_overcrowded": np.zeros(self.n_TUs, dtype=np.bool_),
                    "total_rna_init": 0,
                    "max_p": 0.0,
                },
                "ribosome_data": {"total_rna_init": 0},
                "rnap_data": {
                    "did_initialize": 0,
                    "rna_init_event": np.zeros(self.n_TUs, dtype=np.int64),
                },
            },
            "active_RNAPs": {},
            "full_chromosomes": {},
            "promoters": {},
            "RNAs": {},
        }

        if len(states["full_chromosomes"]) == 0:
            return update

        TU_index, domain_index_promoters = attrs(
            states["promoters"], ["TU_index", "domain_index"]
        )

        n_promoters = states["promoters"]["_entryState"].sum()
        TU_to_promoter = scipy.sparse.csr_matrix(
            (np.ones(n_promoters), (TU_index, np.arange(n_promoters))),
            shape=(self.n_TUs, n_promoters),
            dtype=np.int8,
        )

        target_TU_synth_probs = TU_to_promoter.dot(promoter_init_probs)
        update["listeners"]["rna_synth_prob"]["target_rna_synth_prob"] = (
            target_TU_synth_probs
        )

        self.activationProb = self._calculateActivationProb(
            states["timestep"],
            self.fracActiveRnap,
            self.rnaLengths,
            (units.nt / units.s) * self.elongation_rates,
            target_TU_synth_probs,
        )

        n_RNAPs_to_activate = np.int64(
            self.activationProb * counts(states["bulk"], self.inactive_RNAP_idx)
        )

        if n_RNAPs_to_activate == 0:
            return update

        max_p = (
            self.rnaPolymeraseElongationRate
            / self.active_rnap_footprint_size
            * (units.s)
            * states["timestep"]
            / n_RNAPs_to_activate
        ).asNumber()
        update["listeners"]["rna_synth_prob"]["max_p"] = max_p
        is_overcrowded = promoter_init_probs > max_p

        while np.any(promoter_init_probs > max_p):
            promoter_init_probs[is_overcrowded] = max_p
            scale_the_rest_by = (
                1.0 - promoter_init_probs[is_overcrowded].sum()
            ) / promoter_init_probs[~is_overcrowded].sum()
            promoter_init_probs[~is_overcrowded] *= scale_the_rest_by
            is_overcrowded |= promoter_init_probs > max_p

        actual_TU_synth_probs = TU_to_promoter.dot(promoter_init_probs)
        tu_is_overcrowded = TU_to_promoter.dot(is_overcrowded).astype(bool)
        update["listeners"]["rna_synth_prob"]["actual_rna_synth_prob"] = (
            actual_TU_synth_probs
        )
        update["listeners"]["rna_synth_prob"]["tu_is_overcrowded"] = tu_is_overcrowded

        n_initiations = self.random_state.multinomial(
            n_RNAPs_to_activate, promoter_init_probs
        )

        TU_index_partial_RNAs = np.repeat(TU_index, n_initiations)
        domain_index_rnap = np.repeat(domain_index_promoters, n_initiations)

        coordinates = self.replication_coordinate[TU_index_partial_RNAs]
        is_forward = self.transcription_direction[TU_index_partial_RNAs]

        RNAP_indexes = create_unique_indices(n_RNAPs_to_activate, states["RNAs"])
        update["active_RNAPs"].update(
            {
                "add": {
                    "unique_index": RNAP_indexes,
                    "domain_index": domain_index_rnap,
                    "coordinates": coordinates,
                    "is_forward": is_forward,
                }
            }
        )

        update["bulk"] = [(self.inactive_RNAP_idx, -n_initiations.sum())]

        is_mRNA = np.isin(TU_index_partial_RNAs, self.idx_mRNA)
        update["RNAs"].update(
            {
                "add": {
                    "TU_index": TU_index_partial_RNAs,
                    "transcript_length": np.zeros(cast(int, n_RNAPs_to_activate)),
                    "is_mRNA": is_mRNA,
                    "is_full_transcript": np.zeros(
                        cast(int, n_RNAPs_to_activate), dtype=bool
                    ),
                    "can_translate": is_mRNA,
                    "RNAP_index": RNAP_indexes,
                }
            }
        )

        rna_init_event = TU_to_promoter.dot(n_initiations)
        rRNA_initiations = rna_init_event[self.idx_rRNA]

        update["listeners"]["ribosome_data"] = {
            "rRNA_initiated_TU": rRNA_initiations.astype(int),
            "rRNA_init_prob_TU": rRNA_initiations / float(n_RNAPs_to_activate),
            "total_rna_init": n_RNAPs_to_activate,
        }

        update["listeners"]["rnap_data"] = {
            "did_initialize": n_RNAPs_to_activate,
            "rna_init_event": rna_init_event.astype(np.int64),
        }

        update["listeners"]["rna_synth_prob"]["total_rna_init"] = n_RNAPs_to_activate

        return update

    def _calculateActivationProb(
        self, timestep, fracActiveRnap, rnaLengths,
        rnaPolymeraseElongationRates, synthProb,
    ):
        allTranscriptionTimes = 1.0 / rnaPolymeraseElongationRates * rnaLengths
        timesteps = (1.0 / (timestep * units.s) * allTranscriptionTimes).asNumber()
        allTranscriptionTimestepCounts = np.ceil(timesteps)
        averageTranscriptionTimestepCounts = np.dot(
            synthProb, allTranscriptionTimestepCounts
        )
        expectedTerminationRate = 1.0 / averageTranscriptionTimestepCounts

        allFractionTimeInactive = (
            1
            - (1.0 / (timestep * units.s) * allTranscriptionTimes).asNumber()
            / allTranscriptionTimestepCounts
        )
        averageFractionTimeInactive = np.dot(allFractionTimeInactive, synthProb)
        effectiveFracActiveRnap = fracActiveRnap / (1 - averageFractionTimeInactive)

        activation_prob = (
            effectiveFracActiveRnap
            * expectedTerminationRate
            / (1 - effectiveFracActiveRnap)
        )

        if activation_prob > 1:
            activation_prob = 1

        return activation_prob

    def _rescale_initiation_probs(self, fixed_indexes, fixed_synth_probs, TU_index, promoter_init_probs):
        for idx, synth_prob in zip(fixed_indexes, fixed_synth_probs):
            fixed_mask = TU_index == idx
            promoter_init_probs[fixed_mask] = synth_prob / fixed_mask.sum()


class TranscriptInitiationRequester(Step):
    """Requester step for transcript initiation.

    Calculates promoter initiation probabilities and requests inactive RNAPs.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop("_logic", None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = TranscriptInitiationLogic(config)
        self.process_name = 'ecoli-transcript-initiation'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'full_chromosomes': InPlaceDict(),
            'promoters': InPlaceDict(),
            'listeners': InPlaceDict(),
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

        return {
            'request': {self.process_name: requests},
        }


class TranscriptInitiationEvolver(Step):
    """Evolver step for transcript initiation.

    Reads allocated bulk molecules, activates RNAPs, creates new RNAs.
    RECOMPUTES promoter_init_probs since Requester and Evolver are separate instances.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance from Requester
        self.process = config.pop("_logic", None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = TranscriptInitiationLogic(config)
        self.process_name = 'ecoli-transcript-initiation'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'allocate': InPlaceDict(),
            'environment': InPlaceDict(),
            'full_chromosomes': InPlaceDict(),
            'RNAs': InPlaceDict(),
            'active_RNAPs': InPlaceDict(),
            'promoters': InPlaceDict(),
            'listeners': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(_default=1.0),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'active_RNAPs': InPlaceDict(),
            'RNAs': InPlaceDict(),
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

        # Apply allocation: replace bulk counts with allocated amounts
        allocation = state.pop('allocate', {})
        bulk_alloc = allocation.get('bulk')
        if bulk_alloc is not None and hasattr(bulk_alloc, '__len__') and len(bulk_alloc) > 0 and hasattr(state.get('bulk'), 'dtype'):
            import numpy as np
            alloc_bulk = state['bulk'].copy()
            alloc_bulk['count'][:] = np.array(bulk_alloc, dtype=alloc_bulk['count'].dtype)
            state['bulk'] = alloc_bulk

        # Use promoter_init_probs cached by Requester (shared Logic instance)
        promoter_init_probs = proc.promoter_init_probs.copy()

        # Evolve state
        update = proc.evolve_state(timestep, state, promoter_init_probs)
        update['next_update_time'] = state.get('global_time', 0) + state.get('timestep', 1.0)
        return update


# Keep legacy class for backward compatibility during migration
from v2ecoli.steps.partition import PartitionedProcess

class TranscriptInitiation(PartitionedProcess):
    """Legacy PartitionedProcess wrapper -- will be removed after migration."""

    name = "ecoli-transcript-initiation"
    topology = {
        "environment": ("environment",),
        "full_chromosomes": ("unique", "full_chromosome"),
        "RNAs": ("unique", "RNA"),
        "active_RNAPs": ("unique", "active_RNAP"),
        "promoters": ("unique", "promoter"),
        "bulk": ("bulk",),
        "listeners": ("listeners",),
        "timestep": ("timestep",),
    }
    defaults = {
        "fracActiveRnapDict": {},
        "rnaLengths": np.array([]),
        "rnaPolymeraseElongationRateDict": {},
        "variable_elongation": False,
        "make_elongation_rates": (
            lambda random, rate, timestep, variable: np.array([])
        ),
        "active_rnap_foorprint_size": 1,
        "basal_prob": np.array([]),
        "delta_prob": {"deltaI": [], "deltaJ": [], "deltaV": [], "shape": tuple()},
        "get_delta_prob_matrix": None,
        "perturbations": {},
        "rna_data": {},
        "active_rnap_footprint_size": 24 * units.nt,
        "get_rnap_active_fraction_from_ppGpp": lambda x: 0.1,
        "idx_rRNA": np.array([]),
        "idx_mRNA": np.array([]),
        "idx_tRNA": np.array([]),
        "idx_rprotein": np.array([]),
        "idx_rnap": np.array([]),
        "rnaSynthProbFractions": {},
        "rnaSynthProbRProtein": {},
        "rnaSynthProbRnaPolymerase": {},
        "replication_coordinate": np.array([]),
        "transcription_direction": np.array([]),
        "n_avogadro": 6.02214076e23 / units.mol,
        "cell_density": 1100 * units.g / units.L,
        "ppgpp": "ppGpp",
        "inactive_RNAP": "APORNAP-CPLX[c]",
        "synth_prob": lambda concentration, copy: 0.0,
        "copy_number": lambda x: x,
        "ppgpp_regulation": False,
        "trna_attenuation": False,
        "attenuated_rna_indices": np.array([]),
        "attenuation_adjustments": np.array([]),
        "seed": 0,
        "emit_unique": False,
    }

    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)
        self._logic = TranscriptInitiationLogic(self.parameters)
        self.n_TUs = self._logic.n_TUs
        self.rna_data = self._logic.rna_data
        self.idx_rRNA = self._logic.idx_rRNA

    def ports_schema(self):
        return {
            "environment": {"media_id": {"_default": "", "_updater": "set"}},
            "bulk": numpy_schema("bulk"),
            "full_chromosomes": numpy_schema(
                "full_chromosomes", emit=self.parameters["emit_unique"]
            ),
            "promoters": numpy_schema("promoters", emit=self.parameters["emit_unique"]),
            "RNAs": numpy_schema("RNAs", emit=self.parameters["emit_unique"]),
            "active_RNAPs": numpy_schema(
                "active_RNAPs", emit=self.parameters["emit_unique"]
            ),
            "listeners": {
                "mass": {"cell_mass": {"_default": 0.0}, "dry_mass": {"_default": 0.0}},
                "rna_synth_prob": listener_schema(
                    {
                        "target_rna_synth_prob": [0.0],
                        "actual_rna_synth_prob": [0.0],
                        "tu_is_overcrowded": (
                            [False] * self.n_TUs,
                            self.rna_data["id"],
                        ),
                        "total_rna_init": 0,
                        "max_p": 0.0,
                    }
                ),
                "ribosome_data": listener_schema(
                    {
                        "rRNA_initiated_TU": [0] * len(self.idx_rRNA),
                        "rRNA_init_prob_TU": [0.0] * len(self.idx_rRNA),
                        "total_rna_init": 0,
                    }
                ),
                "rnap_data": listener_schema(
                    {"did_initialize": 0, "rna_init_event": (0, self.rna_data["id"])}
                ),
            },
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        return self._logic.calculate_request(timestep, states)

    def evolve_state(self, timestep, states):
        return self._logic.evolve_state(timestep, states, self._logic.promoter_init_probs)
