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

from bigraph_schema.schema import Float, Overwrite

from v2ecoli.library.schema import (
    create_unique_indices,
    counts,
    attrs,
    bulk_name_to_idx,
    MetadataArray,
)

from v2ecoli.library.units import units
from v2ecoli.library.random import stochasticRound

from v2ecoli.steps.partition import _protect_state, _SafeInvokeMixin
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore


class TranscriptInitiationStep(_SafeInvokeMixin, Step):
    """Transcript Initiation — merged single-step process."""

    config_schema = {}

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

    def initialize(self, config):
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
            # attenuation
            "trna_attenuation": False,
            "attenuated_rna_indices": np.array([]),
            "attenuation_adjustments": np.array([]),
            # random seed
            "seed": 0,
            "emit_unique": False,
            "time_step": 1,
        }
        params = {**defaults, **config}

        # Load parameters
        self.fracActiveRnapDict = params["fracActiveRnapDict"]
        self.rnaLengths = params["rnaLengths"]
        self.rnaPolymeraseElongationRateDict = params[
            "rnaPolymeraseElongationRateDict"
        ]
        self.variable_elongation = params["variable_elongation"]
        self.make_elongation_rates = params["make_elongation_rates"]
        self.active_rnap_footprint_size = params["active_rnap_footprint_size"]

        # Initialize matrices used to calculate synthesis probabilities
        self.basal_prob = params["basal_prob"].copy()
        self.trna_attenuation = params["trna_attenuation"]
        if self.trna_attenuation:
            self.attenuated_rna_indices = params["attenuated_rna_indices"]
            self.attenuation_adjustments = params["attenuation_adjustments"]
            self.basal_prob[self.attenuated_rna_indices] += self.attenuation_adjustments

        self.n_TUs = len(self.basal_prob)
        self.delta_prob = params["delta_prob"]
        if params["get_delta_prob_matrix"] is not None:
            self.delta_prob_matrix = params["get_delta_prob_matrix"](
                dense=True, ppgpp=params["ppgpp_regulation"]
            )
        else:
            # make delta_prob_matrix without adjustments
            self.delta_prob_matrix = scipy.sparse.csr_matrix(
                (
                    self.delta_prob["deltaV"],
                    (self.delta_prob["deltaI"], self.delta_prob["deltaJ"]),
                ),
                shape=self.delta_prob["shape"],
            ).toarray()

        # Determine changes from genetic perturbations
        self.genetic_perturbations = {}
        self.perturbations = params["perturbations"]
        self.rna_data = params["rna_data"]

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
        self.idx_rRNA = params["idx_rRNA"]
        self.idx_mRNA = params["idx_mRNA"]
        self.idx_tRNA = params["idx_tRNA"]
        self.idx_rprotein = params["idx_rprotein"]
        self.idx_rnap = params["idx_rnap"]

        # Synthesis probabilities for different categories of genes
        self.rnaSynthProbFractions = params["rnaSynthProbFractions"]
        self.rnaSynthProbRProtein = params["rnaSynthProbRProtein"]
        self.rnaSynthProbRnaPolymerase = params["rnaSynthProbRnaPolymerase"]

        # Coordinates and transcription directions of transcription units
        self.replication_coordinate = params["replication_coordinate"]
        self.transcription_direction = params["transcription_direction"]

        self.inactive_RNAP = params["inactive_RNAP"]

        # ppGpp control related
        self.n_avogadro = params["n_avogadro"]
        self.cell_density = params["cell_density"]
        self.ppgpp = params["ppgpp"]
        self.synth_prob = params["synth_prob"]
        self.copy_number = params["copy_number"]
        self.ppgpp_regulation = params["ppgpp_regulation"]
        self.get_rnap_active_fraction_from_ppGpp = params[
            "get_rnap_active_fraction_from_ppGpp"
        ]

        self.seed = params["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing (lazy init)
        self.ppgpp_idx = None

        self.time_step = params["time_step"]

    # ------------------------------------------------------------------
    # Helper methods (moved from Logic class)
    # ------------------------------------------------------------------

    def _calculateActivationProb(
        self,
        timestep,
        fracActiveRnap,
        rnaLengths,
        rnaPolymeraseElongationRates,
        synthProb,
    ):
        """
        Calculate expected RNAP termination rate based on RNAP elongation rate.
        """
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

        # Return activation probability that will balance out the expected
        # termination rate
        activation_prob = (
            effectiveFracActiveRnap
            * expectedTerminationRate
            / (1 - effectiveFracActiveRnap)
        )

        if activation_prob > 1:
            activation_prob = 1

        return activation_prob

    def _rescale_initiation_probs(self, fixed_indexes, fixed_synth_probs, TU_index):
        """
        Rescales the initiation probabilities of each promoter such that the
        total synthesis probabilities of certain types of RNAs are fixed to
        a predetermined value.
        """
        for idx, synth_prob in zip(fixed_indexes, fixed_synth_probs):
            fixed_mask = TU_index == idx
            self.promoter_init_probs[fixed_mask] = synth_prob / fixed_mask.sum()

    # ------------------------------------------------------------------
    # Step interface
    # ------------------------------------------------------------------

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'full_chromosomes': UniqueNumpyUpdate(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
            'promoters': UniqueNumpyUpdate(),
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
        # Time-gating
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)
        timestep = state["timestep"]

        # ---- Lazy index initialization ----
        if self.ppgpp_idx is None:
            bulk_ids = state["bulk"]["id"]
            self.ppgpp_idx = bulk_name_to_idx(self.ppgpp, bulk_ids)
            self.inactive_RNAP_idx = bulk_name_to_idx(self.inactive_RNAP, bulk_ids)

        # ==================================================================
        # Phase 1: Requester logic — compute initiation probabilities
        # ==================================================================

        # Read current environment
        current_media_id = state["environment"]["media_id"]

        if state["full_chromosomes"]["_entryState"].sum() > 0:
            # Get attributes of promoters
            TU_index, bound_TF = attrs(state["promoters"], ["TU_index", "bound_TF"])

            if self.ppgpp_regulation:
                cell_mass = state["listeners"]["mass"]["cell_mass"] * units.fg
                cell_volume = cell_mass / self.cell_density
                counts_to_molar = 1 / (self.n_avogadro * cell_volume)
                ppgpp_conc = counts(state["bulk"], self.ppgpp_idx) * counts_to_molar
                basal_prob, _ = self.synth_prob(ppgpp_conc, self.copy_number)
                if self.trna_attenuation:
                    basal_prob[self.attenuated_rna_indices] += (
                        self.attenuation_adjustments
                    )
                fracActiveRnap = self.get_rnap_active_fraction_from_ppGpp(
                    ppgpp_conc
                )
                ppgpp_scale = basal_prob[TU_index]
                # Use original delta prob if no ppGpp basal
                ppgpp_scale[ppgpp_scale == 0] = 1
            else:
                basal_prob = self.basal_prob
                fracActiveRnap = self.fracActiveRnapDict[current_media_id]
                ppgpp_scale = 1

            # Calculate probabilities of the RNAP binding to each promoter
            self.promoter_init_probs = basal_prob[TU_index] + ppgpp_scale * np.multiply(
                self.delta_prob_matrix[TU_index, :], bound_TF
            ).sum(axis=1)

            if len(self.genetic_perturbations) > 0:
                self._rescale_initiation_probs(
                    self.genetic_perturbations["fixedRnaIdxs"],
                    self.genetic_perturbations["fixedSynthProbs"],
                    TU_index,
                )

            # Adjust probabilities to not be negative
            self.promoter_init_probs[self.promoter_init_probs < 0] = 0.0
            self.promoter_init_probs /= self.promoter_init_probs.sum()

            if not self.ppgpp_regulation:
                # Adjust synthesis probabilities depending on environment
                synthProbFractions = self.rnaSynthProbFractions[current_media_id]

                # Create masks for different types of RNAs
                is_mrna = np.isin(TU_index, self.idx_mRNA)
                is_trna = np.isin(TU_index, self.idx_tRNA)
                is_rrna = np.isin(TU_index, self.idx_rRNA)
                is_rprotein = np.isin(TU_index, self.idx_rprotein)
                is_rnap = np.isin(TU_index, self.idx_rnap)
                is_fixed = is_trna | is_rrna | is_rprotein | is_rnap

                # Rescale initiation probabilities based on type of RNA
                self.promoter_init_probs[is_mrna] *= (
                    synthProbFractions["mRna"] / self.promoter_init_probs[is_mrna].sum()
                )
                self.promoter_init_probs[is_trna] *= (
                    synthProbFractions["tRna"] / self.promoter_init_probs[is_trna].sum()
                )
                self.promoter_init_probs[is_rrna] *= (
                    synthProbFractions["rRna"] / self.promoter_init_probs[is_rrna].sum()
                )

                # Set fixed synthesis probabilities for RProteins and RNAPs
                self._rescale_initiation_probs(
                    np.concatenate((self.idx_rprotein, self.idx_rnap)),
                    np.concatenate(
                        (
                            self.rnaSynthProbRProtein[current_media_id],
                            self.rnaSynthProbRnaPolymerase[current_media_id],
                        )
                    ),
                    TU_index,
                )

                assert self.promoter_init_probs[is_fixed].sum() < 1.0

                # Scale remaining synthesis probabilities accordingly
                scaleTheRestBy = (
                    1.0 - self.promoter_init_probs[is_fixed].sum()
                ) / self.promoter_init_probs[~is_fixed].sum()
                self.promoter_init_probs[~is_fixed] *= scaleTheRestBy

        # If there are no chromosomes in the cell, set all probs to zero
        else:
            self.promoter_init_probs = np.zeros(
                state["promoters"]["_entryState"].sum()
            )

        rnaPolymeraseElongationRate = self.rnaPolymeraseElongationRateDict[
            current_media_id
        ]
        elongation_rates = self.make_elongation_rates(
            self.random_state,
            rnaPolymeraseElongationRate.asNumber(units.nt / units.s),
            1,  # want elongation rate, not lengths adjusted for time step
            self.variable_elongation,
        )

        # ==================================================================
        # Phase 2: Evolver logic — activate RNAPs using actual counts
        # ==================================================================

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
            "next_update_time": global_time + timestep,
        }

        # no synthesis if no chromosome
        if len(state["full_chromosomes"]) != 0:
            # Get attributes of promoters
            TU_index, domain_index_promoters = attrs(
                state["promoters"], ["TU_index", "domain_index"]
            )

            n_promoters = state["promoters"]["_entryState"].sum()
            # Construct matrix that maps promoters to transcription units
            TU_to_promoter = scipy.sparse.csr_matrix(
                (np.ones(n_promoters), (TU_index, np.arange(n_promoters))),
                shape=(self.n_TUs, n_promoters),
                dtype=np.int8,
            )

            # Compute target synthesis probabilities of each transcription unit
            target_TU_synth_probs = TU_to_promoter.dot(self.promoter_init_probs)
            update["listeners"]["rna_synth_prob"]["target_rna_synth_prob"] = (
                target_TU_synth_probs
            )

            # Calculate RNA polymerases to activate based on probabilities
            activationProb = self._calculateActivationProb(
                timestep,
                fracActiveRnap,
                self.rnaLengths,
                (units.nt / units.s) * elongation_rates,
                target_TU_synth_probs,
            )

            # Use actual inactive RNAP counts (no allocation)
            n_inactive_RNAPs = counts(state["bulk"], self.inactive_RNAP_idx)
            n_RNAPs_to_activate = np.int64(activationProb * n_inactive_RNAPs)

            if n_RNAPs_to_activate != 0:
                # Cap the initiation probabilities at the maximum level physically
                # allowed from the known RNAP footprint sizes
                max_p = (
                    rnaPolymeraseElongationRate
                    / self.active_rnap_footprint_size
                    * (units.s)
                    * timestep
                    / n_RNAPs_to_activate
                ).asNumber()
                update["listeners"]["rna_synth_prob"]["max_p"] = max_p
                is_overcrowded = self.promoter_init_probs > max_p

                while np.any(self.promoter_init_probs > max_p):
                    self.promoter_init_probs[is_overcrowded] = max_p
                    scale_the_rest_by = (
                        1.0 - self.promoter_init_probs[is_overcrowded].sum()
                    ) / self.promoter_init_probs[~is_overcrowded].sum()
                    self.promoter_init_probs[~is_overcrowded] *= scale_the_rest_by
                    is_overcrowded |= self.promoter_init_probs > max_p

                # Compute actual synthesis probabilities of each transcription unit
                actual_TU_synth_probs = TU_to_promoter.dot(self.promoter_init_probs)
                tu_is_overcrowded = TU_to_promoter.dot(is_overcrowded).astype(bool)
                update["listeners"]["rna_synth_prob"]["actual_rna_synth_prob"] = (
                    actual_TU_synth_probs
                )
                update["listeners"]["rna_synth_prob"]["tu_is_overcrowded"] = tu_is_overcrowded

                # Sample a multinomial distribution of initiation probabilities to
                # determine what promoters are initialized
                n_initiations = self.random_state.multinomial(
                    n_RNAPs_to_activate, self.promoter_init_probs
                )

                # Build array of transcription unit indexes for partially transcribed
                # RNAs and domain indexes for RNAPs
                TU_index_partial_RNAs = np.repeat(TU_index, n_initiations)
                domain_index_rnap = np.repeat(domain_index_promoters, n_initiations)

                # Build arrays of starting coordinates and transcription directions
                coordinates = self.replication_coordinate[TU_index_partial_RNAs]
                is_forward = self.transcription_direction[TU_index_partial_RNAs]

                # new RNAPs
                RNAP_indexes = create_unique_indices(n_RNAPs_to_activate, state["RNAs"])
                update.setdefault("active_RNAPs", {})
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

                # Decrement counts of inactive RNAPs
                update["bulk"] = [(self.inactive_RNAP_idx, -n_initiations.sum())]

                # Add partially transcribed RNAs
                is_mRNA = np.isin(TU_index_partial_RNAs, self.idx_mRNA)
                update.setdefault("RNAs", {})
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

                # Write outputs to listeners
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
