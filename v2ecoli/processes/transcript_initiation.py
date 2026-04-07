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

from v2ecoli.steps.partition import RequesterBase, EvolverBase
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore


class TranscriptInitiationLogic:
    """Transcript Initiation — shared state container for Requester/Evolver.

    **Defaults:**

    - **fracActiveRnapDict** (``dict``): Dictionary with keys corresponding to
      media, values being the fraction of active RNA Polymerase (RNAP)
      for that media.
    - **rnaLengths** (``numpy.ndarray[int]``): lengths of RNAs for each transcription
      unit (TU), in nucleotides
    - **rnaPolymeraseElongationRateDict** (``dict``): Dictionary with keys
      corresponding to media, values being RNAP's elongation rate in
      that media, in nucleotides/s
    - **variable_elongation** (``bool``): Whether to add amplified elongation rates
      for rRNAs. False by default.
    - **make_elongation_rates** (``func``): Function for making elongation rates
      (see :py:meth:`~reconstruction.ecoli.dataclasses.process.transcription.Transcription.make_elongation_rates`).
      Takes PRNG, basal elongation rate, timestep, and ``variable_elongation``.
      Returns an array of elongation rates for all genes.
    - **active_rnap_footprint_size** (``int``): Maximum physical footprint of RNAP
      in nucleotides to cap initiation probabilities
    - **basal_prob** (``numpy.ndarray[float]``): Baseline probability of synthesis for
      every TU.
    - **delta_prob** (``dict``): Dictionary with four keys, used to create a matrix
      encoding the effect of transcription factors (TFs) on transcription
      probabilities::

        {'deltaV' (np.ndarray[float]): deltas associated with the effects of
            TFs on TUs,
        'deltaI' (np.ndarray[int]): index of the affected TU for each delta,
        'deltaJ' (np.ndarray[int]): index of the acting TF for each delta,
        'shape' (tuple): (m, n) = (# of TUs, # of TFs)}

    - **perturbations** (``dict``): Dictionary of genetic perturbations (optional,
      can be empty)
    - **rna_data** (``numpy.ndarray``): Structured array with an entry for each TU,
      where entries look like::

            (id, deg_rate, length (nucleotides), counts_AGCU, mw
            (molecular weight), is_mRNA, is_miscRNA, is_rRNA, is_tRNA,
            is_23S_rRNA, is_16S_rRNA, is_5S_rRNA, is_ribosomal_protein,
            is_RNAP, gene_id, Km_endoRNase, replication_coordinate,
            direction)

    - **idx_rRNA** (``numpy.ndarray[int]``): indexes of TUs corresponding to rRNAs
    - **idx_mRNA** (``numpy.ndarray[int]``): indexes of TUs corresponding to mRNAs
    - **idx_tRNA** (``numpy.ndarray[int]``): indexes of TUs corresponding to tRNAs
    - **idx_rprotein** (``numpy.ndarray[int]``): indexes of TUs corresponding ribosomal
      proteins
    - **idx_rnap** (``numpy.ndarray[int]``): indexes of TU corresponding to RNAP
    - **rnaSynthProbFractions** (``dict``): Dictionary where keys are media types,
      values are sub-dictionaries with keys 'mRna', 'tRna', 'rRna', and
      values being probabilities of synthesis for each RNA type
    - **rnaSynthProbRProtein** (``dict``): Dictionary where keys are media types,
      values are arrays storing the (fixed) probability of synthesis for
      each rProtein TU, under that media condition.
    - **rnaSynthProbRnaPolymerase** (``dict``): Dictionary where keys are media
      types, values are arrays storing the (fixed) probability of
      synthesis for each RNAP TU, under that media condition.
    - **replication_coordinate** (``numpy.ndarray[int]``): Array with chromosome
      coordinates for each TU
    - **transcription_direction** (``numpy.ndarray[bool]``): Array of transcription
      directions for each TU
    - **n_avogadro** (``unum.Unum``): Avogadro's number (constant)
    - **cell_density** (``unum.Unum``): Density of cell (constant)
    - **ppgpp** (``str``): id of ppGpp
    - **inactive_RNAP** (``str``): id of inactive RNAP
    - **synth_prob** (``Callable[[Unum, int], numpy.ndarrray[float]]``):
      Function used in model of ppGpp regulation
      (see :py:func:`~reconstruction.ecoli.dataclasses.process.transcription.Transcription.synth_prob_from_ppgpp`).
      Takes ppGpp concentration (mol/volume) and copy number, returns
      normalized synthesis probability for each gene
    - **copy_number** (``Callable[[Unum, int], numpy.ndarrray[float]]``):
      see :py:func:`~reconstruction.ecoli.dataclasses.process.replication.Replication.get_average_copy_number`.
      Takes expected doubling time in minutes and chromosome coordinates of genes,
      returns average copy number of each gene expected at doubling time
    - **ppgpp_regulation** (``bool``): Whether to include model of ppGpp regulation
    - **get_rnap_active_fraction_from_ppGpp** (``Callable[[Unum], float]``):
      Returns elongation rate for a given ppGpp concentration
    - **seed** (``int``): random seed to initialize PRNG
    """

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
        # attenuation
        "trna_attenuation": False,
        "attenuated_rna_indices": np.array([]),
        "attenuation_adjustments": np.array([]),
        # random seed
        "seed": 0,
        "emit_unique": False,
        "time_step": 1,
    }

    # Constructor
    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.request_set = False

        # Load parameters
        self.fracActiveRnapDict = self.parameters["fracActiveRnapDict"]
        self.rnaLengths = self.parameters["rnaLengths"]
        self.rnaPolymeraseElongationRateDict = self.parameters[
            "rnaPolymeraseElongationRateDict"
        ]
        self.variable_elongation = self.parameters["variable_elongation"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]
        self.active_rnap_footprint_size = self.parameters["active_rnap_footprint_size"]

        # Initialize matrices used to calculate synthesis probabilities
        self.basal_prob = self.parameters["basal_prob"].copy()
        self.trna_attenuation = self.parameters["trna_attenuation"]
        if self.trna_attenuation:
            self.attenuated_rna_indices = self.parameters["attenuated_rna_indices"]
            self.attenuation_adjustments = self.parameters["attenuation_adjustments"]
            self.basal_prob[self.attenuated_rna_indices] += self.attenuation_adjustments

        self.n_TUs = len(self.basal_prob)
        self.delta_prob = self.parameters["delta_prob"]
        if self.parameters["get_delta_prob_matrix"] is not None:
            self.delta_prob_matrix = self.parameters["get_delta_prob_matrix"](
                dense=True, ppgpp=self.parameters["ppgpp_regulation"]
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
        self.perturbations = self.parameters["perturbations"]
        self.rna_data = self.parameters["rna_data"]

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
        self.idx_rRNA = self.parameters["idx_rRNA"]
        self.idx_mRNA = self.parameters["idx_mRNA"]
        self.idx_tRNA = self.parameters["idx_tRNA"]
        self.idx_rprotein = self.parameters["idx_rprotein"]
        self.idx_rnap = self.parameters["idx_rnap"]

        # Synthesis probabilities for different categories of genes
        self.rnaSynthProbFractions = self.parameters["rnaSynthProbFractions"]
        self.rnaSynthProbRProtein = self.parameters["rnaSynthProbRProtein"]
        self.rnaSynthProbRnaPolymerase = self.parameters["rnaSynthProbRnaPolymerase"]

        # Coordinates and transcription directions of transcription units
        self.replication_coordinate = self.parameters["replication_coordinate"]
        self.transcription_direction = self.parameters["transcription_direction"]

        self.inactive_RNAP = self.parameters["inactive_RNAP"]

        # ppGpp control related
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]
        self.ppgpp = self.parameters["ppgpp"]
        self.synth_prob = self.parameters["synth_prob"]
        self.copy_number = self.parameters["copy_number"]
        self.ppgpp_regulation = self.parameters["ppgpp_regulation"]
        self.get_rnap_active_fraction_from_ppGpp = self.parameters[
            "get_rnap_active_fraction_from_ppGpp"
        ]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.ppgpp_idx = None

    def _calculateActivationProb(
        self,
        timestep,
        fracActiveRnap,
        rnaLengths,
        rnaPolymeraseElongationRates,
        synthProb,
    ):
        """
        Calculate expected RNAP termination rate based on RNAP elongation rate
        - allTranscriptionTimes: Vector of times required to transcribe each
        transcript
        - allTranscriptionTimestepCounts: Vector of numbers of timesteps
        required to transcribe each transcript
        - averageTranscriptionTimeStepCounts: Average number of timesteps
        required to transcribe a transcript, weighted by synthesis
        probabilities of each transcript
        - expectedTerminationRate: Average number of terminations in one
        timestep for one transcript
        """
        allTranscriptionTimes = 1.0 / rnaPolymeraseElongationRates * rnaLengths
        timesteps = (1.0 / (timestep * units.s) * allTranscriptionTimes).asNumber()
        allTranscriptionTimestepCounts = np.ceil(timesteps)
        averageTranscriptionTimestepCounts = np.dot(
            synthProb, allTranscriptionTimestepCounts
        )
        expectedTerminationRate = 1.0 / averageTranscriptionTimestepCounts

        """
        Modify given fraction of active RNAPs to take into account early
        terminations in between timesteps
        - allFractionTimeInactive: Vector of probabilities an "active" RNAP
        will in effect be "inactive" because it has terminated during a
        timestep
        - averageFractionTimeInactive: Average probability of an "active" RNAP
        being in effect "inactive", weighted by synthesis probabilities
        - effectiveFracActiveRnap: New higher "goal" for fraction of active
        RNAP, considering that the "effective" fraction is lower than what the
        listener sees
        """
        allFractionTimeInactive = (
            1
            - (1.0 / (timestep * units.s) * allTranscriptionTimes).asNumber()
            / allTranscriptionTimestepCounts
        )
        averageFractionTimeInactive = np.dot(allFractionTimeInactive, synthProb)
        effectiveFracActiveRnap = fracActiveRnap / (1 - averageFractionTimeInactive)

        # Return activation probability that will balance out the expected termination rate
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
        a predetermined value. For instance, if there are two copies of
        promoters for RNA A, whose synthesis probability should be fixed to
        0.1, each promoter is given an initiation probability of 0.05.
        """
        for idx, synth_prob in zip(fixed_indexes, fixed_synth_probs):
            fixed_mask = TU_index == idx
            self.promoter_init_probs[fixed_mask] = synth_prob / fixed_mask.sum()


class TranscriptInitiationRequester(RequesterBase):
    """Reads stores to compute transcript initiation request."""

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'full_chromosomes': UniqueNumpyUpdate(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
            'promoters': UniqueNumpyUpdate(),
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

    def compute_request(self, state, timestep):
        p = self.process
        # At first update, convert all strings to indices
        if p.ppgpp_idx is None:
            bulk_ids = state["bulk"]["id"]
            p.ppgpp_idx = bulk_name_to_idx(p.ppgpp, bulk_ids)
            p.inactive_RNAP_idx = bulk_name_to_idx(p.inactive_RNAP, bulk_ids)

        # Get all inactive RNA polymerases
        request = {
            "bulk": [
                (p.inactive_RNAP_idx, counts(state["bulk"], p.inactive_RNAP_idx))
            ]
        }

        # Read current environment
        current_media_id = state["environment"]["media_id"]

        if state["full_chromosomes"]["_entryState"].sum() > 0:
            # Get attributes of promoters
            TU_index, bound_TF = attrs(state["promoters"], ["TU_index", "bound_TF"])

            if p.ppgpp_regulation:
                cell_mass = state["listeners"]["mass"]["cell_mass"] * units.fg
                cell_volume = cell_mass / p.cell_density
                counts_to_molar = 1 / (p.n_avogadro * cell_volume)
                ppgpp_conc = counts(state["bulk"], p.ppgpp_idx) * counts_to_molar
                basal_prob, _ = p.synth_prob(ppgpp_conc, p.copy_number)
                if p.trna_attenuation:
                    basal_prob[p.attenuated_rna_indices] += (
                        p.attenuation_adjustments
                    )
                p.fracActiveRnap = p.get_rnap_active_fraction_from_ppGpp(
                    ppgpp_conc
                )
                ppgpp_scale = basal_prob[TU_index]
                # Use original delta prob if no ppGpp basal
                ppgpp_scale[ppgpp_scale == 0] = 1
            else:
                basal_prob = p.basal_prob
                p.fracActiveRnap = p.fracActiveRnapDict[current_media_id]
                ppgpp_scale = 1

            # Calculate probabilities of the RNAP binding to each promoter
            p.promoter_init_probs = basal_prob[TU_index] + ppgpp_scale * np.multiply(
                p.delta_prob_matrix[TU_index, :], bound_TF
            ).sum(axis=1)

            if len(p.genetic_perturbations) > 0:
                p._rescale_initiation_probs(
                    p.genetic_perturbations["fixedRnaIdxs"],
                    p.genetic_perturbations["fixedSynthProbs"],
                    TU_index,
                )

            # Adjust probabilities to not be negative
            p.promoter_init_probs[p.promoter_init_probs < 0] = 0.0
            p.promoter_init_probs /= p.promoter_init_probs.sum()

            if not p.ppgpp_regulation:
                # Adjust synthesis probabilities depending on environment
                synthProbFractions = p.rnaSynthProbFractions[current_media_id]

                # Create masks for different types of RNAs
                is_mrna = np.isin(TU_index, p.idx_mRNA)
                is_trna = np.isin(TU_index, p.idx_tRNA)
                is_rrna = np.isin(TU_index, p.idx_rRNA)
                is_rprotein = np.isin(TU_index, p.idx_rprotein)
                is_rnap = np.isin(TU_index, p.idx_rnap)
                is_fixed = is_trna | is_rrna | is_rprotein | is_rnap

                # Rescale initiation probabilities based on type of RNA
                p.promoter_init_probs[is_mrna] *= (
                    synthProbFractions["mRna"] / p.promoter_init_probs[is_mrna].sum()
                )
                p.promoter_init_probs[is_trna] *= (
                    synthProbFractions["tRna"] / p.promoter_init_probs[is_trna].sum()
                )
                p.promoter_init_probs[is_rrna] *= (
                    synthProbFractions["rRna"] / p.promoter_init_probs[is_rrna].sum()
                )

                # Set fixed synthesis probabilities for RProteins and RNAPs
                p._rescale_initiation_probs(
                    np.concatenate((p.idx_rprotein, p.idx_rnap)),
                    np.concatenate(
                        (
                            p.rnaSynthProbRProtein[current_media_id],
                            p.rnaSynthProbRnaPolymerase[current_media_id],
                        )
                    ),
                    TU_index,
                )

                assert p.promoter_init_probs[is_fixed].sum() < 1.0

                # Scale remaining synthesis probabilities accordingly
                scaleTheRestBy = (
                    1.0 - p.promoter_init_probs[is_fixed].sum()
                ) / p.promoter_init_probs[~is_fixed].sum()
                p.promoter_init_probs[~is_fixed] *= scaleTheRestBy

        # If there are no chromosomes in the cell, set all probs to zero
        else:
            p.promoter_init_probs = np.zeros(
                state["promoters"]["_entryState"].sum()
            )

        p.rnaPolymeraseElongationRate = p.rnaPolymeraseElongationRateDict[
            current_media_id
        ]
        p.elongation_rates = p.make_elongation_rates(
            p.random_state,
            p.rnaPolymeraseElongationRate.asNumber(units.nt / units.s),
            1,  # want elongation rate, not lengths adjusted for time step
            p.variable_elongation,
        )
        return request


class TranscriptInitiationEvolver(EvolverBase):
    """Reads allocation, writes bulk/unique/listener updates."""

    def inputs(self):
        return {
            'allocate': InPlaceDict(),
            'bulk': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'listeners': ListenerStore(),
            'full_chromosomes': UniqueNumpyUpdate(),
            'RNAs': UniqueNumpyUpdate(),
            'active_RNAPs': UniqueNumpyUpdate(),
            'promoters': UniqueNumpyUpdate(),
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

    def compute_evolve(self, state, timestep):
        p = self.process
        update = {
            "listeners": {
                "rna_synth_prob": {
                    "target_rna_synth_prob": np.zeros(p.n_TUs),
                    "actual_rna_synth_prob": np.zeros(p.n_TUs),
                    "tu_is_overcrowded": np.zeros(p.n_TUs, dtype=np.bool_),
                    "total_rna_init": 0,
                    "max_p": 0.0,
                },
                "ribosome_data": {"total_rna_init": 0},
                "rnap_data": {
                    "did_initialize": 0,
                    "rna_init_event": np.zeros(p.n_TUs, dtype=np.int64),
                },
            },
            "active_RNAPs": {},
            "full_chromosomes": {},
            "promoters": {},
            "RNAs": {},
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
                shape=(p.n_TUs, n_promoters),
                dtype=np.int8,
            )

            # Compute target synthesis probabilities of each transcription unit
            target_TU_synth_probs = TU_to_promoter.dot(p.promoter_init_probs)
            update["listeners"]["rna_synth_prob"]["target_rna_synth_prob"] = (
                target_TU_synth_probs
            )

            # Calculate RNA polymerases to activate based on probabilities
            p.activationProb = p._calculateActivationProb(
                state["timestep"],
                p.fracActiveRnap,
                p.rnaLengths,
                (units.nt / units.s) * p.elongation_rates,
                target_TU_synth_probs,
            )

            n_RNAPs_to_activate = np.int64(
                p.activationProb * counts(state["bulk"], p.inactive_RNAP_idx)
            )

            if n_RNAPs_to_activate != 0:
                # Cap the initiation probabilities at the maximum level physically
                # allowed from the known RNAP footprint sizes
                max_p = (
                    p.rnaPolymeraseElongationRate
                    / p.active_rnap_footprint_size
                    * (units.s)
                    * state["timestep"]
                    / n_RNAPs_to_activate
                ).asNumber()
                update["listeners"]["rna_synth_prob"]["max_p"] = max_p
                is_overcrowded = p.promoter_init_probs > max_p

                while np.any(p.promoter_init_probs > max_p):
                    p.promoter_init_probs[is_overcrowded] = max_p
                    scale_the_rest_by = (
                        1.0 - p.promoter_init_probs[is_overcrowded].sum()
                    ) / p.promoter_init_probs[~is_overcrowded].sum()
                    p.promoter_init_probs[~is_overcrowded] *= scale_the_rest_by
                    is_overcrowded |= p.promoter_init_probs > max_p

                # Compute actual synthesis probabilities of each transcription unit
                actual_TU_synth_probs = TU_to_promoter.dot(p.promoter_init_probs)
                tu_is_overcrowded = TU_to_promoter.dot(is_overcrowded).astype(bool)
                update["listeners"]["rna_synth_prob"]["actual_rna_synth_prob"] = (
                    actual_TU_synth_probs
                )
                update["listeners"]["rna_synth_prob"]["tu_is_overcrowded"] = tu_is_overcrowded

                # Sample a multinomial distribution of initiation probabilities to
                # determine what promoters are initialized
                n_initiations = p.random_state.multinomial(
                    n_RNAPs_to_activate, p.promoter_init_probs
                )

                # Build array of transcription unit indexes for partially transcribed
                # RNAs and domain indexes for RNAPs
                TU_index_partial_RNAs = np.repeat(TU_index, n_initiations)
                domain_index_rnap = np.repeat(domain_index_promoters, n_initiations)

                # Build arrays of starting coordinates and transcription directions
                coordinates = p.replication_coordinate[TU_index_partial_RNAs]
                is_forward = p.transcription_direction[TU_index_partial_RNAs]

                # new RNAPs
                RNAP_indexes = create_unique_indices(n_RNAPs_to_activate, state["RNAs"])
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
                update["bulk"] = [(p.inactive_RNAP_idx, -n_initiations.sum())]

                # Add partially transcribed RNAs
                is_mRNA = np.isin(TU_index_partial_RNAs, p.idx_mRNA)
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
                rRNA_initiations = rna_init_event[p.idx_rRNA]

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
