"""
=====================
Transcript Initiation
=====================

This process models the binding of RNA polymerase (RNAP) to each gene.

Mathematical Model
------------------
The number of RNAPs to activate per timestep is determined so that the
average active RNAP fraction matches measured values (media-dependent).

1. **Active RNAP target**: Given the media-dependent fraction f_active:

       n_to_activate = round(f_active * n_total_RNAP) - n_currently_active

2. **Synthesis probability**: Each transcription unit (TU) i has a
   probability of initiation p_i that combines a basal probability
   with TF-mediated modulation via a sparse delta_prob matrix:

       p_i = max(0, basal_prob_i + sum_j(delta_prob[i,j] * bound_TF_j))

   The delta_prob matrix is stored in COO format as (deltaV, deltaI,
   deltaJ, shape) and reconstructed into a CSR sparse matrix.

3. **Probabilistic initiation**: The n_to_activate RNAPs are distributed
   across TUs via multinomial sampling weighted by normalized p_i,
   subject to physical constraints:

   - **Footprint constraint**: An RNAP occupies ~24 nt at the promoter.
     Overcrowded TUs (where new initiation would overlap existing RNAPs)
     have their probability set to zero.

   - **Promoter availability**: Only TUs with at least one promoter on
     an existing chromosome can initiate.

4. **ppGpp regulation** (optional): When enabled, ppGpp concentration
   modulates both basal_prob and f_active via fitted functions,
   providing growth-rate-dependent transcriptional regulation.

TODO:
  - use transcription units instead of single genes
  - match sigma factors to promoters
"""

import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from typing import cast

# simulate_process removed

from v2ecoli.library.schema import (
    create_unique_indices,
    listener_schema,
    numpy_schema,
    counts,
    attrs,
    bulk_name_to_idx,
    MetadataArray,
)

from v2ecoli.types.quantity import ureg as units
from wholecell.utils.random import stochasticRound
from wholecell.utils.unit_struct_array import UnitStructArray

from v2ecoli.library.data_predicates import monotonically_decreasing, all_nonnegative
from scipy.stats import chisquare

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.schema_types import (
    FULL_CHROMOSOME_ARRAY,
    RNA_ARRAY,
    ACTIVE_RNAP_ARRAY,
    PROMOTER_ARRAY,
)


# Register default topology for this process, associating it with process name
NAME = "ecoli-transcript-initiation"
TOPOLOGY = {
    "environment": ("environment",),
    "full_chromosomes": ("unique", "full_chromosome"),
    "RNAs": ("unique", "RNA"),
    "active_RNAPs": ("unique", "active_RNAP"),
    "promoters": ("unique", "promoter"),
    "bulk": ("bulk",),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
    "ppgpp_state": ("ppgpp_state",),
}


class TranscriptInitiation(Step):
    """Transcript Initiation PartitionedProcess

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

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'active_rnap_foorprint_size': {'_type': 'integer', '_default': 1},
        'active_rnap_footprint_size': {'_type': 'quantity[nt]', '_default': 24.0},
        'attenuated_rna_indices': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'attenuation_adjustments': {'_type': 'array[float]', '_default': np.array([], dtype=float)},
        'basal_prob': {'_type': 'array[float]', '_default': np.array([], dtype=float)},
        'cell_density': {'_type': 'quantity[g/L]', '_default': 1100.0},
        'copy_number': {'_type': 'array[integer]', '_default': None},
        'delta_prob': {'_type': 'map[node]', '_default': {}},
        'emit_unique': {'_type': 'boolean', '_default': False},
        'fracActiveRnapDict': {'_type': 'map[float]', '_default': {}},
        'get_delta_prob_matrix': 'method',
        'get_rnap_active_fraction_from_ppGpp': {'_type': 'method', '_default': None},
        'idx_mRNA': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'idx_rRNA': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'idx_rnap': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'idx_rprotein': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'idx_tRNA': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'inactive_RNAP': {'_type': 'string', '_default': 'APORNAP-CPLX[c]'},
        'make_elongation_rates': {'_type': 'method', '_default': None},
        'n_avogadro': {'_type': 'quantity[float,1/mol]', '_default': 6.02214076e+23},
        'perturbations': {'_type': 'map[node]', '_default': {}},
        'ppgpp': {'_type': 'string', '_default': 'ppGpp'},
        'ppgpp_regulation': {'_type': 'boolean', '_default': False},
        'replication_coordinate': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'rnaLengths': {'_type': 'quantity[array[integer],nt]', '_default': np.array([], dtype=float)},
        'rnaPolymeraseElongationRateDict': {'_type': 'map[quantity[float,nucleotide/s]]', '_default': {}},
        'rnaSynthProbFractions': {'_type': 'map[float]', '_default': {}},
        'rnaSynthProbRProtein': {'_type': 'map[float]', '_default': {}},
        'rnaSynthProbRnaPolymerase': {'_type': 'map[float]', '_default': {}},
        'rna_data': {'_type': 'units_array', '_default': {}},
        'seed': {'_type': 'integer', '_default': 0},
        'synth_prob': {'_type': 'array[float]', '_default': None},
        'transcription_direction': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'trna_attenuation': {'_type': 'boolean', '_default': False},
        'variable_elongation': {'_type': 'boolean', '_default': False},
    }

    def initialize(self, config):

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

        self.n_TUs = len(self.basal_prob)
        self.delta_prob = self.parameters["delta_prob"]
        if self.parameters["get_delta_prob_matrix"] is not None:
            self.delta_prob_matrix = self.parameters["get_delta_prob_matrix"](
                dense=True, ppgpp=self.parameters.get("ppgpp_regulation", False)
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

        # Constants for volume calculations
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]
        self.ppgpp = self.parameters["ppgpp"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Helper indices for Numpy indexing
        self.ppgpp_idx = None

    def inputs(self):
        return {
            'environment': {
                'media_id': {'_type': 'string', '_default': ''},
            },
            'full_chromosomes': {'_type': FULL_CHROMOSOME_ARRAY, '_default': []},
            'RNAs': {'_type': RNA_ARRAY, '_default': []},
            'active_RNAPs': {'_type': ACTIVE_RNAP_ARRAY, '_default': []},
            'promoters': {'_type': PROMOTER_ARRAY, '_default': []},
            'bulk': {'_type': 'bulk_array', '_default': []},
            'listeners': {
                'mass': {
                    'cell_mass': {'_type': 'quantity[float,fg]', '_default': 0.0},
                },
            },
            'timestep': {'_type': 'integer', '_default': 1.0},
            'ppgpp_state': {
                'basal_prob': {'_type': 'array[float]', '_default': []},
                'frac_active_rnap': {'_type': 'float', '_default': 0.0},
            },
        }

    def outputs(self):
        return (
            {
                'bulk': 'bulk_array',
                'RNAs': RNA_ARRAY,
                'active_RNAPs': ACTIVE_RNAP_ARRAY,
                'listeners':                 {
                    'rna_synth_prob':                     {
                        'target_rna_synth_prob': {'_type': 'overwrite[array[float]]', '_default': []},
                        'actual_rna_synth_prob': {'_type': 'overwrite[array[float]]', '_default': []},
                        'max_p': {'_type': 'overwrite[float]', '_default': 0.0},
                        'tu_is_overcrowded': {'_type': 'overwrite[array[boolean]]', '_default': []},
                        'total_rna_init': {'_type': 'overwrite[integer]', '_default': 0},
                    },
                    'ribosome_data':                     {
                        'rRNA_initiated_TU': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'rRNA_init_prob_TU': {'_type': 'overwrite[array[float]]', '_default': []},
                        'total_rna_init': {'_type': 'overwrite[integer]', '_default': 0},
                    },
                    'rnap_data':                     {
                        'did_initialize': {'_type': 'overwrite[integer]', '_default': 0},
                        'rna_init_event': {'_type': 'overwrite[array[integer]]', '_default': []},
                    },
                },
            }
        )
    def update(self, states, interval=None):
        self._prepare(states)
        return self._evolve(states)

    def _prepare(self, states):
        # At first update, convert all strings to indices
        if self.ppgpp_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.ppgpp_idx = bulk_name_to_idx(self.ppgpp, bulk_ids)
            self.inactive_RNAP_idx = bulk_name_to_idx(self.inactive_RNAP, bulk_ids)

        # Read current environment
        current_media_id = states["environment"]["media_id"]

        if states["full_chromosomes"]["_entryState"].sum() > 0:
            # Get attributes of promoters
            TU_index, bound_TF = attrs(states["promoters"], ["TU_index", "bound_TF"])

            # Check if ppGpp regulation module has written state
            ppgpp = states.get("ppgpp_state", {})
            ppgpp_basal = ppgpp.get("basal_prob")
            has_ppgpp = (ppgpp_basal is not None
                         and hasattr(ppgpp_basal, '__len__')
                         and len(ppgpp_basal) > 0)

            if has_ppgpp:
                basal_prob = ppgpp_basal
                self.fracActiveRnap = ppgpp["frac_active_rnap"]
                ppgpp_scale = basal_prob[TU_index]
                ppgpp_scale[ppgpp_scale == 0] = 1
            else:
                basal_prob = self.basal_prob
                self.fracActiveRnap = self.fracActiveRnapDict[current_media_id]
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

            if not has_ppgpp:
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
                states["promoters"]["_entryState"].sum()
            )

        self.rnaPolymeraseElongationRate = self.rnaPolymeraseElongationRateDict[
            current_media_id
        ]
        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.rnaPolymeraseElongationRate.to(units.nt / units.s).magnitude,
            1,  # want elongation rate, not lengths adjusted for time step
            self.variable_elongation,
        )

    def _evolve(self, states):
        timestep = states["timestep"]
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

        # no synthesis if no chromosome
        if len(states["full_chromosomes"]) == 0:
            return update

        # Get attributes of promoters
        TU_index, domain_index_promoters = attrs(
            states["promoters"], ["TU_index", "domain_index"]
        )

        n_promoters = states["promoters"]["_entryState"].sum()
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
        # Note: ideally we should be using the actual TU synthesis probabilities
        # here, but the calculation of actual probabilities requires the number
        # of RNAPs to activate. The difference should be small.
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

        # Cap the initiation probabilities at the maximum level physically
        # allowed from the known RNAP footprint sizes
        max_p = (
            self.rnaPolymeraseElongationRate
            / self.active_rnap_footprint_size
            * (units.s)
            * states["timestep"]
            / n_RNAPs_to_activate
        ).magnitude
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

        # Decrement counts of inactive RNAPs
        update["bulk"] = [(self.inactive_RNAP_idx, -n_initiations.sum())]

        # Add partially transcribed RNAs
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
        timesteps = (1.0 / (timestep * units.s) * allTranscriptionTimes).magnitude
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
            - (1.0 / (timestep * units.s) * allTranscriptionTimes).magnitude
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
