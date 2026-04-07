"""
======================
Polypeptide Elongation
======================

This process models the polymerization of amino acids into polypeptides
by ribosomes using an mRNA transcript as a template. Elongation terminates
once a ribosome has reached the end of an mRNA transcript. Polymerization
occurs across all ribosomes simultaneously and resources are allocated to
maximize the progress of all ribosomes within the limits of the maximum ribosome
elongation rate, available amino acids and GTP, and the length of the transcript.
"""

from typing import Any, Callable, Optional, Tuple

from numba import njit
import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
from unum import Unum

# wcEcoli imports
from process_bigraph import Step

from v2ecoli.library.polymerize import buildSequences, polymerize, computeMassIncrease
from v2ecoli.library.random import stochasticRound
from v2ecoli.library.units import units

# vivarium imports
from v2ecoli.steps.partition import _protect_state, _SafeInvokeMixin, deep_merge
from vivarium.library.units import units as vivunits
# vivarium-ecoli imports
from v2ecoli.library.schema import (
    counts,
    attrs,
    bulk_name_to_idx,
)
from v2ecoli.processes.metabolism import CONC_UNITS, TIME_UNITS
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore


MICROMOLAR_UNITS = units.umol / units.L
"""Units used for all concentrations."""
REMOVED_FROM_CHARGING = {"L-SELENOCYSTEINE[c]"}
"""Amino acids to remove from charging when running with
``steady_state_trna_charging``"""


DEFAULT_AA_NAMES = [
    "L-ALPHA-ALANINE[c]",
    "ARG[c]",
    "ASN[c]",
    "L-ASPARTATE[c]",
    "CYS[c]",
    "GLT[c]",
    "GLN[c]",
    "GLY[c]",
    "HIS[c]",
    "ILE[c]",
    "LEU[c]",
    "LYS[c]",
    "MET[c]",
    "PHE[c]",
    "PRO[c]",
    "SER[c]",
    "THR[c]",
    "TRP[c]",
    "TYR[c]",
    "L-SELENOCYSTEINE[c]",
    "VAL[c]",
]


class PolypeptideElongationStep(_SafeInvokeMixin, Step):
    """Polypeptide Elongation — merged single Step."""

    config_schema = {}

    topology = {
        "environment": ("environment",),
        "boundary": ("boundary",),
        "listeners": ("listeners",),
        "active_ribosome": ("unique", "active_ribosome"),
        "bulk": ("bulk",),
        "polypeptide_elongation": ("process_state", "polypeptide_elongation"),
        # Non-partitioned counts
        "bulk_total": ("bulk",),
        "timestep": ("timestep",),
    }

    def initialize(self, config):
        defaults = {
            "time_step": 1,
            "n_avogadro": 6.02214076e23 / units.mol,
            "proteinIds": np.array([]),
            "proteinLengths": np.array([]),
            "proteinSequences": np.array([[]]),
            "aaWeightsIncorporated": np.array([]),
            "endWeight": np.array([2.99146113e-08]),
            "variable_elongation": False,
            "make_elongation_rates": (
                lambda random, rate, timestep, variable: np.array([])
            ),
            "next_aa_pad": 1,
            "ribosomeElongationRate": 17.388824902723737,
            "translation_aa_supply": {"minimal": np.array([])},
            "import_threshold": 1e-05,
            "aa_from_trna": np.zeros(21),
            "gtpPerElongation": 4.2,
            "aa_supply_in_charging": False,
            "mechanistic_translation_supply": False,
            "mechanistic_aa_transport": False,
            "ppgpp_regulation": False,
            "disable_ppgpp_elongation_inhibition": False,
            "trna_charging": False,
            "translation_supply": False,
            "mechanistic_supply": False,
            "ribosome30S": "ribosome30S",
            "ribosome50S": "ribosome50S",
            "amino_acids": DEFAULT_AA_NAMES,
            "aa_exchange_names": DEFAULT_AA_NAMES,
            "basal_elongation_rate": 22.0,
            "ribosomeElongationRateDict": {
                "minimal": 17.388824902723737 * units.aa / units.s
            },
            "uncharged_trna_names": np.array([]),
            "aaNames": DEFAULT_AA_NAMES,
            "aa_enzymes": [],
            "proton": "PROTON",
            "water": "H2O",
            "cellDensity": 1100 * units.g / units.L,
            "elongation_max": 22 * units.aa / units.s,
            "aa_from_synthetase": np.array([[]]),
            "charging_stoich_matrix": np.array([[]]),
            "charged_trna_names": [],
            "charging_molecule_names": [],
            "synthetase_names": [],
            "ppgpp_reaction_names": [],
            "ppgpp_reaction_metabolites": [],
            "ppgpp_reaction_stoich": np.array([[]]),
            "ppgpp_synthesis_reaction": "GDPPYPHOSKIN-RXN",
            "ppgpp_degradation_reaction": "PPGPPSYN-RXN",
            "aa_importers": [],
            "amino_acid_export": None,
            "synthesis_index": 0,
            "aa_exporters": [],
            "get_pathway_enzyme_counts_per_aa": None,
            "import_constraint_threshold": 0,
            "unit_conversion": 0,
            "elong_rate_by_ppgpp": 0,
            "amino_acid_import": None,
            "degradation_index": 1,
            "amino_acid_synthesis": None,
            "rela": "RELA",
            "spot": "SPOT",
            "ppgpp": "ppGpp",
            "kS": 100.0,
            "KMtf": 1.0,
            "KMaa": 100.0,
            "krta": 1.0,
            "krtf": 500.0,
            "KD_RelA": 0.26,
            "k_RelA": 75.0,
            "k_SpoT_syn": 2.6,
            "k_SpoT_deg": 0.23,
            "KI_SpoT": 20.0,
            "aa_supply_scaling": lambda aa_conc, aa_in_media: 0,
            "seed": 0,
            "emit_unique": False,
        }
        params = {**defaults, **config}

        # Simulation options
        self.aa_supply_in_charging = params["aa_supply_in_charging"]
        self.mechanistic_translation_supply = params[
            "mechanistic_translation_supply"
        ]
        self.mechanistic_aa_transport = params["mechanistic_aa_transport"]
        self.ppgpp_regulation = params["ppgpp_regulation"]
        self.disable_ppgpp_elongation_inhibition = params[
            "disable_ppgpp_elongation_inhibition"
        ]
        self.variable_elongation = params["variable_elongation"]
        self.variable_polymerize = self.ppgpp_regulation or self.variable_elongation
        translation_supply = params["translation_supply"]
        trna_charging = params["trna_charging"]

        # Load parameters
        self.n_avogadro = params["n_avogadro"]
        self.proteinIds = params["proteinIds"]
        self.protein_lengths = params["proteinLengths"]
        self.proteinSequences = params["proteinSequences"]
        self.aaWeightsIncorporated = params["aaWeightsIncorporated"]
        self.endWeight = params["endWeight"]
        self.make_elongation_rates = params["make_elongation_rates"]
        self.next_aa_pad = params["next_aa_pad"]

        self.ribosome30S = params["ribosome30S"]
        self.ribosome50S = params["ribosome50S"]
        self.amino_acids = params["amino_acids"]
        self.aa_exchange_names = params["aa_exchange_names"]
        self.aa_environment_names = [aa[:-3] for aa in self.aa_exchange_names]
        self.aa_enzymes = params["aa_enzymes"]

        self.ribosomeElongationRate = params["ribosomeElongationRate"]

        # Amino acid supply calculations
        self.translation_aa_supply = params["translation_aa_supply"]
        self.import_threshold = params["import_threshold"]

        # Used for figure in publication
        self.trpAIndex = np.where(self.proteinIds == "TRYPSYN-APROTEIN[c]")[0][0]

        self.elngRateFactor = 1.0

        # Data structures for charging
        self.aa_from_trna = params["aa_from_trna"]

        # Set modeling method
        if trna_charging:
            self.elongation_model = SteadyStateElongationModel(params, self)
        elif translation_supply:
            self.elongation_model = TranslationSupplyElongationModel(params, self)
        else:
            self.elongation_model = BaseElongationModel(params, self)

        # Growth associated maintenance energy requirements for elongations
        self.gtpPerElongation = params["gtpPerElongation"]
        if not trna_charging:
            self.gtpPerElongation += 2

        # basic molecule names
        self.proton = params["proton"]
        self.water = params["water"]
        self.rela = params["rela"]
        self.spot = params["spot"]
        self.ppgpp = params["ppgpp"]
        self.aa_importers = params["aa_importers"]
        self.aa_exporters = params["aa_exporters"]
        # Numpy index for bulk molecule
        self.proton_idx = None

        # Names of molecules associated with tRNA charging
        self.ppgpp_reaction_metabolites = params["ppgpp_reaction_metabolites"]
        self.uncharged_trna_names = params["uncharged_trna_names"]
        self.charged_trna_names = params["charged_trna_names"]
        self.charging_molecule_names = params["charging_molecule_names"]
        self.synthetase_names = params["synthetase_names"]

        self.seed = params["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Store params for elongation model access
        self.parameters = params

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'boundary': InPlaceDict(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'polypeptide_elongation': InPlaceDict(),
            'timestep': InPlaceDict(),
            'global_time': InPlaceDict(),
            'next_update_time': InPlaceDict(),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'polypeptide_elongation': InPlaceDict(),
            'boundary': InPlaceDict(),
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
        if self.proton_idx is None:
            bulk_ids = state["bulk"]["id"]
            self.proton_idx = bulk_name_to_idx(self.proton, bulk_ids)
            self.water_idx = bulk_name_to_idx(self.water, bulk_ids)
            self.rela_idx = bulk_name_to_idx(self.rela, bulk_ids)
            self.spot_idx = bulk_name_to_idx(self.spot, bulk_ids)
            self.ppgpp_idx = bulk_name_to_idx(self.ppgpp, bulk_ids)
            self.monomer_idx = bulk_name_to_idx(self.proteinIds, bulk_ids)
            self.amino_acid_idx = bulk_name_to_idx(self.amino_acids, bulk_ids)
            self.aa_enzyme_idx = bulk_name_to_idx(self.aa_enzymes, bulk_ids)
            self.ppgpp_rxn_metabolites_idx = bulk_name_to_idx(
                self.ppgpp_reaction_metabolites, bulk_ids
            )
            self.uncharged_trna_idx = bulk_name_to_idx(
                self.uncharged_trna_names, bulk_ids
            )
            self.charged_trna_idx = bulk_name_to_idx(self.charged_trna_names, bulk_ids)
            self.charging_molecule_idx = bulk_name_to_idx(
                self.charging_molecule_names, bulk_ids
            )
            self.synthetase_idx = bulk_name_to_idx(self.synthetase_names, bulk_ids)
            self.ribosome30S_idx = bulk_name_to_idx(self.ribosome30S, bulk_ids)
            self.ribosome50S_idx = bulk_name_to_idx(self.ribosome50S, bulk_ids)
            self.aa_importer_idx = bulk_name_to_idx(self.aa_importers, bulk_ids)
            self.aa_exporter_idx = bulk_name_to_idx(self.aa_exporters, bulk_ids)

        # --- Requester logic ---

        # MODEL SPECIFIC: get ribosome elongation rate
        self.ribosomeElongationRate = self.elongation_model.elongation_rate(state)

        update = {
            "listeners": {"ribosome_data": {}, "growth_limits": {}},
            "polypeptide_elongation": {},
            "bulk": [],
        }

        # Set values for metabolism in case of early return
        update["polypeptide_elongation"]["gtp_to_hydrolyze"] = 0
        update["polypeptide_elongation"]["aa_count_diff"] = np.zeros(
            len(self.amino_acids), dtype=np.float64
        )

        # Get number of active ribosomes
        n_active_ribosomes = state["active_ribosome"]["_entryState"].sum()
        update["listeners"]["growth_limits"]["active_ribosome_allocated"] = (
            n_active_ribosomes
        )
        update["listeners"]["growth_limits"]["aa_allocated"] = counts(
            state["bulk"], self.amino_acid_idx
        )

        # If there are no active ribosomes, return immediately
        if n_active_ribosomes == 0:
            request_listeners = {}
            ribosome_data_listener = request_listeners.setdefault("ribosome_data", {})
            growth_limits_listener = request_listeners.setdefault("growth_limits", {})
            update["listeners"] = deep_merge(update["listeners"], request_listeners)
            update["polypeptide_elongation"].setdefault(
                "aa_exchange_rates", np.zeros(len(self.amino_acids))
            )
            update["next_update_time"] = global_time + timestep
            return update

        # Build sequences to request appropriate amount of amino acids
        (
            proteinIndexes,
            peptideLengths,
        ) = attrs(state["active_ribosome"], ["protein_index", "peptide_length"])

        elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.ribosomeElongationRate,
            timestep,
            self.variable_elongation,
        )

        sequences_for_request = buildSequences(
            self.proteinSequences, proteinIndexes, peptideLengths, elongation_rates
        )

        sequenceHasAA = sequences_for_request != polymerize.PAD_VALUE
        aasInSequences = np.bincount(sequences_for_request[sequenceHasAA], minlength=21)

        # Calculate AA supply for expected doubling of protein
        dryMass = state["listeners"]["mass"]["dry_mass"] * units.fg
        current_media_id = state["environment"]["media_id"]
        translation_supply_rate = (
            self.translation_aa_supply[current_media_id] * self.elngRateFactor
        )
        mol_aas_supplied = (
            translation_supply_rate * dryMass * timestep * units.s
        )
        self.aa_supply = units.strip_empty_units(mol_aas_supplied * self.n_avogadro)

        # MODEL SPECIFIC: Calculate AA request (also computes cached values)
        fraction_charged, aa_counts_for_translation, request = (
            self.elongation_model.request(state, aasInSequences)
        )

        # Merge request listeners into update
        request_listeners = request.get("listeners", {})
        ribosome_data_listener = request_listeners.setdefault("ribosome_data", {})
        ribosome_data_listener["translation_supply"] = (
            translation_supply_rate.asNumber()
        )
        growth_limits_listener = request_listeners.setdefault("growth_limits", {})
        growth_limits_listener["fraction_trna_charged"] = np.dot(
            fraction_charged, self.aa_from_trna
        )
        growth_limits_listener["aa_pool_size"] = counts(
            state["bulk_total"], self.amino_acid_idx
        )
        growth_limits_listener["aa_request_size"] = aa_counts_for_translation
        update["listeners"] = deep_merge(update["listeners"], request_listeners)

        # Merge polypeptide_elongation data from request
        request_proc = request.get("polypeptide_elongation", {})
        request_proc.setdefault("aa_exchange_rates", np.zeros(len(self.amino_acids)))
        update["polypeptide_elongation"].update(request_proc)

        # --- Evolver logic ---

        # Store elongation_rates for evolver use
        self.elongation_rates = elongation_rates

        # Polypeptide elongation requires counts to be updated in real-time
        # so make a writeable copy of bulk counts to do so
        state["bulk"] = counts(state["bulk"], range(len(state["bulk"])))

        # Build amino acids sequences for each ribosome to polymerize
        protein_indexes, peptide_lengths, positions_on_mRNA = attrs(
            state["active_ribosome"],
            ["protein_index", "peptide_length", "pos_on_mRNA"],
        )

        all_sequences = buildSequences(
            self.proteinSequences,
            protein_indexes,
            peptide_lengths,
            elongation_rates + self.next_aa_pad,
        )
        sequences = all_sequences[:, : -self.next_aa_pad].copy()

        if sequences.size != 0:
            # Calculate elongation resource capacity
            aaCountInSequence = np.bincount(sequences[(sequences != polymerize.PAD_VALUE)])
            total_aa_counts = counts(state["bulk"], self.amino_acid_idx)
            charged_trna_counts = counts(state["bulk"], self.charged_trna_idx)

            # MODEL SPECIFIC: Get amino acid counts
            aa_counts_for_translation = self.elongation_model.final_amino_acids(
                total_aa_counts, charged_trna_counts
            )

            # Using polymerization algorithm elongate each ribosome
            result = polymerize(
                sequences,
                aa_counts_for_translation,
                10000000,
                self.random_state,
                elongation_rates[protein_indexes],
                variable_elongation=self.variable_polymerize,
            )

            sequence_elongations = result.sequenceElongation
            aas_used = result.monomerUsages
            nElongations = result.nReactions

            next_amino_acid = all_sequences[
                np.arange(len(sequence_elongations)), sequence_elongations
            ]
            next_amino_acid_count = np.bincount(
                next_amino_acid[next_amino_acid != polymerize.PAD_VALUE], minlength=21
            )

            # Update masses of ribosomes attached to polymerizing polypeptides
            added_protein_mass = computeMassIncrease(
                sequences, sequence_elongations, self.aaWeightsIncorporated
            )

            updated_lengths = peptide_lengths + sequence_elongations
            updated_positions_on_mRNA = positions_on_mRNA + 3 * sequence_elongations

            didInitialize = (sequence_elongations > 0) & (peptide_lengths == 0)

            added_protein_mass[didInitialize] += self.endWeight

            # Write current average elongation to listener
            currElongRate = (sequence_elongations.sum() / n_active_ribosomes) / timestep

            # Ribosomes that reach the end of their sequences are terminated
            terminalLengths = self.protein_lengths[protein_indexes]

            didTerminate = updated_lengths == terminalLengths

            terminatedProteins = np.bincount(
                protein_indexes[didTerminate], minlength=self.proteinSequences.shape[0]
            )

            (protein_mass,) = attrs(state["active_ribosome"], ["massDiff_protein"])
            update.setdefault("active_ribosome", {})
            update["active_ribosome"].update(
                {
                    "delete": np.where(didTerminate)[0],
                    "set": {
                        "massDiff_protein": protein_mass + added_protein_mass,
                        "peptide_length": updated_lengths,
                        "pos_on_mRNA": updated_positions_on_mRNA,
                    },
                }
            )

            update["bulk"].append((self.monomer_idx, terminatedProteins))
            state["bulk"][self.monomer_idx] += terminatedProteins

            nTerminated = didTerminate.sum()
            nInitialized = didInitialize.sum()

            update["bulk"].append((self.ribosome30S_idx, nTerminated))
            update["bulk"].append((self.ribosome50S_idx, nTerminated))
            state["bulk"][self.ribosome30S_idx] += nTerminated
            state["bulk"][self.ribosome50S_idx] += nTerminated

            # MODEL SPECIFIC: evolve
            net_charged, aa_count_diff, evolve_update = self.elongation_model.evolve(
                state,
                total_aa_counts,
                aas_used,
                next_amino_acid_count,
                nElongations,
                nInitialized,
            )

            evolve_bulk_update = evolve_update.pop("bulk")
            update = deep_merge(update, evolve_update)
            update["bulk"].extend(evolve_bulk_update)

            update["polypeptide_elongation"]["aa_count_diff"] = aa_count_diff
            update["polypeptide_elongation"]["gtp_to_hydrolyze"] = (
                self.gtpPerElongation * nElongations
            )

            # Write data to listeners
            update["listeners"]["growth_limits"]["net_charged"] = net_charged
            update["listeners"]["growth_limits"]["aas_used"] = aas_used
            update["listeners"]["growth_limits"]["aa_count_diff"] = aa_count_diff

            ribosome_data_listener = update["listeners"].setdefault("ribosome_data", {})
            ribosome_data_listener["effective_elongation_rate"] = currElongRate
            ribosome_data_listener["aa_count_in_sequence"] = aaCountInSequence
            ribosome_data_listener["aa_counts"] = aa_counts_for_translation
            ribosome_data_listener["actual_elongations"] = sequence_elongations.sum()
            ribosome_data_listener["actual_elongation_hist"] = np.histogram(
                sequence_elongations, bins=np.arange(0, 23)
            )[0]
            ribosome_data_listener["elongations_non_terminating_hist"] = np.histogram(
                sequence_elongations[~didTerminate], bins=np.arange(0, 23)
            )[0]
            ribosome_data_listener["did_terminate"] = didTerminate.sum()
            ribosome_data_listener["termination_loss"] = (
                terminalLengths - peptide_lengths
            )[didTerminate].sum()
            ribosome_data_listener["num_trpA_terminated"] = terminatedProteins[
                self.trpAIndex
            ]
            ribosome_data_listener["process_elongation_rate"] = (
                self.ribosomeElongationRate / timestep
            )

        update["next_update_time"] = global_time + timestep
        return update


class BaseElongationModel(object):
    """
    Base Model: Request amino acids according to upcoming sequence, assuming
    max ribosome elongation.
    """

    def __init__(self, parameters, process):
        self.parameters = parameters
        self.process = process
        self.basal_elongation_rate = self.parameters["basal_elongation_rate"]
        self.ribosomeElongationRateDict = self.parameters["ribosomeElongationRateDict"]

    def elongation_rate(self, states):
        """
        Sets ribosome elongation rate according to the media; returns
        max value of 22 amino acids/second.
        """
        current_media_id = states["environment"]["media_id"]
        rate = self.process.elngRateFactor * self.ribosomeElongationRateDict[
            current_media_id
        ].asNumber(units.aa / units.s)
        return np.min([self.basal_elongation_rate, rate])

    def amino_acid_counts(self, aasInSequences):
        return aasInSequences

    def request(
        self, states: dict, aasInSequences: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict]:
        aa_counts_for_translation = self.amino_acid_counts(aasInSequences)

        requests = {
            "bulk": [
                (
                    self.process.amino_acid_idx,
                    aa_counts_for_translation.astype(np.int64),
                )
            ]
        }

        # Not modeling charging so set fraction charged to 0 for all tRNA
        fraction_charged = np.zeros(len(self.process.amino_acid_idx))

        return fraction_charged, aa_counts_for_translation.astype(float), requests

    def final_amino_acids(self, total_aa_counts, charged_trna_counts):
        return total_aa_counts

    def evolve(
        self,
        states,
        total_aa_counts,
        aas_used,
        next_amino_acid_count,
        nElongations,
        nInitialized,
    ):
        # Update counts of amino acids and water to reflect polymerization
        # reactions
        net_charged = np.zeros(
            len(self.parameters["uncharged_trna_names"]), dtype=np.int64
        )
        return (
            net_charged,
            np.zeros(len(self.process.amino_acids), dtype=np.float64),
            {
                "bulk": [
                    (self.process.amino_acid_idx, -aas_used),
                    (self.process.water_idx, nElongations - nInitialized),
                ]
            },
        )


class TranslationSupplyElongationModel(BaseElongationModel):
    """
    Translation Supply Model: Requests minimum of 1) upcoming amino acid
    sequence assuming max ribosome elongation (ie. Base Model) and 2)
    estimation based on doubling the proteome in one cell cycle (does not
    use ribosome elongation, computed in Parca).
    """

    def __init__(self, parameters, process):
        super().__init__(parameters, process)

    def elongation_rate(self, states):
        """
        Sets ribosome elongation rate according to the media; returns
        max value of 22 amino acids/second.
        """
        return self.basal_elongation_rate

    def amino_acid_counts(self, aasInSequences):
        return np.fmin(self.process.aa_supply, aasInSequences)


class SteadyStateElongationModel(TranslationSupplyElongationModel):
    """
    Steady State Charging Model: Requests amino acids based on the
    Michaelis-Menten competitive inhibition model.
    """

    def __init__(self, parameters, process):
        super().__init__(parameters, process)

        # Cell parameters
        self.cellDensity = self.parameters["cellDensity"]

        # Names of molecules associated with tRNA charging
        self.charged_trna_names = self.parameters["charged_trna_names"]
        self.charging_molecule_names = self.parameters["charging_molecule_names"]
        self.synthetase_names = self.parameters["synthetase_names"]

        # Data structures for charging
        self.aa_from_synthetase = self.parameters["aa_from_synthetase"]
        self.charging_stoich_matrix = self.parameters["charging_stoich_matrix"]
        self.charging_molecules_not_aa = np.array(
            [
                mol not in set(self.parameters["amino_acids"])
                for mol in self.charging_molecule_names
            ]
        )

        # ppGpp synthesis
        self.ppgpp_reaction_metabolites = self.parameters["ppgpp_reaction_metabolites"]
        self.elong_rate_by_ppgpp = self.parameters["elong_rate_by_ppgpp"]

        # Parameters for tRNA charging, ribosome elongation and ppGpp reactions
        self.charging_params = {
            "kS": self.parameters["kS"],
            "KMaa": self.parameters["KMaa"],
            "KMtf": self.parameters["KMtf"],
            "krta": self.parameters["krta"],
            "krtf": self.parameters["krtf"],
            "max_elong_rate": float(
                self.parameters["elongation_max"].asNumber(units.aa / units.s)
            ),
            "charging_mask": np.array(
                [
                    aa not in REMOVED_FROM_CHARGING
                    for aa in self.parameters["amino_acids"]
                ]
            ),
            "unit_conversion": self.parameters["unit_conversion"],
        }
        self.ppgpp_params = {
            "KD_RelA": self.parameters["KD_RelA"],
            "k_RelA": self.parameters["k_RelA"],
            "k_SpoT_syn": self.parameters["k_SpoT_syn"],
            "k_SpoT_deg": self.parameters["k_SpoT_deg"],
            "KI_SpoT": self.parameters["KI_SpoT"],
            "ppgpp_reaction_stoich": self.parameters["ppgpp_reaction_stoich"],
            "synthesis_index": self.parameters["synthesis_index"],
            "degradation_index": self.parameters["degradation_index"],
        }

        # Amino acid supply calculations
        self.aa_supply_scaling = self.parameters["aa_supply_scaling"]

        self.amino_acid_synthesis = self.parameters["amino_acid_synthesis"]
        self.amino_acid_import = self.parameters["amino_acid_import"]
        self.amino_acid_export = self.parameters["amino_acid_export"]
        self.get_pathway_enzyme_counts_per_aa = self.parameters[
            "get_pathway_enzyme_counts_per_aa"
        ]

        # Comparing two values with units is faster than converting units
        self.import_constraint_threshold = (
            self.parameters["import_constraint_threshold"] * vivunits.mM
        )

    def elongation_rate(self, states):
        if (
            self.process.ppgpp_regulation
            and not self.process.disable_ppgpp_elongation_inhibition
        ):
            cell_mass = states["listeners"]["mass"]["cell_mass"] * units.fg
            cell_volume = cell_mass / self.cellDensity
            counts_to_molar = 1 / (self.process.n_avogadro * cell_volume)
            ppgpp_count = counts(states["bulk"], self.process.ppgpp_idx)
            ppgpp_conc = ppgpp_count * counts_to_molar
            rate = self.elong_rate_by_ppgpp(
                ppgpp_conc, self.basal_elongation_rate
            ).asNumber(units.aa / units.s)
        else:
            rate = super().elongation_rate(states)
        return rate

    def request(
        self, states: dict, aasInSequences: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict]:
        # Conversion from counts to molarity
        cell_mass = states["listeners"]["mass"]["cell_mass"] * units.fg
        dry_mass = states["listeners"]["mass"]["dry_mass"] * units.fg
        cell_volume = cell_mass / self.cellDensity
        self.counts_to_molar = 1 / (self.process.n_avogadro * cell_volume)

        # ppGpp related concentrations
        ppgpp_conc = self.counts_to_molar * counts(
            states["bulk_total"], self.process.ppgpp_idx
        )
        rela_conc = self.counts_to_molar * counts(
            states["bulk_total"], self.process.rela_idx
        )
        spot_conc = self.counts_to_molar * counts(
            states["bulk_total"], self.process.spot_idx
        )

        # Get counts and convert synthetase and tRNA to a per AA basis
        synthetase_counts = np.dot(
            self.aa_from_synthetase,
            counts(states["bulk_total"], self.process.synthetase_idx),
        )
        aa_counts = counts(states["bulk_total"], self.process.amino_acid_idx)
        uncharged_trna_array = counts(
            states["bulk_total"], self.process.uncharged_trna_idx
        )
        charged_trna_array = counts(states["bulk_total"], self.process.charged_trna_idx)
        uncharged_trna_counts = np.dot(self.process.aa_from_trna, uncharged_trna_array)
        charged_trna_counts = np.dot(self.process.aa_from_trna, charged_trna_array)
        ribosome_counts = states["active_ribosome"]["_entryState"].sum()

        # Get concentration
        f = aasInSequences / aasInSequences.sum()
        synthetase_conc = self.counts_to_molar * synthetase_counts
        aa_conc = self.counts_to_molar * aa_counts
        uncharged_trna_conc = self.counts_to_molar * uncharged_trna_counts
        charged_trna_conc = self.counts_to_molar * charged_trna_counts
        ribosome_conc = self.counts_to_molar * ribosome_counts

        # Calculate amino acid supply
        aa_in_media = np.array(
            [
                states["boundary"]["external"][aa] > self.import_constraint_threshold
                for aa in self.process.aa_environment_names
            ]
        )
        fwd_enzyme_counts, rev_enzyme_counts = self.get_pathway_enzyme_counts_per_aa(
            counts(states["bulk_total"], self.process.aa_enzyme_idx)
        )
        importer_counts = counts(states["bulk_total"], self.process.aa_importer_idx)
        exporter_counts = counts(states["bulk_total"], self.process.aa_exporter_idx)
        synthesis, fwd_saturation, rev_saturation = self.amino_acid_synthesis(
            fwd_enzyme_counts, rev_enzyme_counts, aa_conc
        )
        import_rates = self.amino_acid_import(
            aa_in_media,
            dry_mass,
            aa_conc,
            importer_counts,
            self.process.mechanistic_aa_transport,
        )
        export_rates = self.amino_acid_export(
            exporter_counts, aa_conc, self.process.mechanistic_aa_transport
        )
        exchange_rates = import_rates - export_rates

        supply_function = get_charging_supply_function(
            self.process.aa_supply_in_charging,
            self.process.mechanistic_translation_supply,
            self.process.mechanistic_aa_transport,
            self.amino_acid_synthesis,
            self.amino_acid_import,
            self.amino_acid_export,
            self.aa_supply_scaling,
            self.counts_to_molar,
            self.process.aa_supply,
            fwd_enzyme_counts,
            rev_enzyme_counts,
            dry_mass,
            importer_counts,
            exporter_counts,
            aa_in_media,
        )

        # Calculate steady state tRNA levels and resulting elongation rate
        self.charging_params["max_elong_rate"] = self.elongation_rate(states)
        (
            fraction_charged,
            v_rib,
            synthesis_in_charging,
            import_in_charging,
            export_in_charging,
        ) = calculate_trna_charging(
            synthetase_conc,
            uncharged_trna_conc,
            charged_trna_conc,
            aa_conc,
            ribosome_conc,
            f,
            self.charging_params,
            supply=supply_function,
            limit_v_rib=True,
            time_limit=states["timestep"],
        )

        # Use the supply calculated from each sub timestep while solving the charging steady state
        if self.process.aa_supply_in_charging:
            conversion = (
                1 / self.counts_to_molar.asNumber(MICROMOLAR_UNITS) / states["timestep"]
            )
            synthesis = conversion * synthesis_in_charging
            import_rates = conversion * import_in_charging
            export_rates = conversion * export_in_charging
            self.process.aa_supply = synthesis + import_rates - export_rates
        elif self.process.mechanistic_translation_supply:
            self.process.aa_supply = states["timestep"] * (synthesis + exchange_rates)
        else:
            self.process.aa_supply *= self.aa_supply_scaling(
                self.charging_params["unit_conversion"] * aa_conc.asNumber(CONC_UNITS),
                aa_in_media,
            )

        aa_counts_for_translation = (
            v_rib
            * f
            * states["timestep"]
            / self.counts_to_molar.asNumber(MICROMOLAR_UNITS)
        )

        total_trna = charged_trna_array + uncharged_trna_array
        final_charged_trna = stochasticRound(
            self.process.random_state,
            np.dot(fraction_charged, self.process.aa_from_trna * total_trna),
        )

        # Request charged tRNA that will become uncharged
        charged_trna_request = charged_trna_array - final_charged_trna
        charged_trna_request[charged_trna_request < 0] = 0
        uncharged_trna_request = final_charged_trna - charged_trna_array
        uncharged_trna_request[uncharged_trna_request < 0] = 0
        self.uncharged_trna_to_charge = uncharged_trna_request

        self.aa_counts_for_translation = np.array(aa_counts_for_translation)

        fraction_trna_per_aa = total_trna / np.dot(
            np.dot(self.process.aa_from_trna, total_trna), self.process.aa_from_trna
        )
        total_charging_reactions = stochasticRound(
            self.process.random_state,
            np.dot(aa_counts_for_translation, self.process.aa_from_trna)
            * fraction_trna_per_aa
            + uncharged_trna_request,
        )

        # Only request molecules that will be consumed in the charging reactions
        aa_from_uncharging = -self.charging_stoich_matrix @ charged_trna_request
        aa_from_uncharging[self.charging_molecules_not_aa] = 0
        requested_molecules = (
            -np.dot(self.charging_stoich_matrix, total_charging_reactions)
            - aa_from_uncharging
        )
        requested_molecules[requested_molecules < 0] = 0
        self.uncharged_trna_to_charge = uncharged_trna_request

        # ppGpp reactions based on charged tRNA
        bulk_request = [
            (
                self.process.charging_molecule_idx,
                requested_molecules.astype(int),
            ),
            (self.process.charged_trna_idx, charged_trna_request.astype(int)),
            (self.process.water_idx, int(aa_counts_for_translation.sum())),
        ]
        if self.process.ppgpp_regulation:
            total_trna_conc = self.counts_to_molar * (
                uncharged_trna_counts + charged_trna_counts
            )
            updated_charged_trna_conc = total_trna_conc * fraction_charged
            updated_uncharged_trna_conc = total_trna_conc - updated_charged_trna_conc
            delta_metabolites, *_ = ppgpp_metabolite_changes(
                updated_uncharged_trna_conc,
                updated_charged_trna_conc,
                ribosome_conc,
                f,
                rela_conc,
                spot_conc,
                ppgpp_conc,
                self.counts_to_molar,
                v_rib,
                self.charging_params,
                self.ppgpp_params,
                states["timestep"],
                request=True,
                random_state=self.process.random_state,
            )

            request_ppgpp_metabolites = -delta_metabolites.astype(int)
            ppgpp_request = counts(states["bulk"], self.process.ppgpp_idx)
            bulk_request.append((self.process.ppgpp_idx, ppgpp_request))
            bulk_request.append(
                (
                    self.process.ppgpp_rxn_metabolites_idx,
                    request_ppgpp_metabolites,
                )
            )

        return (
            fraction_charged,
            aa_counts_for_translation,
            {
                "bulk": bulk_request,
                "listeners": {
                    "growth_limits": {
                        "original_aa_supply": self.process.aa_supply,
                        "aa_in_media": aa_in_media,
                        "synthetase_conc": synthetase_conc.asNumber(MICROMOLAR_UNITS),
                        "uncharged_trna_conc": uncharged_trna_conc.asNumber(
                            MICROMOLAR_UNITS
                        ),
                        "charged_trna_conc": charged_trna_conc.asNumber(
                            MICROMOLAR_UNITS
                        ),
                        "aa_conc": aa_conc.asNumber(MICROMOLAR_UNITS),
                        "ribosome_conc": ribosome_conc.asNumber(MICROMOLAR_UNITS),
                        "fraction_aa_to_elongate": f,
                        "aa_supply": self.process.aa_supply,
                        "aa_synthesis": synthesis * states["timestep"],
                        "aa_import": import_rates * states["timestep"],
                        "aa_export": export_rates * states["timestep"],
                        "aa_supply_enzymes_fwd": fwd_enzyme_counts,
                        "aa_supply_enzymes_rev": rev_enzyme_counts,
                        "aa_importers": importer_counts,
                        "aa_exporters": exporter_counts,
                        "aa_supply_aa_conc": aa_conc.asNumber(units.mmol / units.L),
                        "aa_supply_fraction_fwd": fwd_saturation,
                        "aa_supply_fraction_rev": rev_saturation,
                        "ppgpp_conc": ppgpp_conc.asNumber(MICROMOLAR_UNITS),
                        "rela_conc": rela_conc.asNumber(MICROMOLAR_UNITS),
                        "spot_conc": spot_conc.asNumber(MICROMOLAR_UNITS),
                    }
                },
                "polypeptide_elongation": {
                    "aa_exchange_rates": (
                        self.counts_to_molar / units.s * (import_rates - export_rates)
                    ).asNumber(CONC_UNITS / TIME_UNITS)
                },
            },
        )

    def final_amino_acids(self, total_aa_counts, charged_trna_counts):
        charged_counts_to_uncharge = self.process.aa_from_trna @ charged_trna_counts
        return np.fmin(
            total_aa_counts + charged_counts_to_uncharge, self.aa_counts_for_translation
        )

    def evolve(
        self,
        states,
        total_aa_counts,
        aas_used,
        next_amino_acid_count,
        nElongations,
        nInitialized,
    ):
        update = {
            "bulk": [],
            "listeners": {"growth_limits": {}},
        }

        # Get tRNA counts
        uncharged_trna = counts(states["bulk"], self.process.uncharged_trna_idx)
        charged_trna = counts(states["bulk"], self.process.charged_trna_idx)
        total_trna = uncharged_trna + charged_trna

        # Adjust molecules for number of charging reactions that occurred
        charged_and_elongated_per_aa = np.fmax(
            0, (aas_used - self.process.aa_from_trna @ charged_trna)
        )
        aa_for_charging = total_aa_counts - charged_and_elongated_per_aa
        n_aa_charged = np.fmin(
            aa_for_charging,
            np.dot(
                self.process.aa_from_trna,
                np.fmin(self.uncharged_trna_to_charge, uncharged_trna),
            ),
        )
        n_uncharged_per_aa = aas_used - charged_and_elongated_per_aa

        ## Calculate changes in tRNA based on limitations
        n_trna_charged = self.distribution_from_aa(n_aa_charged, uncharged_trna, True)
        n_trna_uncharged = self.distribution_from_aa(
            n_uncharged_per_aa, charged_trna, True
        )

        charged_and_elongated = self.distribution_from_aa(
            charged_and_elongated_per_aa, total_trna
        )

        ## Determine total number of reactions that occur
        total_uncharging_reactions = charged_and_elongated + n_trna_uncharged
        total_charging_reactions = charged_and_elongated + n_trna_charged
        net_charged = total_charging_reactions - total_uncharging_reactions
        charging_mol_delta = np.dot(
            self.charging_stoich_matrix, total_charging_reactions
        ).astype(int)
        update["bulk"].append((self.process.charging_molecule_idx, charging_mol_delta))
        states["bulk"][self.process.charging_molecule_idx] += charging_mol_delta

        ## Account for uncharging of tRNA during elongation
        update["bulk"].append(
            (self.process.charged_trna_idx, -total_uncharging_reactions)
        )
        update["bulk"].append(
            (self.process.uncharged_trna_idx, total_uncharging_reactions)
        )
        states["bulk"][self.process.charged_trna_idx] += -total_uncharging_reactions
        states["bulk"][self.process.uncharged_trna_idx] += total_uncharging_reactions

        # Update proton counts
        update["bulk"].append((self.process.proton_idx, nElongations))
        update["bulk"].append((self.process.water_idx, -nInitialized))
        states["bulk"][self.process.proton_idx] += nElongations
        states["bulk"][self.process.water_idx] += -nInitialized

        # Create or degrade ppGpp
        if self.process.ppgpp_regulation:
            v_rib = (nElongations * self.counts_to_molar).asNumber(
                MICROMOLAR_UNITS
            ) / states["timestep"]
            ribosome_conc = (
                self.counts_to_molar * states["active_ribosome"]["_entryState"].sum()
            )
            updated_uncharged_trna_counts = (
                counts(states["bulk_total"], self.process.uncharged_trna_idx)
                - net_charged
            )
            updated_charged_trna_counts = (
                counts(states["bulk_total"], self.process.charged_trna_idx)
                + net_charged
            )
            uncharged_trna_conc = self.counts_to_molar * np.dot(
                self.process.aa_from_trna, updated_uncharged_trna_counts
            )
            charged_trna_conc = self.counts_to_molar * np.dot(
                self.process.aa_from_trna, updated_charged_trna_counts
            )
            ppgpp_conc = self.counts_to_molar * counts(
                states["bulk_total"], self.process.ppgpp_idx
            )
            rela_conc = self.counts_to_molar * counts(
                states["bulk_total"], self.process.rela_idx
            )
            spot_conc = self.counts_to_molar * counts(
                states["bulk_total"], self.process.spot_idx
            )

            aa_at_ribosome = aas_used + next_amino_acid_count
            f = aa_at_ribosome / aa_at_ribosome.sum()
            limits = counts(states["bulk"], self.process.ppgpp_rxn_metabolites_idx)
            (
                delta_metabolites,
                ppgpp_syn,
                ppgpp_deg,
                rela_syn,
                spot_syn,
                spot_deg,
                spot_deg_inhibited,
            ) = ppgpp_metabolite_changes(
                uncharged_trna_conc,
                charged_trna_conc,
                ribosome_conc,
                f,
                rela_conc,
                spot_conc,
                ppgpp_conc,
                self.counts_to_molar,
                v_rib,
                self.charging_params,
                self.ppgpp_params,
                states["timestep"],
                random_state=self.process.random_state,
                limits=limits,
            )

            update["listeners"]["growth_limits"] = {
                "rela_syn": rela_syn,
                "spot_syn": spot_syn,
                "spot_deg": spot_deg,
                "spot_deg_inhibited": spot_deg_inhibited,
            }

            update["bulk"].append(
                (self.process.ppgpp_rxn_metabolites_idx, delta_metabolites.astype(int))
            )
            states["bulk"][self.process.ppgpp_rxn_metabolites_idx] += (
                delta_metabolites.astype(int)
            )

        aa_used_trna = np.dot(self.process.aa_from_trna, total_charging_reactions)
        aa_diff = self.process.aa_supply - aa_used_trna

        update["listeners"]["growth_limits"]["trna_charged"] = aa_used_trna.astype(int)

        return (
            net_charged,
            aa_diff,
            update,
        )

    def distribution_from_aa(
        self,
        n_aa: npt.NDArray[np.int64],
        n_trna: npt.NDArray[np.int64],
        limited: bool = False,
    ) -> npt.NDArray[np.int64]:
        """
        Distributes counts of amino acids to tRNAs that are associated with
        each amino acid.
        """

        with np.errstate(invalid="ignore"):
            f_trna = n_trna / np.dot(
                np.dot(self.process.aa_from_trna, n_trna), self.process.aa_from_trna
            )
        f_trna[~np.isfinite(f_trna)] = 0

        trna_counts = np.zeros(f_trna.shape, np.int64)
        for count, row in zip(n_aa, self.process.aa_from_trna):
            idx = row == 1
            frac = f_trna[idx]

            counts = np.floor(frac * count)
            diff = int(count - counts.sum())

            if diff > 0:
                if limited:
                    for _ in range(diff):
                        frac[(n_trna[idx] - counts) == 0] = 0
                        frac /= frac.sum()
                        adjustment = self.process.random_state.multinomial(1, frac)
                        counts += adjustment
                else:
                    adjustment = self.process.random_state.multinomial(diff, frac)
                    counts += adjustment

            trna_counts[idx] = counts

        return trna_counts


def ppgpp_metabolite_changes(
    uncharged_trna_conc: Unum,
    charged_trna_conc: Unum,
    ribosome_conc: Unum,
    f: npt.NDArray[np.float64],
    rela_conc: Unum,
    spot_conc: Unum,
    ppgpp_conc: Unum,
    counts_to_molar: Unum,
    v_rib: Unum,
    charging_params: dict[str, Any],
    ppgpp_params: dict[str, Any],
    time_step: float,
    request: bool = False,
    limits: Optional[npt.NDArray[np.float64]] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> tuple[npt.NDArray[np.int64], int, int, Unum, Unum, Unum, Unum]:
    """
    Calculates the changes in metabolite counts based on ppGpp synthesis and
    degradation reactions.

    Args:
        uncharged_trna_conc: concentration (MICROMOLAR_UNITS) of uncharged tRNA
        charged_trna_conc: concentration (MICROMOLAR_UNITS) of charged tRNA
        ribosome_conc: concentration (MICROMOLAR_UNITS) of active ribosomes
        f: fraction of each amino acid to be incorporated
        rela_conc: concentration (MICROMOLAR_UNITS) of RelA
        spot_conc: concentration (MICROMOLAR_UNITS) of SpoT
        ppgpp_conc: concentration (MICROMOLAR_UNITS) of ppGpp
        counts_to_molar: conversion factor from counts to molarity
        v_rib: rate of amino acid incorporation at the ribosome (uM/s)
        charging_params: parameters used in charging equations
        ppgpp_params: parameters used in ppGpp reactions
        time_step: length of the current time step
        request: if True, only considers reactant stoichiometry
        limits: counts of molecules that are available to prevent negative totals
        random_state: random state for the process

    Returns:
        7-element tuple containing delta_metabolites, n_syn_reactions,
        n_deg_reactions, v_rela_syn, v_spot_syn, v_deg, v_deg_inhibited
    """

    if random_state is None:
        random_state = np.random.RandomState()

    uncharged_trna_conc = uncharged_trna_conc.asNumber(MICROMOLAR_UNITS)
    charged_trna_conc = charged_trna_conc.asNumber(MICROMOLAR_UNITS)
    ribosome_conc = ribosome_conc.asNumber(MICROMOLAR_UNITS)
    rela_conc = rela_conc.asNumber(MICROMOLAR_UNITS)
    spot_conc = spot_conc.asNumber(MICROMOLAR_UNITS)
    ppgpp_conc = ppgpp_conc.asNumber(MICROMOLAR_UNITS)
    counts_to_micromolar = counts_to_molar.asNumber(MICROMOLAR_UNITS)

    numerator = (
        1
        + charged_trna_conc / charging_params["krta"]
        + uncharged_trna_conc / charging_params["krtf"]
    )
    saturated_charged = charged_trna_conc / charging_params["krta"] / numerator
    saturated_uncharged = uncharged_trna_conc / charging_params["krtf"] / numerator
    if v_rib == 0:
        ribosome_conc_a_site = f * ribosome_conc
    else:
        ribosome_conc_a_site = (
            f * v_rib / (saturated_charged * charging_params["max_elong_rate"])
        )
    ribosomes_bound_to_uncharged = ribosome_conc_a_site * saturated_uncharged

    # Handle rare cases when tRNA concentrations are 0
    mask = ~np.isfinite(ribosomes_bound_to_uncharged)
    ribosomes_bound_to_uncharged[mask] = (
        ribosome_conc
        * f[mask]
        * np.array(uncharged_trna_conc[mask] + charged_trna_conc[mask] > 0)
    )

    # Calculate active fraction of RelA
    competitive_inhibition = 1 + ribosomes_bound_to_uncharged / ppgpp_params["KD_RelA"]
    inhibition_product = np.prod(competitive_inhibition)
    with np.errstate(divide="ignore"):
        frac_rela = 1 / (
            ppgpp_params["KD_RelA"]
            / ribosomes_bound_to_uncharged
            * inhibition_product
            / competitive_inhibition
            + 1
        )

    # Calculate rates for synthesis and degradation
    v_rela_syn = ppgpp_params["k_RelA"] * rela_conc * frac_rela
    v_spot_syn = ppgpp_params["k_SpoT_syn"] * spot_conc
    v_syn = v_rela_syn.sum() + v_spot_syn
    max_deg = ppgpp_params["k_SpoT_deg"] * spot_conc * ppgpp_conc
    fractions = uncharged_trna_conc / ppgpp_params["KI_SpoT"]
    v_deg = max_deg / (1 + fractions.sum())
    v_deg_inhibited = (max_deg - v_deg) * fractions / fractions.sum()

    # Convert to discrete reactions
    n_syn_reactions = stochasticRound(
        random_state, v_syn * time_step / counts_to_micromolar
    )[0]
    n_deg_reactions = stochasticRound(
        random_state, v_deg * time_step / counts_to_micromolar
    )[0]

    # Only look at reactant stoichiometry if requesting molecules to use
    if request:
        ppgpp_reaction_stoich = np.zeros_like(ppgpp_params["ppgpp_reaction_stoich"])
        reactants = ppgpp_params["ppgpp_reaction_stoich"] < 0
        ppgpp_reaction_stoich[reactants] = ppgpp_params["ppgpp_reaction_stoich"][
            reactants
        ]
    else:
        ppgpp_reaction_stoich = ppgpp_params["ppgpp_reaction_stoich"]

    # Calculate the change in metabolites and adjust to limits if provided
    max_iterations = int(n_deg_reactions + n_syn_reactions + 1)
    old_counts = None
    for it in range(max_iterations):
        delta_metabolites = (
            ppgpp_reaction_stoich[:, ppgpp_params["synthesis_index"]] * n_syn_reactions
            + ppgpp_reaction_stoich[:, ppgpp_params["degradation_index"]]
            * n_deg_reactions
        )

        if limits is None:
            break
        else:
            final_counts = delta_metabolites + limits

            if np.all(final_counts >= 0) or (
                old_counts is not None and np.all(final_counts == old_counts)
            ):
                break

            limited_index = np.argmin(final_counts)
            if (
                ppgpp_reaction_stoich[limited_index, ppgpp_params["synthesis_index"]]
                < 0
            ):
                limited = np.ceil(
                    final_counts[limited_index]
                    / ppgpp_reaction_stoich[
                        limited_index, ppgpp_params["synthesis_index"]
                    ]
                )
                n_syn_reactions -= min(limited, n_syn_reactions)
            if (
                ppgpp_reaction_stoich[limited_index, ppgpp_params["degradation_index"]]
                < 0
            ):
                limited = np.ceil(
                    final_counts[limited_index]
                    / ppgpp_reaction_stoich[
                        limited_index, ppgpp_params["degradation_index"]
                    ]
                )
                n_deg_reactions -= min(limited, n_deg_reactions)

            old_counts = final_counts
    else:
        raise ValueError("Failed to meet molecule limits with ppGpp reactions.")

    return (
        delta_metabolites,
        n_syn_reactions,
        n_deg_reactions,
        v_rela_syn,
        v_spot_syn,
        v_deg,
        v_deg_inhibited,
    )


def calculate_trna_charging(
    synthetase_conc: Unum,
    uncharged_trna_conc: Unum,
    charged_trna_conc: Unum,
    aa_conc: Unum,
    ribosome_conc: Unum,
    f: Unum,
    params: dict[str, Any],
    supply: Optional[Callable] = None,
    time_limit: float = 1000,
    limit_v_rib: bool = False,
    use_disabled_aas: bool = False,
) -> tuple[Unum, float, Unum, Unum, Unum]:
    """
    Calculates the steady state value of tRNA based on charging and
    incorporation through polypeptide elongation.

    Args:
        synthetase_conc: concentration of synthetases
        uncharged_trna_conc: concentration of uncharged tRNA
        charged_trna_conc: concentration of charged tRNA
        aa_conc: concentration of each amino acid
        ribosome_conc: concentration of active ribosomes
        f: fraction of each amino acid to be incorporated
        params: parameters used in charging equations
        supply: function to get the rate of amino acid supply
        time_limit: time limit to reach steady state
        limit_v_rib: if True, v_rib is limited to available amino acids
        use_disabled_aas: if False, excluded amino acids are excluded from charging

    Returns:
        5-element tuple containing new_fraction_charged, v_rib,
        total_synthesis, total_import, total_export
    """

    def negative_check(trna1: npt.NDArray[np.float64], trna2: npt.NDArray[np.float64]):
        mask = trna1 < 0
        trna2[mask] = trna1[mask] + trna2[mask]
        trna1[mask] = 0

    def dcdt(t: float, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        v_charging, dtrna, daa = dcdt_jit(
            t,
            c,
            n_aas_masked,
            n_aas,
            mask,
            params["kS"],
            synthetase_conc,
            params["KMaa"],
            params["KMtf"],
            f,
            params["krta"],
            params["krtf"],
            params["max_elong_rate"],
            ribosome_conc,
            limit_v_rib,
            aa_rate_limit,
            v_rib_max,
        )

        if supply is None:
            v_synthesis = np.zeros(n_aas)
            v_import = np.zeros(n_aas)
            v_export = np.zeros(n_aas)
        else:
            aa_conc = c[2 * n_aas_masked : 2 * n_aas_masked + n_aas]
            v_synthesis, v_import, v_export = supply(unit_conversion * aa_conc)
            v_supply = v_synthesis + v_import - v_export
            daa[mask] = v_supply[mask] - v_charging

        return np.hstack((-dtrna, dtrna, daa, v_synthesis, v_import, v_export))

    # Convert inputs for integration
    synthetase_conc = synthetase_conc.asNumber(MICROMOLAR_UNITS)
    uncharged_trna_conc = uncharged_trna_conc.asNumber(MICROMOLAR_UNITS)
    charged_trna_conc = charged_trna_conc.asNumber(MICROMOLAR_UNITS)
    aa_conc = aa_conc.asNumber(MICROMOLAR_UNITS)
    ribosome_conc = ribosome_conc.asNumber(MICROMOLAR_UNITS)
    unit_conversion = params["unit_conversion"]

    # Remove disabled amino acids from calculations
    n_total_aas = len(aa_conc)
    if use_disabled_aas:
        mask = np.ones(n_total_aas, bool)
    else:
        mask = params["charging_mask"]
    synthetase_conc = synthetase_conc[mask]
    original_uncharged_trna_conc = uncharged_trna_conc[mask]
    original_charged_trna_conc = charged_trna_conc[mask]
    original_aa_conc = aa_conc[mask]
    f = f[mask]

    n_aas = len(aa_conc)
    n_aas_masked = len(original_aa_conc)

    # Limits for integration
    aa_rate_limit = original_aa_conc / time_limit
    trna_rate_limit = original_charged_trna_conc / time_limit
    v_rib_max = max(0, ((aa_rate_limit + trna_rate_limit) / f).min())

    # Integrate rates of charging and elongation
    c_init = np.hstack(
        (
            original_uncharged_trna_conc,
            original_charged_trna_conc,
            aa_conc,
            np.zeros(n_aas),
            np.zeros(n_aas),
            np.zeros(n_aas),
        )
    )
    sol = solve_ivp(dcdt, [0, time_limit], c_init, method="BDF")
    c_sol = sol.y.T

    # Determine new values from integration results
    final_uncharged_trna_conc = c_sol[-1, :n_aas_masked]
    final_charged_trna_conc = c_sol[-1, n_aas_masked : 2 * n_aas_masked]
    total_synthesis = c_sol[-1, 2 * n_aas_masked + n_aas : 2 * n_aas_masked + 2 * n_aas]
    total_import = c_sol[
        -1, 2 * n_aas_masked + 2 * n_aas : 2 * n_aas_masked + 3 * n_aas
    ]
    total_export = c_sol[
        -1, 2 * n_aas_masked + 3 * n_aas : 2 * n_aas_masked + 4 * n_aas
    ]

    negative_check(final_uncharged_trna_conc, final_charged_trna_conc)
    negative_check(final_charged_trna_conc, final_uncharged_trna_conc)

    fraction_charged = final_charged_trna_conc / (
        final_uncharged_trna_conc + final_charged_trna_conc
    )
    numerator_ribosome = 1 + np.sum(
        f
        * (
            params["krta"] / final_charged_trna_conc
            + final_uncharged_trna_conc
            / final_charged_trna_conc
            * params["krta"]
            / params["krtf"]
        )
    )
    v_rib = params["max_elong_rate"] * ribosome_conc / numerator_ribosome
    if limit_v_rib:
        v_rib_max = max(
            0,
            (
                (
                    original_aa_conc
                    + (original_charged_trna_conc - final_charged_trna_conc)
                )
                / time_limit
                / f
            ).min(),
        )
        v_rib = min(v_rib, v_rib_max)

    # Replace SEL fraction charged with average
    new_fraction_charged = np.zeros(n_total_aas)
    new_fraction_charged[mask] = fraction_charged
    new_fraction_charged[~mask] = fraction_charged.mean()

    return new_fraction_charged, v_rib, total_synthesis, total_import, total_export


@njit(error_model="numpy")
def dcdt_jit(
    t,
    c,
    n_aas_masked,
    n_aas,
    mask,
    kS,
    synthetase_conc,
    KMaa,
    KMtf,
    f,
    krta,
    krtf,
    max_elong_rate,
    ribosome_conc,
    limit_v_rib,
    aa_rate_limit,
    v_rib_max,
):
    uncharged_trna_conc = c[:n_aas_masked]
    charged_trna_conc = c[n_aas_masked : 2 * n_aas_masked]
    aa_conc = c[2 * n_aas_masked : 2 * n_aas_masked + n_aas]
    masked_aa_conc = aa_conc[mask]

    v_charging = (
        kS
        * synthetase_conc
        * uncharged_trna_conc
        * masked_aa_conc
        / (KMaa[mask] * KMtf[mask])
        / (
            1
            + uncharged_trna_conc / KMtf[mask]
            + masked_aa_conc / KMaa[mask]
            + uncharged_trna_conc * masked_aa_conc / KMtf[mask] / KMaa[mask]
        )
    )
    numerator_ribosome = 1 + np.sum(
        f
        * (
            krta / charged_trna_conc
            + uncharged_trna_conc / charged_trna_conc * krta / krtf
        )
    )
    v_rib = max_elong_rate * ribosome_conc / numerator_ribosome

    # Handle case when f is 0 and charged_trna_conc is 0
    if not np.isfinite(v_rib):
        v_rib = 0

    # Limit v_rib and v_charging to the amount of available amino acids
    if limit_v_rib:
        v_charging = np.fmin(v_charging, aa_rate_limit)
        v_rib = min(v_rib, v_rib_max)

    dtrna = v_charging - v_rib * f
    daa = np.zeros(n_aas)

    return v_charging, dtrna, daa


def get_charging_supply_function(
    supply_in_charging: bool,
    mechanistic_supply: bool,
    mechanistic_aa_transport: bool,
    amino_acid_synthesis: Callable,
    amino_acid_import: Callable,
    amino_acid_export: Callable,
    aa_supply_scaling: Callable,
    counts_to_molar: Unum,
    aa_supply: npt.NDArray[np.float64],
    fwd_enzyme_counts: npt.NDArray[np.int64],
    rev_enzyme_counts: npt.NDArray[np.int64],
    dry_mass: Unum,
    importer_counts: npt.NDArray[np.int64],
    exporter_counts: npt.NDArray[np.int64],
    aa_in_media: npt.NDArray[np.bool_],
) -> Optional[Callable[[npt.NDArray[np.float64]], Tuple[Unum, Unum, Unum]]]:
    """
    Get a function mapping internal amino acid concentrations to the amount of
    amino acid supply expected.
    """

    supply_function = None
    if supply_in_charging:
        counts_to_molar = counts_to_molar.asNumber(MICROMOLAR_UNITS)
        zeros = counts_to_molar * np.zeros_like(aa_supply)
        if mechanistic_supply:
            if mechanistic_aa_transport:

                def supply_function(aa_conc):
                    return (
                        counts_to_molar
                        * amino_acid_synthesis(
                            fwd_enzyme_counts, rev_enzyme_counts, aa_conc
                        )[0],
                        counts_to_molar
                        * amino_acid_import(
                            aa_in_media,
                            dry_mass,
                            aa_conc,
                            importer_counts,
                            mechanistic_aa_transport,
                        ),
                        counts_to_molar
                        * amino_acid_export(
                            exporter_counts, aa_conc, mechanistic_aa_transport
                        ),
                    )
            else:

                def supply_function(aa_conc):
                    return (
                        counts_to_molar
                        * amino_acid_synthesis(
                            fwd_enzyme_counts, rev_enzyme_counts, aa_conc
                        )[0],
                        counts_to_molar
                        * amino_acid_import(
                            aa_in_media,
                            dry_mass,
                            aa_conc,
                            importer_counts,
                            mechanistic_aa_transport,
                        ),
                        zeros,
                    )
        else:

            def supply_function(aa_conc):
                return (
                    counts_to_molar
                    * aa_supply
                    * aa_supply_scaling(aa_conc, aa_in_media),
                    zeros,
                    zeros,
                )

    return supply_function
