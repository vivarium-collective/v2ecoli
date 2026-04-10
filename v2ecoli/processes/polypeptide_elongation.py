"""Process-bigraph partitioned process: polypeptide_elongation."""

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

from unum import Unum
from v2ecoli.processes.metabolism import CONC_UNITS, TIME_UNITS

MICROMOLAR_UNITS = units.umol / units.L


def _apply_config_defaults(config_schema, parameters):
    """Merge config_schema defaults with provided parameters."""
    merged = {}
    for key, spec in config_schema.items():
        if isinstance(spec, dict) and "_default" in spec:
            merged[key] = spec["_default"]
    merged.update(parameters or {})
    return merged

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


class PolypeptideElongationLogic:
    """Polypeptide Elongation — shared state container for Requester/Evolver.

    defaults:
        proteinIds: array length n of protein names
    """

    name = "ecoli-polypeptide-elongation"
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
    config_schema = {
        "time_step": {"_default": 1},
        "n_avogadro": {"_default": None},
        "proteinIds": {"_default": None},
        "proteinLengths": {"_default": None},
        "proteinSequences": {"_default": None},
        "aaWeightsIncorporated": {"_default": None},
        "endWeight": {"_default": None},
        "variable_elongation": {"_default": False},
        "make_elongation_rates": {"_default": None},
        "next_aa_pad": {"_default": 1},
        "ribosomeElongationRate": {"_default": 17.388824902723737},
        "translation_aa_supply": {"_default": None},
        "import_threshold": {"_default": 1e-05},
        "aa_from_trna": {"_default": None},
        "gtpPerElongation": {"_default": 4.2},
        "aa_supply_in_charging": {"_default": False},
        "mechanistic_translation_supply": {"_default": False},
        "mechanistic_aa_transport": {"_default": False},
        "ppgpp_regulation": {"_default": False},
        "disable_ppgpp_elongation_inhibition": {"_default": False},
        "trna_charging": {"_default": False},
        "translation_supply": {"_default": False},
        "mechanistic_supply": {"_default": False},
        "ribosome30S": {"_default": "ribosome30S"},
        "ribosome50S": {"_default": "ribosome50S"},
        "amino_acids": {"_default": None},
        "aa_exchange_names": {"_default": None},
        "basal_elongation_rate": {"_default": 22.0},
        "ribosomeElongationRateDict": {"_default": None},
        "uncharged_trna_names": {"_default": None},
        "aaNames": {"_default": None},
        "aa_enzymes": {"_default": []},
        "proton": {"_default": "PROTON"},
        "water": {"_default": "H2O"},
        "cellDensity": {"_default": None},
        "elongation_max": {"_default": None},
        "aa_from_synthetase": {"_default": None},
        "charging_stoich_matrix": {"_default": None},
        "charged_trna_names": {"_default": []},
        "charging_molecule_names": {"_default": []},
        "synthetase_names": {"_default": []},
        "ppgpp_reaction_names": {"_default": []},
        "ppgpp_reaction_metabolites": {"_default": []},
        "ppgpp_reaction_stoich": {"_default": None},
        "ppgpp_synthesis_reaction": {"_default": "GDPPYPHOSKIN-RXN"},
        "ppgpp_degradation_reaction": {"_default": "PPGPPSYN-RXN"},
        "aa_importers": {"_default": []},
        "amino_acid_export": {"_default": None},
        "synthesis_index": {"_default": 0},
        "aa_exporters": {"_default": []},
        "get_pathway_enzyme_counts_per_aa": {"_default": None},
        "import_constraint_threshold": {"_default": 0},
        "unit_conversion": {"_default": 0},
        "elong_rate_by_ppgpp": {"_default": 0},
        "amino_acid_import": {"_default": None},
        "degradation_index": {"_default": 1},
        "amino_acid_synthesis": {"_default": None},
        "rela": {"_default": "RELA"},
        "spot": {"_default": "SPOT"},
        "ppgpp": {"_default": "ppGpp"},
        "kS": {"_default": 100.0},
        "KMtf": {"_default": 1.0},
        "KMaa": {"_default": 100.0},
        "krta": {"_default": 1.0},
        "krtf": {"_default": 500.0},
        "KD_RelA": {"_default": 0.26},
        "k_RelA": {"_default": 75.0},
        "k_SpoT_syn": {"_default": 2.6},
        "k_SpoT_deg": {"_default": 0.23},
        "KI_SpoT": {"_default": 20.0},
        "aa_supply_scaling": {"_default": None},
        "seed": {"_default": 0},
        "emit_unique": {"_default": False},
    }

    def __init__(self, parameters=None):
        self.parameters = _apply_config_defaults(self.config_schema, parameters)
        self.request_set = False

        # Simulation options
        self.aa_supply_in_charging = self.parameters["aa_supply_in_charging"]
        self.mechanistic_translation_supply = self.parameters[
            "mechanistic_translation_supply"
        ]
        self.mechanistic_aa_transport = self.parameters["mechanistic_aa_transport"]
        self.ppgpp_regulation = self.parameters["ppgpp_regulation"]
        self.disable_ppgpp_elongation_inhibition = self.parameters[
            "disable_ppgpp_elongation_inhibition"
        ]
        self.variable_elongation = self.parameters["variable_elongation"]
        self.variable_polymerize = self.ppgpp_regulation or self.variable_elongation
        translation_supply = self.parameters["translation_supply"]
        trna_charging = self.parameters["trna_charging"]

        # Load parameters
        self.n_avogadro = self.parameters["n_avogadro"]
        self.proteinIds = self.parameters["proteinIds"]
        self.protein_lengths = self.parameters["proteinLengths"]
        self.proteinSequences = self.parameters["proteinSequences"]
        self.aaWeightsIncorporated = self.parameters["aaWeightsIncorporated"]
        self.endWeight = self.parameters["endWeight"]
        self.make_elongation_rates = self.parameters["make_elongation_rates"]
        self.next_aa_pad = self.parameters["next_aa_pad"]

        self.ribosome30S = self.parameters["ribosome30S"]
        self.ribosome50S = self.parameters["ribosome50S"]
        self.amino_acids = self.parameters["amino_acids"]
        self.aa_exchange_names = self.parameters["aa_exchange_names"]
        self.aa_environment_names = [aa[:-3] for aa in self.aa_exchange_names]
        self.aa_enzymes = self.parameters["aa_enzymes"]

        self.ribosomeElongationRate = self.parameters["ribosomeElongationRate"]

        # Amino acid supply calculations
        self.translation_aa_supply = self.parameters["translation_aa_supply"]
        self.import_threshold = self.parameters["import_threshold"]

        # Used for figure in publication
        self.trpAIndex = np.where(self.proteinIds == "TRYPSYN-APROTEIN[c]")[0][0]

        self.elngRateFactor = 1.0

        # Data structures for charging
        self.aa_from_trna = self.parameters["aa_from_trna"]

        # Set modeling method
        # TODO: Test that these models all work properly
        if trna_charging:
            self.elongation_model = SteadyStateElongationModel(self.parameters, self)
        elif translation_supply:
            self.elongation_model = TranslationSupplyElongationModel(
                self.parameters, self
            )
        else:
            self.elongation_model = BaseElongationModel(self.parameters, self)

        # Growth associated maintenance energy requirements for elongations
        self.gtpPerElongation = self.parameters["gtpPerElongation"]
        # Need to account for ATP hydrolysis for charging that has been
        # removed from measured GAM (ATP -> AMP is 2 hydrolysis reactions)
        # if charging reactions are not explicitly modeled
        if not trna_charging:
            self.gtpPerElongation += 2

        # basic molecule names
        self.proton = self.parameters["proton"]
        self.water = self.parameters["water"]
        self.rela = self.parameters["rela"]
        self.spot = self.parameters["spot"]
        self.ppgpp = self.parameters["ppgpp"]
        self.aa_importers = self.parameters["aa_importers"]
        self.aa_exporters = self.parameters["aa_exporters"]
        # Numpy index for bulk molecule
        self.proton_idx = None

        # Names of molecules associated with tRNA charging
        self.ppgpp_reaction_metabolites = self.parameters["ppgpp_reaction_metabolites"]
        self.uncharged_trna_names = self.parameters["uncharged_trna_names"]
        self.charged_trna_names = self.parameters["charged_trna_names"]
        self.charging_molecule_names = self.parameters["charging_molecule_names"]
        self.synthetase_names = self.parameters["synthetase_names"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)



class PolypeptideElongationRequester(_SafeInvokeMixin, Step):
    """Reads stores to compute polypeptide elongation request."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'boundary': InPlaceDict(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'polypeptide_elongation': InPlaceDict(),
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
        if p.proton_idx is None:
            bulk_ids = state["bulk"]["id"]
            p.proton_idx = bulk_name_to_idx(p.proton, bulk_ids)
            p.water_idx = bulk_name_to_idx(p.water, bulk_ids)
            p.rela_idx = bulk_name_to_idx(p.rela, bulk_ids)
            p.spot_idx = bulk_name_to_idx(p.spot, bulk_ids)
            p.ppgpp_idx = bulk_name_to_idx(p.ppgpp, bulk_ids)
            p.monomer_idx = bulk_name_to_idx(p.proteinIds, bulk_ids)
            p.amino_acid_idx = bulk_name_to_idx(p.amino_acids, bulk_ids)
            p.aa_enzyme_idx = bulk_name_to_idx(p.aa_enzymes, bulk_ids)
            p.ppgpp_rxn_metabolites_idx = bulk_name_to_idx(
                p.ppgpp_reaction_metabolites, bulk_ids
            )
            p.uncharged_trna_idx = bulk_name_to_idx(
                p.uncharged_trna_names, bulk_ids
            )
            p.charged_trna_idx = bulk_name_to_idx(p.charged_trna_names, bulk_ids)
            p.charging_molecule_idx = bulk_name_to_idx(
                p.charging_molecule_names, bulk_ids
            )
            p.synthetase_idx = bulk_name_to_idx(p.synthetase_names, bulk_ids)
            p.ribosome30S_idx = bulk_name_to_idx(p.ribosome30S, bulk_ids)
            p.ribosome50S_idx = bulk_name_to_idx(p.ribosome50S, bulk_ids)
            p.aa_importer_idx = bulk_name_to_idx(p.aa_importers, bulk_ids)
            p.aa_exporter_idx = bulk_name_to_idx(p.aa_exporters, bulk_ids)

        # MODEL SPECIFIC: get ribosome elongation rate
        p.ribosomeElongationRate = p.elongation_model.elongation_rate(state)

        # If there are no active ribosomes, return immediately
        if state["active_ribosome"]["_entryState"].sum() == 0:
            request = {"listeners": {"ribosome_data": {}, "growth_limits": {}}}
        else:
            # Build sequences to request appropriate amount of amino acids to
            # polymerize for next timestep
            (
                proteinIndexes,
                peptideLengths,
            ) = attrs(state["active_ribosome"], ["protein_index", "peptide_length"])

            p.elongation_rates = p.make_elongation_rates(
                p.random_state,
                p.ribosomeElongationRate,
                state["timestep"],
                p.variable_elongation,
            )

            sequences = buildSequences(
                p.proteinSequences, proteinIndexes, peptideLengths, p.elongation_rates
            )

            sequenceHasAA = sequences != polymerize.PAD_VALUE
            aasInSequences = np.bincount(sequences[sequenceHasAA], minlength=21)

            # Calculate AA supply for expected doubling of protein
            dryMass = state["listeners"]["mass"]["dry_mass"] * units.fg
            current_media_id = state["environment"]["media_id"]
            translation_supply_rate = (
                p.translation_aa_supply[current_media_id] * p.elngRateFactor
            )
            mol_aas_supplied = (
                translation_supply_rate * dryMass * state["timestep"] * units.s
            )
            p.aa_supply = units.strip_empty_units(mol_aas_supplied * p.n_avogadro)

            # MODEL SPECIFIC: Calculate AA request
            fraction_charged, aa_counts_for_translation, request = (
                p.elongation_model.request(state, aasInSequences)
            )

            # Write to listeners
            listeners = request.setdefault("listeners", {})
            ribosome_data_listener = listeners.setdefault("ribosome_data", {})
            ribosome_data_listener["translation_supply"] = (
                translation_supply_rate.asNumber()
            )
            growth_limits_listener = request["listeners"].setdefault("growth_limits", {})
            growth_limits_listener["fraction_trna_charged"] = np.dot(
                fraction_charged, p.aa_from_trna
            )
            growth_limits_listener["aa_pool_size"] = counts(
                state["bulk_total"], p.amino_acid_idx
            )
            growth_limits_listener["aa_request_size"] = aa_counts_for_translation
            # Simulations without mechanistic translation supply need this to be
            # manually zeroed after division
            proc_data = request.setdefault("polypeptide_elongation", {})
            proc_data.setdefault("aa_exchange_rates", np.zeros(len(p.amino_acids)))
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


class PolypeptideElongationEvolver(_SafeInvokeMixin, Step):
    """Reads allocation, writes bulk/unique/listener/boundary updates."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': SetStore(),
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'environment': InPlaceDict(),
            'boundary': InPlaceDict(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'polypeptide_elongation': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(
                _default=self.process.parameters.get('time_step', 1.0)),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'active_ribosome': UniqueNumpyUpdate(),
            'polypeptide_elongation': InPlaceDict(),
            'boundary': InPlaceDict(),
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
        update = {
            "listeners": {"ribosome_data": {}, "growth_limits": {}},
            "polypeptide_elongation": {},
            "active_ribosome": {},
            "bulk": [],
        }

        # Begin wcEcoli evolveState()
        # Set values for metabolism in case of early return
        update["polypeptide_elongation"]["gtp_to_hydrolyze"] = 0
        update["polypeptide_elongation"]["aa_count_diff"] = np.zeros(
            len(p.amino_acids), dtype=np.float64
        )

        # Get number of active ribosomes
        n_active_ribosomes = state["active_ribosome"]["_entryState"].sum()
        update["listeners"]["growth_limits"]["active_ribosome_allocated"] = (
            n_active_ribosomes
        )
        update["listeners"]["growth_limits"]["aa_allocated"] = counts(
            state["bulk"], p.amino_acid_idx
        )

        # If there are no active ribosomes, return immediately
        if n_active_ribosomes != 0:
            # Polypeptide elongation requires counts to be updated in real-time
            # so make a writeable copy of bulk counts to do so
            state["bulk"] = counts(state["bulk"], range(len(state["bulk"])))

            # Build amino acids sequences for each ribosome to polymerize
            protein_indexes, peptide_lengths, positions_on_mRNA = attrs(
                state["active_ribosome"],
                ["protein_index", "peptide_length", "pos_on_mRNA"],
            )

            all_sequences = buildSequences(
                p.proteinSequences,
                protein_indexes,
                peptide_lengths,
                p.elongation_rates + p.next_aa_pad,
            )
            sequences = all_sequences[:, : -p.next_aa_pad].copy()

            if sequences.size != 0:
                # Calculate elongation resource capacity
                aaCountInSequence = np.bincount(sequences[(sequences != polymerize.PAD_VALUE)])
                total_aa_counts = counts(state["bulk"], p.amino_acid_idx)
                charged_trna_counts = counts(state["bulk"], p.charged_trna_idx)

                # MODEL SPECIFIC: Get amino acid counts
                aa_counts_for_translation = p.elongation_model.final_amino_acids(
                    total_aa_counts, charged_trna_counts
                )

                # Using polymerization algorithm elongate each ribosome up to the limits
                # of amino acids, sequence, and GTP
                result = polymerize(
                    sequences,
                    aa_counts_for_translation,
                    10000000,  # Set to a large number, the limit is now taken care of in metabolism
                    p.random_state,
                    p.elongation_rates[protein_indexes],
                    variable_elongation=p.variable_polymerize,
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
                    sequences, sequence_elongations, p.aaWeightsIncorporated
                )

                updated_lengths = peptide_lengths + sequence_elongations
                updated_positions_on_mRNA = positions_on_mRNA + 3 * sequence_elongations

                didInitialize = (sequence_elongations > 0) & (peptide_lengths == 0)

                added_protein_mass[didInitialize] += p.endWeight

                # Write current average elongation to listener
                currElongRate = (sequence_elongations.sum() / n_active_ribosomes) / state[
                    "timestep"
                ]

                # Ribosomes that reach the end of their sequences are terminated and
                # dissociated into 30S and 50S subunits. The polypeptide that they are
                # polymerizing is converted into a protein in BulkMolecules
                terminalLengths = p.protein_lengths[protein_indexes]

                didTerminate = updated_lengths == terminalLengths

                terminatedProteins = np.bincount(
                    protein_indexes[didTerminate], minlength=p.proteinSequences.shape[0]
                )

                (protein_mass,) = attrs(state["active_ribosome"], ["massDiff_protein"])
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

                update["bulk"].append((p.monomer_idx, terminatedProteins))
                state["bulk"][p.monomer_idx] += terminatedProteins

                nTerminated = didTerminate.sum()
                nInitialized = didInitialize.sum()

                update["bulk"].append((p.ribosome30S_idx, nTerminated))
                update["bulk"].append((p.ribosome50S_idx, nTerminated))
                state["bulk"][p.ribosome30S_idx] += nTerminated
                state["bulk"][p.ribosome50S_idx] += nTerminated

                # MODEL SPECIFIC: evolve
                net_charged, aa_count_diff, evolve_update = p.elongation_model.evolve(
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
                # GTP hydrolysis is carried out in Metabolism process for growth
                # associated maintenance. This is passed to metabolism.
                update["polypeptide_elongation"]["gtp_to_hydrolyze"] = (
                    p.gtpPerElongation * nElongations
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
                    p.trpAIndex
                ]
                ribosome_data_listener["process_elongation_rate"] = (
                    p.ribosomeElongationRate / state["timestep"]
                )
        # --- end inlined ---
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
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
        Sets ribosome elongation rate accordint to the media; returns
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

        # Bulk requests have to be integers (wcEcoli implicitly casts floats to ints)
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
        Sets ribosome elongation rate accordint to the media; returns
        max value of 22 amino acids/second.
        """
        return self.basal_elongation_rate

    def amino_acid_counts(self, aasInSequences):
        # Check if this is required. It is a better request but there may be
        # fewer elongations.
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

        # Store threshold as plain float (mM magnitude) to avoid pint
        # registry mismatch when comparing with deserialized state values
        threshold = self.parameters["import_constraint_threshold"]
        if hasattr(threshold, 'magnitude'):
            self.import_constraint_threshold = float(threshold.magnitude)
        elif hasattr(threshold, 'asNumber'):
            self.import_constraint_threshold = float(threshold.asNumber())
        else:
            self.import_constraint_threshold = float(threshold)

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
        from v2ecoli.processes.metabolism import _to_mM
        aa_in_media = np.array(
            [
                _to_mM(states["boundary"]["external"][aa])
                > self.import_constraint_threshold
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
        # Use the supply calculated from the starting amino acid concentrations only
        elif self.process.mechanistic_translation_supply:
            # Set supply based on mechanistic synthesis and supply
            self.process.aa_supply = states["timestep"] * (synthesis + exchange_rates)
        else:
            # Adjust aa_supply higher if amino acid concentrations are low
            # Improves stability of charging and mimics amino acid synthesis
            # inhibition and export
            # Polypeptide elongation operates using concentration units of CONC_UNITS (uM)
            # but aa_supply_scaling uses M units, so convert using unit_conversion (1e-6)
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
            # Request water for transfer of AA from tRNA for initial polypeptide.
            # This is severe overestimate assuming the worst case that every
            # elongation is initializing a polypeptide. This excess of water
            # shouldn't matter though.
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
        ## Determine limitations for charging and uncharging reactions
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

        ## Determine reactions that are charged and elongated in same time step without changing
        ## charged or uncharged counts
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

        # Update proton counts to reflect polymerization reactions and transfer of AA from tRNA
        # Peptide bond formation releases a water but transferring AA from tRNA consumes a OH-
        # Net production of H+ for each elongation, consume extra water for each initialization
        # since a peptide bond doesn't form
        update["bulk"].append((self.process.proton_idx, nElongations))
        update["bulk"].append((self.process.water_idx, -nInitialized))
        states["bulk"][self.process.proton_idx] += nElongations
        states["bulk"][self.process.water_idx] += -nInitialized

        # Create or degrade ppGpp
        # This should come after all countInc/countDec calls since it shares some molecules with
        # other views and those counts should be updated to get the proper limits on ppGpp reactions
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

            # Need to include the next amino acid the ribosome sees for certain
            # cases where elongation does not occur, otherwise f will be NaN
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

        # Use the difference between (expected AA supply based on expected
        # doubling time and current DCW) and AA used to charge tRNA to update
        # the concentration target in metabolism during the next time step
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
        each amino acid. Uses self.process.aa_from_trna mapping to distribute
        from amino acids to tRNA based on the fraction that each tRNA species
        makes up for all tRNA species that code for the same amino acid.

        Args:
            n_aa: counts of each amino acid to distribute to each tRNA
            n_trna: counts of each tRNA to determine the distribution
            limited: optional, if True, limits the amino acids
                distributed to each tRNA to the number of tRNA that are
                available (n_trna)

        Returns:
            Distributed counts for each tRNA
        """

        # Determine the fraction each tRNA species makes up out of all tRNA of
        # the associated amino acid
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

            # Add additional counts to get up to counts to distribute
            # Prevent adding over the number of tRNA available if limited
            if diff > 0:
                if limited:
                    for _ in range(diff):
                        frac[(n_trna[idx] - counts) == 0] = 0
                        # normalize for multinomial distribution
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
        uncharged_trna_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
            of uncharged tRNA associated with each amino acid
        charged_trna_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
            of charged tRNA associated with each amino acid
        ribosome_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
            of active ribosomes
        f: fraction of each amino acid to be incorporated
            to total amino acids incorporated
        rela_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`) of RelA
        spot_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`) of SpoT
        ppgpp_conc: concentration (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`) of ppGpp
        counts_to_molar: conversion factor
            from counts to molarity (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
        v_rib: rate of amino acid incorporation at the ribosome (units of uM/s)
        charging_params: parameters used in charging equations
        ppgpp_params: parameters used in ppGpp reactions
        time_step: length of the current time step
        request: if True, only considers reactant stoichiometry,
            otherwise considers reactants and products. For use in
            calculateRequest. GDP appears as both a reactant and product
            and the request can be off the actual use if not handled in this
            manner.
        limits: counts of molecules that are available to prevent
            negative total counts as a result of delta_metabolites.
            If None, no limits are placed on molecule changes.
        random_state: random state for the process
    Returns:
        7-element tuple containing

        - **delta_metabolites**: the change in counts of each metabolite
          involved in ppGpp reactions
        - **n_syn_reactions**: the number of ppGpp synthesis reactions
        - **n_deg_reactions**: the number of ppGpp degradation reactions
        - **v_rela_syn**: rate of synthesis from RelA per amino
          acid tRNA species
        - **v_spot_syn**: rate of synthesis from SpoT
        - **v_deg**: rate of degradation from SpoT
        - **v_deg_inhibited**: rate of degradation from SpoT per
          amino acid tRNA species
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
    # Can result in inf and nan so assume a fraction of ribosomes
    # bind to the uncharged tRNA if any tRNA are present or 0 if not
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
    # Possible reactions are adjusted down to limits if the change in any
    # metabolites would result in negative counts
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
    incorporation through polypeptide elongation. The fraction of
    charged/uncharged is also used to determine how quickly the
    ribosome is elongating. All concentrations are given in units of
    :py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`.

    Args:
        synthetase_conc: concentration of synthetases associated with
            each amino acid
        uncharged_trna_conc: concentration of uncharged tRNA associated
            with each amino acid
        charged_trna_conc: concentration of charged tRNA associated with
            each amino acid
        aa_conc: concentration of each amino acid
        ribosome_conc: concentration of active ribosomes
        f: fraction of each amino acid to be incorporated to total amino
            acids incorporated
        params: parameters used in charging equations
        supply: function to get the rate of amino acid supply (synthesis
            and import) based on amino acid concentrations. If None, amino
            acid concentrations remain constant during charging
        time_limit: time limit to reach steady state
        limit_v_rib: if True, v_rib is limited to the number of amino acids
            that are available
        use_disabled_aas: if False, amino acids in
            :py:data:`~ecoli.processes.polypeptide_elongation.REMOVED_FROM_CHARGING`
            are excluded from charging

    Returns:
        5-element tuple containing

        - **new_fraction_charged**: fraction of total tRNA that is charged for each
          amino acid species
        - **v_rib**: ribosomal elongation rate in units of uM/s
        - **total_synthesis**: the total amount of amino acids synthesized during charging
          in units of MICROMOLAR_UNITS. Will be zeros if supply function is not given.
        - **total_import**: the total amount of amino acids imported during charging
          in units of MICROMOLAR_UNITS. Will be zeros if supply function is not given.
        - **total_export**: the total amount of amino acids exported during charging
          in units of MICROMOLAR_UNITS. Will be zeros if supply function is not given.
    """

    def negative_check(trna1: npt.NDArray[np.float64], trna2: npt.NDArray[np.float64]):
        """
        Check for floating point precision issues that can lead to small
        negative numbers instead of 0. Adjusts both species of tRNA to
        bring concentration of trna1 to 0 and keep the same total concentration.

        Args:
            trna1: concentration of one tRNA species (charged or uncharged)
            trna2: concentration of another tRNA species (charged or uncharged)
        """

        mask = trna1 < 0
        trna2[mask] = trna1[mask] + trna2[mask]
        trna1[mask] = 0

    def dcdt(t: float, c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Function for solve_ivp to integrate

        Args:
            c: 1D array of concentrations of uncharged and charged tRNAs
                dims: 2 * number of amino acids (uncharged tRNA come first, then charged)
            t: time of integration step

        Returns:
            Array of dc/dt for tRNA concentrations
                dims: 2 * number of amino acids (uncharged tRNA come first, then charged)
        """

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

    Args:
        supply_in_charging: True if using aa_supply_in_charging option
        mechanistic_supply: True if using mechanistic_translation_supply option
        mechanistic_aa_transport: True if using mechanistic_aa_transport option
        amino_acid_synthesis: function to provide rates of synthesis for amino
            acids based on the internal state
        amino_acid_import: function to provide import rates for amino
            acids based on the internal and external state
        amino_acid_export: function to provide export rates for amino
            acids based on the internal state
        aa_supply_scaling: function to scale the amino acid supply based
            on the internal state
        counts_to_molar: conversion factor for counts to molar
            (:py:data:`~ecoli.processes.polypeptide_elongation.MICROMOLAR_UNITS`)
        aa_supply: rate of amino acid supply expected
        fwd_enzyme_counts: enzyme counts in forward reactions for each amino acid
        rev_enzyme_counts: enzyme counts in loss reactions for each amino acid
        dry_mass: dry mass of the cell with mass units
        importer_counts: counts for amino acid importers
        exporter_counts: counts for amino acid exporters
        aa_in_media: True for each amino acid that is present in the media
    Returns:
        Function that provides the amount of supply (synthesis, import, export)
        for each amino acid based on the internal state of the cell
    """

    # Create functions that are only dependent on amino acid concentrations for more stable
    # charging and amino acid concentrations.  If supply_in_charging is not set, then
    # setting None will maintain constant amino acid concentrations throughout charging.
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
