"""
======================
Polypeptide Elongation
======================

This process models the polymerization of amino acids into polypeptides by
ribosomes using an mRNA transcript as a template.

Mathematical Model
------------------
**Amino acid polymerization**

Each active ribosome extends its polypeptide by incorporating amino acids
from the template sequence. The ``polymerize`` algorithm distributes
limited amino acid pools across all ribosomes to maximize total elongation:

    sequences = buildSequences(protein_seqs, ribosome_positions, v * dt)
    result = polymerize(sequences, aa_counts, rate_limit)

Each amino acid incorporation also consumes GTP for EF-Tu/EF-G cycling:

    GTP_consumed = n_elongations * gtpPerElongation  (default 4.2 GTP/aa)

**ppGpp regulation** (optional, via ``include_ppgpp``)

ppGpp inhibits elongation rate through a fitted relationship:

    v_elong = f_ppgpp([ppGpp])   [aa/s]

ppGpp synthesis/degradation is modeled via RelA and SpoT enzymes:

    d[ppGpp]/dt = k_RelA * [RelA] * [uncharged_tRNA] / (KD + [uncharged_tRNA])
                - k_SpoT * [SpoT] * [ppGpp] / (KI + [ppGpp])
                + k_SpoT_syn * [SpoT]

**tRNA charging** (optional, via ``steady_state_trna_charging``)

For each amino acid species a, the fraction of charged tRNA is tracked:

    f_charged_a = [charged_tRNA_a] / ([charged_tRNA_a] + [uncharged_tRNA_a])

The effective elongation rate is modulated by the minimum f_charged across
all amino acid species, coupling translation speed to tRNA availability.

Charging reactions consume amino acids, ATP, and uncharged tRNAs, producing
charged tRNAs, AMP, and PPi, governed by aminoacyl-tRNA synthetase kinetics.

**Amino acid supply** (optional, via ``mechanistic_aa_transport``)

Amino acid pools are replenished by synthesis, import, and recycled from
degradation. Export rates are also tracked. Supply rates inform the
metabolism process via kinetic constraints.
"""

from typing import Any, Callable, Optional, Tuple

from numba import njit
import numpy as np
import numpy.typing as npt
from scipy.integrate import solve_ivp
# wcEcoli imports
from wholecell.utils.polymerize import buildSequences, polymerize, computeMassIncrease
from wholecell.utils.random import stochasticRound
from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.unit_bridge import unum_to_pint

# vivarium imports
# simulate_process removed
from bigraph_schema import deep_merge
from v2ecoli.library.unit_defs import units as vivunits
# plot_variables removed

# vivarium-ecoli imports
from v2ecoli.library.schema import (
    listener_schema,
    numpy_schema,
    counts,
    attrs,
    bulk_name_to_idx,
)
# topology_registry removed
from v2ecoli.steps.partition import PartitionedProcess
from v2ecoli.processes.metabolism import CONC_UNITS, TIME_UNITS
from v2ecoli.library.schema_types import ACTIVE_RIBOSOME_ARRAY

# Shared constants and kinetics helpers (split into polypeptide/ subpackage)
from v2ecoli.processes.polypeptide.common import (
    MICROMOLAR_UNITS,
    REMOVED_FROM_CHARGING,
)
from v2ecoli.processes.polypeptide.kinetics import (
    ppgpp_metabolite_changes,
    calculate_trna_charging,
    dcdt_jit,
    get_charging_supply_function,
)
from v2ecoli.processes.polypeptide.elongation_models import (
    BaseElongationModel,
    TranslationSupplyElongationModel,
    SteadyStateElongationModel,
)


# Register default topology for this process, associating it with process name
NAME = "ecoli-polypeptide-elongation"
TOPOLOGY = {
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


class PolypeptideElongation(PartitionedProcess):
    """Polypeptide Elongation PartitionedProcess

    defaults:
        proteinIds: array length n of protein names
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        'KD_RelA': {'_type': 'float', '_default': 0.26},
        'KI_SpoT': {'_type': 'float', '_default': 20.0},
        'KMaa': {'_type': 'float', '_default': 100.0},
        'KMtf': {'_type': 'float', '_default': 1.0},
        'aaNames': {'_type': 'list[string]', '_default': []},
        'aaWeightsIncorporated': {'_type': 'array[float]', '_default': np.array([], dtype=float)},
        'aa_enzymes': {'_type': 'list[string]', '_default': []},
        'aa_exchange_names': {'_type': 'list[string]', '_default': []},
        'aa_exporters': {'_type': 'list[string]', '_default': []},
        'aa_from_synthetase': {'_type': 'array[float]', '_default': np.array([], dtype=float)},
        'aa_from_trna': {'_type': 'array[float]', '_default': np.zeros(21, dtype=np.float64)},
        'aa_importers': {'_type': 'list[string]', '_default': []},
        'aa_supply_in_charging': {'_type': 'boolean', '_default': False},
        'aa_supply_scaling': {'_type': 'map[float]', '_default': None},
        'amino_acid_export': 'map[node]',
        'amino_acid_import': 'map[node]',
        'amino_acid_synthesis': 'map[node]',
        'amino_acids': {'_type': 'list[string]', '_default': []},
        'basal_elongation_rate': {'_type': 'float', '_default': 22.0},
        'cellDensity': {'_type': 'unum[g/L]', '_default': 1100.0},
        'charged_trna_names': {'_type': 'list[string]', '_default': []},
        'charging_molecule_names': {'_type': 'list[string]', '_default': []},
        'charging_stoich_matrix': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'degradation_index': {'_type': 'integer', '_default': 1},
        'disable_ppgpp_elongation_inhibition': {'_type': 'boolean', '_default': False},
        'elong_rate_by_ppgpp': {'_type': 'float', '_default': 0},
        'elongation_max': {'_type': 'unum[aa/s]', '_default': 22.0},
        'emit_unique': {'_type': 'boolean', '_default': False},
        'endWeight': {'_type': 'array[float]', '_default': np.array([2.99146113e-08])},
        'get_pathway_enzyme_counts_per_aa': 'method',
        'gtpPerElongation': {'_type': 'float', '_default': 4.2},
        'import_constraint_threshold': {'_type': 'float', '_default': 0},
        'import_threshold': {'_type': 'float', '_default': 1e-05},
        'kS': {'_type': 'float', '_default': 100.0},
        'k_RelA': {'_type': 'float', '_default': 75.0},
        'k_SpoT_deg': {'_type': 'float', '_default': 0.23},
        'k_SpoT_syn': {'_type': 'float', '_default': 2.6},
        'krta': {'_type': 'float', '_default': 1.0},
        'krtf': {'_type': 'float', '_default': 500.0},
        'make_elongation_rates': {'_type': 'method', '_default': None},
        'mechanistic_aa_transport': {'_type': 'boolean', '_default': False},
        'mechanistic_supply': {'_type': 'boolean', '_default': False},
        'mechanistic_translation_supply': {'_type': 'boolean', '_default': False},
        'n_avogadro': {'_type': 'unum[1/mol]', '_default': 6.02214076e+23},
        'next_aa_pad': {'_type': 'integer', '_default': 1},
        'ppgpp': {'_type': 'string', '_default': 'ppGpp'},
        'ppgpp_degradation_reaction': {'_type': 'string', '_default': 'PPGPPSYN-RXN'},
        'ppgpp_reaction_metabolites': {'_type': 'list[string]', '_default': []},
        'ppgpp_reaction_names': {'_type': 'list[string]', '_default': []},
        'ppgpp_reaction_stoich': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'ppgpp_regulation': {'_type': 'boolean', '_default': False},
        'ppgpp_synthesis_reaction': {'_type': 'string', '_default': 'GDPPYPHOSKIN-RXN'},
        'proteinIds': {'_type': 'array[string]', '_default': np.array([], dtype=float)},
        'proteinLengths': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'proteinSequences': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'proton': {'_type': 'string', '_default': 'PROTON'},
        'rela': {'_type': 'string', '_default': 'RELA'},
        'ribosome30S': {'_type': 'string', '_default': 'ribosome30S'},
        'ribosome50S': {'_type': 'string', '_default': 'ribosome50S'},
        'ribosomeElongationRate': {'_type': 'float', '_default': 17.388824902723737},
        'ribosomeElongationRateDict': {'_type': 'map[float]', '_default': {}},
        'seed': {'_type': 'integer', '_default': 0},
        'spot': {'_type': 'string', '_default': 'SPOT'},
        'synthesis_index': {'_type': 'integer', '_default': 0},
        'synthetase_names': {'_type': 'list[string]', '_default': []},
        'time_step': {'_type': 'integer', '_default': 1},
        'translation_aa_supply': {'_type': 'map[float]', '_default': {}},
        'translation_supply': {'_type': 'boolean', '_default': False},
        'trna_charging': {'_type': 'boolean', '_default': False},
        'uncharged_trna_names': {'_type': 'array[string]', '_default': np.array([], dtype=float)},
        'unit_conversion': {'_type': 'float', '_default': 0},
        'variable_elongation': {'_type': 'boolean', '_default': False},
        'water': {'_type': 'string', '_default': 'H2O'},
    }

    def inputs(self):
        return {
            'environment': {'media_id': 'string'},
            'boundary': 'node',
            'listeners': {
                'mass': {
                    'cell_mass': 'float[fg]',
                    'dry_mass': 'float[fg]',
                },
            },
            'active_ribosome': ACTIVE_RIBOSOME_ARRAY,
            'bulk': 'bulk_array',
            'bulk_total': 'bulk_array',
            'polypeptide_elongation': {
                'gtp_to_hydrolyze': 'float',
                'aa_count_diff': 'array[float]',
                'aa_exchange_rates': 'array[float]',
            },
            'timestep': 'integer',
        }

    def outputs(self):
        return {
            'bulk': 'bulk_array',
            'active_ribosome': ACTIVE_RIBOSOME_ARRAY,
            'listeners': {
                'growth_limits': {
                    # Concentrations — micromolar (uM = umol/L)
                    'synthetase_conc': {'_type': 'overwrite[array[float[uM]]]', '_default': []},
                    'uncharged_trna_conc': {'_type': 'overwrite[array[float[uM]]]', '_default': []},
                    'charged_trna_conc': {'_type': 'overwrite[array[float[uM]]]', '_default': []},
                    'aa_conc': {'_type': 'overwrite[array[float[uM]]]', '_default': []},
                    'ribosome_conc': {'_type': 'overwrite[float[uM]]', '_default': 0.0},
                    'ppgpp_conc': {'_type': 'overwrite[float[uM]]', '_default': 0.0},
                    'rela_conc': {'_type': 'overwrite[float[uM]]', '_default': 0.0},
                    'spot_conc': {'_type': 'overwrite[float[uM]]', '_default': 0.0},
                    # Concentration — millimolar (mM = mmol/L)
                    'aa_supply_aa_conc': {'_type': 'overwrite[array[float[mM]]]', '_default': []},
                    # Count deltas and pool sizes
                    'aa_count_diff': {'_type': 'overwrite[array[float]]', '_default': []},
                    'aas_used': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'net_charged': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'aa_allocated': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'aa_pool_size': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'aa_request_size': {'_type': 'overwrite[array[float]]', '_default': []},
                    'active_ribosome_allocated': {'_type': 'overwrite[integer]', '_default': 0},
                    # Saturation fractions and supply
                    'fraction_trna_charged': {'_type': 'overwrite[array[float]]', '_default': []},
                    'fraction_aa_to_elongate': {'_type': 'overwrite[array[float]]', '_default': []},
                    'aa_supply_fraction_fwd': {'_type': 'overwrite[array[float]]', '_default': []},
                    'aa_supply_fraction_rev': {'_type': 'overwrite[array[float]]', '_default': []},
                    'original_aa_supply': {'_type': 'overwrite[array[float]]', '_default': []},
                    'aa_in_media': {'_type': 'overwrite[array[boolean]]', '_default': []},
                    'aa_supply': {'_type': 'overwrite[array[float]]', '_default': []},
                    'aa_synthesis': {'_type': 'overwrite[array[float]]', '_default': []},
                    'aa_import': {'_type': 'overwrite[array[float]]', '_default': []},
                    'aa_export': {'_type': 'overwrite[array[float]]', '_default': []},
                    'aa_supply_enzymes_fwd': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'aa_supply_enzymes_rev': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'aa_importers': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'aa_exporters': {'_type': 'overwrite[array[integer]]', '_default': []},
                    # ppGpp synthesis/degradation rates (umol/L/s)
                    'rela_syn': {'_type': 'overwrite[array[float]]', '_default': []},
                    'spot_syn': {'_type': 'overwrite[float]', '_default': 0.0},
                    'spot_deg': {'_type': 'overwrite[float]', '_default': 0.0},
                    'spot_deg_inhibited': {'_type': 'overwrite[array[float]]', '_default': []},
                    # Charging
                    'trna_charged': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'ntp_used': {'_type': 'overwrite[array[integer]]', '_default': []},  # written by transcript_elongation but lives here too
                },
                'ribosome_data': {
                    # Read by polypeptide_initiation next timestep
                    'effective_elongation_rate': {'_type': 'overwrite[float[aa/s]]', '_default': 0.0},
                    'translation_supply': {'_type': 'overwrite[array[float]]', '_default': []},
                    'aa_count_in_sequence': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'aa_counts': {'_type': 'overwrite[array[float]]', '_default': []},
                    'actual_elongations': {'_type': 'overwrite[integer]', '_default': 0},
                    'actual_elongation_hist': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'elongations_non_terminating_hist': {'_type': 'overwrite[array[integer]]', '_default': []},
                    'did_terminate': {'_type': 'overwrite[integer]', '_default': 0},
                    'termination_loss': {'_type': 'overwrite[integer]', '_default': 0},
                    'num_trpA_terminated': {'_type': 'overwrite[integer]', '_default': 0},
                    'process_elongation_rate': {'_type': 'overwrite[float[aa/s]]', '_default': 0.0},
                },
            },
            'polypeptide_elongation': {
                'gtp_to_hydrolyze': {'_type': 'overwrite[float]', '_default': 0.0},
                'aa_count_diff': {'_type': 'overwrite[array[float]]', '_default': []},
                'aa_exchange_rates': {'_type': 'overwrite[array[float]]', '_default': []},
            },
        }



    def initialize(self, config):

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
        self.n_avogadro = unum_to_pint(self.parameters["n_avogadro"])
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
        self.translation_aa_supply = {
            media: unum_to_pint(rate)
            for media, rate in self.parameters["translation_aa_supply"].items()
        }
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

    def inputs(self):
        return (
            {
                'environment':                 {
                    'media_id': {'_type': 'string', '_default': ''},
                },
                'boundary': 'node',
                'listeners':                 {
                    'mass':                     {
                        'cell_mass': {'_type': 'float[fg]', '_default': 0.0},
                        'dry_mass': {'_type': 'float[fg]', '_default': 0.0},
                    },
                },
                'active_ribosome': {'_type': ACTIVE_RIBOSOME_ARRAY, '_default': []},
                'bulk': {'_type': 'bulk_array', '_default': []},
                'bulk_total': {'_type': 'bulk_array', '_default': []},
                'polypeptide_elongation':                 {
                    'gtp_to_hydrolyze': {'_type': 'float', '_default': 0.0},
                    'aa_count_diff': {'_type': 'array[float]', '_default': []},
                    'aa_exchange_rates': {'_type': 'array[float]', '_default': []},
                },
                'timestep': {'_type': 'integer', '_default': 1},
            }
        )

    def outputs(self):
        return (
            {
                'bulk': {'_type': 'bulk_array', '_default': []},
                'active_ribosome': {'_type': ACTIVE_RIBOSOME_ARRAY, '_default': []},
                'listeners':                 {
                    'growth_limits':                     {
                        'synthetase_conc': {'_type': 'overwrite[array[float[uM]]]', '_default': []},
                        'uncharged_trna_conc': {'_type': 'overwrite[array[float[uM]]]', '_default': []},
                        'charged_trna_conc': {'_type': 'overwrite[array[float[uM]]]', '_default': []},
                        'aa_conc': {'_type': 'overwrite[array[float[uM]]]', '_default': []},
                        'ribosome_conc': {'_type': 'overwrite[float[uM]]', '_default': 0.0},
                        'ppgpp_conc': {'_type': 'overwrite[float[uM]]', '_default': 0.0},
                        'rela_conc': {'_type': 'overwrite[float[uM]]', '_default': 0.0},
                        'spot_conc': {'_type': 'overwrite[float[uM]]', '_default': 0.0},
                        'aa_supply_aa_conc': {'_type': 'overwrite[array[float[mM]]]', '_default': []},
                        'aa_count_diff': {'_type': 'overwrite[array[float]]', '_default': []},
                        'aas_used': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'net_charged': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'aa_allocated': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'aa_pool_size': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'aa_request_size': {'_type': 'overwrite[array[float]]', '_default': []},
                        'active_ribosome_allocated': {'_type': 'overwrite[integer]', '_default': 0},
                        'fraction_trna_charged': {'_type': 'overwrite[array[float]]', '_default': []},
                        'fraction_aa_to_elongate': {'_type': 'overwrite[array[float]]', '_default': []},
                        'aa_supply_fraction_fwd': {'_type': 'overwrite[array[float]]', '_default': []},
                        'aa_supply_fraction_rev': {'_type': 'overwrite[array[float]]', '_default': []},
                        'original_aa_supply': {'_type': 'overwrite[array[float]]', '_default': []},
                        'aa_in_media': {'_type': 'overwrite[array[boolean]]', '_default': []},
                        'aa_supply': {'_type': 'overwrite[array[float]]', '_default': []},
                        'aa_synthesis': {'_type': 'overwrite[array[float]]', '_default': []},
                        'aa_import': {'_type': 'overwrite[array[float]]', '_default': []},
                        'aa_export': {'_type': 'overwrite[array[float]]', '_default': []},
                        'aa_supply_enzymes_fwd': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'aa_supply_enzymes_rev': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'aa_importers': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'aa_exporters': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'rela_syn': {'_type': 'overwrite[array[float]]', '_default': []},
                        'spot_syn': {'_type': 'overwrite[float]', '_default': 0.0},
                        'spot_deg': {'_type': 'overwrite[float]', '_default': 0.0},
                        'spot_deg_inhibited': {'_type': 'overwrite[array[float]]', '_default': []},
                        'trna_charged': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'ntp_used': {'_type': 'overwrite[array[integer]]', '_default': []},
                    },
                    'ribosome_data':                     {
                        'effective_elongation_rate': {'_type': 'overwrite[float[aa/s]]', '_default': 0.0},
                        'translation_supply': {'_type': 'overwrite[array[float]]', '_default': []},
                        'aa_count_in_sequence': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'aa_counts': {'_type': 'overwrite[array[float]]', '_default': []},
                        'actual_elongations': {'_type': 'overwrite[integer]', '_default': 0},
                        'actual_elongation_hist': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'elongations_non_terminating_hist': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'did_terminate': {'_type': 'overwrite[integer]', '_default': 0},
                        'termination_loss': {'_type': 'overwrite[integer]', '_default': 0},
                        'num_trpA_terminated': {'_type': 'overwrite[integer]', '_default': 0},
                        'process_elongation_rate': {'_type': 'overwrite[float[aa/s]]', '_default': 0.0},
                    },
                },
                'polypeptide_elongation':                 {
                    'gtp_to_hydrolyze': {'_type': 'overwrite[float]', '_default': 0.0},
                    'aa_count_diff': {'_type': 'overwrite[array[float]]', '_default': []},
                    'aa_exchange_rates': {'_type': 'overwrite[array[float]]', '_default': []},
                },
            }
        )

    def calculate_request(self, timestep, states):
        """
        Set ribosome elongation rate based on simulation medium environment and elongation rate factor
        which is used to create single-cell variability in growth rate
        The maximum number of amino acids that can be elongated in a single timestep is set to 22
        intentionally as the minimum number of padding values on the protein sequence matrix is set to 22.
        If timesteps longer than 1.0s are used, this feature will lead to errors in the effective ribosome
        elongation rate.
        """

        if self.proton_idx is None:
            bulk_ids = states["bulk"]["id"]
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

        # MODEL SPECIFIC: get ribosome elongation rate
        self.ribosomeElongationRate = self.elongation_model.elongation_rate(states)

        # If there are no active ribosomes, return immediately
        if states["active_ribosome"]["_entryState"].sum() == 0:
            return {"listeners": {"ribosome_data": {}, "growth_limits": {}}}

        # Build sequences to request appropriate amount of amino acids to
        # polymerize for next timestep
        (
            proteinIndexes,
            peptideLengths,
        ) = attrs(states["active_ribosome"], ["protein_index", "peptide_length"])

        self.elongation_rates = self.make_elongation_rates(
            self.random_state,
            self.ribosomeElongationRate,
            states["timestep"],
            self.variable_elongation,
        )

        sequences = buildSequences(
            self.proteinSequences, proteinIndexes, peptideLengths, self.elongation_rates
        )

        sequenceHasAA = sequences != polymerize.PAD_VALUE
        aasInSequences = np.bincount(sequences[sequenceHasAA], minlength=21)

        # Calculate AA supply for expected doubling of protein
        dryMass = states["listeners"]["mass"]["dry_mass"] * units.fg
        current_media_id = states["environment"]["media_id"]
        translation_supply_rate = (
            self.translation_aa_supply[current_media_id] * self.elngRateFactor
        )
        mol_aas_supplied = (
            translation_supply_rate * dryMass * states["timestep"] * units.s
        )
        # mol_aas_supplied has units mol; * n_avogadro (1/mol) gives a count.
        self.aa_supply = (mol_aas_supplied * self.n_avogadro).to("dimensionless").magnitude

        # MODEL SPECIFIC: Calculate AA request
        fraction_charged, aa_counts_for_translation, requests = (
            self.elongation_model.request(states, aasInSequences)
        )

        # Write to listeners
        listeners = requests.setdefault("listeners", {})
        ribosome_data_listener = listeners.setdefault("ribosome_data", {})
        ribosome_data_listener["translation_supply"] = (
            translation_supply_rate.magnitude
        )
        growth_limits_listener = requests["listeners"].setdefault("growth_limits", {})
        growth_limits_listener["fraction_trna_charged"] = np.dot(
            fraction_charged, self.aa_from_trna
        )
        growth_limits_listener["aa_pool_size"] = counts(
            states["bulk_total"], self.amino_acid_idx
        )
        growth_limits_listener["aa_request_size"] = aa_counts_for_translation
        # Simulations without mechanistic translation supply need this to be
        # manually zeroed after division
        proc_data = requests.setdefault("polypeptide_elongation", {})
        proc_data.setdefault("aa_exchange_rates", np.zeros(len(self.amino_acids)))

        return requests

    def evolve_state(self, timestep, states):
        """
        Set ribosome elongation rate based on simulation medium environment and elongation rate factor
        which is used to create single-cell variability in growth rate
        The maximum number of amino acids that can be elongated in a single timestep is set to 22
        intentionally as the minimum number of padding values on the protein sequence matrix is set to 22.
        If timesteps longer than 1.0s are used, this feature will lead to errors in the effective ribosome
        elongation rate.
        """

        update = {
            "listeners": {"ribosome_data": {}, "growth_limits": {}},
            "polypeptide_elongation": {},
            "active_ribosome": {},
            "bulk": [],
        }

        # Begin wcEcoli evolveState()
        # Set values for metabolism in case of early return
        update["polypeptide_elongation"]["gtp_to_hydrolyze"] = 0.0
        update["polypeptide_elongation"]["aa_count_diff"] = np.zeros(
            len(self.amino_acids), dtype=np.float64
        )

        # Get number of active ribosomes
        n_active_ribosomes = states["active_ribosome"]["_entryState"].sum()
        update["listeners"]["growth_limits"]["active_ribosome_allocated"] = (
            n_active_ribosomes
        )
        update["listeners"]["growth_limits"]["aa_allocated"] = counts(
            states["bulk"], self.amino_acid_idx
        )

        # If there are no active ribosomes, return immediately
        if n_active_ribosomes == 0:
            return update

        # Polypeptide elongation requires counts to be updated in real-time
        # so make a writeable copy of bulk counts to do so
        states["bulk"] = counts(states["bulk"], range(len(states["bulk"])))

        # Build amino acids sequences for each ribosome to polymerize
        protein_indexes, peptide_lengths, positions_on_mRNA = attrs(
            states["active_ribosome"],
            ["protein_index", "peptide_length", "pos_on_mRNA"],
        )

        all_sequences = buildSequences(
            self.proteinSequences,
            protein_indexes,
            peptide_lengths,
            self.elongation_rates + self.next_aa_pad,
        )
        sequences = all_sequences[:, : -self.next_aa_pad].copy()

        if sequences.size == 0:
            return update

        # Calculate elongation resource capacity
        aaCountInSequence = np.bincount(sequences[(sequences != polymerize.PAD_VALUE)])
        total_aa_counts = counts(states["bulk"], self.amino_acid_idx)
        charged_trna_counts = counts(states["bulk"], self.charged_trna_idx)

        # MODEL SPECIFIC: Get amino acid counts
        aa_counts_for_translation = self.elongation_model.final_amino_acids(
            total_aa_counts, charged_trna_counts
        )

        # Using polymerization algorithm elongate each ribosome up to the limits
        # of amino acids, sequence, and GTP
        result = polymerize(
            sequences,
            aa_counts_for_translation,
            10000000,  # Set to a large number, the limit is now taken care of in metabolism
            self.random_state,
            self.elongation_rates[protein_indexes],
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
        currElongRate = (sequence_elongations.sum() / n_active_ribosomes) / states[
            "timestep"
        ]

        # Ribosomes that reach the end of their sequences are terminated and
        # dissociated into 30S and 50S subunits. The polypeptide that they are
        # polymerizing is converted into a protein in BulkMolecules
        terminalLengths = self.protein_lengths[protein_indexes]

        didTerminate = updated_lengths == terminalLengths

        terminatedProteins = np.bincount(
            protein_indexes[didTerminate], minlength=self.proteinSequences.shape[0]
        )

        (protein_mass,) = attrs(states["active_ribosome"], ["massDiff_protein"])
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
        states["bulk"][self.monomer_idx] += terminatedProteins

        nTerminated = didTerminate.sum()
        nInitialized = didInitialize.sum()

        update["bulk"].append((self.ribosome30S_idx, nTerminated))
        update["bulk"].append((self.ribosome50S_idx, nTerminated))
        states["bulk"][self.ribosome30S_idx] += nTerminated
        states["bulk"][self.ribosome50S_idx] += nTerminated

        # MODEL SPECIFIC: evolve
        net_charged, aa_count_diff, evolve_update = self.elongation_model.evolve(
            states,
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
            self.ribosomeElongationRate / states["timestep"]
        )

        return update



def test_polypeptide_elongation(return_data=False):
    def make_elongation_rates(random, base, time_step, variable_elongation=False):
        size = 1
        lengths = time_step * np.full(size, base, dtype=np.int64)
        lengths = stochasticRound(random, lengths) if random else np.round(lengths)
        return lengths.astype(np.int64)

    test_config = {
        "time_step": 2,
        "proteinIds": np.array(["TRYPSYN-APROTEIN[c]"]),
        "ribosome30S": "CPLX0-3953[c]",
        "ribosome50S": "CPLX0-3962[c]",
        "make_elongation_rates": make_elongation_rates,
        "proteinLengths": np.array(
            [245]
        ),  # this is the length of proteins in proteinSequences
        "translation_aa_supply": {
            "minimal": (units.mol / units.fg / units.min)
            * np.array(
                [
                    6.73304301e-21,
                    3.63835219e-21,
                    2.89772671e-21,
                    3.88086822e-21,
                    5.04645651e-22,
                    4.45295877e-21,
                    2.64600664e-21,
                    5.35711230e-21,
                    1.26817689e-21,
                    3.81168405e-21,
                    5.66834531e-21,
                    4.30576056e-21,
                    1.70428208e-21,
                    2.24878356e-21,
                    2.49335033e-21,
                    3.47019761e-21,
                    3.83858460e-21,
                    6.34564026e-22,
                    1.86880523e-21,
                    1.40959498e-27,
                    5.20884460e-21,
                ]
            )
        },
        "proteinSequences": np.array(
            [
                [
                    12,
                    10,
                    18,
                    9,
                    13,
                    1,
                    10,
                    9,
                    9,
                    16,
                    20,
                    9,
                    18,
                    15,
                    9,
                    10,
                    20,
                    4,
                    20,
                    13,
                    7,
                    15,
                    9,
                    18,
                    4,
                    10,
                    13,
                    15,
                    14,
                    1,
                    2,
                    14,
                    11,
                    8,
                    20,
                    0,
                    16,
                    13,
                    7,
                    8,
                    12,
                    13,
                    7,
                    1,
                    10,
                    0,
                    14,
                    10,
                    13,
                    7,
                    10,
                    11,
                    20,
                    5,
                    4,
                    1,
                    11,
                    14,
                    16,
                    3,
                    0,
                    5,
                    15,
                    18,
                    7,
                    2,
                    0,
                    9,
                    18,
                    9,
                    0,
                    2,
                    8,
                    6,
                    2,
                    2,
                    18,
                    3,
                    12,
                    20,
                    16,
                    0,
                    15,
                    2,
                    9,
                    20,
                    6,
                    14,
                    14,
                    16,
                    20,
                    16,
                    20,
                    7,
                    11,
                    11,
                    15,
                    10,
                    10,
                    17,
                    9,
                    14,
                    13,
                    13,
                    7,
                    6,
                    10,
                    18,
                    17,
                    10,
                    16,
                    7,
                    2,
                    10,
                    10,
                    9,
                    3,
                    1,
                    2,
                    2,
                    1,
                    16,
                    11,
                    0,
                    8,
                    7,
                    16,
                    9,
                    0,
                    5,
                    20,
                    20,
                    2,
                    8,
                    13,
                    11,
                    11,
                    1,
                    1,
                    9,
                    15,
                    9,
                    17,
                    12,
                    13,
                    14,
                    5,
                    7,
                    16,
                    1,
                    15,
                    1,
                    7,
                    1,
                    7,
                    10,
                    10,
                    14,
                    13,
                    11,
                    16,
                    7,
                    0,
                    13,
                    8,
                    0,
                    0,
                    9,
                    0,
                    0,
                    7,
                    20,
                    14,
                    9,
                    9,
                    14,
                    20,
                    4,
                    20,
                    15,
                    16,
                    16,
                    15,
                    2,
                    11,
                    9,
                    2,
                    10,
                    2,
                    1,
                    10,
                    8,
                    2,
                    7,
                    10,
                    20,
                    9,
                    20,
                    5,
                    12,
                    10,
                    14,
                    14,
                    9,
                    3,
                    20,
                    15,
                    6,
                    18,
                    7,
                    11,
                    3,
                    6,
                    20,
                    1,
                    5,
                    10,
                    0,
                    0,
                    8,
                    4,
                    1,
                    15,
                    9,
                    12,
                    5,
                    6,
                    11,
                    9,
                    0,
                    5,
                    10,
                    3,
                    11,
                    5,
                    20,
                    0,
                    5,
                    1,
                    5,
                    0,
                    0,
                    7,
                    11,
                    20,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]
            ]
        ).astype(np.int8),
    }

    polypep_elong = PolypeptideElongation(test_config)

    initial_state = {
        "environment": {"media_id": "minimal"},
        "bulk": np.array(
            [
                ("CPLX0-3953[c]", 100),
                ("CPLX0-3962[c]", 100),
                ("TRYPSYN-APROTEIN[c]", 0),
                ("RELA", 0),
                ("SPOT", 0),
                ("H2O", 0),
                ("PROTON", 0),
                ("ppGpp", 0),
            ]
            + [(aa, 100) for aa in DEFAULT_AA_NAMES],
            dtype=[("id", "U40"), ("count", int)],
        ),
        "unique": {
            "active_ribosome": np.array(
                [(1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)],
                dtype=[
                    ("_entryState", np.bool_),
                    ("unique_index", int),
                    ("protein_index", int),
                    ("peptide_length", int),
                    ("pos_on_mRNA", int),
                    ("massDiff_DNA", "<f8"),
                    ("massDiff_mRNA", "<f8"),
                    ("massDiff_metabolite", "<f8"),
                    ("massDiff_miscRNA", "<f8"),
                    ("massDiff_nonspecific_RNA", "<f8"),
                    ("massDiff_protein", "<f8"),
                    ("massDiff_rRNA", "<f8"),
                    ("massDiff_tRNA", "<f8"),
                    ("massDiff_water", "<f8"),
                ],
            )
        },
        "listeners": {"mass": {"dry_mass": 350.0}},
    }

    settings = {"total_time": 200, "initial_state": initial_state, "topology": TOPOLOGY}
    data = simulate_process(polypep_elong, settings)

    if return_data:
        return data, test_config


def run_plot(data, config):
    # plot a list of variables
    bulk_ids = [
        "CPLX0-3953[c]",
        "CPLX0-3962[c]",
        "TRYPSYN-APROTEIN[c]",
        "RELA",
        "SPOT",
        "H2O",
        "PROTON",
        "ppGpp",
    ] + [aa for aa in DEFAULT_AA_NAMES]
    variables = [(bulk_id,) for bulk_id in bulk_ids]

    # format data
    bulk_timeseries = np.array(data["bulk"])
    for i, bulk_id in enumerate(bulk_ids):
        data[bulk_id] = bulk_timeseries[:, i]

    plot_variables(
        data,
        variables=variables,
        out_dir="out/processes/polypeptide_elongation",
        filename="variables",
    )


def main():
    data, config = test_polypeptide_elongation(return_data=True)
    run_plot(data, config)


if __name__ == "__main__":
    main()
