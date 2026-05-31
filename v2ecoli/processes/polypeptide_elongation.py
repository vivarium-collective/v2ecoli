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
from v2ecoli.library.polymerize import buildSequences, polymerize, computeMassIncrease
from wholecell.utils.random import stochasticRound
from v2ecoli.types.quantity import ureg as units
from v2ecoli.library.quantity_helpers import as_quantity, fg_magnitude
from v2ecoli.library.unit_bridge import unum_to_pint, pint_to_unum

from bigraph_schema import deep_merge

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


class BasePolypeptideElongation(PartitionedProcess):
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
        'cellDensity': {'_type': 'quantity[g/L]', '_default': 1100.0},
        'charged_trna_names': {'_type': 'list[string]', '_default': []},
        'charging_molecule_names': {'_type': 'list[string]', '_default': []},
        'charging_stoich_matrix': {'_type': 'array[integer]', '_default': np.array([], dtype=float)},
        'degradation_index': {'_type': 'integer', '_default': 1},
        'disable_ppgpp_elongation_inhibition': {'_type': 'boolean', '_default': False},
        'elong_rate_by_ppgpp': {'_type': 'float', '_default': 0},
        'elongation_max': {'_type': 'quantity[aa/s]', '_default': 22.0},
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
        'n_avogadro': {'_type': 'quantity[float,1/mol]', '_default': 6.02214076e+23},
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
        'ribosomeElongationRateDict': {'_type': 'map[quantity[float,amino_acid/s]]', '_default': {}},
        'seed': {'_type': 'integer', '_default': 0},
        'spot': {'_type': 'string', '_default': 'SPOT'},
        'synthesis_index': {'_type': 'integer', '_default': 0},
        'synthetase_names': {'_type': 'list[string]', '_default': []},
        'time_step': {'_type': 'integer', '_default': 1},
        'translation_aa_supply': {'_type': 'map[quantity[array[float],mol/fg/min]]', '_default': {}},
        'uncharged_trna_names': {'_type': 'array[string]', '_default': np.array([], dtype=float)},
        'unit_conversion': {'_type': 'float', '_default': 0},
        'variable_elongation': {'_type': 'boolean', '_default': False},
        'water': {'_type': 'string', '_default': 'H2O'},
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

        # Pre-strip the per-media supply rates to plain numpy arrays in
        # mol/(fg·s). The original schema declares them as Quantities in
        # mol/(fg·min); convert and divide by 60 here once instead of
        # constructing a fresh Quantity chain each tick inside
        # calculate_request(). Listener output is preserved in the original
        # mol/(fg·min) basis via _translation_aa_supply_min_mag below.
        self._translation_aa_supply_per_s = {
            media: float_arr
            for media, q in self.translation_aa_supply.items()
            for float_arr in (
                np.asarray(
                    q.to(units.mol / units.fg / units.s).magnitude,
                    dtype=np.float64),
            )
        }
        self._translation_aa_supply_min_mag = {
            media: np.asarray(q.magnitude, dtype=np.float64)
            for media, q in self.translation_aa_supply.items()
        }
        # Avogadro as a plain float — used as a count-per-mole multiplier.
        self._n_avogadro_mag = float(self.n_avogadro.to(
            (units.mol) ** -1).magnitude)

        # Used for figure in publication
        self.trpAIndex = np.where(self.proteinIds == "TRYPSYN-APROTEIN[c]")[0][0]

        self.elngRateFactor = 1.0

        # Data structures for charging
        self.aa_from_trna = self.parameters["aa_from_trna"]

        # Elongation-model parameters (formerly BaseElongationModel.__init__)
        self.basal_elongation_rate = self.parameters["basal_elongation_rate"]
        self.ribosomeElongationRateDict = self.parameters["ribosomeElongationRateDict"]

        # Growth associated maintenance energy requirements for elongations.
        # Base/Supply do NOT model charging, so they add +2 to account for ATP
        # hydrolysis for charging that has been removed from measured GAM
        # (ATP -> AMP is 2 hydrolysis reactions). SteadyState overrides this.
        self.gtpPerElongation = self.parameters["gtpPerElongation"] + 2

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
                'listeners':                 {
                    'mass':                     {
                        'cell_mass': {'_type': 'quantity[float,fg]', '_default': 0.0},
                        'dry_mass': {'_type': 'quantity[float,fg]', '_default': 0.0},
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
                        'aa_count_diff': {'_type': 'overwrite[array[float]]', '_default': []},
                        'aas_used': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'net_charged': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'aa_allocated': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'aa_pool_size': {'_type': 'overwrite[array[integer]]', '_default': []},
                        'aa_request_size': {'_type': 'overwrite[array[float]]', '_default': []},
                        'active_ribosome_allocated': {'_type': 'overwrite[integer]', '_default': 0},
                        'fraction_trna_charged': {'_type': 'overwrite[array[float]]', '_default': []},
                        'ntp_used': {'_type': 'overwrite[array[integer]]', '_default': []},
                    },
                    'ribosome_data':                     {
                        'effective_elongation_rate': {'_type': 'overwrite[quantity[float,amino_acid/s]]', '_default': 0.0},
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

    def _init_bulk_indices(self, bulk_ids):
        """Resolve every molecule-name -> bulk-array-index map this process uses.

        Called once, on the first tick, when the bulk store's id ordering is
        first available. Cached on ``self`` (guarded by ``self.proton_idx is
        None``) so it never runs on the per-tick hot path.
        """
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
        self.uncharged_trna_idx = bulk_name_to_idx(self.uncharged_trna_names, bulk_ids)
        self.charged_trna_idx = bulk_name_to_idx(self.charged_trna_names, bulk_ids)
        self.charging_molecule_idx = bulk_name_to_idx(
            self.charging_molecule_names, bulk_ids
        )
        self.synthetase_idx = bulk_name_to_idx(self.synthetase_names, bulk_ids)
        self.ribosome30S_idx = bulk_name_to_idx(self.ribosome30S, bulk_ids)
        self.ribosome50S_idx = bulk_name_to_idx(self.ribosome50S, bulk_ids)
        self.aa_importer_idx = bulk_name_to_idx(self.aa_importers, bulk_ids)
        self.aa_exporter_idx = bulk_name_to_idx(self.aa_exporters, bulk_ids)

    def calculate_request(self, timestep, states):
        """Phase 1 of the partitioned step: REQUEST resources for this tick.

        Computes the ribosome elongation rate (model-specific, possibly ppGpp- or
        tRNA-charging-modulated), builds the per-ribosome amino-acid sequences for
        the coming timestep, and from them requests the amino acids (and, for the
        charging model, tRNAs/ATP) needed to elongate. The partitioner then
        allocates the cell's actual pools against all processes' requests; the
        granted counts arrive in ``evolve_state``.

        Elongation is capped at 22 aa/tick: the protein-sequence matrix is padded
        with 22 PAD_VALUEs, so timesteps > 1.0 s would under-count the effective
        elongation rate.
        """

        # One-time, on the first tick once the bulk layout is known: resolve the
        # molecule-name -> bulk-array-index maps used throughout request/evolve.
        if self.proton_idx is None:
            self._init_bulk_indices(states["bulk"]["id"])

        # MODEL SPECIFIC: get ribosome elongation rate
        self.ribosomeElongationRate = self.elongation_rate(states)

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

        # Calculate AA supply for expected doubling of protein.
        # Pure-float path: every input is in known units, no per-tick
        # Quantity construction. translation_supply_rate_per_s carries
        # the unit factor (min→s, /60) baked in at init.
        # dry_mass may be a pint Quantity (quantity[float,fg] port) or a
        # bare float; strip to a plain float so the product stays a plain
        # numpy array even when pint Quantities are flowing through the store.
        current_media_id = states["environment"]["media_id"]
        dry_mass_fg = fg_magnitude(states["listeners"]["mass"]["dry_mass"])
        timestep_s = states["timestep"]
        translation_supply_rate_per_s = (
            self._translation_aa_supply_per_s[current_media_id]
            * self.elngRateFactor
        )
        self.aa_supply = (
            translation_supply_rate_per_s
            * dry_mass_fg
            * timestep_s
            * self._n_avogadro_mag
        )

        # MODEL SPECIFIC: Calculate AA request
        fraction_charged, aa_counts_for_translation, requests = (
            self.request(states, aasInSequences)
        )

        # Write to listeners
        listeners = requests.setdefault("listeners", {})
        ribosome_data_listener = listeners.setdefault("ribosome_data", {})
        # Listener output preserved in the original mol/(fg·min) basis —
        # use the pre-stripped magnitude rather than reconstructing a
        # Quantity just to call .magnitude on it.
        ribosome_data_listener["translation_supply"] = (
            self._translation_aa_supply_min_mag[current_media_id]
            * self.elngRateFactor
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
        """Phase 2 of the partitioned step: EVOLVE under the allocated resources.

        Runs the ``polymerize`` algorithm over the per-ribosome sequences using
        the amino acids the partitioner actually granted (model-specific
        ``final_amino_acids``), then applies the universal ribosome mechanics:
        advance each ribosome's peptide length and mRNA position by the elongation
        it achieved, add the incorporated amino-acid mass, terminate ribosomes that
        reached the end of their protein (dissociating them into 30S/50S subunits
        and releasing the finished monomer), and report GTP to hydrolyze. The
        model-specific ``evolve`` then settles tRNA charging / ppGpp / AA supply,
        and the results are written to the growth_limits and ribosome_data
        listeners (incl. the effective elongation rate read by
        polypeptide_initiation next tick).
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
        aa_counts_for_translation = self.final_amino_acids(
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
        net_charged, aa_count_diff, evolve_update = self.evolve(
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
        # Emit as a pint Quantity[amino_acid/s] so the unit travels through the
        # store and serialization (read back by polypeptide_initiation, and
        # serialized to {units, magnitude} columns by the emitter).
        ribosome_data_listener["effective_elongation_rate"] = (
            currElongRate * units.amino_acid / units.s
        )
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

    # ------------------------------------------------------------------
    # Elongation-model hooks (Base Model: request amino acids according to
    # upcoming sequence, assuming max ribosome elongation).
    # ------------------------------------------------------------------

    def elongation_rate(self, states):
        """
        Sets ribosome elongation rate accordint to the media; returns
        max value of 22 amino acids/second.
        """
        current_media_id = states["environment"]["media_id"]
        rate = (
            self.elngRateFactor
            * self.ribosomeElongationRateDict[current_media_id]
            .to(units.aa / units.s).magnitude
        )
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
                    self.amino_acid_idx,
                    aa_counts_for_translation.astype(np.int64),
                )
            ]
        }

        # Not modeling charging so set fraction charged to 0 for all tRNA
        fraction_charged = np.zeros(len(self.amino_acid_idx))

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
            np.zeros(len(self.amino_acids), dtype=np.float64),
            {
                "bulk": [
                    (self.amino_acid_idx, -aas_used),
                    (self.water_idx, nElongations - nInitialized),
                ]
            },
        )


class TranslationSupplyPolypeptideElongation(BasePolypeptideElongation):
    """
    Translation Supply Model: Requests minimum of 1) upcoming amino acid
    sequence assuming max ribosome elongation (ie. Base Model) and 2)
    estimation based on doubling the proteome in one cell cycle (does not
    use ribosome elongation, computed in Parca).
    """

    name = NAME
    topology = TOPOLOGY

    def elongation_rate(self, states):
        """
        Sets ribosome elongation rate accordint to the media; returns
        max value of 22 amino acids/second.
        """
        return self.basal_elongation_rate

    def amino_acid_counts(self, aasInSequences):
        # Check if this is required. It is a better request but there may be
        # fewer elongations.
        return np.fmin(self.aa_supply, aasInSequences)


class SteadyStatePolypeptideElongation(TranslationSupplyPolypeptideElongation):
    """Steady-state tRNA-charging elongation model (Michaelis-Menten competitive
    inhibition). This is the growth-rate-control model: the ribosome elongation
    rate emerges from amino-acid availability rather than being imposed.

    It composes three growth-rate-control modules, each a focused unit:

    * **tRNA charging** — the steady-state charged-fraction / elongation-rate
      solve (``kinetics.calculate_trna_charging``), plus the per-tRNA
      distribution bookkeeping (``distribution_from_aa``) applied in
      ``request`` (charge requests) and ``evolve`` (realized charging reactions).
    * **ppGpp (RelA/SpoT)** — ``elongation_rate`` (ppGpp-modulated rate),
      ``_ppgpp_request`` (predicted ppGpp turnover → bulk requests) and
      ``_ppgpp_evolve`` (realized ppGpp synthesis/degradation), over
      ``kinetics.ppgpp_metabolite_changes``.
    * **amino-acid supply** — ``_amino_acid_supply`` (synthesis / import / export
      rates + the supply closure the charging solve integrates).

    ``request`` and ``evolve`` orchestrate these three in sequence per the
    partitioned request→allocate→evolve contract.
    """

    name = NAME
    topology = TOPOLOGY

    def initialize(self, config):
        super().initialize(config)

        # SteadyState models charging explicitly, so it does NOT add +2 to GAM
        # (Base/Supply add +2 to account for charging removed from measured GAM).
        self.gtpPerElongation = self.parameters["gtpPerElongation"]

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
                self.parameters["elongation_max"].to(units.aa / units.s).magnitude
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

        # Store as plain float (boundary.external values are plain floats in mM)
        self.import_constraint_threshold = float(
            self.parameters["import_constraint_threshold"]
        )

    def inputs(self):
        # SteadyState additionally reads the boundary store (external media
        # concentrations) via _amino_acid_supply. Base/Supply do not.
        base = super().inputs()
        return deep_merge(base, {'boundary': 'node'})

    def outputs(self):
        # SteadyState additionally writes the tRNA-charging / ppGpp /
        # amino-acid-supply growth_limits listener fields that Base/Supply
        # never emit.
        base = super().outputs()
        return deep_merge(
            base,
            {
                'listeners': {
                    'growth_limits': {
                        'synthetase_conc': {'_type': 'overwrite[array[float[uM]]]', '_default': []},
                        'uncharged_trna_conc': {'_type': 'overwrite[array[float[uM]]]', '_default': []},
                        'charged_trna_conc': {'_type': 'overwrite[array[float[uM]]]', '_default': []},
                        'aa_conc': {'_type': 'overwrite[array[float[uM]]]', '_default': []},
                        'ribosome_conc': {'_type': 'overwrite[float[uM]]', '_default': 0.0},
                        'ppgpp_conc': {'_type': 'overwrite[float[uM]]', '_default': 0.0},
                        'rela_conc': {'_type': 'overwrite[float[uM]]', '_default': 0.0},
                        'spot_conc': {'_type': 'overwrite[float[uM]]', '_default': 0.0},
                        'aa_supply_aa_conc': {'_type': 'overwrite[array[float[mM]]]', '_default': []},
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
                    },
                },
            },
        )

    def elongation_rate(self, states):
        if (
            self.ppgpp_regulation
            and not self.disable_ppgpp_elongation_inhibition
        ):
            cell_mass = as_quantity(states["listeners"]["mass"]["cell_mass"], units.fg)
            cell_volume = cell_mass / self.cellDensity
            counts_to_molar = 1 / (self.n_avogadro * cell_volume)
            ppgpp_count = counts(states["bulk"], self.ppgpp_idx)
            ppgpp_conc = ppgpp_count * counts_to_molar
            # elong_rate_by_ppgpp is the function-registry wrapper, which
            # accepts pint and returns a pint Quantity in aa/s.
            rate = self.elong_rate_by_ppgpp(
                ppgpp_conc, self.basal_elongation_rate
            ).to(units.aa / units.s).magnitude
        else:
            rate = super().elongation_rate(states)
        return rate

    def _amino_acid_supply(self, states, aa_conc, dry_mass, counts_to_uM_mag):
        """Growth-control module: amino-acid supply.

        Computes per-amino-acid synthesis / import / export rates from the
        current enzyme, importer and exporter counts and the boundary media,
        and builds the supply closure that the tRNA-charging solve integrates
        over the timestep. Pure read of ``states`` — returns a bundle consumed
        by ``request`` (supply reconciliation + listeners); makes no updates.
        """
        aa_in_media = np.array(
            [
                states["boundary"]["external"][aa] > self.import_constraint_threshold
                for aa in self.aa_environment_names
            ]
        )
        fwd_enzyme_counts, rev_enzyme_counts = self.get_pathway_enzyme_counts_per_aa(
            counts(states["bulk_total"], self.aa_enzyme_idx)
        )
        importer_counts = counts(states["bulk_total"], self.aa_importer_idx)
        exporter_counts = counts(states["bulk_total"], self.aa_exporter_idx)
        # amino_acid_synthesis/import/export are upstream Numba-compiled and
        # Unum-native; convert pint args at the boundary.
        aa_conc_unum = pint_to_unum(aa_conc)
        dry_mass_unum = pint_to_unum(dry_mass)
        synthesis, fwd_saturation, rev_saturation = self.amino_acid_synthesis(
            fwd_enzyme_counts, rev_enzyme_counts, aa_conc_unum
        )
        import_rates = self.amino_acid_import(
            aa_in_media,
            dry_mass_unum,
            aa_conc_unum,
            importer_counts,
            self.mechanistic_aa_transport,
        )
        export_rates = self.amino_acid_export(
            exporter_counts, aa_conc_unum, self.mechanistic_aa_transport
        )
        exchange_rates = import_rates - export_rates

        # The closure produced here calls the upstream Unum-native
        # amino_acid_synthesis/import/export with dry_mass and aa_conc, so
        # convert dry_mass at the boundary.
        supply_function = get_charging_supply_function(
            self.aa_supply_in_charging,
            self.mechanistic_translation_supply,
            self.mechanistic_aa_transport,
            self.amino_acid_synthesis,
            self.amino_acid_import,
            self.amino_acid_export,
            self.aa_supply_scaling,
            counts_to_uM_mag,
            self.aa_supply,
            fwd_enzyme_counts,
            rev_enzyme_counts,
            pint_to_unum(dry_mass),
            importer_counts,
            exporter_counts,
            aa_in_media,
        )
        return {
            "aa_in_media": aa_in_media,
            "fwd_enzyme_counts": fwd_enzyme_counts,
            "rev_enzyme_counts": rev_enzyme_counts,
            "importer_counts": importer_counts,
            "exporter_counts": exporter_counts,
            "synthesis": synthesis,
            "import_rates": import_rates,
            "export_rates": export_rates,
            "exchange_rates": exchange_rates,
            "fwd_saturation": fwd_saturation,
            "rev_saturation": rev_saturation,
            "supply_function": supply_function,
        }

    def request(
        self, states: dict, aasInSequences: npt.NDArray[np.int64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], dict]:
        # Conversion from counts to molarity
        cell_mass = as_quantity(states["listeners"]["mass"]["cell_mass"], units.fg)
        dry_mass = as_quantity(states["listeners"]["mass"]["dry_mass"], units.fg)
        cell_volume = cell_mass / self.cellDensity
        self.counts_to_molar = 1 / (self.n_avogadro * cell_volume)
        # Strip counts_to_molar to a plain float in MICROMOLAR_UNITS once
        # per tick. All downstream uses that feed calculate_trna_charging /
        # ppgpp_metabolite_changes / get_charging_supply_function (which
        # now accept μM magnitudes directly) multiply this float instead
        # of constructing a fresh pint Quantity per concentration.
        counts_to_uM_mag = self.counts_to_molar.to(MICROMOLAR_UNITS).magnitude
        self._counts_to_uM_mag = counts_to_uM_mag  # for evolve() reuse

        # ppGpp related concentrations (now μM-magnitude numpy arrays).
        ppgpp_conc = counts_to_uM_mag * counts(
            states["bulk_total"], self.ppgpp_idx
        )
        rela_conc = counts_to_uM_mag * counts(
            states["bulk_total"], self.rela_idx
        )
        spot_conc = counts_to_uM_mag * counts(
            states["bulk_total"], self.spot_idx
        )

        # Get counts and convert synthetase and tRNA to a per AA basis
        synthetase_counts = np.dot(
            self.aa_from_synthetase,
            counts(states["bulk_total"], self.synthetase_idx),
        )
        aa_counts = counts(states["bulk_total"], self.amino_acid_idx)
        uncharged_trna_array = counts(
            states["bulk_total"], self.uncharged_trna_idx
        )
        charged_trna_array = counts(states["bulk_total"], self.charged_trna_idx)
        uncharged_trna_counts = np.dot(self.aa_from_trna, uncharged_trna_array)
        charged_trna_counts = np.dot(self.aa_from_trna, charged_trna_array)
        ribosome_counts = states["active_ribosome"]["_entryState"].sum()

        # Get concentration. calculate_trna_charging and
        # ppgpp_metabolite_changes both accept numpy magnitude arrays in
        # MICROMOLAR_UNITS — convert the per-tick counts_to_molar to μM
        # once and multiply through. Skips per-input pint Quantity
        # construction + downstream .to(MICROMOLAR_UNITS).magnitude stripping
        # inside those functions.
        f = aasInSequences / aasInSequences.sum()
        counts_to_uM_mag = self.counts_to_molar.to(MICROMOLAR_UNITS).magnitude
        synthetase_conc = counts_to_uM_mag * synthetase_counts
        aa_conc = counts_to_uM_mag * aa_counts
        uncharged_trna_conc = counts_to_uM_mag * uncharged_trna_counts
        charged_trna_conc = counts_to_uM_mag * charged_trna_counts
        ribosome_conc = counts_to_uM_mag * ribosome_counts

        # GROWTH-CONTROL MODULE: amino-acid supply (synthesis / import / export
        # rates + the supply closure threaded into the charging solve).
        sup = self._amino_acid_supply(states, aa_conc, dry_mass, counts_to_uM_mag)
        aa_in_media = sup["aa_in_media"]
        fwd_enzyme_counts = sup["fwd_enzyme_counts"]
        rev_enzyme_counts = sup["rev_enzyme_counts"]
        importer_counts = sup["importer_counts"]
        exporter_counts = sup["exporter_counts"]
        synthesis = sup["synthesis"]
        import_rates = sup["import_rates"]
        export_rates = sup["export_rates"]
        exchange_rates = sup["exchange_rates"]
        fwd_saturation = sup["fwd_saturation"]
        rev_saturation = sup["rev_saturation"]
        supply_function = sup["supply_function"]

        # GROWTH-CONTROL MODULE: tRNA charging — calculate steady state tRNA
        # levels and the resulting elongation rate (see kinetics.calculate_trna_charging)
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
        if self.aa_supply_in_charging:
            # counts_to_uM_mag is the μM-magnitude float computed above;
            # 1/counts_to_uM_mag/timestep is the per-timestep conversion.
            conversion = 1 / counts_to_uM_mag / states["timestep"]
            synthesis = conversion * synthesis_in_charging
            import_rates = conversion * import_in_charging
            export_rates = conversion * export_in_charging
            self.aa_supply = synthesis + import_rates - export_rates
        # Use the supply calculated from the starting amino acid concentrations only
        elif self.mechanistic_translation_supply:
            # Set supply based on mechanistic synthesis and supply
            self.aa_supply = states["timestep"] * (synthesis + exchange_rates)
        else:
            # Adjust aa_supply higher if amino acid concentrations are low
            # Improves stability of charging and mimics amino acid synthesis
            # inhibition and export.
            # aa_conc is in μM (MICROMOLAR_UNITS = CONC_UNITS); pass
            # magnitude directly to aa_supply_scaling.
            self.aa_supply *= self.aa_supply_scaling(
                self.charging_params["unit_conversion"] * aa_conc,
                aa_in_media,
            )

        # counts_to_uM_mag is the cached per-tick conversion factor;
        # dividing here matches the previous Quantity-stripping path.
        aa_counts_for_translation = (
            v_rib
            * f
            * states["timestep"]
            / counts_to_uM_mag
        )

        total_trna = charged_trna_array + uncharged_trna_array
        final_charged_trna = stochasticRound(
            self.random_state,
            np.dot(fraction_charged, self.aa_from_trna * total_trna),
        )

        # Request charged tRNA that will become uncharged
        charged_trna_request = charged_trna_array - final_charged_trna
        charged_trna_request[charged_trna_request < 0] = 0
        uncharged_trna_request = final_charged_trna - charged_trna_array
        uncharged_trna_request[uncharged_trna_request < 0] = 0
        self.uncharged_trna_to_charge = uncharged_trna_request

        self.aa_counts_for_translation = np.array(aa_counts_for_translation)

        fraction_trna_per_aa = total_trna / np.dot(
            np.dot(self.aa_from_trna, total_trna), self.aa_from_trna
        )
        total_charging_reactions = stochasticRound(
            self.random_state,
            np.dot(aa_counts_for_translation, self.aa_from_trna)
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
                self.charging_molecule_idx,
                requested_molecules.astype(int),
            ),
            (self.charged_trna_idx, charged_trna_request.astype(int)),
            # Request water for transfer of AA from tRNA for initial polypeptide.
            # This is severe overestimate assuming the worst case that every
            # elongation is initializing a polypeptide. This excess of water
            # shouldn't matter though.
            (self.water_idx, int(aa_counts_for_translation.sum())),
        ]
        # GROWTH-CONTROL MODULE: ppGpp (request side) — RelA/SpoT-driven ppGpp
        # turnover given the charging solve's predicted charged fraction.
        bulk_request += self._ppgpp_request(
            states,
            counts_to_uM_mag,
            uncharged_trna_counts,
            charged_trna_counts,
            fraction_charged,
            ribosome_conc,
            f,
            rela_conc,
            spot_conc,
            ppgpp_conc,
            v_rib,
        )

        return (
            fraction_charged,
            aa_counts_for_translation,
            {
                "bulk": bulk_request,
                "listeners": {
                    "growth_limits": {
                        "original_aa_supply": self.aa_supply,
                        "aa_in_media": aa_in_media,
                        # These five concentrations are now plain μM-
                        # magnitude numpy arrays (see counts_to_uM_mag
                        # above). The listener output basis is unchanged
                        # — μM directly, no per-tick .to().magnitude
                        # round-trip needed.
                        "synthetase_conc": synthetase_conc,
                        "uncharged_trna_conc": uncharged_trna_conc,
                        "charged_trna_conc": charged_trna_conc,
                        "aa_conc": aa_conc,
                        "ribosome_conc": ribosome_conc,
                        "fraction_aa_to_elongate": f,
                        "aa_supply": self.aa_supply,
                        "aa_synthesis": synthesis * states["timestep"],
                        "aa_import": import_rates * states["timestep"],
                        "aa_export": export_rates * states["timestep"],
                        "aa_supply_enzymes_fwd": fwd_enzyme_counts,
                        "aa_supply_enzymes_rev": rev_enzyme_counts,
                        "aa_importers": importer_counts,
                        "aa_exporters": exporter_counts,
                        # aa_conc is in μM (MICROMOLAR_UNITS); convert to
                        # mM (mmol/L) for this listener field — × 1e-3.
                        "aa_supply_aa_conc": aa_conc * 1e-3,
                        "aa_supply_fraction_fwd": fwd_saturation,
                        "aa_supply_fraction_rev": rev_saturation,
                        # ppgpp_conc / rela_conc / spot_conc are now plain
                        # μM-magnitude values (computed via counts_to_uM_mag *
                        # counts earlier in this method); no per-tick
                        # .to(MICROMOLAR_UNITS).magnitude round-trip needed.
                        "ppgpp_conc": ppgpp_conc,
                        "rela_conc": rela_conc,
                        "spot_conc": spot_conc,
                    }
                },
                "polypeptide_elongation": {
                    "aa_exchange_rates": (
                        self.counts_to_molar / units.s * (import_rates - export_rates)
                    ).to(CONC_UNITS / TIME_UNITS).magnitude
                },
            },
        )

    def final_amino_acids(self, total_aa_counts, charged_trna_counts):
        charged_counts_to_uncharge = self.aa_from_trna @ charged_trna_counts
        return np.fmin(
            total_aa_counts + charged_counts_to_uncharge, self.aa_counts_for_translation
        )

    def _ppgpp_request(
        self, states, counts_to_uM_mag, uncharged_trna_counts, charged_trna_counts,
        fraction_charged, ribosome_conc, f, rela_conc, spot_conc, ppgpp_conc, v_rib,
    ):
        """Growth-control module: ppGpp (request side).

        From the charging solve's predicted charged fraction, estimate the
        RelA/SpoT-driven ppGpp metabolite turnover and return the extra bulk
        requests it implies (ppGpp itself + the ppGpp-reaction metabolites).
        Returns an empty list when ppGpp regulation is off.
        """
        if not self.ppgpp_regulation:
            return []
        # Compute total_trna_conc in μM-magnitude form, matching the
        # ppgpp_metabolite_changes contract (numpy magnitudes).
        total_trna_conc = counts_to_uM_mag * (
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
            counts_to_uM_mag,
            v_rib,
            self.charging_params,
            self.ppgpp_params,
            states["timestep"],
            request=True,
            random_state=self.random_state,
        )
        request_ppgpp_metabolites = -delta_metabolites.astype(int)
        ppgpp_request = counts(states["bulk"], self.ppgpp_idx)
        return [
            (self.ppgpp_idx, ppgpp_request),
            (self.ppgpp_rxn_metabolites_idx, request_ppgpp_metabolites),
        ]

    def _ppgpp_evolve(
        self, states, update, net_charged, nElongations, aas_used,
        next_amino_acid_count,
    ):
        """Growth-control module: ppGpp (evolve side).

        Create/degrade ppGpp via RelA/SpoT from the realized post-elongation
        tRNA charging state. Mutates ``update`` (the ppGpp-reaction bulk delta +
        the rela/spot synthesis/degradation listener fields) and the in-place
        ``states['bulk']`` working copy. No-op when ppGpp regulation is off.
        """
        if not self.ppgpp_regulation:
            return
        # Use the same μM-magnitude conversion factor request() cached
        # for this tick — see request() for the derivation. evolve()
        # runs in the same tick as request() so self._counts_to_uM_mag
        # is set.
        counts_to_uM_mag = self._counts_to_uM_mag
        v_rib = (nElongations * counts_to_uM_mag) / states["timestep"]
        ribosome_conc = (
            counts_to_uM_mag * states["active_ribosome"]["_entryState"].sum()
        )
        updated_uncharged_trna_counts = (
            counts(states["bulk_total"], self.uncharged_trna_idx)
            - net_charged
        )
        updated_charged_trna_counts = (
            counts(states["bulk_total"], self.charged_trna_idx)
            + net_charged
        )
        uncharged_trna_conc = counts_to_uM_mag * np.dot(
            self.aa_from_trna, updated_uncharged_trna_counts
        )
        charged_trna_conc = counts_to_uM_mag * np.dot(
            self.aa_from_trna, updated_charged_trna_counts
        )
        ppgpp_conc = counts_to_uM_mag * counts(
            states["bulk_total"], self.ppgpp_idx
        )
        rela_conc = counts_to_uM_mag * counts(
            states["bulk_total"], self.rela_idx
        )
        spot_conc = counts_to_uM_mag * counts(
            states["bulk_total"], self.spot_idx
        )

        # Need to include the next amino acid the ribosome sees for certain
        # cases where elongation does not occur, otherwise f will be NaN
        aa_at_ribosome = aas_used + next_amino_acid_count
        f = aa_at_ribosome / aa_at_ribosome.sum()
        limits = counts(states["bulk"], self.ppgpp_rxn_metabolites_idx)
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
            counts_to_uM_mag,
            v_rib,
            self.charging_params,
            self.ppgpp_params,
            states["timestep"],
            random_state=self.random_state,
            limits=limits,
        )

        update["listeners"]["growth_limits"] = {
            "rela_syn": rela_syn,
            "spot_syn": spot_syn,
            "spot_deg": spot_deg,
            "spot_deg_inhibited": spot_deg_inhibited,
        }

        update["bulk"].append(
            (self.ppgpp_rxn_metabolites_idx, delta_metabolites.astype(int))
        )
        states["bulk"][self.ppgpp_rxn_metabolites_idx] += (
            delta_metabolites.astype(int)
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
        uncharged_trna = counts(states["bulk"], self.uncharged_trna_idx)
        charged_trna = counts(states["bulk"], self.charged_trna_idx)
        total_trna = uncharged_trna + charged_trna

        # Adjust molecules for number of charging reactions that occurred
        ## Determine limitations for charging and uncharging reactions
        charged_and_elongated_per_aa = np.fmax(
            0, (aas_used - self.aa_from_trna @ charged_trna)
        )
        aa_for_charging = total_aa_counts - charged_and_elongated_per_aa
        n_aa_charged = np.fmin(
            aa_for_charging,
            np.dot(
                self.aa_from_trna,
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
        update["bulk"].append((self.charging_molecule_idx, charging_mol_delta))
        states["bulk"][self.charging_molecule_idx] += charging_mol_delta

        ## Account for uncharging of tRNA during elongation
        update["bulk"].append(
            (self.charged_trna_idx, -total_uncharging_reactions)
        )
        update["bulk"].append(
            (self.uncharged_trna_idx, total_uncharging_reactions)
        )
        states["bulk"][self.charged_trna_idx] += -total_uncharging_reactions
        states["bulk"][self.uncharged_trna_idx] += total_uncharging_reactions

        # Update proton counts to reflect polymerization reactions and transfer of AA from tRNA
        # Peptide bond formation releases a water but transferring AA from tRNA consumes a OH-
        # Net production of H+ for each elongation, consume extra water for each initialization
        # since a peptide bond doesn't form
        update["bulk"].append((self.proton_idx, nElongations))
        update["bulk"].append((self.water_idx, -nInitialized))
        states["bulk"][self.proton_idx] += nElongations
        states["bulk"][self.water_idx] += -nInitialized

        # GROWTH-CONTROL MODULE: ppGpp (evolve side) — create/degrade ppGpp from
        # the realized post-elongation tRNA state. Runs after all bulk updates
        # above so the ppGpp reaction limits see current counts.
        self._ppgpp_evolve(
            states, update, net_charged, nElongations, aas_used, next_amino_acid_count
        )

        # Use the difference between (expected AA supply based on expected
        # doubling time and current DCW) and AA used to charge tRNA to update
        # the concentration target in metabolism during the next time step
        aa_used_trna = np.dot(self.aa_from_trna, total_charging_reactions)
        aa_diff = self.aa_supply - aa_used_trna

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
        each amino acid. Uses self.aa_from_trna mapping to distribute
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
                np.dot(self.aa_from_trna, n_trna), self.aa_from_trna
            )
        f_trna[~np.isfinite(f_trna)] = 0

        trna_counts = np.zeros(f_trna.shape, np.int64)
        for count, row in zip(n_aa, self.aa_from_trna):
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
                        adjustment = self.random_state.multinomial(1, frac)
                        counts += adjustment
                else:
                    adjustment = self.random_state.multinomial(diff, frac)
                    counts += adjustment

            trna_counts[idx] = counts

        return trna_counts
