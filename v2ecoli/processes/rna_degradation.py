"""Process-bigraph partitioned process: rna_degradation."""

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

class RnaDegradationLogic:
    """RNA Degradation — shared state container for Requester/Evolver."""

    name = "ecoli-rna-degradation"
    topology = {
    "bulk": ("bulk",),
    "RNAs": ("unique", "RNA"),
    "active_ribosome": ("unique", "active_ribosome"),
    "listeners": ("listeners",),
    "timestep": ("timestep",),
}
    config_schema = {
        "rna_ids": {"_default": []},
        "mature_rna_ids": {"_default": []},
        "cistron_ids": {"_default": []},
        "cistron_tu_mapping_matrix": {"_default": []},
        "mature_rna_cistron_indexes": {"_default": []},
        "all_rna_ids": {"_default": []},
        "n_total_RNAs": {"_default": 0},
        "n_avogadro": {"_default": 0.0},
        "cell_density": {"_default": None},
        "endoRNase_ids": {"_default": []},
        "exoRNase_ids": {"_default": []},
        "kcat_exoRNase": {"_default": None},
        "Kcat_endoRNases": {"_default": None},
        "charged_trna_names": {"_default": []},
        "uncharged_trna_indexes": {"_default": []},
        "rna_deg_rates": {"_default": []},
        "is_mRNA": {"_default": None},
        "is_rRNA": {"_default": None},
        "is_tRNA": {"_default": None},
        "is_miscRNA": {"_default": None},
        "degrade_misc": {"_default": False},
        "rna_lengths": {"_default": None},
        "nt_counts": {"_default": None},
        "polymerized_ntp_ids": {"_default": []},
        "water_id": {"_default": "h2o"},
        "ppi_id": {"_default": "ppi"},
        "proton_id": {"_default": "h+"},
        "nmp_ids": {"_default": []},
        "rrfa_idx": {"_default": 0},
        "rrla_idx": {"_default": 0},
        "rrsa_idx": {"_default": 0},
        "ribosome30S": {"_default": "ribosome30S"},
        "ribosome50S": {"_default": "ribosome50S"},
        "Kms": {"_default": None},
        "seed": {"_default": 0},
        "emit_unique": {"_default": False},
        "time_step": {"_default": 1},
    }

    def __init__(self, parameters=None):
        self.parameters = _apply_config_defaults(self.config_schema, parameters)
        self.request_set = False

        self.rna_ids = self.parameters["rna_ids"]
        self.mature_rna_ids = self.parameters["mature_rna_ids"]
        self.n_transcribed_rnas = len(self.rna_ids)
        self.mature_rna_exists = len(self.mature_rna_ids) > 0
        self.cistron_ids = self.parameters["cistron_ids"]
        self.cistron_tu_mapping_matrix = self.parameters["cistron_tu_mapping_matrix"]
        self.mature_rna_cistron_indexes = self.parameters["mature_rna_cistron_indexes"]
        self.all_rna_ids = self.parameters["all_rna_ids"]
        self.n_total_RNAs = self.parameters["n_total_RNAs"]

        # Load constants
        self.n_avogadro = self.parameters["n_avogadro"]
        self.cell_density = self.parameters["cell_density"]

        # Load RNase kinetic data
        self.endoRNase_ids = self.parameters["endoRNase_ids"]
        self.exoRNase_ids = self.parameters["exoRNase_ids"]
        self.kcat_exoRNase = self.parameters["kcat_exoRNase"]
        self.Kcat_endoRNases = self.parameters["Kcat_endoRNases"]

        # Load information about uncharged/charged tRNA
        self.uncharged_trna_indexes = self.parameters["uncharged_trna_indexes"]
        self.charged_trna_names = self.parameters["charged_trna_names"]

        # Load first-order RNA degradation rates
        # (estimated by mRNA half-life data)
        self.rna_deg_rates = self.parameters["rna_deg_rates"]

        self.is_mRNA = self.parameters["is_mRNA"]
        self.is_rRNA = self.parameters["is_rRNA"]
        self.is_tRNA = self.parameters["is_tRNA"]

        # NEW to vivarium-ecoli
        self.is_miscRNA = self.parameters["is_miscRNA"]
        self.degrade_misc = self.parameters["degrade_misc"]

        self.rna_lengths = self.parameters["rna_lengths"]
        self.nt_counts = self.parameters["nt_counts"]

        # Build stoichiometric matrix
        self.polymerized_ntp_ids = self.parameters["polymerized_ntp_ids"]
        self.nmp_ids = self.parameters["nmp_ids"]
        self.water_id = self.parameters["water_id"]
        self.ppi_id = self.parameters["ppi_id"]
        self.proton_id = self.parameters["proton_id"]

        self.end_cleavage_metabolite_ids = self.polymerized_ntp_ids + [
            self.water_id,
            self.ppi_id,
            self.proton_id,
        ]
        nmp_idx = list(range(4))
        water_idx = self.end_cleavage_metabolite_ids.index(self.water_id)
        ppi_idx = self.end_cleavage_metabolite_ids.index(self.ppi_id)
        proton_idx = self.end_cleavage_metabolite_ids.index(self.proton_id)
        self.endo_degradation_stoich_matrix = np.zeros(
            (len(self.end_cleavage_metabolite_ids), self.n_total_RNAs), np.int64
        )
        self.endo_degradation_stoich_matrix[nmp_idx, :] = self.nt_counts.T
        self.endo_degradation_stoich_matrix[water_idx, :] = 0
        self.endo_degradation_stoich_matrix[ppi_idx, :] = 1
        self.endo_degradation_stoich_matrix[proton_idx, :] = 0

        # Load Michaelis-Menten constants fitted to recapitulate
        # first-order RNA decay model
        self.Kms = self.parameters["Kms"]

        self.seed = self.parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Numpy indices for bulk molecules
        self.water_idx = None

    def _calculate_total_n_to_degrade(
        self, timestep, specificity, total_kcat_endornase
    ):
        """
        Calculate the total number of RNAs to degrade for a specific class of
        RNAs, based on the specificity of endoRNases on that specific class and
        the total kcat value of the endoRNases.

        Args:
            specificity: Sum of fraction of active endoRNases for all RNAs
                in a given class
            total_kcat_endornase: The summed kcat of all existing endoRNases
        Returns:
            Total number of RNAs to degrade for the given class of RNAs
        """
        return np.round(
            (specificity * total_kcat_endornase * (units.s * timestep)).asNumber()
        )

    def _get_rnas_to_degrade(self, n_total_rnas_to_degrade, rna_deg_probs, rna_counts):
        """
        Distributes the total count of RNAs to degrade for each class of RNAs
        into individual RNAs, based on the given degradation probabilities
        of individual RNAs. The upper bound is set by the current count of the
        specific RNA.

        Args:
            n_total_rnas_to_degrade: Total number of RNAs to degrade for the
                given class of RNAs (integer, scalar)
            rna_deg_probs: Degradation probabilities of each RNA (vector of
                equal length to the total number of different RNAs)
            rna_counts: Current counts of each RNA molecule (vector of equal
                length to the total number of different RNAs)
        Returns:
            Vector of equal length to rna_counts, specifying the number of
            molecules to degrade for each RNA
        """
        n_rnas_to_degrade = np.zeros_like(rna_counts)
        remaining_rna_counts = rna_counts

        while (
            n_rnas_to_degrade.sum() < n_total_rnas_to_degrade
            and remaining_rna_counts.sum() != 0
        ):
            n_rnas_to_degrade += np.fmin(
                self.random_state.multinomial(
                    n_total_rnas_to_degrade - n_rnas_to_degrade.sum(), rna_deg_probs
                ),
                remaining_rna_counts,
            )
            remaining_rna_counts = rna_counts - n_rnas_to_degrade

        return n_rnas_to_degrade


class RnaDegradationRequester(_SafeInvokeMixin, Step):
    """Reads stores to compute RNA degradation request."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']
        self.process_name = config.get('process_name', self.process.name)

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'RNAs': UniqueNumpyUpdate(),
            'active_ribosome': UniqueNumpyUpdate(),
            'listeners': ListenerStore(),
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
        if p.water_idx is None:
            bulk_ids = state["bulk"]["id"]
            p.charged_trna_idx = bulk_name_to_idx(p.charged_trna_names, bulk_ids)
            p.bulk_rnas_idx = bulk_name_to_idx(p.all_rna_ids, bulk_ids)
            p.nmps_idx = bulk_name_to_idx(p.nmp_ids, bulk_ids)
            p.fragment_metabolites_idx = bulk_name_to_idx(
                p.end_cleavage_metabolite_ids, bulk_ids
            )
            p.fragment_bases_idx = bulk_name_to_idx(
                p.polymerized_ntp_ids, bulk_ids
            )
            p.endoRNase_idx = bulk_name_to_idx(p.endoRNase_ids, bulk_ids)
            p.exoRNase_idx = bulk_name_to_idx(p.exoRNase_ids, bulk_ids)
            p.water_idx = bulk_name_to_idx(p.water_id, bulk_ids)
            p.proton_idx = bulk_name_to_idx(p.proton_id, bulk_ids)

        # Compute factor that convert counts into concentration, and vice versa
        cell_mass = state["listeners"]["mass"]["cell_mass"] * units.fg
        cell_volume = cell_mass / p.cell_density
        counts_to_molar = 1 / (p.n_avogadro * cell_volume)

        # Get total counts of RNAs including free rRNAs, uncharged and charged tRNAs, and
        # active (translatable) unique mRNAs
        bulk_RNA_counts = counts(state["bulk"], p.bulk_rnas_idx)
        bulk_RNA_counts[p.uncharged_trna_indexes] += counts(
            state["bulk"], p.charged_trna_idx
        )

        TU_index, can_translate, is_full_transcript = attrs(
            state["RNAs"], ["TU_index", "can_translate", "is_full_transcript"]
        )

        TU_index_translatable_mRNAs = TU_index[can_translate]
        unique_RNA_counts = np.bincount(
            TU_index_translatable_mRNAs, minlength=p.n_total_RNAs
        )
        total_RNA_counts = bulk_RNA_counts + unique_RNA_counts

        # Compute RNA concentrations
        rna_conc_molar = counts_to_molar * total_RNA_counts

        # Get counts of endoRNases
        endoRNase_counts = counts(state["bulk"], p.endoRNase_idx)
        total_kcat_endoRNase = units.dot(p.Kcat_endoRNases, endoRNase_counts)

        # Calculate the fraction of active endoRNases for each RNA based on
        # Michaelis-Menten kinetics
        frac_endoRNase_saturated = (
            rna_conc_molar / p.Kms / (1 + units.sum(rna_conc_molar / p.Kms))
        ).asNumber()

        # Calculate difference in degradation rates from first-order decay
        # and the number of EndoRNases per one molecule of RNA
        total_endoRNase_counts = np.sum(endoRNase_counts)
        diff_relative_first_order_decay = units.sum(
            units.abs(
                p.rna_deg_rates * total_RNA_counts
                - total_kcat_endoRNase * frac_endoRNase_saturated
            )
        )
        endoRNase_per_rna = total_endoRNase_counts / np.sum(total_RNA_counts)

        request = {"listeners": {"rna_degradation_listener": {}}}
        request["listeners"]["rna_degradation_listener"][
            "fraction_active_endoRNases"
        ] = np.sum(frac_endoRNase_saturated)
        request["listeners"]["rna_degradation_listener"][
            "diff_relative_first_order_decay"
        ] = diff_relative_first_order_decay.asNumber()
        request["listeners"]["rna_degradation_listener"]["fract_endo_rrna_counts"] = (
            endoRNase_per_rna
        )

        # Dissect RNAse specificity into mRNA, tRNA, and rRNA
        # NEW to vivarium-ecoli: Degrade miscRNAs and mRNAs together
        if p.degrade_misc:
            is_transient_rna = p.is_mRNA | p.is_miscRNA
            mrna_specificity = np.dot(frac_endoRNase_saturated, is_transient_rna)
        else:
            mrna_specificity = np.dot(frac_endoRNase_saturated, p.is_mRNA)
        trna_specificity = np.dot(frac_endoRNase_saturated, p.is_tRNA)
        rrna_specificity = np.dot(frac_endoRNase_saturated, p.is_rRNA)

        n_total_mrnas_to_degrade = p._calculate_total_n_to_degrade(
            state["timestep"], mrna_specificity, total_kcat_endoRNase
        )
        n_total_trnas_to_degrade = p._calculate_total_n_to_degrade(
            state["timestep"], trna_specificity, total_kcat_endoRNase
        )
        n_total_rrnas_to_degrade = p._calculate_total_n_to_degrade(
            state["timestep"], rrna_specificity, total_kcat_endoRNase
        )

        # Compute RNAse specificity
        rna_specificity = frac_endoRNase_saturated / np.sum(frac_endoRNase_saturated)

        # Boolean variable that tracks existence of each RNA
        rna_exists = (total_RNA_counts > 0).astype(np.int64)

        # Compute degradation probabilities of each RNA: for mRNAs and rRNAs, this
        # is based on the specificity of each mRNA. For tRNAs and rRNAs,
        # this is distributed evenly.
        if p.degrade_misc:
            mrna_deg_probs = (
                1.0
                / np.dot(rna_specificity, is_transient_rna * rna_exists)
                * rna_specificity
                * is_transient_rna
                * rna_exists
            )
        else:
            mrna_deg_probs = (
                1.0
                / np.dot(rna_specificity, p.is_mRNA * rna_exists)
                * rna_specificity
                * p.is_mRNA
                * rna_exists
            )
        rrna_deg_probs = (
            1.0
            / np.dot(rna_specificity, p.is_rRNA * rna_exists)
            * rna_specificity
            * p.is_rRNA
            * rna_exists
        )
        trna_deg_probs = (
            1.0 / np.dot(p.is_tRNA, rna_exists) * p.is_tRNA * rna_exists
        )

        # Mask RNA counts into each class of RNAs
        if p.degrade_misc:
            mrna_counts = total_RNA_counts * is_transient_rna
        else:
            mrna_counts = total_RNA_counts * p.is_mRNA
        trna_counts = total_RNA_counts * p.is_tRNA
        rrna_counts = total_RNA_counts * p.is_rRNA

        # Determine number of individual RNAs to be degraded for each class
        # of RNA.
        n_mrnas_to_degrade = p._get_rnas_to_degrade(
            n_total_mrnas_to_degrade, mrna_deg_probs, mrna_counts
        )
        n_trnas_to_degrade = p._get_rnas_to_degrade(
            n_total_trnas_to_degrade, trna_deg_probs, trna_counts
        )
        n_rrnas_to_degrade = p._get_rnas_to_degrade(
            n_total_rrnas_to_degrade, rrna_deg_probs, rrna_counts
        )
        n_RNAs_to_degrade = n_mrnas_to_degrade + n_trnas_to_degrade + n_rrnas_to_degrade

        # Bulk RNAs (tRNAs and rRNAs) are degraded immediately. Unique RNAs
        # (mRNAs) are immediately deactivated (becomes unable to bind
        # ribosomes), but not degraded until transcription is finished and the
        # mRNA becomes a full transcript to simplify the transcript elongation
        # process.
        n_bulk_RNAs_to_degrade = n_RNAs_to_degrade.copy()
        n_bulk_RNAs_to_degrade[p.is_mRNA.astype(bool)] = 0
        p.n_unique_RNAs_to_deactivate = n_RNAs_to_degrade.copy()
        p.n_unique_RNAs_to_deactivate[np.logical_not(p.is_mRNA.astype(bool))] = 0

        request.setdefault("bulk", []).extend(
            [
                (p.bulk_rnas_idx, n_bulk_RNAs_to_degrade),
                (
                    p.fragment_bases_idx,
                    counts(state["bulk"], p.fragment_bases_idx),
                ),
            ]
        )

        # Calculate the amount of water required for total RNA hydrolysis by
        # endo and exonucleases. We first calculate the number of unique RNAs
        # that should be degraded at this timestep.
        p.unique_mRNAs_to_degrade = np.logical_and(
            np.logical_not(can_translate), is_full_transcript
        )
        p.n_unique_RNAs_to_degrade = np.bincount(
            TU_index[p.unique_mRNAs_to_degrade], minlength=p.n_total_RNAs
        )

        # Assuming complete hydrolysis for now. Note that one additional water
        # molecule is needed for each RNA to hydrolyze the 5' diphosphate.
        water_for_degraded_rnas = np.dot(
            n_bulk_RNAs_to_degrade + p.n_unique_RNAs_to_degrade, p.rna_lengths
        )
        water_for_fragments = counts(state["bulk"], p.fragment_bases_idx).sum()
        request["bulk"].append(
            (p.water_idx, water_for_degraded_rnas + water_for_fragments)
        )
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


class RnaDegradationEvolver(_SafeInvokeMixin, Step):
    """Reads allocation, writes bulk/unique/listener updates."""

    config_schema = {}

    def initialize(self, config):
        self.process = config['process']

    def inputs(self):
        return {
            'allocate': SetStore(),
            'bulk': BulkNumpyUpdate(),
            'RNAs': UniqueNumpyUpdate(),
            'active_ribosome': UniqueNumpyUpdate(),
            'listeners': ListenerStore(),
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
            'active_ribosome': UniqueNumpyUpdate(),
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
        # Get vector of numbers of RNAs to degrade for each RNA species
        n_degraded_bulk_RNA = counts(state["bulk"], p.bulk_rnas_idx)
        n_degraded_unique_RNA = p.n_unique_RNAs_to_degrade
        n_degraded_RNA = n_degraded_bulk_RNA + n_degraded_unique_RNA

        # Deactivate and degrade unique RNAs
        TU_index, can_translate = attrs(state["RNAs"], ["TU_index", "can_translate"])
        can_translate = can_translate.copy()
        n_deactivated_unique_RNA = p.n_unique_RNAs_to_deactivate

        # Deactive unique RNAs
        non_zero_deactivation = n_deactivated_unique_RNA > 0

        for index, n_degraded in zip(
            np.arange(n_deactivated_unique_RNA.size)[non_zero_deactivation],
            n_deactivated_unique_RNA[non_zero_deactivation],
        ):
            # Get mask for translatable mRNAs belonging to the degraded species
            mask = np.logical_and(TU_index == index, can_translate)

            # Choose n_degraded indexes randomly to deactivate
            can_translate[
                p.random_state.choice(
                    size=n_degraded, a=np.where(mask)[0], replace=False
                )
            ] = False

        count_RNA_degraded_per_cistron = p.cistron_tu_mapping_matrix.dot(
            n_degraded_RNA[: p.n_transcribed_rnas]
        )
        # Add degraded counts from mature RNAs
        if p.mature_rna_exists:
            count_RNA_degraded_per_cistron[p.mature_rna_cistron_indexes] += (
                n_degraded_RNA[p.n_transcribed_rnas :]
            )

        update = {
            "listeners": {
                "rna_degradation_listener": {
                    "count_rna_degraded": n_degraded_RNA,
                    "nucleotides_from_degradation": np.dot(
                        n_degraded_RNA, p.rna_lengths
                    ),
                    "count_RNA_degraded_per_cistron": count_RNA_degraded_per_cistron,
                }
            },
            # Degrade bulk RNAs
            "bulk": [(p.bulk_rnas_idx, -n_degraded_bulk_RNA)],
            "RNAs": {
                "set": {"can_translate": can_translate},
                # Degrade full mRNAs that are inactive
                "delete": np.where(p.unique_mRNAs_to_degrade)[0],
            },
        }

        # Modeling assumption: Once a RNA is cleaved by an endonuclease its
        # resulting nucleotides are lumped together as "polymerized fragments".
        # These fragments can carry over from previous timesteps. We are also
        # assuming that during endonucleolytic cleavage the 5'terminal
        # phosphate is removed. This is modeled as all of the fragments being
        # one long linear chain of "fragment bases".

        # Example:
        # PPi-Base-PO4(-)-Base-PO4(-)-Base-OH
        # ==>
        # Pi-FragmentBase-PO4(-)-FragmentBase-PO4(-)-FragmentBase + PPi
        # Note: Lack of -OH on 3' end of chain
        metabolites_endo_cleavage = np.dot(
            p.endo_degradation_stoich_matrix, n_degraded_RNA
        )

        # Increase polymerized fragment counts
        update["bulk"].append(
            (p.fragment_metabolites_idx, metabolites_endo_cleavage)
        )
        # fragment_metabolites overlaps with fragment_bases
        bulk_count_copy = state["bulk"].copy()
        if len(bulk_count_copy.dtype) > 1:
            bulk_count_copy = bulk_count_copy["count"]
        bulk_count_copy[p.fragment_metabolites_idx] += metabolites_endo_cleavage
        fragment_bases = bulk_count_copy[p.fragment_bases_idx]

        # Check if exonucleolytic digestion can happen
        if fragment_bases.sum() != 0:
            # Calculate exolytic cleavage events

            # Modeling assumption: We model fragments as one long fragment chain of
            # polymerized nucleotides. We are also assuming that there is no
            # sequence specificity or bias towards which nucleotides are
            # hydrolyzed.

            # Example:
            # Pi-FragmentBase-PO4(-)-FragmentBase-PO4(-)-FragmentBase + 3 H2O
            # ==>
            # 3 NMP + 3 H(+)
            # Note: Lack of -OH on 3' end of chain

            n_exoRNases = counts(state["bulk"], p.exoRNase_idx)
            n_fragment_bases = fragment_bases
            n_fragment_bases_sum = n_fragment_bases.sum()

            exornase_capacity = (
                n_exoRNases.sum() * p.kcat_exoRNase * (units.s * state["timestep"])
            )

            if exornase_capacity >= n_fragment_bases_sum:
                update["bulk"].extend(
                    [
                        (p.nmps_idx, n_fragment_bases),
                        (p.water_idx, -n_fragment_bases_sum),
                        (p.proton_idx, n_fragment_bases_sum),
                        (p.fragment_bases_idx, -n_fragment_bases),
                    ]
                )
                total_fragment_bases_digested = n_fragment_bases_sum

            else:
                fragment_specificity = n_fragment_bases / n_fragment_bases_sum
                possible_bases_to_digest = p.random_state.multinomial(
                    exornase_capacity, fragment_specificity
                )
                n_fragment_bases_digested = n_fragment_bases - np.fmax(
                    n_fragment_bases - possible_bases_to_digest, 0
                )

                total_fragment_bases_digested = n_fragment_bases_digested.sum()

                update["bulk"].extend(
                    [
                        (p.nmps_idx, n_fragment_bases_digested),
                        (p.water_idx, -total_fragment_bases_digested),
                        (p.proton_idx, total_fragment_bases_digested),
                        (p.fragment_bases_idx, -n_fragment_bases_digested),
                    ]
                )

            update["listeners"]["rna_degradation_listener"]["fragment_bases_digested"] = (
                total_fragment_bases_digested
            )

        # Note that once mRNAs have been degraded,
        # chromosome_structure.py will handle deleting the active
        # ribosomes that were translating those mRNAs.
        # --- end inlined ---
        update['next_update_time'] = state.get('global_time', 0.0) + timestep
        return update
