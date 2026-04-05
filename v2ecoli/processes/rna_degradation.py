"""
===============
RNA Degradation
===============

Mathematical formulations

* ``dr/dt = Kb - kcatEndoRNase * EndoRNase * r/Km / (1 + Sum(r/Km))``

where

* r = RNA counts
* Kb = RNA production given a RNAP synthesis rate
* kcatEndoRNase = enzymatic activity for EndoRNases
* Km = Michaelis-Menten constants fitted to recapitulate first-order
* RNA decay: ``kd * r = kcatEndoRNase * EndoRNase * r/Km / (1 + sum(r/Km))``

This sub-model encodes molecular simulation of RNA degradation as two main
steps guided by RNases, "endonucleolytic cleavage" and "exonucleolytic
digestion":

1. Compute total counts of RNA to be degraded (D) and total capacity for
   endo-cleavage (C) at each time point
2. Evaluate C and D. If C > D, then define a fraction of active endoRNases
3. Dissect RNA degraded into different species (mRNA, tRNA, and rRNA) by
   accounting endoRNases specificity
4. Update RNA fragments (assumption: fragments are represented as a pool of
   nucleotides) created because of endonucleolytic cleavage
5. Compute total capacity of exoRNases and determine fraction of nucleotides
   that can be digested
6. Update pool of metabolites (H and H2O) created because of exonucleolytic
   digestion
"""

import numpy as np

from process_bigraph import Step
from process_bigraph.composite import SyncUpdate
from bigraph_schema.schema import Float, Overwrite, Node

from v2ecoli.library.schema import (
    bulk_name_to_idx,
    counts,
    attrs,
    numpy_schema,
    listener_schema,
)

from v2ecoli.library.units import units
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.unique_numpy import UniqueNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore
from v2ecoli.steps.partition import _protect_state


class RnaDegradationLogic:
    """Biological logic for RNA degradation.

    Extracted from the original PartitionedProcess so that
    Requester and Evolver each get their own instance.
    """

    def __init__(self, parameters):
        self.parameters = {**RnaDegradation.defaults, **(parameters or {})}
        parameters = self.parameters
        self.rna_ids = parameters["rna_ids"]
        self.mature_rna_ids = parameters["mature_rna_ids"]
        self.n_transcribed_rnas = len(self.rna_ids)
        self.mature_rna_exists = len(self.mature_rna_ids) > 0
        self.cistron_ids = parameters["cistron_ids"]
        self.cistron_tu_mapping_matrix = parameters["cistron_tu_mapping_matrix"]
        self.mature_rna_cistron_indexes = parameters["mature_rna_cistron_indexes"]
        self.all_rna_ids = parameters["all_rna_ids"]
        self.n_total_RNAs = parameters["n_total_RNAs"]

        self.n_avogadro = parameters["n_avogadro"]
        self.cell_density = parameters["cell_density"]

        self.endoRNase_ids = parameters["endoRNase_ids"]
        self.exoRNase_ids = parameters["exoRNase_ids"]
        self.kcat_exoRNase = parameters["kcat_exoRNase"]
        self.Kcat_endoRNases = parameters["Kcat_endoRNases"]

        self.uncharged_trna_indexes = parameters["uncharged_trna_indexes"]
        self.charged_trna_names = parameters["charged_trna_names"]

        self.rna_deg_rates = parameters["rna_deg_rates"]

        self.is_mRNA = parameters["is_mRNA"]
        self.is_rRNA = parameters["is_rRNA"]
        self.is_tRNA = parameters["is_tRNA"]
        self.is_miscRNA = parameters["is_miscRNA"]
        self.degrade_misc = parameters["degrade_misc"]

        self.rna_lengths = parameters["rna_lengths"]
        self.nt_counts = parameters["nt_counts"]

        self.polymerized_ntp_ids = parameters["polymerized_ntp_ids"]
        self.nmp_ids = parameters["nmp_ids"]
        self.water_id = parameters["water_id"]
        self.ppi_id = parameters["ppi_id"]
        self.proton_id = parameters["proton_id"]

        self.end_cleavage_metabolite_ids = self.polymerized_ntp_ids + [
            self.water_id, self.ppi_id, self.proton_id,
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

        self.Kms = parameters["Kms"]

        self.seed = parameters["seed"]
        self.random_state = np.random.RandomState(seed=self.seed)

        # Numpy indices
        self.water_idx = None

    def _init_indices(self, bulk_ids):
        if self.water_idx is None:
            self.charged_trna_idx = bulk_name_to_idx(self.charged_trna_names, bulk_ids)
            self.bulk_rnas_idx = bulk_name_to_idx(self.all_rna_ids, bulk_ids)
            self.nmps_idx = bulk_name_to_idx(self.nmp_ids, bulk_ids)
            self.fragment_metabolites_idx = bulk_name_to_idx(
                self.end_cleavage_metabolite_ids, bulk_ids
            )
            self.fragment_bases_idx = bulk_name_to_idx(
                self.polymerized_ntp_ids, bulk_ids
            )
            self.endoRNase_idx = bulk_name_to_idx(self.endoRNase_ids, bulk_ids)
            self.exoRNase_idx = bulk_name_to_idx(self.exoRNase_ids, bulk_ids)
            self.water_idx = bulk_name_to_idx(self.water_id, bulk_ids)
            self.proton_idx = bulk_name_to_idx(self.proton_id, bulk_ids)

    def calculate_request(self, timestep, states):
        """Calculate bulk request and cache values needed by evolve_state."""
        self._init_indices(states["bulk"]["id"])

        cell_mass = states["listeners"]["mass"]["cell_mass"] * units.fg
        cell_volume = cell_mass / self.cell_density
        counts_to_molar = 1 / (self.n_avogadro * cell_volume)

        bulk_RNA_counts = counts(states["bulk"], self.bulk_rnas_idx)
        bulk_RNA_counts[self.uncharged_trna_indexes] += counts(
            states["bulk"], self.charged_trna_idx
        )

        TU_index, can_translate, is_full_transcript = attrs(
            states["RNAs"], ["TU_index", "can_translate", "is_full_transcript"]
        )

        TU_index_translatable_mRNAs = TU_index[can_translate]
        unique_RNA_counts = np.bincount(
            TU_index_translatable_mRNAs, minlength=self.n_total_RNAs
        )
        total_RNA_counts = bulk_RNA_counts + unique_RNA_counts

        rna_conc_molar = counts_to_molar * total_RNA_counts

        endoRNase_counts = counts(states["bulk"], self.endoRNase_idx)
        total_kcat_endoRNase = units.dot(self.Kcat_endoRNases, endoRNase_counts)

        frac_endoRNase_saturated = (
            rna_conc_molar / self.Kms / (1 + units.sum(rna_conc_molar / self.Kms))
        ).asNumber()

        total_endoRNase_counts = np.sum(endoRNase_counts)
        diff_relative_first_order_decay = units.sum(
            units.abs(
                self.rna_deg_rates * total_RNA_counts
                - total_kcat_endoRNase * frac_endoRNase_saturated
            )
        )
        endoRNase_per_rna = total_endoRNase_counts / np.sum(total_RNA_counts)

        requests = {"listeners": {"rna_degradation_listener": {}}}
        requests["listeners"]["rna_degradation_listener"][
            "fraction_active_endoRNases"
        ] = np.sum(frac_endoRNase_saturated)
        requests["listeners"]["rna_degradation_listener"][
            "diff_relative_first_order_decay"
        ] = diff_relative_first_order_decay.asNumber()
        requests["listeners"]["rna_degradation_listener"]["fract_endo_rrna_counts"] = (
            endoRNase_per_rna
        )

        if self.degrade_misc:
            is_transient_rna = self.is_mRNA | self.is_miscRNA
            mrna_specificity = np.dot(frac_endoRNase_saturated, is_transient_rna)
        else:
            mrna_specificity = np.dot(frac_endoRNase_saturated, self.is_mRNA)
        trna_specificity = np.dot(frac_endoRNase_saturated, self.is_tRNA)
        rrna_specificity = np.dot(frac_endoRNase_saturated, self.is_rRNA)

        n_total_mrnas_to_degrade = self._calculate_total_n_to_degrade(
            states["timestep"], mrna_specificity, total_kcat_endoRNase
        )
        n_total_trnas_to_degrade = self._calculate_total_n_to_degrade(
            states["timestep"], trna_specificity, total_kcat_endoRNase
        )
        n_total_rrnas_to_degrade = self._calculate_total_n_to_degrade(
            states["timestep"], rrna_specificity, total_kcat_endoRNase
        )

        rna_specificity = frac_endoRNase_saturated / np.sum(frac_endoRNase_saturated)
        rna_exists = (total_RNA_counts > 0).astype(np.int64)

        if self.degrade_misc:
            mrna_deg_probs = (
                1.0
                / np.dot(rna_specificity, is_transient_rna * rna_exists)
                * rna_specificity * is_transient_rna * rna_exists
            )
        else:
            mrna_deg_probs = (
                1.0
                / np.dot(rna_specificity, self.is_mRNA * rna_exists)
                * rna_specificity * self.is_mRNA * rna_exists
            )
        rrna_deg_probs = (
            1.0
            / np.dot(rna_specificity, self.is_rRNA * rna_exists)
            * rna_specificity * self.is_rRNA * rna_exists
        )
        trna_deg_probs = (
            1.0 / np.dot(self.is_tRNA, rna_exists) * self.is_tRNA * rna_exists
        )

        if self.degrade_misc:
            mrna_counts = total_RNA_counts * is_transient_rna
        else:
            mrna_counts = total_RNA_counts * self.is_mRNA
        trna_counts = total_RNA_counts * self.is_tRNA
        rrna_counts = total_RNA_counts * self.is_rRNA

        n_mrnas_to_degrade = self._get_rnas_to_degrade(
            n_total_mrnas_to_degrade, mrna_deg_probs, mrna_counts
        )
        n_trnas_to_degrade = self._get_rnas_to_degrade(
            n_total_trnas_to_degrade, trna_deg_probs, trna_counts
        )
        n_rrnas_to_degrade = self._get_rnas_to_degrade(
            n_total_rrnas_to_degrade, rrna_deg_probs, rrna_counts
        )
        n_RNAs_to_degrade = n_mrnas_to_degrade + n_trnas_to_degrade + n_rrnas_to_degrade

        n_bulk_RNAs_to_degrade = n_RNAs_to_degrade.copy()
        n_bulk_RNAs_to_degrade[self.is_mRNA.astype(bool)] = 0
        self.n_unique_RNAs_to_deactivate = n_RNAs_to_degrade.copy()
        self.n_unique_RNAs_to_deactivate[np.logical_not(self.is_mRNA.astype(bool))] = 0

        requests.setdefault("bulk", []).extend(
            [
                (self.bulk_rnas_idx, n_bulk_RNAs_to_degrade),
                (
                    self.fragment_bases_idx,
                    counts(states["bulk"], self.fragment_bases_idx),
                ),
            ]
        )

        self.unique_mRNAs_to_degrade = np.logical_and(
            np.logical_not(can_translate), is_full_transcript
        )
        self.n_unique_RNAs_to_degrade = np.bincount(
            TU_index[self.unique_mRNAs_to_degrade], minlength=self.n_total_RNAs
        )

        water_for_degraded_rnas = np.dot(
            n_bulk_RNAs_to_degrade + self.n_unique_RNAs_to_degrade, self.rna_lengths
        )
        water_for_fragments = counts(states["bulk"], self.fragment_bases_idx).sum()
        requests["bulk"].append(
            (self.water_idx, water_for_degraded_rnas + water_for_fragments)
        )
        return requests

    def evolve_state(self, timestep, states):
        """Evolve state using cached values from calculate_request."""
        self._init_indices(states["bulk"]["id"])

        n_degraded_bulk_RNA = counts(states["bulk"], self.bulk_rnas_idx)
        n_degraded_unique_RNA = self.n_unique_RNAs_to_degrade
        n_degraded_RNA = n_degraded_bulk_RNA + n_degraded_unique_RNA

        TU_index, can_translate = attrs(states["RNAs"], ["TU_index", "can_translate"])
        can_translate = can_translate.copy()
        n_deactivated_unique_RNA = self.n_unique_RNAs_to_deactivate

        non_zero_deactivation = n_deactivated_unique_RNA > 0

        for index, n_degraded in zip(
            np.arange(n_deactivated_unique_RNA.size)[non_zero_deactivation],
            n_deactivated_unique_RNA[non_zero_deactivation],
        ):
            mask = np.logical_and(TU_index == index, can_translate)
            can_translate[
                self.random_state.choice(
                    size=n_degraded, a=np.where(mask)[0], replace=False
                )
            ] = False

        count_RNA_degraded_per_cistron = self.cistron_tu_mapping_matrix.dot(
            n_degraded_RNA[: self.n_transcribed_rnas]
        )
        if self.mature_rna_exists:
            count_RNA_degraded_per_cistron[self.mature_rna_cistron_indexes] += (
                n_degraded_RNA[self.n_transcribed_rnas :]
            )

        update = {
            "listeners": {
                "rna_degradation_listener": {
                    "count_rna_degraded": n_degraded_RNA,
                    "nucleotides_from_degradation": np.dot(
                        n_degraded_RNA, self.rna_lengths
                    ),
                    "count_RNA_degraded_per_cistron": count_RNA_degraded_per_cistron,
                }
            },
            "bulk": [(self.bulk_rnas_idx, -n_degraded_bulk_RNA)],
            "RNAs": {
                "set": {"can_translate": can_translate},
                "delete": np.where(self.unique_mRNAs_to_degrade)[0],
            },
        }

        metabolites_endo_cleavage = np.dot(
            self.endo_degradation_stoich_matrix, n_degraded_RNA
        )

        update["bulk"].append(
            (self.fragment_metabolites_idx, metabolites_endo_cleavage)
        )
        bulk_count_copy = states["bulk"].copy()
        if len(bulk_count_copy.dtype) > 1:
            bulk_count_copy = bulk_count_copy["count"]
        bulk_count_copy[self.fragment_metabolites_idx] += metabolites_endo_cleavage
        fragment_bases = bulk_count_copy[self.fragment_bases_idx]

        if fragment_bases.sum() == 0:
            return update

        n_exoRNases = counts(states["bulk"], self.exoRNase_idx)
        n_fragment_bases = fragment_bases
        n_fragment_bases_sum = n_fragment_bases.sum()

        exornase_capacity = (
            n_exoRNases.sum() * self.kcat_exoRNase * (units.s * states["timestep"])
        )

        if exornase_capacity >= n_fragment_bases_sum:
            update["bulk"].extend(
                [
                    (self.nmps_idx, n_fragment_bases),
                    (self.water_idx, -n_fragment_bases_sum),
                    (self.proton_idx, n_fragment_bases_sum),
                    (self.fragment_bases_idx, -n_fragment_bases),
                ]
            )
            total_fragment_bases_digested = n_fragment_bases_sum
        else:
            fragment_specificity = n_fragment_bases / n_fragment_bases_sum
            possible_bases_to_digest = self.random_state.multinomial(
                exornase_capacity, fragment_specificity
            )
            n_fragment_bases_digested = n_fragment_bases - np.fmax(
                n_fragment_bases - possible_bases_to_digest, 0
            )
            total_fragment_bases_digested = n_fragment_bases_digested.sum()
            update["bulk"].extend(
                [
                    (self.nmps_idx, n_fragment_bases_digested),
                    (self.water_idx, -total_fragment_bases_digested),
                    (self.proton_idx, total_fragment_bases_digested),
                    (self.fragment_bases_idx, -n_fragment_bases_digested),
                ]
            )

        update["listeners"]["rna_degradation_listener"]["fragment_bases_digested"] = (
            total_fragment_bases_digested
        )

        return update

    def _calculate_total_n_to_degrade(self, timestep, specificity, total_kcat_endornase):
        return np.round(
            (specificity * total_kcat_endornase * (units.s * timestep)).asNumber()
        )

    def _get_rnas_to_degrade(self, n_total_rnas_to_degrade, rna_deg_probs, rna_counts):
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


class RnaDegradationRequester(Step):
    """Requester step for RNA degradation."""

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance or create own
        self.process = config.pop("_logic", None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = RnaDegradationLogic(config)
        self.process_name = 'ecoli-rna-degradation'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'RNAs': InPlaceDict(),
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

        # Extract listener data from the request
        listener_data = requests.pop('listeners', {})
        return {
            'request': {self.process_name: requests},
            'listeners': listener_data,
        }


class RnaDegradationEvolver(Step):
    """Evolver step for RNA degradation.

    RECOMPUTES cached values since Requester and Evolver are separate instances.
    """

    config_schema = {}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        # Accept shared Logic instance from Requester
        self.process = config.pop("_logic", None) if isinstance(config, dict) else None
        if self.process is None:
            self.process = RnaDegradationLogic(config)
        self.process_name = 'ecoli-rna-degradation'

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'allocate': InPlaceDict(),
            'RNAs': InPlaceDict(),
            'active_ribosome': InPlaceDict(),
            'listeners': InPlaceDict(),
            'timestep': Float(_default=self.process.parameters.get('time_step', 1.0)),
            'global_time': Float(_default=0.0),
            'next_update_time': Float(_default=1.0),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
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

class RnaDegradation(PartitionedProcess):
    """Legacy PartitionedProcess wrapper -- will be removed after migration."""

    name = "ecoli-rna-degradation"
    topology = {
        "bulk": ("bulk",),
        "RNAs": ("unique", "RNA"),
        "active_ribosome": ("unique", "active_ribosome"),
        "listeners": ("listeners",),
        "timestep": ("timestep",),
    }
    defaults = {
        "rna_ids": [],
        "mature_rna_ids": [],
        "cistron_ids": [],
        "cistron_tu_mapping_matrix": [],
        "mature_rna_cistron_indexes": [],
        "all_rna_ids": [],
        "n_total_RNAs": 0,
        "n_avogadro": 0.0,
        "cell_density": 1100 * units.g / units.L,
        "endoRNase_ids": [],
        "exoRNase_ids": [],
        "kcat_exoRNase": np.array([]) / units.s,
        "Kcat_endoRNases": np.array([]) / units.s,
        "charged_trna_names": [],
        "uncharged_trna_indexes": [],
        "rna_deg_rates": [],
        "is_mRNA": np.array([]),
        "is_rRNA": np.array([]),
        "is_tRNA": np.array([]),
        "is_miscRNA": np.array([]),
        "degrade_misc": False,
        "rna_lengths": np.array([]),
        "nt_counts": np.array([[]]),
        "polymerized_ntp_ids": [],
        "water_id": "h2o",
        "ppi_id": "ppi",
        "proton_id": "h+",
        "nmp_ids": [],
        "rrfa_idx": 0,
        "rrla_idx": 0,
        "rrsa_idx": 0,
        "ribosome30S": "ribosome30S",
        "ribosome50S": "ribosome50S",
        "Kms": np.array([]) * units.mol / units.L,
        "seed": 0,
        "emit_unique": False,
    }

    def __init__(self, parameters=None, **kwargs):
        super().__init__(parameters, **kwargs)
        self._logic = RnaDegradationLogic(self.parameters)

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "active_ribosome": numpy_schema(
                "active_ribosome", emit=self.parameters["emit_unique"]
            ),
            "RNAs": numpy_schema("RNAs", emit=self.parameters["emit_unique"]),
            "listeners": {
                "mass": listener_schema({"cell_mass": 0.0, "dry_mass": 0.0}),
                "rna_degradation_listener": listener_schema(
                    {
                        "fraction_active_endornases": 0.0,
                        "diff_relative_first_order_decay": 0.0,
                        "fract_endo_rrna_counts": 0.0,
                        "count_rna_degraded": (
                            [0] * len(self._logic.all_rna_ids),
                            self._logic.all_rna_ids,
                        ),
                        "count_RNA_degraded_per_cistron": (
                            [0] * len(self._logic.cistron_ids),
                            self._logic.cistron_ids,
                        ),
                        "nucleotides_from_degradation": 0,
                        "fragment_bases_digested": 0,
                    }
                ),
            },
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def calculate_request(self, timestep, states):
        return self._logic.calculate_request(timestep, states)

    def evolve_state(self, timestep, states):
        return self._logic.evolve_state(timestep, states)
