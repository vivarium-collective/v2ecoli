"""
RnaMaturation process
=====================
- Converts unprocessed tRNA/rRNA molecules into mature tRNA/rRNAs
- Consolidates the different variants of 23S, 16S, and 5S rRNAs into the single
  variant that is used for ribosomal subunits
"""

import numpy as np

from process_bigraph import Step

from v2ecoli.steps.partition import _protect_state, _SafeInvokeMixin
from v2ecoli.library.schema import counts, bulk_name_to_idx
from v2ecoli.types.bulk_numpy import BulkNumpyUpdate
from v2ecoli.types.stores import InPlaceDict, ListenerStore


class RnaMaturationStep(_SafeInvokeMixin, Step):
    """RNA Maturation — single-step enzyme-gated maturation + rRNA consolidation."""

    config_schema = {}

    topology = {"bulk": ("bulk",), "bulk_total": ("bulk",), "listeners": ("listeners",)}

    def initialize(self, config):
        self.stoich_matrix = config["stoich_matrix"]
        self.enzyme_matrix = config["enzyme_matrix"]
        self.n_required_enzymes = config["n_required_enzymes"]
        self.degraded_nt_counts = config["degraded_nt_counts"]
        self.n_ppi_added = config["n_ppi_added"]

        self.main_23s_rRNA_id = config["main_23s_rRNA_id"]
        self.main_16s_rRNA_id = config["main_16s_rRNA_id"]
        self.main_5s_rRNA_id = config["main_5s_rRNA_id"]

        self.variant_23s_rRNA_ids = config["variant_23s_rRNA_ids"]
        self.variant_16s_rRNA_ids = config["variant_16s_rRNA_ids"]
        self.variant_5s_rRNA_ids = config["variant_5s_rRNA_ids"]

        self.delta_nt_counts_23s = config["delta_nt_counts_23s"]
        self.delta_nt_counts_16s = config["delta_nt_counts_16s"]
        self.delta_nt_counts_5s = config["delta_nt_counts_5s"]

        self.unprocessed_rna_ids = config["unprocessed_rna_ids"]
        self.mature_rna_ids = config["mature_rna_ids"]
        self.rna_maturation_enzyme_ids = config["rna_maturation_enzyme_ids"]
        self.fragment_bases = config["fragment_bases"]
        self.ppi = config["ppi"]
        self.water = config["water"]
        self.nmps = config["nmps"]
        self.proton = config["proton"]

        self.ppi_idx = None

    def inputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'bulk_total': BulkNumpyUpdate(),
            'global_time': InPlaceDict(),
            'next_update_time': InPlaceDict(),
        }

    def outputs(self):
        return {
            'bulk': BulkNumpyUpdate(),
            'listeners': ListenerStore(),
            'next_update_time': InPlaceDict(),
        }

    def update(self, state, interval=None):
        next_time = state.get('next_update_time', 0.0)
        global_time = state.get('global_time', 0.0)
        if next_time > global_time:
            return {}

        state = _protect_state(state)

        # Initialize indices on first call
        if self.ppi_idx is None:
            bulk_ids = state["bulk"]["id"]
            self.unprocessed_rna_idx = bulk_name_to_idx(self.unprocessed_rna_ids, bulk_ids)
            self.mature_rna_idx = bulk_name_to_idx(self.mature_rna_ids, bulk_ids)
            self.rna_maturation_enzyme_idx = bulk_name_to_idx(
                self.rna_maturation_enzyme_ids, bulk_ids)
            self.fragment_base_idx = bulk_name_to_idx(self.fragment_bases, bulk_ids)
            self.ppi_idx = bulk_name_to_idx(self.ppi, bulk_ids)
            self.water_idx = bulk_name_to_idx(self.water, bulk_ids)
            self.nmps_idx = bulk_name_to_idx(self.nmps, bulk_ids)
            self.proton_idx = bulk_name_to_idx(self.proton, bulk_ids)
            self.main_23s_rRNA_idx = bulk_name_to_idx(self.main_23s_rRNA_id, bulk_ids)
            self.main_16s_rRNA_idx = bulk_name_to_idx(self.main_16s_rRNA_id, bulk_ids)
            self.main_5s_rRNA_idx = bulk_name_to_idx(self.main_5s_rRNA_id, bulk_ids)
            self.variant_23s_rRNA_idx = bulk_name_to_idx(self.variant_23s_rRNA_ids, bulk_ids)
            self.variant_16s_rRNA_idx = bulk_name_to_idx(self.variant_16s_rRNA_ids, bulk_ids)
            self.variant_5s_rRNA_idx = bulk_name_to_idx(self.variant_5s_rRNA_ids, bulk_ids)

        # Check enzyme availability to determine which reactions are possible
        enzyme_availability = counts(
            state["bulk_total"], self.rna_maturation_enzyme_idx
        ).astype(bool)
        reaction_is_off = (
            self.enzyme_matrix.dot(enzyme_availability) < self.n_required_enzymes
        )

        # Get counts of unprocessed RNAs, zeroing those without enzymes
        unprocessed_rna_counts = counts(state["bulk_total"], self.unprocessed_rna_idx)
        unprocessed_rna_counts[reaction_is_off] = 0

        # Calculate mature RNAs and fragment bases from maturation
        n_mature_rnas = self.stoich_matrix.dot(unprocessed_rna_counts)
        n_added_bases_from_maturation = np.dot(
            self.degraded_nt_counts.T, unprocessed_rna_counts
        )
        ppi_update = self.n_ppi_added.dot(unprocessed_rna_counts)

        update = {
            "bulk": [
                (self.mature_rna_idx, n_mature_rnas),
                (self.unprocessed_rna_idx, -unprocessed_rna_counts),
                (self.ppi_idx, -ppi_update),
            ],
            "listeners": {
                "rna_maturation_listener": {
                    "total_maturation_events": unprocessed_rna_counts.sum(),
                    "total_degraded_ntps": n_added_bases_from_maturation.sum(dtype=int),
                    "unprocessed_rnas_consumed": unprocessed_rna_counts,
                    "mature_rnas_generated": n_mature_rnas,
                    "maturation_enzyme_counts": counts(
                        state["bulk_total"], self.rna_maturation_enzyme_idx
                    ),
                }
            },
        }

        # Consolidate variant rRNAs
        variant_23s_rRNA_counts = counts(state["bulk_total"], self.variant_23s_rRNA_idx)
        variant_16s_rRNA_counts = counts(state["bulk_total"], self.variant_16s_rRNA_idx)
        variant_5s_rRNA_counts = counts(state["bulk_total"], self.variant_5s_rRNA_idx)

        n_added_bases_from_consolidation = (
            self.delta_nt_counts_23s.T.dot(variant_23s_rRNA_counts)
            + self.delta_nt_counts_16s.T.dot(variant_16s_rRNA_counts)
            + self.delta_nt_counts_5s.T.dot(variant_5s_rRNA_counts)
        )

        update["bulk"].extend([
            (self.main_23s_rRNA_idx, variant_23s_rRNA_counts.sum()),
            (self.main_16s_rRNA_idx, variant_16s_rRNA_counts.sum()),
            (self.main_5s_rRNA_idx, variant_5s_rRNA_counts.sum()),
            (self.variant_23s_rRNA_idx, -variant_23s_rRNA_counts),
            (self.variant_16s_rRNA_idx, -variant_16s_rRNA_counts),
            (self.variant_5s_rRNA_idx, -variant_5s_rRNA_counts),
        ])

        # Balance mass with NMPs/water/protons
        n_added_bases = (
            n_added_bases_from_maturation + n_added_bases_from_consolidation
        ).astype(int)
        n_total_added_bases = n_added_bases.sum()

        update["bulk"].extend([
            (self.nmps_idx, n_added_bases),
            (self.water_idx, -n_total_added_bases),
            (self.proton_idx, n_total_added_bases),
        ])

        update["next_update_time"] = global_time + 1.0
        return update
