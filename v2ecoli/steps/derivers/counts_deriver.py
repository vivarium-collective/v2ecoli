"""
=============
Counts Deriver
=============

Consolidated deriver for molecule-count readouts. Computes, in one step,
the count observables formerly split across three separate steps — all
derived from the same ``bulk`` / ``unique`` state and written to disjoint
sub-keys of ``listeners``:

  - ``rna_counts``            (was RNACounts)            — mRNA/rRNA TU + cistron counts
  - ``monomer_counts``        (was MonomerCounts)        — protein monomers incl. complexed/active
  - ``unique_molecule_counts``(was UniqueMoleculeCounts) — per-type active unique-molecule totals

This is a *genuine single step*: one ``update()`` body with the three
computations inlined verbatim from the originals (not a wrapper delegating
to sub-instances). Verified byte-identical against the pre-consolidation
baseline signature.

Config: the three former configs are merged into one flat dict by the
composite. ``monomer_counts`` and ``unique_molecule_counts`` both carried a
``unique_ids`` key with identical contents (the 11 unique-molecule types),
so the flat merge is unambiguous; ``unique_ids`` is read only by the
unique-molecule section.
"""

import numpy as np

from bigraph_schema.contract import ProcessContract
from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.types.labeled_array import register_labeled_array
from v2ecoli.library.schema import attrs, bulk_name_to_idx, counts
from v2ecoli.library.schema_types import (
    RNA_ARRAY,
    ACTIVE_RIBOSOME_ARRAY,
    ACTIVE_RNAP_ARRAY,
    ACTIVE_REPLISOME_ARRAY,
    PROMOTER_ARRAY,
)

NAME = "counts_deriver"
TOPOLOGY = {
    "listeners": ("listeners",),
    "bulk": ("bulk",),
    "unique": ("unique",),
    "RNAs": ("unique", "RNA"),
    "promoters": ("unique", "promoter"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}


class CountsDeriver(Step):
    """Single deriver for RNA, monomer, and unique-molecule counts."""

    description = (
        "Tallies molecule counts for downstream readouts: mRNA/rRNA "
        "transcription-unit and cistron counts (partial + full transcripts), "
        "protein monomer counts (including monomers sequestered in complexes "
        "and in active ribosomes/RNAPs/replisomes), and per-type active "
        "unique-molecule totals. Consolidates the former RNACounts, "
        "MonomerCounts, and UniqueMoleculeCounts listeners into one step."
    )

    contract = ProcessContract(
        summary=(
            "Count-observable deriver: tallies mRNA/rRNA transcription-unit and "
            "cistron counts (partial vs full transcripts), protein monomer "
            "counts (decomposing bulk complexes and folding in subunits "
            "sequestered in active ribosomes/RNAPs/replisomes and promoter-bound "
            "TFs), and per-type active unique-molecule totals. Pure read-out."
        ),
        math=[
            "n_monomer = bulk_free + S_cplx·(-n_complex) + S_eq·(-n_eqcplx) + S_tcs·(-n_tcs) + subunits(active ribosome/RNAP/replisome + bound_TF)",
            "TU_counts = bincount(RNA.TU_index over translatable + rRNA);  cistron_counts = M_cistron_TU · TU_counts;  partial = all - full",
            "unique_type_count = sum(_entryState) per unique-molecule type",
        ],
        symbols={
            "n_monomer": "protein monomer counts, including monomers sequestered in complexes and active machinery (count)",
            "bulk_free": "free bulk molecule counts before complex decomposition (count)",
            "S_cplx": "complexation stoichiometry (molecules × complexes); unpacks bulk complexes into constituent monomers",
            "n_complex": "counts of bulk complexation complexes being decomposed (count)",
            "S_eq": "equilibrium-binding stoichiometry; unpacks equilibrium complexes into monomers",
            "n_eqcplx": "counts of equilibrium complexes (count)",
            "S_tcs": "two-component-system stoichiometry; unpacks phosphotransfer complexes into monomers",
            "n_tcs": "counts of two-component-system complexes (count)",
            "TU_counts": "per-transcription-unit RNA counts from bincount over unique RNA TU_index (count)",
            "M_cistron_TU": "cistron × transcription-unit mapping matrix",
            "cistron_counts": "per-cistron RNA counts (count)",
            "_entryState": "per-row active flag of a unique-molecule array; summed to count active molecules of a type",
        },
        inputs={
            "RNAs": "Reads unique RNA TU_index, can_translate, and is_full_transcript to tally mRNA/rRNA transcription-unit and cistron counts (partial vs full transcripts).",
            "bulk": "Reads bulk molecule counts, then decomposes complexes (TCS → equilibrium → complexation) to recover protein monomer totals.",
            "unique": "Reads active_ribosome/active_RNAP/active_replisome _entryState to fold sequestered subunits into monomer totals and to report per-type active counts.",
            "promoters": "Reads bound_TF to fold promoter-bound transcription-factor subunits back into monomer totals.",
            "global_time": "Clock read used with timestep to gate the update cadence.",
            "timestep": "Update cadence divisor (update runs when global_time % timestep == 0).",
        },
        outputs={
            "listeners": "Writes rna_counts (mRNA/rRNA TU and cistron counts, partial/full), monomer_counts (overwrite/SET semantics), and unique_molecule_counts (per-type active totals).",
        },
        config={
            "cistron_tu_mapping_matrix": "Cistron × transcription-unit matrix M_cistron_TU mapping TU counts to cistron counts.",
            "complexation_stoich": "Complexation stoichiometry S_cplx; unpacks bulk complexes into constituent monomers.",
            "equilibrium_stoich": "Equilibrium-binding stoichiometry S_eq; unpacks equilibrium complexes.",
            "two_component_system_stoich": "Two-component-system stoichiometry S_tcs; unpacks phosphotransfer complexes.",
            "monomer_ids": "Protein monomer species whose final counts are reported.",
            "unique_ids": "Unique-molecule types whose active _entryState totals are reported.",
            "tf_ids": "Transcription-factor subunit ids folded back into monomer totals from promoter-bound TFs.",
            "ribosome_50s_subunits": "50S ribosomal subunit ids + stoichiometry for active-ribosome subunit accounting.",
            "ribosome_30s_subunits": "30S ribosomal subunit ids + stoichiometry for active-ribosome subunit accounting.",
            "rnap_subunits": "RNA-polymerase subunit ids + stoichiometry for active-RNAP subunit accounting.",
        },
        assumptions=[
            "Pure derivation/read-out: it does not mutate bulk/unique state; monomer_counts uses overwrite (SET) semantics to avoid step-wise accumulation.",
            "Complexes are decomposed in order TCS → equilibrium → complexation (nested subunits) after adding active-machinery and bound-TF subunits, so nested complexes fully resolve to monomers.",
            "Consolidates the former RNACounts, MonomerCounts, and UniqueMoleculeCounts listeners; verified byte-identical to the pre-consolidation baseline.",
        ],
    )

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # --- RNA counts ---
        'rna_ids': 'list[string]',
        'mrna_indexes': 'array[integer]',
        'all_TU_ids': 'list[string]',
        'mRNA_indexes': 'array[integer]',
        'mRNA_TU_ids': 'list[string]',
        'rRNA_indexes': 'array[integer]',
        'rRNA_TU_ids': 'list[string]',
        'all_cistron_ids': 'list[string]',
        'cistron_is_mRNA': 'array[integer]',
        'mRNA_cistron_ids': 'list[string]',
        'cistron_is_rRNA': 'array[integer]',
        'rRNA_cistron_ids': 'list[string]',
        'cistron_tu_mapping_matrix': 'csr_matrix',
        # --- monomer counts ---
        'bulk_molecule_ids': 'list[string]',
        'complexation_molecule_ids': 'list[string]',
        'complexation_complex_ids': 'list[string]',
        'equilibrium_molecule_ids': 'list[string]',
        'equilibrium_complex_ids': 'list[string]',
        'monomer_ids': 'list[string]',
        'tf_ids': 'list[string]',
        'two_component_system_molecule_ids': 'list[string]',
        'two_component_system_complex_ids': 'list[string]',
        'ribosome_50s_subunits': 'list[string]',
        'ribosome_30s_subunits': 'list[string]',
        'rnap_subunits': 'list[string]',
        'replisome_trimer_subunits': 'list[string]',
        'replisome_monomer_subunits': 'list[string]',
        'complexation_stoich': 'csr_matrix',
        'equilibrium_stoich': 'csr_matrix',
        'two_component_system_stoich': 'csr_matrix',
        # --- unique-molecule counts ---
        'unique_ids': 'list[string]',
        # --- shared ---
        'time_step': 'float{1.0}',
        'emit_unique': 'boolean{false}',
    }

    def inputs(self):
        return {
            'RNAs': {'_type': RNA_ARRAY, '_default': []},
            'bulk': {'_type': 'bulk_array', '_default': []},
            'unique': {
                'active_ribosome': {'_type': ACTIVE_RIBOSOME_ARRAY, '_default': []},
                'active_RNAP': {'_type': ACTIVE_RNAP_ARRAY, '_default': []},
                'active_replisome': {'_type': ACTIVE_REPLISOME_ARRAY, '_default': []},
            },
            'promoters': {'_type': PROMOTER_ARRAY, '_default': []},
            'global_time': {'_type': 'float', '_default': 0.0},
            'timestep': {'_type': 'float', '_default': 1.0},
        }

    def outputs(self):
        return {
            'listeners': {
                'rna_counts': {
                    'mRNA_counts': {'_type': f'overwrite[array[{self.n_mRNA_TU},integer]]', '_default': []},
                    'full_mRNA_counts': {'_type': f'overwrite[array[{self.n_mRNA_TU},integer]]', '_default': []},
                    'partial_mRNA_counts': {'_type': f'overwrite[array[{self.n_mRNA_TU},integer]]', '_default': []},
                    'mRNA_cistron_counts': {'_type': f'overwrite[array[{self.n_mRNA_cistron},integer]]', '_default': []},
                    'full_mRNA_cistron_counts': {'_type': f'overwrite[array[{self.n_mRNA_cistron},integer]]', '_default': []},
                    'partial_mRNA_cistron_counts': {'_type': f'overwrite[array[{self.n_mRNA_cistron},integer]]', '_default': []},
                    'partial_rRNA_counts': {'_type': f'overwrite[array[{self.n_rRNA_TU},integer]]', '_default': []},
                    'partial_rRNA_cistron_counts': {'_type': f'overwrite[array[{self.n_rRNA_cistron},integer]]', '_default': []},
                },
                # ``overwrite[...]`` gives SET semantics. The bare Array type's
                # default apply is ACCUMULATE (element-wise add), which would
                # add each step's true monomer-count vector to the prior store
                # value — producing a linear ramp (~N x the real count after N
                # steps) and inflating monomer_counts by ~10^3 x. The sibling
                # rna_counts fields are protected the same way (overwrite[...]).
                'monomer_counts': 'overwrite[monomer_counts_vec]',
                'unique_molecule_counts': 'map[integer]',
            },
        }

    def initialize(self, config):
        p = self.parameters
        # ---------------- RNA counts ----------------
        self.all_TU_ids = p["all_TU_ids"]
        self.mRNA_indexes = p["mRNA_indexes"]
        self.mRNA_TU_ids = p["mRNA_TU_ids"]
        self.rRNA_indexes = p["rRNA_indexes"]
        self.rRNA_TU_ids = p["rRNA_TU_ids"]
        self.all_cistron_ids = p["all_cistron_ids"]
        self.cistron_is_mRNA = p["cistron_is_mRNA"]
        self.mRNA_cistron_ids = p["mRNA_cistron_ids"]
        self.cistron_is_rRNA = p["cistron_is_rRNA"]
        self.rRNA_cistron_ids = p["rRNA_cistron_ids"]
        self.cistron_tu_mapping_matrix = p["cistron_tu_mapping_matrix"]
        self.n_mRNA_TU = len(self.mRNA_TU_ids)
        self.n_rRNA_TU = len(self.rRNA_TU_ids)
        self.n_mRNA_cistron = len(self.mRNA_cistron_ids)
        self.n_rRNA_cistron = len(self.rRNA_cistron_ids)

        # ---------------- monomer counts (verbatim from MonomerCounts) -----
        self.bulk_molecule_ids = p["bulk_molecule_ids"]
        self.complexation_molecule_ids = p["complexation_molecule_ids"]
        self.complexation_complex_ids = p["complexation_complex_ids"]
        self.equilibrium_molecule_ids = p["equilibrium_molecule_ids"]
        self.equilibrium_complex_ids = p["equilibrium_complex_ids"]
        self.monomer_ids = p["monomer_ids"]
        self.n_monomers = len(self.monomer_ids)
        self.two_component_system_molecule_ids = p["two_component_system_molecule_ids"]
        self.two_component_system_complex_ids = p["two_component_system_complex_ids"]
        # Transcription-factor subunit IDs, for folding promoter-bound TFs back
        # into the monomer total (compartment-tagged in the listener config).
        self.tf_subunit_ids = p["tf_ids"]

        ribosome_50s_subunits = p["ribosome_50s_subunits"]
        ribosome_30s_subunits = p["ribosome_30s_subunits"]
        self.ribosome_subunit_ids = (
            ribosome_50s_subunits["subunitIds"].tolist()
            + ribosome_30s_subunits["subunitIds"].tolist()
        )
        rnap_subunits = p["rnap_subunits"]
        self.rnap_subunit_ids = rnap_subunits["subunitIds"].tolist()
        replisome_trimer_subunits = p["replisome_trimer_subunits"]
        replisome_monomer_subunits = p["replisome_monomer_subunits"]
        self.replisome_subunit_ids = (
            replisome_trimer_subunits + replisome_monomer_subunits
        )

        self.complexation_stoich = p["complexation_stoich"]
        self.equilibrium_stoich = p["equilibrium_stoich"]
        self.two_component_system_stoich = p["two_component_system_stoich"]
        self.ribosome_stoich = np.hstack(
            (
                ribosome_50s_subunits["subunitStoich"],
                ribosome_30s_subunits["subunitStoich"],
            )
        )
        self.rnap_stoich = rnap_subunits["subunitStoich"]
        self.replisome_stoich = np.hstack(
            (
                3 * np.ones(len(replisome_trimer_subunits)),
                np.ones(len(replisome_monomer_subunits)),
            )
        )
        self.monomer_idx = None

        # Register the named labeled-array type so the output_metadata walker
        # can recover monomer IDs without needing sim_data.  Registering in
        # initialize() (not outputs()) ensures the type is in the core before
        # Composite() resolves port schemas.
        if self.core is not None:
            register_labeled_array(
                self.core, 'monomer_counts_vec',
                shape=(self.n_monomers,),
                data=np.dtype('int64'),
                labels=tuple(self.monomer_ids),
            )

        # ---------------- unique-molecule counts ----------------
        self.unique_ids = p["unique_ids"]

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def update(self, states, interval=None):
        # ===================== RNA counts =====================
        TU_indexes, can_translate, is_full_transcript = attrs(
            states["RNAs"], ["TU_index", "can_translate", "is_full_transcript"]
        )
        is_rRNA = np.isin(TU_indexes, self.rRNA_indexes)
        all_TU_counts = np.bincount(
            TU_indexes[np.logical_or(can_translate, is_rRNA)],
            minlength=len(self.all_TU_ids),
        )
        mRNA_counts = all_TU_counts[self.mRNA_indexes]
        full_TU_counts = np.bincount(
            TU_indexes[np.logical_and(can_translate, is_full_transcript)],
            minlength=len(self.all_TU_ids),
        )
        full_mRNA_counts = full_TU_counts[self.mRNA_indexes]
        partial_TU_counts = all_TU_counts - full_TU_counts
        partial_mRNA_counts = mRNA_counts - full_mRNA_counts
        partial_rRNA_counts = all_TU_counts[self.rRNA_indexes]
        cistron_counts = self.cistron_tu_mapping_matrix.dot(all_TU_counts)
        mRNA_cistron_counts = cistron_counts[self.cistron_is_mRNA]
        full_mRNA_cistron_counts = self.cistron_tu_mapping_matrix.dot(full_TU_counts)[
            self.cistron_is_mRNA
        ]
        partial_mRNA_cistron_counts = self.cistron_tu_mapping_matrix.dot(
            partial_TU_counts
        )[self.cistron_is_mRNA]
        partial_rRNA_cistron_counts = cistron_counts[self.cistron_is_rRNA]

        # ===================== monomer counts =====================
        if self.monomer_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.bulk_molecule_idx = bulk_name_to_idx(self.bulk_molecule_ids, bulk_ids)
            self.monomer_idx = bulk_name_to_idx(self.monomer_ids, bulk_ids)
            self.complexation_molecule_idx = bulk_name_to_idx(
                self.complexation_molecule_ids, bulk_ids
            )
            self.complexation_complex_idx = bulk_name_to_idx(
                self.complexation_complex_ids, bulk_ids
            )
            self.equilibrium_molecule_idx = bulk_name_to_idx(
                self.equilibrium_molecule_ids, bulk_ids
            )
            self.equilibrium_complex_idx = bulk_name_to_idx(
                self.equilibrium_complex_ids, bulk_ids
            )
            self.two_component_system_molecule_idx = bulk_name_to_idx(
                self.two_component_system_molecule_ids, bulk_ids
            )
            self.two_component_system_complex_idx = bulk_name_to_idx(
                self.two_component_system_complex_ids, bulk_ids
            )
            self.ribosome_subunit_idx = bulk_name_to_idx(
                self.ribosome_subunit_ids, bulk_ids
            )
            self.rnap_subunit_idx = bulk_name_to_idx(self.rnap_subunit_ids, bulk_ids)
            self.replisome_subunit_idx = bulk_name_to_idx(
                self.replisome_subunit_ids, bulk_ids
            )
            self.tf_subunit_idx = bulk_name_to_idx(self.tf_subunit_ids, bulk_ids)

        bulkMoleculeCounts = counts(states["bulk"], self.bulk_molecule_idx)
        n_active_ribosome = states["unique"]["active_ribosome"]["_entryState"].sum()
        n_active_rnap = states["unique"]["active_RNAP"]["_entryState"].sum()
        n_active_replisome = states["unique"]["active_replisome"]["_entryState"].sum()
        (n_bound_TFs,) = attrs(states["promoters"], ["bound_TF"])

        # Counts of subunits sequestered in active unique-molecule complexes
        # (and promoter-bound transcription factors):
        n_ribosome_subunit = n_active_ribosome * self.ribosome_stoich
        n_rnap_subunit = n_active_rnap * self.rnap_stoich
        n_replisome_subunit = n_active_replisome * self.replisome_stoich
        n_bound_TF_subunit = n_bound_TFs.astype(np.int32).sum(axis=0)

        # Add active unique-molecule + bound-TF subunit counts to the free bulk
        # counts FIRST: some of these subunits are themselves bulk complexes
        # that the unpacking steps below further decompose (the old code added
        # them last, so nested subunits inside TCS/equilibrium/complexation
        # complexes were never cascaded — and bound TFs were never counted).
        bulkMoleculeCounts[self.ribosome_subunit_idx] += n_ribosome_subunit.astype(np.int32)
        bulkMoleculeCounts[self.rnap_subunit_idx] += n_rnap_subunit.astype(np.int32)
        bulkMoleculeCounts[self.replisome_subunit_idx] += n_replisome_subunit.astype(np.int32)
        bulkMoleculeCounts[self.tf_subunit_idx] += n_bound_TF_subunit.astype(np.int32)

        # Unpack two-component-system complexes (their subunits can be
        # equilibrium or complexation complexes, so unpack before both):
        two_component_monomer_counts = np.dot(
            self.two_component_system_stoich,
            np.negative(bulkMoleculeCounts[self.two_component_system_complex_idx]),
        )
        bulkMoleculeCounts[self.two_component_system_molecule_idx] += (
            two_component_monomer_counts.astype(np.int32)
        )

        # Unpack equilibrium complexes (their subunits can be complexation
        # complexes, so unpack before those):
        equilibrium_monomer_counts = np.dot(
            self.equilibrium_stoich,
            np.negative(bulkMoleculeCounts[self.equilibrium_complex_idx]),
        )
        bulkMoleculeCounts[self.equilibrium_molecule_idx] += (
            equilibrium_monomer_counts.astype(np.int32)
        )

        # Unpack bulk complexation complexes:
        complex_monomer_counts = np.dot(
            self.complexation_stoich,
            np.negative(bulkMoleculeCounts[self.complexation_complex_idx]),
        )
        bulkMoleculeCounts[self.complexation_molecule_idx] += (
            complex_monomer_counts.astype(np.int32)
        )

        monomer_counts = bulkMoleculeCounts[self.monomer_idx]

        # ===================== unique-molecule counts =====================
        unique = states["unique"]
        unique_molecule_counts = {
            str(unique_id): (
                unique[unique_id]["_entryState"].sum()
                if unique_id in unique else 0
            )
            for unique_id in self.unique_ids
        }

        return {
            "listeners": {
                "rna_counts": {
                    "mRNA_counts": mRNA_counts,
                    "full_mRNA_counts": full_mRNA_counts,
                    "partial_mRNA_counts": partial_mRNA_counts,
                    "partial_rRNA_counts": partial_rRNA_counts,
                    "mRNA_cistron_counts": mRNA_cistron_counts,
                    "full_mRNA_cistron_counts": full_mRNA_cistron_counts,
                    "partial_mRNA_cistron_counts": partial_mRNA_cistron_counts,
                    "partial_rRNA_cistron_counts": partial_rRNA_cistron_counts,
                },
                "monomer_counts": monomer_counts,
                "unique_molecule_counts": unique_molecule_counts,
            },
        }
