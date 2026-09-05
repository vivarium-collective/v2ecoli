"""
SimulationData relation functions
"""

import numpy as np


class Relation(object):
    """Relation"""

    def __init__(self, raw_data, sim_data):
        self._build_cistron_to_monomer_mapping(raw_data, sim_data)
        self._build_monomer_to_mRNA_cistron_mapping(raw_data, sim_data)
        self._build_monomer_to_tu_mapping(raw_data, sim_data)
        self._build_RNA_to_tf_mapping(raw_data, sim_data)
        self._build_tf_to_RNA_mapping(raw_data, sim_data)

    def _build_cistron_to_monomer_mapping(self, raw_data, sim_data):
        """
        Build a vector that can map vectors that describe a property for RNA
        cistrons into a vector that describes the same property for the
        corresponding monomers if used as an index array. Assumes that each
        monomer maps to a single RNA cistron (A single RNA can map to multiple
        monomers).

        e.g.
        monomer_property = RNA_cistron_property[
                sim_data.relation.cistron_to_monomer_mapping]
        """
        # Map cistron IDs to indexes given in cistron_data (rnas.tsv)
        cistron_id_to_index = {
            cistron_id: i
            for i, cistron_id in enumerate(
                sim_data.process.transcription.cistron_data["id"]
            )
        }

        # List the cistron_data indexes of cistron IDs in the order of
        # corresponding cistrons given in monomer_data (proteins.tsv)
        self.cistron_to_monomer_mapping = np.array(
            [
                cistron_id_to_index[cistron_id]
                for cistron_id in sim_data.process.translation.monomer_data[
                    "cistron_id"
                ]
            ]
        )

    def _build_monomer_to_mRNA_cistron_mapping(self, raw_data, sim_data):
        """
        Builds a sparse matrix that can map vectors that describe a property
        for protein monomers into a vector that describes the same property for
        the corresponding mRNA cistrons if multiplied to the right of the
        original vector. The transformed property must be additive (i.e. if two
        proteins map to the same cistron, the values given for the two proteins
        are added to yield a value for the cistron).

        The full matrix can be returned by calling
        monomer_to_mRNA_cistron_mapping().
        """
        # Initialize sparse matrix variables
        self._monomer_to_mRNA_cistron_mapping_i = []
        self._monomer_to_mRNA_cistron_mapping_j = []
        self._monomer_to_mRNA_cistron_mapping_v = []
        self._monomer_to_mRNA_cistron_mapping_shape = (
            len(sim_data.process.translation.monomer_data),
            sim_data.process.transcription.cistron_data["is_mRNA"].sum(),
        )

        # Build mapping from mRNA ID to mRNA index
        mRNA_data = sim_data.process.transcription.cistron_data[
            sim_data.process.transcription.cistron_data["is_mRNA"]
        ]
        mRNA_id_to_index = {mRNA["id"]: j for j, mRNA in enumerate(mRNA_data)}

        # Build sparse matrix
        for i, monomer in enumerate(sim_data.process.translation.monomer_data):
            self._monomer_to_mRNA_cistron_mapping_i.append(i)
            self._monomer_to_mRNA_cistron_mapping_j.append(
                mRNA_id_to_index[monomer["cistron_id"]]
            )
            self._monomer_to_mRNA_cistron_mapping_v.append(1)

    def monomer_to_mRNA_cistron_mapping(self):
        """
        Returns the full version of the sparse matrix built by
        _build_monomer_to_mRNA_cistron_mapping().

        e.g.
        mRNA_property = sim_data.relation.monomer_to_mRNA_cistron_mapping().T.dot(
                monomer_property)
        """
        out = np.zeros(self._monomer_to_mRNA_cistron_mapping_shape, dtype=np.float64)
        out[
            self._monomer_to_mRNA_cistron_mapping_i,
            self._monomer_to_mRNA_cistron_mapping_j,
        ] = self._monomer_to_mRNA_cistron_mapping_v
        return out

    def _build_monomer_to_tu_mapping(self, raw_data, sim_data):
        """
        Builds a dictionary that maps monomer IDs to a list of all transcription
        unit IDs that the monomer can be translated from.
        """
        self.monomer_index_to_tu_indexes = {
            i: sim_data.process.transcription.cistron_id_to_rna_indexes(
                monomer["cistron_id"]
            )
            for i, monomer in enumerate(sim_data.process.translation.monomer_data)
        }

    def _build_RNA_to_tf_mapping(self, raw_data, sim_data):
        """
        Builds a dictionary that maps RNA IDs to a list of all transcription
        factor IDs that regulate the given RNA. All TFs that target any of the
        constituent cistrons in the RNA are added to each list.

        With promoter-specific attribution enabled (see
        ``_load_tf_tu_attribution``), a TF is additionally dropped from any TU
        that EcoCyc CONTRADICTS -- one where the factor is recorded acting at
        the same operon, but on a different transcription unit. Assignments
        EcoCyc confirms, and assignments it says nothing about, are left alone:
        absence of a record is not evidence against the fold-change data that
        put the edge there.
        """
        cistron_ids = sim_data.process.transcription.cistron_data["id"]
        attribution = self._load_tf_tu_attribution()

        self.rna_id_to_regulating_tfs = {}
        self.tf_tu_reattribution_dropped = []
        for rna_id in sim_data.process.transcription.rna_data["id"]:
            tf_list = []
            for (
                cistron_index
            ) in sim_data.process.transcription.rna_id_to_cistron_indexes(rna_id):
                tf_list.extend(
                    sim_data.process.transcription_regulation.target_tf.get(
                        cistron_ids[cistron_index], []
                    )
                )

            tf_list = sorted(set(tf_list))
            if attribution is not None:
                genes = {
                    str(cistron_ids[i]).replace("_RNA", "")
                    for i in sim_data.process.transcription.rna_id_to_cistron_indexes(
                        rna_id
                    )
                }
                this_tu = str(rna_id)[:-3] if str(rna_id).endswith("[c]") else str(rna_id)
                kept = []
                for tf in tf_list:
                    recorded = set()
                    for gene in genes:
                        recorded |= attribution.get((tf, gene), set())
                    # No record at this operon -> absence of evidence, keep it.
                    # Recorded here -> keep. Recorded only elsewhere -> drop.
                    if not recorded or this_tu in recorded:
                        kept.append(tf)
                    else:
                        self.tf_tu_reattribution_dropped.append(
                            (str(rna_id), tf, sorted(recorded))
                        )
                tf_list = kept

            self.rna_id_to_regulating_tfs[rna_id] = tf_list

    @staticmethod
    def _load_tf_tu_attribution():
        """
        Load references/tf_tu_reattribution.tsv as {(tf_id, gene): {tu, ...}}.

        Returns None -- leaving the cistron-content assignment untouched --
        unless V2ECOLI_PROMOTER_SPECIFIC_TF is set to a truthy value. The table
        is derived from the EcoCyc PGDB by
        scripts/build_tf_tu_reattribution.py; the PGDB export itself is
        licensed and not committed.
        """
        import os

        if os.environ.get("V2ECOLI_PROMOTER_SPECIFIC_TF", "").lower() not in (
            "1",
            "true",
            "yes",
        ):
            return None

        here = os.path.dirname(os.path.abspath(__file__))
        root = here
        for _ in range(8):
            candidate = os.path.join(root, "references", "tf_tu_reattribution.tsv")
            if os.path.exists(candidate):
                break
            root = os.path.dirname(root)
        else:
            raise FileNotFoundError(
                "V2ECOLI_PROMOTER_SPECIFIC_TF is set but "
                "references/tf_tu_reattribution.tsv was not found. Generate it "
                "with scripts/build_tf_tu_reattribution.py."
            )

        table = {}
        with open(candidate) as handle:
            for line in handle:
                if line.startswith("#") or line.startswith("tf_id\t"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 4:
                    continue
                tf_id, _eco, tu, gene = parts
                table.setdefault((tf_id, gene), set()).add(tu)
        return table

    def _build_tf_to_RNA_mapping(self, raw_data, sim_data):
        """
        Builds a dictionary that maps transcription factor IDs to a list of all
        RNA IDs that are targeted by the given TF. All RNA transcription units
        that contain any of the cistrons regulated by the TF are added to each
        list.
        """
        self.tf_id_to_target_RNAs = {}
        for rna_id, tf_list in self.rna_id_to_regulating_tfs.items():
            for tf_id in tf_list:
                self.tf_id_to_target_RNAs.setdefault(tf_id, []).append(rna_id)
