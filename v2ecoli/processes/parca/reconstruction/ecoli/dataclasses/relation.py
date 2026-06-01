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

        Phase 3 of Path 3: ``tf_tu_routing.tsv`` entries with
        ``route_only=true`` restrict a (TF, target_gene) regulation to a
        specific TU; TFs are removed from other TUs containing the cistron.
        See reports/regulation_data_pipeline_v2ecoli.html §10.
        """
        cistron_ids = sim_data.process.transcription.cistron_data["id"]

        # Build (tf_active_id, cistron_rna_id) -> set of allowed TU_ids from
        # tf_tu_routing.tsv. When a (TF, cistron) pair is in this map, only
        # the listed TUs may inherit the TF; other TUs containing the cistron
        # are filtered out.
        restrict = {}
        if getattr(raw_data, "tf_tu_routing", None):
            # Map TSV short names -> active complex IDs via
            # transcription_factors.tsv (column "activeId")
            tf_short_to_active = {
                tf["TF"]: tf["activeId"]
                for tf in raw_data.transcription_factors
                if tf.get("activeId")
            }
            # Map gene short names -> cistron RNA IDs via raw_data.genes
            gene_short_to_rna = {
                g["symbol"]: g["rna_ids"][0] for g in raw_data.genes
            }
            for row in raw_data.tf_tu_routing:
                if str(row.get("route_only", "")).lower() not in ("true", "1"):
                    continue
                tf_id = tf_short_to_active.get(row["TF"])
                cistron = gene_short_to_rna.get(row["target_gene"])
                if not tf_id or not cistron:
                    continue
                restrict.setdefault((tf_id, cistron), set()).add(row["TU_id"])

        self.rna_id_to_regulating_tfs = {}
        for rna_id in sim_data.process.transcription.rna_data["id"]:
            tu_id_no_loc = str(rna_id).split("[")[0]
            tf_list = []
            for (
                cistron_index
            ) in sim_data.process.transcription.rna_id_to_cistron_indexes(rna_id):
                cistron = cistron_ids[cistron_index]
                for tf in sim_data.process.transcription_regulation.target_tf.get(
                    cistron, []
                ):
                    # Apply tf_tu_routing restriction: if a restriction
                    # exists for this (tf, cistron), skip TUs not in the
                    # allowed set.
                    allowed_tus = restrict.get((tf, cistron))
                    if allowed_tus is not None and tu_id_no_loc not in allowed_tus:
                        continue
                    tf_list.append(tf)

            # Remove duplicates and sort
            self.rna_id_to_regulating_tfs[rna_id] = sorted(set(tf_list))

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
