"""
SimulationData for transcription regulation

"""

from typing import Union

import numpy as np
from scipy import sparse


class TranscriptionRegulation(object):
    """
    SimulationData for transcription regulation
    """

    def __init__(self, raw_data, sim_data):
        # Build lookups
        self._build_lookups(raw_data)

        # Store list of transcription factor IDs
        self.tf_ids = list(sorted(sim_data.tf_to_active_inactive_conditions.keys()))

        # Build dictionary mapping RNA targets to its regulators
        self.target_tf = {}

        for tf in sorted(sim_data.tf_to_fold_change):
            targets = sim_data.tf_to_fold_change[tf]
            targetsToRemove = []

            for target in targets:
                if target not in self.target_tf:
                    self.target_tf[target] = []

                self.target_tf[target].append(tf)

            for targetToRemove in targetsToRemove:
                sim_data.tf_to_fold_change[tf].pop(targetToRemove)

        # Build dictionaries mapping transcription factors to their bound form,
        # and to their regulating type
        self.active_to_bound = {
            x["active TF"]: x["metabolite bound form"]
            for x in raw_data.tf_one_component_bound
        }
        self.tf_to_tf_type = {
            x["active TF"]: x["TF type"] for x in raw_data.condition.tf_condition
        }
        self.tf_to_gene_id = {
            x["active TF"]: x["TF"] for x in raw_data.condition.tf_condition
        }

        # Values set after promoter fitting in parca with calculateRnapRecruitment()
        self.basal_prob = None
        self.delta_prob = None

        # Promoter-keyed views of the above, built by
        # build_promoter_keyed_probs() at the end of Step 7. Additive — the
        # runtime still reads the TU-keyed arrays. See
        # docs/promoter_transcript_split_scope.html.
        self.promoter_ids = None
        self.promoter_to_TU = None
        self.promoter_initiation_share = None
        self.promoter_basal_prob = None
        self.promoter_delta_prob = None

    def p_promoter_bound_tf(self, tfActive, tfInactive):
        """
        Computes probability of a transcription factor binding promoter.
        """
        return float(tfActive) / (float(tfActive) + float(tfInactive))

    def p_promoter_bound_SKd(self, signal, Kd, power):
        """
        Computes probability of a one-component transcription factor binding
        promoter.
        """
        return float(signal) ** power / (float(signal) ** power + float(Kd) ** power)

    def build_promoter_keyed_probs(self, sim_data):
        """
        Build promoter-keyed views of ``basal_prob`` and ``delta_prob``.

        Phase 1 of the promoter/transcript split
        (docs/promoter_transcript_split_scope.html). Additive: the runtime
        still reads the TU-keyed arrays. Nothing here changes behaviour.

        A transcription unit currently serves as both promoter and
        transcript, so ``basal_prob`` carries one synthesis probability per
        TU. Under the split a transcript may be driven by several
        promoters, and the transcript's synthesis has to be divided among
        them by their **initiation share**:

            promoter_basal_prob[p] = basal_prob[TU(p)] * share[p]
            promoter_delta_prob[p, tf] = delta_prob[TU(p), tf] * share[p]

        ``delta_prob`` is scaled by the same share so that the split is
        exactly conservative:

            sum over p of TU(p)==T  of  promoter_basal_prob[p] == basal_prob[T]

        and likewise per TF. That invariant is what lets the flip to
        promoter-keyed indexing be behaviour-preserving in aggregate; a
        later ``tf_promoter_routing`` would assign a TF's whole delta to
        one promoter instead of spreading it, which is where per-promoter
        regulation actually starts to differ.

        Shares are uniform (1/N) unless a curated source supplies them —
        this branch carries no per-promoter measurements yet, so every
        share is uniform today; the hook is kept for when promoter-level
        TSS data lands. Uniform is not a guess
        about biology — with N promoters feeding one transcript, any split
        summing to 1 reproduces the same transcript-level synthesis. It
        only matters once a TF is routed to a specific promoter, and that
        needs data, which is what the flat file is for.

        Transitional note: ``getter.get_promoter_records()`` maps promoters
        to canonical transcripts under *pure* dedup, but ``rna_data`` still
        contains the exemption's extra transcripts. Until the exemption is
        removed a promoter is mapped to itself when it appears in
        ``rna_data``, and to its canonical transcript otherwise. That keeps
        the conservation invariant exact in both the current and target
        states.
        """
        rna_ids = [str(x).split("[")[0] for x in
                   sim_data.process.transcription.rna_data["id"]]
        tu_index = {rna_id: i for i, rna_id in enumerate(rna_ids)}
        records = sim_data.getter.get_promoter_records()

        # Curated initiation shares, keyed by promoter id (basal condition).
        curated_share = {}
        for row in getattr(sim_data.process.transcription,
                           "_per_promoter_ratios", []) or []:
            if row.get("condition") == "basal":
                try:
                    curated_share[row["TU_id"]] = float(row["ratio"])
                except (KeyError, TypeError, ValueError):
                    continue

        # Resolve each promoter onto the transcript it drives *today*.
        promoter_ids, promoter_to_TU = [], []
        for record in records:
            target = (record["id"] if record["id"] in tu_index
                      else record["transcript_id"])
            if target not in tu_index:
                continue
            promoter_ids.append(record["id"])
            promoter_to_TU.append(tu_index[target])

        promoter_to_TU = np.asarray(promoter_to_TU, dtype=np.int64)
        n_promoters = len(promoter_ids)

        # Share within each transcript's promoter set.
        members = {}
        for p, tu in enumerate(promoter_to_TU):
            members.setdefault(int(tu), []).append(p)
        shares = np.zeros(n_promoters, dtype=np.float64)
        for group in members.values():
            curated = [curated_share.get(promoter_ids[p]) for p in group]
            if all(c is not None for c in curated) and sum(curated) > 0:
                total = sum(curated)
                for p, c in zip(group, curated):
                    shares[p] = c / total
            else:
                for p in group:
                    shares[p] = 1.0 / len(group)

        self.promoter_ids = promoter_ids
        self.promoter_to_TU = promoter_to_TU
        self.promoter_initiation_share = shares
        self.promoter_basal_prob = self.basal_prob[promoter_to_TU] * shares

        # Re-key the sparse delta_prob rows from TU space to promoter space.
        tu_to_promoters = members
        delta_i, delta_j, delta_v = [], [], []
        for tu, j, v in zip(self.delta_prob["deltaI"],
                            self.delta_prob["deltaJ"],
                            self.delta_prob["deltaV"]):
            for p in tu_to_promoters.get(int(tu), ()):
                delta_i.append(p)
                delta_j.append(j)
                delta_v.append(v * shares[p])
        self.promoter_delta_prob = {
            "deltaI": np.asarray(delta_i, dtype=np.int64),
            "deltaJ": np.asarray(delta_j, dtype=np.int64),
            "deltaV": np.asarray(delta_v, dtype=np.float64),
            "shape": (n_promoters, self.delta_prob["shape"][1]),
        }

    def get_delta_prob_matrix(
        self, dense=False, ppgpp=False
    ) -> Union[sparse.csr_matrix, np.ndarray]:
        """
        Returns the delta probability matrix mapping the promoter binding effect
        of each TF to each gene.

        Args:
                dense: If True, returns a dense matrix, otherwise csr sparse
                ppgpp: If True, normalizes delta probabilities to be on the same
                        scale as ppGpp normalized probabilities since delta_prob is
                        calculated based on basal_prob which is not normalized to 1

        Returns:
                delta_prob: matrix of probabilities changes expected with a TF
                        binding to a promoter for each gene (n genes, m TFs)
        """

        ppgpp_scaling = self.basal_prob[self.delta_prob["deltaI"]]
        ppgpp_scaling[ppgpp_scaling == 0] = 1
        scaling_factor = ppgpp_scaling if ppgpp else 1.0
        delta_prob = sparse.csr_matrix(
            (
                self.delta_prob["deltaV"] / scaling_factor,
                (self.delta_prob["deltaI"], self.delta_prob["deltaJ"]),
            ),
            shape=self.delta_prob["shape"],
        )

        if dense:
            delta_prob = delta_prob.toarray()

        return delta_prob

    def _build_lookups(self, raw_data):
        """
        Builds dictionaries for mapping transcription factor abbreviations to
        their RNA IDs, and to their active form.
        """
        gene_id_to_cistron_id = {x["id"]: x["rna_ids"][0] for x in raw_data.genes}

        self.abbr_to_rna_id = {}
        for lookupInfo in raw_data.transcription_factors:
            if (
                len(lookupInfo["geneId"]) == 0
                or lookupInfo["geneId"] not in gene_id_to_cistron_id
            ):
                continue
            self.abbr_to_rna_id[lookupInfo["TF"]] = gene_id_to_cistron_id[
                lookupInfo["geneId"]
            ]

        self.abbr_to_active_id = {
            x["TF"]: x["activeId"].split(", ")
            for x in raw_data.transcription_factors
            if len(x["activeId"]) > 0
        }
