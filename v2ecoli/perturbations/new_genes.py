"""New-gene (heterologous) expression + translation perturbations.

The heterologous counterpart to :mod:`v2ecoli.perturbations.translation`. That
module covers **native** genes and is translation-only: a per-monomer multiplier
on translation efficiency, applied to the cached process config through
``baseline``'s ``config_overrides`` seam. New genes need a second axis it cannot
reach — **expression** — because a new gene is inserted *silent*:

    sim_data.adjust_new_gene_final_expression: "the baseline new gene expression
    values need to be set to small non-zero values ... as new genes are knocked
    out by default."

Measured on a real heterologous insertion: its new-gene entries in
``rna_expression['basal']`` are **exactly 0**, against a median 1.06e-06 across
native entries of that same array. ``rna_expression`` is the array the simulation actually consumes
(``initial_conditions.py`` derives cistron-level counts from it); note that
``cistron_expression`` is a reconstruction-time quantity with no runtime reader,
and it stays 0 even after a successful induction — so it is the wrong thing to
check when confirming this function worked.

With no transcript there is nothing to translate, so the enzymes are never
synthesised, the pathway carries no flux, and a design screen over translation
efficiency alone would rank every arm at zero — **while appearing to work**, since
the arms differ in their inputs and are identical in their output. Turning the
construct on is therefore not a convenience knob; it is the first half of the
design axis.

Surface, and why it differs from the sibling module
---------------------------------------------------
``translation.py`` patches a cached process-config array and never touches
sim_data. This module **mutates sim_data directly**, because the expression
half reaches sim_data fields that have no cached-config equivalent — synthesis
probability, expression and free/ppGpp expression are modified; attenuation
adjustments and the basal/delta promoter probabilities are asserted empty rather
than written. That also matches the surface
``sim_data.internal_shift_dict`` expects: the loader
(``v2ecoli/library/sim_data.py``) stores ``{generation: (func, params)}`` and
applies ``func(sim_data, *params)`` once the lineage reaches that generation.
⚠ **In v2ecoli that path is not wired.** Nothing here populates the dict, and the
loader's branch is gated on ``"agent_id" in kwargs``, which no v2ecoli
``LoadSimData`` call site passes — so a callable stored there would be kept and
never fired. It schedules a deferred induction in vEcoli; a v2ecoli driver
applies the modification before the cache is built instead.

⚠ **Composition with the sibling module.**
``translation.translation_efficiency_override`` returns a **full replacement**
translation-efficiency array. If new-gene efficiencies are set here and a native
override is then computed from a *different* cache, the whole-array replacement
silently discards them. Compose both axes against the **same** cache, or apply
this function to sim_data **before** the cache is built so the values are
already in the array the override is derived from.

The reference is ``CovertLab/vEcoli``'s
``ecoli/variants/new_gene_internal_shift.py``, whose ``modify_new_gene_exp_trl``
performs the same two operations (``adjust_new_gene_final_expression`` for the
expression half, direct assignment into
``translation_efficiencies_by_monomer`` for the efficiency half). This module
adds the per-target **relative weight vectors** — one factor per new-gene RNA and
one per monomer — so the two halves can be swept independently across a design
grid rather than moved together by a single scalar.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

__all__ = ["new_gene_indices", "new_gene_operon_structure",
           "set_new_gene_expression"]


def new_gene_indices(sim_data: Any) -> tuple[list[str], list[int], list[str], list[int]]:
    """Return ``(rna_ids, rna_indices, monomer_ids, monomer_indices)`` for new genes.

    New-gene *cistrons* and *monomers* are identified by the authoritative
    ``is_new_gene`` flag on ``cistron_data``. New-gene **RNAs** are identified by
    the ``NG`` id prefix instead: ``rna_data`` carries no equivalent flag (the
    fork records the same gap as a TODO), and a cistron does not map 1:1 to an
    RNA once operons are involved.

    Raises:
        ValueError: if the sim_data carries no new genes — a clearer failure
            than silently perturbing nothing.
    """
    cistrons = sim_data.process.transcription.cistron_data.struct_array
    monomers = sim_data.process.translation.monomer_data.struct_array

    new_cistron_ids = cistrons[cistrons["is_new_gene"]]["id"].tolist()
    if not new_cistron_ids:
        raise ValueError(
            "no new-gene cistrons in this sim_data — was it built with "
            "new_genes_option / the `new_genes` ParCa parameter set?"
        )

    cistron_to_monomer = dict(zip(monomers["cistron_id"], monomers["id"]))
    new_monomer_ids = [
        cistron_to_monomer[c] for c in new_cistron_ids if c in cistron_to_monomer
    ]
    if len(new_monomer_ids) != len(new_cistron_ids):
        raise ValueError(
            f"{len(new_cistron_ids)} new-gene cistrons but "
            f"{len(new_monomer_ids)} monomers — every new gene should encode one."
        )
    monomer_index = {m: i for i, m in enumerate(monomers["id"])}
    new_monomer_indices = [monomer_index[m] for m in new_monomer_ids]

    rna_ids = list(sim_data.process.transcription.rna_data["id"])
    # rna_data ids carry a compartment suffix ("...[c]"); the fork strips it
    # before matching, so do the same rather than matching the raw id.
    rna_index = {str(r)[:-3]: i for i, r in enumerate(rna_ids)}
    new_rna_ids = [r for r in rna_index if r.startswith("NG")]
    new_rna_indices = [rna_index[r] for r in new_rna_ids]
    if not new_rna_indices:
        raise ValueError(
            "new-gene cistrons exist but no rna_data id starts with 'NG' — the "
            "id convention this function relies on does not hold for this build."
        )
    # The RNA list and the monomer list are built from DIFFERENT orderings — RNAs
    # by id prefix over ``rna_data``, monomers from ``new_cistron_ids`` in
    # ``cistron_data`` order. Callers pair per-target weight vectors against these
    # two lists positionally, so if the orderings ever diverge a weight meant for
    # one gene would be applied to another — silently, and a length check cannot
    # see it, because the counts still match.
    #
    # ⚠ That hazard is real only when the two lists describe the SAME set, i.e. a
    # monocistronic insertion. Requiring equality outright assumed every new gene
    # is its own transcription unit, which this function's own docstring says is
    # not so — and it rejected the polycistronic case outright rather than pairing
    # it. An operon has one TU and N cistrons, so the two vectors index different
    # spaces and there is no cross-vector correspondence to protect.
    if len(new_rna_ids) == len(new_cistron_ids):
        # Monocistronic: the lists describe the same genes, so a positional
        # weight for RNA i and monomer i must mean the same gene. Unchanged.
        if new_rna_ids != new_cistron_ids:
            raise ValueError(
                f"new-gene RNA order {new_rna_ids} and cistron order "
                f"{new_cistron_ids} do not correspond; per-target weight vectors "
                "are paired positionally against these two lists, so applying "
                "them would mis-assign weights."
            )
    else:
        # Polycistronic: verify every new cistron is transcribed from a new TU
        # and every new TU carries at least one new cistron. Without that, an
        # ``NG``-prefixed RNA unrelated to these cistrons would silently receive
        # an expression weight meant for the construct.
        _check_operon_coverage(sim_data, new_cistron_ids, new_rna_ids)
    return new_rna_ids, new_rna_indices, new_monomer_ids, new_monomer_indices


def new_gene_operon_structure(sim_data: Any) -> dict[str, list[str]]:
    """Map each new-gene transcription unit to the new cistrons it carries.

    ⚠ Recorded as provenance because **the same pathway can be built either way**:
    a five-gene insertion may reconstruct as five monocistronic TUs or as one
    operon, depending on the new-gene data, and nothing downstream announces
    which. The two are different constructs — an operon's transcription cannot be
    tuned per gene, so a design vector meant for one topology means something else
    against the other. A cache built on the wrong one produces a complete,
    plausible, wrong arm.

    Returns ``{}`` when the sim_data exposes no cistron→TU mapping, rather than
    guessing.
    """
    matrix = getattr(sim_data.process.transcription, "cistron_tu_mapping_matrix", None)
    if matrix is None:
        return {}
    cistrons = sim_data.process.transcription.cistron_data.struct_array
    new_cistron_ids = cistrons[cistrons["is_new_gene"]]["id"].tolist()
    cistron_row = {c: i for i, c in enumerate(cistrons["id"])}
    rna_ids = [str(r)[:-3] for r in sim_data.process.transcription.rna_data["id"]]

    dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    structure: dict[str, list[str]] = {}
    for cistron in new_cistron_ids:
        row = dense[cistron_row[cistron]]
        for col, value in enumerate(row):
            if value:
                structure.setdefault(rna_ids[col], []).append(str(cistron))
    return structure


def _check_operon_coverage(
    sim_data: Any, new_cistron_ids: list[str], new_rna_ids: list[str]
) -> None:
    """Every new cistron sits on a new TU, and every new TU carries one."""
    structure = new_gene_operon_structure(sim_data)
    if not structure:
        raise ValueError(
            f"this build has {len(new_rna_ids)} new-gene RNA(s) and "
            f"{len(new_cistron_ids)} new-gene cistron(s) — a polycistronic "
            "insertion — but sim_data exposes no cistron_tu_mapping_matrix, so "
            "which cistrons belong to which transcription unit cannot be "
            "verified. Refusing rather than pairing weights on an assumption."
        )
    covered = {c for cistrons in structure.values() for c in cistrons}
    orphan_cistrons = sorted(set(new_cistron_ids) - covered)
    if orphan_cistrons:
        raise ValueError(
            f"new-gene cistron(s) {orphan_cistrons} are not transcribed from any "
            "new-gene transcription unit; an expression weight applied to the new "
            "RNAs would not reach them."
        )
    empty_tus = sorted(set(new_rna_ids) - set(structure))
    if empty_tus:
        raise ValueError(
            f"new-gene RNA(s) {empty_tus} carry no new-gene cistron; they would "
            "receive an expression weight meant for the construct. The 'NG' id "
            "prefix may be matching an unrelated RNA."
        )


def set_new_gene_expression(
    sim_data: Any,
    expression: float,
    translation_efficiency: float,
    rel_exp_adj: Sequence[float] | None = None,
    rel_trl_eff_adj: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Turn new genes on, in place, at a chosen expression and efficiency.

    Args:
        sim_data: a ``SimulationDataEcoli`` built with a new-gene insertion.
        expression: multiplier applied to the baseline new-gene expression.
            The per-gene factor is ``expression * rel_exp_adj[i]``.
        translation_efficiency: efficiency assigned to each new-gene monomer.
            The per-monomer value is ``translation_efficiency * rel_trl_eff_adj[i]``
            — assigned, not multiplied into the existing value.
        rel_exp_adj: per-RNA relative expression weights; defaults to all 1.0.
        rel_trl_eff_adj: per-monomer relative efficiency weights; defaults to all
            1.0. **This is the vector a design screen sweeps.**

    Returns:
        What was applied — ids, indices and resolved per-target values — so a
        caller can record it as run provenance rather than re-deriving it.

    Modifies ``sim_data.process.transcription`` (``rna_synth_prob``,
    ``rna_expression``, ``exp_free``, ``exp_ppgpp``,
    ``attenuation_basal_prob_adjustments``),
    ``sim_data.process.transcription_regulation`` (``basal_prob``,
    ``delta_prob``) and
    ``sim_data.process.translation.translation_efficiencies_by_monomer``.
    """
    rna_ids, rna_indices, monomer_ids, monomer_indices = new_gene_indices(sim_data)

    rel_exp_adj = _resolve_weights(rel_exp_adj, len(rna_indices), "rel_exp_adj", "RNA")
    rel_trl_eff_adj = _resolve_weights(
        rel_trl_eff_adj, len(monomer_indices), "rel_trl_eff_adj", "monomer"
    )

    # Expression: ONE batched call, not one call per gene.
    # ``adjust_new_gene_final_expression`` sets each target from its baseline
    # (``arr[i] = baseline_i * factor_i``) and then renormalizes the WHOLE
    # transcriptome. Calling it once per gene therefore renormalizes N times, and
    # each gene is renormalized a different number of times, so equal weights come
    # out unequal and the result depends on call order. Measured on ONE real
    # 5-gene insertion with five IDENTICAL weights, which should give identical
    # results: spread 0.0014% at expression 1e4, 0.14% at 1e6, 13.6% at 1e8
    # (new-gene share of the transcriptome 1.7e-05, 1.7e-03, 1.5e-01). Those
    # magnitudes come from that one build and are indicative; what is exact is
    # that the batched call gives ZERO spread at every level. Negligible at low
    # induction, material once the construct takes a non-trivial share of the
    # transcriptome — which is where a design screen goes. The batched call renormalizes once and
    # is exact at every level; the fork's function already accepts lists.
    exp_applied = [expression * w for w in rel_exp_adj]
    sim_data.adjust_new_gene_final_expression(list(rna_indices), exp_applied)

    # Translation efficiency: assigned outright, not multiplied.
    te = sim_data.process.translation.translation_efficiencies_by_monomer
    te_applied = []
    for monomer_idx, weight in zip(monomer_indices, rel_trl_eff_adj):
        value = translation_efficiency * weight
        te[monomer_idx] = value
        te_applied.append(value)

    # ⚠ Provenance for the build TOPOLOGY, not just the values. The same
    # insertion can reconstruct as N monocistronic TUs or as one operon, and the
    # design vectors mean different things against each — one expression weight
    # for a whole operon is not one weight per gene. Recorded so an arm built on
    # the wrong topology is visible in the manifest instead of only in the result.
    structure = new_gene_operon_structure(sim_data)
    return {
        "rna_ids": rna_ids,
        "rna_indices": rna_indices,
        "expression_factors": exp_applied,
        "monomer_ids": [str(m) for m in monomer_ids],
        "monomer_indices": monomer_indices,
        "translation_efficiencies": te_applied,
        "operon_structure": structure,
        "is_polycistronic": bool(structure) and len(rna_ids) != len(monomer_ids),
    }


def _resolve_weights(
    weights: Iterable[float] | None, n: int, name: str, what: str
) -> list[float]:
    if weights is None:
        return [1.0] * n
    weights = [float(w) for w in weights]
    if len(weights) != n:
        raise ValueError(
            f"{name} has {len(weights)} entries but this build has {n} new-gene "
            f"{what}(s); a screen that silently mis-pairs its design vector "
            f"would rank arms it never applied."
        )
    return weights
