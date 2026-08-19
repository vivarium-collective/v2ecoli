"""New-gene (heterologous) expression + translation perturbations.

The heterologous counterpart to :mod:`v2ecoli.perturbations.translation`. That
module covers **native** genes and is translation-only: a per-monomer multiplier
on translation efficiency, applied to the cached process config through
``baseline``'s ``config_overrides`` seam. New genes need a second axis it cannot
reach — **expression** — because a new gene is inserted *silent*:

    sim_data.adjust_new_gene_final_expression: "the baseline new gene expression
    values need to be set to small non-zero values ... as new genes are knocked
    out by default."

Measured on a real heterologous insertion: its new-gene cistrons carry **exactly
0** expression in ``cistron_expression['basal']`` (a native reference cistron
carries 5.9e-5), so the enzymes are never synthesised, the pathway carries no
flux, and a design screen over translation efficiency alone would rank every arm
at zero. Turning the construct on is therefore not a convenience knob — it is the
first half of the design axis.

Surface, and why it differs from the sibling module
---------------------------------------------------
``translation.py`` patches a cached process-config array and never touches
sim_data. This module **mutates sim_data directly**, because the expression
half spans seven fields that have no cached-config equivalent (synthesis
probability, expression, free/ppGpp expression, attenuation adjustments, and
basal/delta promoter probabilities). That is also the surface
``sim_data.internal_shift_dict`` expects: the loader
(``v2ecoli/library/sim_data.py``) stores ``{generation: (func, params)}`` and
applies ``func(sim_data, *params)`` once the lineage reaches that generation, so
a callable of this shape is what schedules an induction.

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

__all__ = ["new_gene_indices", "set_new_gene_expression"]


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
    return new_rna_ids, new_rna_indices, new_monomer_ids, new_monomer_indices


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

    # Expression: one call per gene, matching the reference. adjust_new_gene_
    # final_expression scales FROM the baseline, so this is idempotent in intent
    # (the same call twice sets the same level, it does not compound).
    exp_applied = []
    for rna_idx, weight in zip(rna_indices, rel_exp_adj):
        factor = expression * weight
        sim_data.adjust_new_gene_final_expression([rna_idx], [factor])
        exp_applied.append(factor)

    # Translation efficiency: assigned outright, not multiplied.
    te = sim_data.process.translation.translation_efficiencies_by_monomer
    te_applied = []
    for monomer_idx, weight in zip(monomer_indices, rel_trl_eff_adj):
        value = translation_efficiency * weight
        te[monomer_idx] = value
        te_applied.append(value)

    return {
        "rna_ids": rna_ids,
        "rna_indices": rna_indices,
        "expression_factors": exp_applied,
        "monomer_ids": [str(m) for m in monomer_ids],
        "monomer_indices": monomer_indices,
        "translation_efficiencies": te_applied,
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
