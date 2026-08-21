"""Scale native genes' translation efficiency on ``sim_data``, before caching.

:mod:`v2ecoli.perturbations.translation` already perturbs native genes, but it
works on a **cache bundle** and returns a ``config_overrides`` entry. This module
does the same arithmetic one layer earlier, on a live ``SimulationDataEcoli``, so
the perturbation is baked into the cache a composite is built from.

Why the earlier layer matters
-----------------------------
**A build-time argument does not survive division.** ``Division`` rebuilds each
daughter with ``baseline(cache_dir=…)`` and threads neither ``config_overrides``
nor ``knockouts`` (``v2ecoli/steps/division.py``), so a perturbation supplied as
an override applies to generation 1 and every daughter silently reverts to the
cached value. Anything resident in the cache is inherited by construction.

⇒ For a **single-generation** run the bundle route is fine and remains the right
tool. For a **multi-generation** study — which is what a design screen is — the
perturbation has to be in the cache, and that is this module.

How the two routes actually differ
-----------------------------------
⚠ **Not by the arithmetic.** ``LoadSimData.get_polypeptide_initiation_config``
stores ``normalize(translation_efficiencies_by_monomer)``, and it is tempting to
conclude that patching before normalisation and patching after it produce
different quantities. They do not. With raw array ``r``, ``S = sum(r)`` and
perturbed ``p``, the bundle route yields ``p/S`` and this route ``p/T`` where
``T = sum(p)`` — **exactly proportional**. The only consumer,
``processes/polypeptide_initiation.py:381``, computes
``normalize(cistron_counts * translation_efficiencies)``, and a positive global
scalar cancels under ``normalize``. Measured on a perturbed array, the resulting
``protein_init_prob`` differ by ``8.7e-19``.

The routes *are* genuinely different, for two other reasons:

1. **Initial state.** ``_write_sim_input_bundle`` calls
   ``generate_initial_state()`` (``v2ecoli/core.py:294``), which reads the
   **raw** ``translation_efficiencies_by_monomer`` off ``sim_data``
   (``library/initial_conditions.py:285``, ``:1676``). This route perturbs the
   array those reads see; the bundle route, patching only a process config,
   leaves the initial state at wild type.
2. **Division inheritance** — the reason above: a cache-resident perturbation is
   inherited by every daughter, a build-time argument is not.

This route matches the reference implementation, which mutates
``sim_data.process.translation.translation_efficiencies_by_monomer`` directly.
⚠ Comparing arms across the two routes is still not advisable — but because of
(1) and (2), **not** because the efficiency arrays disagree.

Multiplier convention, unchanged from the sibling: ``0`` is a knockout,
``0 < m < 1`` a knockdown, ``m > 1`` an overexpression.
"""
from __future__ import annotations

from typing import Any, Mapping

from v2ecoli.perturbations.translation import UnknownPerturbationTarget


def resolve_native_targets(sim_data: Any, targets: list[str]) -> dict[str, int]:
    """Resolve EcoCyc gene ids to monomer indices on a live ``sim_data``.

    The join is ``cistron_data["gene_id"] -> cistron_data["id"] ->
    monomer_data["cistron_id"] -> monomer_data`` index, which is the same path
    the reference implementation walks.

    Raises:
        UnknownPerturbationTarget: listing **every** unresolved target rather
            than the first. A screen declares many genes at once, and failing on
            one at a time turns a single fix into N build attempts. Targets that
            resolve to a non-coding cistron are reported separately, because
            "this gene makes no protein" is a different mistake from a typo.
    """
    cistrons = sim_data.process.transcription.cistron_data.struct_array
    monomers = sim_data.process.translation.monomer_data.struct_array

    gene_to_cistron = dict(zip(cistrons["gene_id"], cistrons["id"]))
    cistron_to_monomer = dict(zip(monomers["cistron_id"], monomers["id"]))
    monomer_index = {m: i for i, m in enumerate(monomers["id"])}

    resolved: dict[str, int] = {}
    unknown: list[str] = []
    non_coding: list[str] = []
    for target in targets:
        cistron = gene_to_cistron.get(target)
        if cistron is None:
            unknown.append(target)
            continue
        monomer = cistron_to_monomer.get(cistron)
        if monomer is None:
            non_coding.append(target)
            continue
        resolved[target] = monomer_index[monomer]

    if unknown or non_coding:
        parts = []
        if unknown:
            parts.append(f"not in this sim_data: {sorted(unknown)}")
        if non_coding:
            parts.append(
                f"resolve to a cistron with no monomer (non-coding): {sorted(non_coding)}")
        raise UnknownPerturbationTarget("; ".join(parts))
    return resolved


def set_native_translation_efficiency(
    sim_data: Any, perturbations: Mapping[str, float]
) -> dict[str, Any]:
    """Scale native genes' translation efficiency in place, before caching.

    Args:
        sim_data: a live ``SimulationDataEcoli``. **Mutated in place** — callers
            running a grid should isolate first, as
            :func:`v2ecoli.perturbations.build_new_gene_cache` does.
        perturbations: ``{EcoCyc gene id: multiplier}``. ``0`` knocks the gene
            out. Empty is a no-op returning empty provenance, so a caller can
            apply it unconditionally.

    Returns:
        Provenance suitable for recording verbatim: the resolved ids, indices,
        multipliers, and the **as-assigned** efficiencies.
        ⚠ As-assigned, not as-cached — the cache normalises afterwards, so an
        as-cached value would be relative to every other monomer and would move
        whenever any of them moved.

    Raises:
        UnknownPerturbationTarget: if any gene id does not resolve.
        ValueError: on a negative or non-finite multiplier — those would produce
            a nonsensical rate rather than an error further downstream.
    """
    if not perturbations:
        return {"gene_ids": [], "monomer_indices": [],
                "multipliers": [], "translation_efficiencies": []}

    bad = {g: m for g, m in perturbations.items()
           if not isinstance(m, (int, float)) or m < 0 or m != m or m in (float("inf"),)}
    if bad:
        raise ValueError(
            f"multipliers must be finite and non-negative: {bad}")

    resolved = resolve_native_targets(sim_data, list(perturbations))
    te = sim_data.process.translation.translation_efficiencies_by_monomer

    gene_ids, indices, multipliers, applied = [], [], [], []
    # Sorted so provenance ordering is stable across runs regardless of the
    # caller's dict ordering — a screen's manifests get diffed.
    for gene in sorted(resolved):
        idx = resolved[gene]
        multiplier = float(perturbations[gene])
        # Scale, not assign: the declaration is a multiplier on this gene's
        # fitted efficiency, so the baseline value has to be read first.
        value = float(te[idx]) * multiplier
        te[idx] = value
        gene_ids.append(str(gene))
        indices.append(int(idx))
        multipliers.append(multiplier)
        applied.append(value)

    return {
        "gene_ids": gene_ids,
        "monomer_indices": indices,
        "multipliers": multipliers,
        "translation_efficiencies": applied,
    }
