"""Build the cache for one stage of a design-variant plan.

:func:`v2ecoli.perturbations.plan_design_variant` says *which* caches a grid
point needs; this builds one of them. It composes the two perturbation halves
onto a single isolated copy of ``sim_data`` and writes the bundle once:

    deep-copy  ->  native multipliers  ->  new-gene induction  ->  save_sim_input

**Why both halves go into the cache rather than being passed at build time.**
``Division`` rebuilds each daughter with ``baseline(cache_dir=…)`` and threads
neither ``config_overrides`` nor ``knockouts``, so a perturbation supplied as a
build-time argument applies to generation 1 and every daughter silently reverts.
A design screen is multi-generation by construction, so both halves have to be
resident in the cache. See :mod:`v2ecoli.perturbations.native_genes` for the
native half's reasoning and the ⚠ note that the cache route and the
``config_overrides`` route are not numerically equivalent.

Relationship to :func:`~v2ecoli.perturbations.build_new_gene_cache`
-------------------------------------------------------------------
That function is the narrower path — one perturbation, its own CLI
(``scripts/build_new_gene_cache.py``), and it predates this one. This is the
screen path, where a stage may carry a native perturbation, a heterologous
induction, both, or neither. They deliberately duplicate ~10 lines of
copy-and-save rather than one wrapping the other, because the merged narrower
path has its own tests and consumer and churning it for tidiness would trade
real risk for neatness. ⚠ **If a third cache builder appears, factor them** —
that is the signal, not this.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

#: Filename the design point is written under, inside the cache directory.
#: Read it to learn what a cache encodes; ``metadata.json`` will not tell you.
DESIGN_POINT_FILE = "design_point.json"

from v2ecoli.perturbations.design_variant import CacheSpec
from v2ecoli.perturbations.native_genes import set_native_translation_efficiency
from v2ecoli.perturbations.new_genes import set_new_gene_expression


def build_variant_cache(
    sim_data: Any,
    cache_dir: str,
    spec: CacheSpec,
    *,
    seed: int = 0,
    fixed_media: str | None = None,
) -> dict[str, Any]:
    """Apply one stage's perturbations to an isolated copy and write its cache.

    Args:
        sim_data: a live ``SimulationDataEcoli``. **Not mutated** — deep-copied
            first, so a grid loop can reuse one loaded object across every stage
            of every point without grid point *k* inheriting *k-1*.
        cache_dir: where the bundle is written; a composite is then built with
            ``baseline(cache_dir=cache_dir)``.
        spec: the stage's :class:`~v2ecoli.perturbations.design_variant.CacheSpec`.
            ``spec.condition`` selects the ParCa nutrient condition;
            ``spec.native_perturbations`` and ``spec.new_gene`` may each be empty
            or absent, and an entirely unperturbed spec is valid — that is the
            silent stage of an induction plan.
        seed: forwarded to ``save_sim_input`` (selects the generated initial
            state).
        fixed_media: media id pinned for the run, if any.

    Returns:
        ``{"cache_dir", "condition", "seed", "fixed_media", "label",
        "native", "new_gene"}``. The last two are the **as-assigned** provenance
        dicts from the two perturbation functions, empty when that half was not
        applied — suitable for recording verbatim in a run manifest.

    Raises:
        ValueError / UnknownPerturbationTarget: propagated from either half. A
            stage that cannot be built fails here rather than producing a cache
            that silently differs from its declaration.
    """
    # Imported here, not at module scope: ``v2ecoli.core`` pulls in the whole
    # composite/cache stack, and this package is otherwise dependency-free
    # arithmetic that a test can import in milliseconds.
    from v2ecoli.core import save_sim_input

    perturbed = copy.deepcopy(sim_data)

    # ORDER: native first, then new-gene. The two touch disjoint monomer indices
    # (native genes vs new genes) and only the new-gene half renormalizes the
    # transcriptome, which the native half does not read — so the result is
    # order-independent, and there is a test asserting exactly that. The order
    # is fixed anyway so that provenance is reproducible rather than incidental.
    native = set_native_translation_efficiency(
        perturbed, dict(spec.native_perturbations or {}))

    new_gene: dict[str, Any] = {}
    if spec.new_gene is not None:
        ng = spec.new_gene
        new_gene = set_new_gene_expression(
            perturbed,
            ng.expression,
            ng.translation_efficiency,
            list(ng.rel_exp_adj) if ng.rel_exp_adj is not None else None,
            list(ng.rel_trl_eff_adj) if ng.rel_trl_eff_adj is not None else None,
        )

    # ORDER IS LOAD-BEARING: both halves must already be in the sim_data arrays
    # when the bundle is extracted, or the cache is a pre-perturbation cache
    # wearing a perturbed cache's name.
    save_sim_input(perturbed, cache_dir, seed=seed,
                   condition=spec.condition, fixed_media=fixed_media)

    provenance = {
        "cache_dir": cache_dir,
        "label": spec.label,
        "condition": spec.condition,
        "seed": seed,
        "fixed_media": fixed_media,
        "native": native,
        "new_gene": new_gene,
    }

    # Persist it INTO the cache, not just return it. A caller that keeps the
    # returned dict has provenance; a consumer handed only ``cache_dir`` — which
    # is every consumer, because ``baseline(cache_dir=…)`` is the whole
    # interface — previously had none. The cache's own ``metadata.json`` records
    # ``unique_names`` and ``media_id`` and nothing about the perturbation, so a
    # perturbed cache and an unperturbed one were indistinguishable on disk.
    #
    # ⚠ That is the failure this file's own docstring warns about, one layer
    # out: not "a pre-perturbation cache wearing a perturbed cache's name", but
    # a perturbed cache that cannot say which perturbation it carries. A design
    # screen ranks over exactly that distinction.
    try:
        (Path(cache_dir) / DESIGN_POINT_FILE).write_text(
            json.dumps(provenance, indent=1, default=str))
    except OSError as exc:  # pragma: no cover - disk-full / read-only cache dir
        # Best-effort: a cache that built is more valuable than one refused for
        # want of a sidecar, and the return value still carries the record.
        print(f"[variant_cache] could not write {DESIGN_POINT_FILE}: {exc}")

    return provenance
