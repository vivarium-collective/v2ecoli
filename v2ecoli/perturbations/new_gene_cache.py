"""Build a named ParCa cache with the new genes turned on.

The caller :mod:`v2ecoli.perturbations.new_genes` was missing. That module is
pure arithmetic on a live ``SimulationDataEcoli`` — it says *what* to write into
the expression and translation-efficiency arrays — but v2ecoli composites do not
build from a live sim_data. ``baseline`` builds from a **cache bundle**, so a
perturbation that only mutates sim_data reaches no simulation unless something
saves the mutated sim_data to a cache. This module is that something.

    load sim_data  ->  deep-copy  ->  set_new_gene_expression  ->  save_sim_input
                                                                   -> named cache dir

That is the shape v2ecoli already uses for a *conditional* refit
(``scripts/build_condition_cache.py``: hydrate -> patch -> ``save_sim_input`` ->
manifest); this is the same shape with the new-gene perturbation as the patch.
The one consumer is ``scripts/build_new_gene_cache.py``. Deliberately no general
interface for hypothetical callers — the second consumer can force whatever
generalisation turns out to be real.

Why the modification is applied **before** the cache is built
-------------------------------------------------------------
vEcoli defers a new-gene induction to a chosen generation through
``sim_data.internal_shift_dict``. ⚠ In v2ecoli that path is not wired (see
:mod:`v2ecoli.perturbations.new_genes`), and ``baseline`` bakes sim_data into a
precomputed cache bundle, so a callable left in that dict is stored and never
fired. Applying at generation 0, before the bundle is written, is the v2ecoli
equivalent — and it is also what makes the sibling module's composition trap go
away: ``translation.translation_efficiency_override`` returns a **full
replacement** efficiency array, so if a native override were computed from a
cache built *before* these values were assigned, the whole-array replacement
would silently discard them. Values that are already in the array the override
is derived from survive it.

⚠ Isolation
-----------
``set_new_gene_expression`` mutates sim_data in place, and the use case is a
design **grid** — one loaded sim_data, many expression levels. Mutating the
caller's object would let grid point *k* inherit grid point *k-1*'s expression
silently, which is expensive to find and cheap to prevent, so this function
deep-copies first via ``pickle.loads(pickle.dumps(...))``. That idiom is not
invented here: it mirrors ``SimDataInjector.materialize``
(``pbg_v2ecoli/uq_sim_data_injection.py:243``), which isolates per UQ sample the
same way. The two are kept separate on purpose — that class is scalar-sampling
shaped (``materialize(sample: dict[str, float])``, ``(bundle, overrides,
cleanup)`` over a **temp** dir) while this writes a durable, named cache for a
design variant — but the duplication is deliberate and visible rather than
hidden. A third copy is a signal to factor it out.

⚠ ``translation_efficiency`` is a WEIGHT, not an achieved rate
--------------------------------------------------------------
The value assigned here is not the value the simulation consumes.
``LoadSimData.get_polypeptide_initiation_config`` stores
``normalize(translation_efficiencies_by_monomer)``
(``v2ecoli/library/sim_data.py:1051``), i.e. the cached array is **L1-normalised
across every monomer**. Only *ratios* survive the cache, and the normaliser
itself moves when any entry moves. So **10x the translation efficiency is not
10x the protein**, and two caches built at different absolute efficiencies but
identical ratios are the same cache.

Consequently the provenance returned by ``build_new_gene_cache`` records the
**as-assigned** values only. An "as-cached" number would be cache-relative — it
shifts with the whole array — so anyone reading it as the achieved value would
be wrong. One number, one meaning; measuring what was achieved is a job for a
simulation read-out, not for this record.
"""

from __future__ import annotations

import pickle
from typing import Any, Sequence

from v2ecoli.perturbations.new_genes import set_new_gene_expression

__all__ = ["build_new_gene_cache"]


def build_new_gene_cache(
    sim_data: Any,
    cache_dir: str,
    *,
    expression: float,
    translation_efficiency: float,
    rel_exp_adj: Sequence[float] | None = None,
    rel_trl_eff_adj: Sequence[float] | None = None,
    seed: int = 0,
    condition: str | None = None,
    fixed_media: str | None = None,
) -> dict[str, Any]:
    """Apply a new-gene induction to ``sim_data`` and save the result as a cache.

    Args:
        sim_data: a live ``SimulationDataEcoli`` built with a new-gene
            insertion. **Not modified** — a deep copy is perturbed instead, so
            one loaded sim_data can drive a whole design grid.
        cache_dir: directory the cache bundle is written to. A composite is then
            built with ``baseline(cache_dir=cache_dir)``.
        expression: multiplier on the baseline new-gene expression; the per-gene
            factor is ``expression * rel_exp_adj[i]``.
        translation_efficiency: efficiency assigned to each new-gene monomer;
            the per-monomer value is ``translation_efficiency *
            rel_trl_eff_adj[i]``. ⚠ A weight, not an achieved rate — see the
            module docstring on L1 normalisation.
        rel_exp_adj: per-RNA relative expression weights; defaults to all 1.0.
        rel_trl_eff_adj: per-monomer relative efficiency weights; defaults to
            all 1.0.
        seed: forwarded to ``save_sim_input`` (selects the generated initial
            state).
        condition: ParCa nutrient condition (e.g. ``"acetate"``); ``None`` keeps
            the default basal fit.
        fixed_media: media id pinned for the run (e.g. ``"minimal_acetate"``).

    Returns:
        ``{"cache_dir", "applied", "seed", "condition", "fixed_media"}``.
        ``applied`` is what ``set_new_gene_expression`` returned — the ids,
        indices and resolved **as-assigned** per-target values — suitable for
        recording verbatim as run provenance.

    Raises:
        ValueError: propagated from ``new_gene_indices`` when the sim_data
            carries no new genes, or from the weight-vector length checks.
    """
    # Imported here, not at module scope: ``v2ecoli.core`` pulls in the whole
    # composite/cache stack, and this package is otherwise dependency-free
    # arithmetic that a test can import in milliseconds.
    from v2ecoli.core import save_sim_input

    # Isolate before mutating — see the module docstring. Mirrors
    # ``SimDataInjector.materialize`` (pbg_v2ecoli/uq_sim_data_injection.py:243).
    perturbed = pickle.loads(pickle.dumps(sim_data))

    applied = set_new_gene_expression(
        perturbed,
        expression,
        translation_efficiency,
        rel_exp_adj,
        rel_trl_eff_adj,
    )

    # ORDER IS LOAD-BEARING: the values must already be in the sim_data arrays
    # when the bundle is extracted, or the cache is a pre-perturbation cache
    # that looks like a perturbed one.
    save_sim_input(
        perturbed,
        cache_dir,
        seed=seed,
        condition=condition,
        fixed_media=fixed_media,
    )

    return {
        "cache_dir": cache_dir,
        "applied": applied,
        "seed": seed,
        "condition": condition,
        "fixed_media": fixed_media,
    }
