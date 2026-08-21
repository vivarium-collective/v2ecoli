"""Genotype perturbations for v2ecoli composites.

The straightforward, no-multi-ParCa tranche of the genotype-perturbation suite
(design: docs/superpowers/specs/2026-07-21-genotype-perturbations-design.md,
PR #341): translation-level knockouts. A per-monomer multiplier on translation
efficiency — 0 for a knockout — applied as a process-config patch on the cached
``ecoli-polypeptide-initiation`` config, so it works from the existing ParCa
cache with no re-fit.

Heterologous (new-gene) insertions need a second axis that module cannot reach —
expression — because ParCa inserts a new gene silent by design. See
:mod:`v2ecoli.perturbations.new_genes` for that arithmetic, and
:mod:`v2ecoli.perturbations.new_gene_cache` for the driver that applies it to
sim_data and saves the result as a cache a composite can be built from.

See :mod:`v2ecoli.perturbations.translation`.
"""

from v2ecoli.perturbations.new_gene_cache import build_new_gene_cache
from v2ecoli.perturbations.new_genes import (
    new_gene_indices,
    set_new_gene_expression,
)
from v2ecoli.perturbations.translation import (
    UnknownPerturbationTarget,
    resolve_targets,
    translation_efficiency_override,
)

__all__ = [
    "UnknownPerturbationTarget",
    "build_new_gene_cache",
    "new_gene_indices",
    "set_new_gene_expression",
    "resolve_targets",
    "translation_efficiency_override",
]
