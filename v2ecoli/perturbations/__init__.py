"""Genotype perturbations for v2ecoli composites.

The straightforward, no-multi-ParCa tranche of the genotype-perturbation suite
(design: docs/superpowers/specs/2026-07-21-genotype-perturbations-design.md,
PR #341): translation-level knockouts. A per-monomer multiplier on translation
efficiency — 0 for a knockout — applied as a process-config patch on the cached
``ecoli-polypeptide-initiation`` config, so it works from the existing ParCa
cache with no re-fit.

See :mod:`v2ecoli.perturbations.translation`.
"""

from v2ecoli.perturbations.translation import (
    UnknownPerturbationTarget,
    resolve_targets,
    translation_efficiency_override,
)

__all__ = [
    "UnknownPerturbationTarget",
    "resolve_targets",
    "translation_efficiency_override",
]
