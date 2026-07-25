"""KO_baseline — a baseline whole-cell E. coli model with genes knocked out.

The straightforward genotype-perturbation composite: translation-level knockouts
(design PR #341, Tranche A). Each named gene's translation efficiency is zeroed
on the cached ``ecoli-polypeptide-initiation`` config — no ribosome initiates on
its mRNA, so no protein is made — then the standard :func:`baseline` document is
built on top. Works from the existing ParCa cache; no re-fit.

The knockout is the only thing this adds over ``baseline``; every other knob is
``baseline``'s, threaded straight through. See
:mod:`v2ecoli.perturbations.translation` for the mechanism and its caveats
(translation-only: the mRNA is still transcribed; operon-mates are untouched;
non-coding genes are rejected).
"""
from __future__ import annotations

from typing import Any

from viva_superpowers.composite_generator import composite_generator

from v2ecoli.composites._helpers import DEFAULT_SINGLE_CELL_VISUALIZATIONS
from v2ecoli.composites.baseline import baseline
from v2ecoli.core import build_core, load_cache_bundle
from v2ecoli.perturbations import translation_efficiency_override


def _merge_ko_override(cache_dir: str, knockouts: list[str] | None,
                       config_overrides: dict | None,
                       bundle: dict | None = None) -> tuple[dict, dict]:
    """Return ``(bundle, merged_config_overrides)`` with the KO patch folded in.

    Loads the cache once (memoized), resolves the knockouts to a
    ``translation_efficiencies`` override, and merges it under any caller
    ``config_overrides`` (the caller's win on an exact key clash — they are
    being explicit). The bundle is returned so ``baseline`` reuses it rather
    than re-reading the cache.
    """
    if bundle is None:
        bundle = load_cache_bundle(cache_dir)
    ko = translation_efficiency_override(bundle, list(knockouts or []))
    merged = {**ko, **(config_overrides or {})}
    return bundle, merged


@composite_generator(
    name="KO_baseline",
    description=(
        "Whole-cell E. coli baseline with translation-level gene knockouts. "
        "Each gene in `knockouts` (an EcoCyc gene id like EG10526, or a monomer "
        "id like LACY-MONOMER[c]) has its translation efficiency zeroed on the "
        "cached polypeptide-initiation config, so no protein is made — a "
        "functional knockout with no ParCa re-fit. All other parameters are the "
        "baseline composite's."
    ),
    parameters={
        "knockouts": {
            "type": "list",
            "default": [],
            "description": "Genes to knock out at the translation level — "
                           "EcoCyc gene ids (EG10526) or monomer ids "
                           "(LACY-MONOMER[c]). Empty = plain baseline.",
        },
        "seed": {
            "type": "integer",
            "default": 0,
            "description": "RNG seed for stochastic initialization.",
        },
        "cache_dir": {
            "type": "string",
            "default": "out/cache",
            "description": "Path to the ParCa cache directory.",
        },
        "emitter": {
            "type": "string",
            "choices": ["parquet", "sqlite", "xarray", "null"],
            "default": "parquet",
            "description": "Observation sink for the internal emitter step.",
        },
        "config_overrides": {
            "type": "map",
            "default": {},
            "description": "Declarative '<process>.<key>': value overrides "
                           "(applied on top of the knockout patch).",
        },
    },
    default_n_steps=2700,
    visualizations=DEFAULT_SINGLE_CELL_VISUALIZATIONS,
)
def KO_baseline(
    core: Any = None,
    *,
    knockouts: list[str] | None = None,
    seed: int = 0,
    cache_dir: str = "out/cache",
    emitter: str = "parquet",
    config_overrides: dict | None = None,
) -> dict:
    """Build the KO_baseline document (see module docstring).

    Args:
        core: bigraph-schema core; a fresh one (``build_core()``) if None.
        knockouts: genes to knock out (EcoCyc gene ids or monomer ids).
        seed: RNG seed.
        cache_dir: ParCa cache directory.
        emitter: internal emitter sink.
        config_overrides: extra process-config overrides, applied over the KO.

    Returns:
        The ``baseline`` document with the knockouts applied.
    """
    if core is None:
        core = build_core()
    bundle, merged = _merge_ko_override(cache_dir, knockouts, config_overrides)
    return baseline(
        core=core, seed=seed, cache_dir=cache_dir, emitter=emitter,
        config_overrides=merged, bundle=bundle,
    )
