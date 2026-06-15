"""Composite generators for v2ecoli architectures.

Importing this package forces the per-architecture modules to load, which
fires their ``@composite_generator`` decorators and registers the generators
in ``pbg_superpowers.composite_generator._REGISTRY``.
"""

from pbg_superpowers.composite_generator import composite_generator as _composite_generator

from v2ecoli.composites import (  # noqa: F401
    baseline,
    baseline_millard,
    baseline_population,
    baseline_time_varying_env,
    colony,
    millard_fba_bridge_harness,
    millard_pdmp_baseline,
    parca,
    reactor_bird_coupled,
    reactor_bird_coupled_millard,
)
from v2ecoli.structural import composite as parsimony_ecoli  # noqa: F401 — registers "parsimony-ecoli"


# --- Package-level alias for the central baseline composite ------------------
# ``baseline.py``'s ``@composite_generator(name="baseline")`` registers under
# the id ``{__module__}.{name}`` == ``v2ecoli.composites.baseline.baseline``.
# Because the module is itself named ``baseline``, that id's ``.composites.``
# trailing segment is the DOUBLED ``baseline.baseline`` — so the studies' short
# composite ref ``baseline`` fails the dashboard's trailing-segment resolution
# (``ref == id.rsplit(".composites.")[-1]``) and the study UI shows
# "composite not found in registry: baseline".
#
# Re-register the same builder from THIS package module so it also lives under
# the clean id ``v2ecoli.composites.baseline`` (module ``v2ecoli.composites`` +
# name ``baseline``), whose trailing segment is exactly ``baseline``. The alias
# reuses the original generator's declared metadata (parameters, default steps,
# visualizations, emitters) and delegates to the real ``baseline()`` builder, so
# the Composite Explorer can both RESOLVE and BUILD it.
_baseline_entry = baseline.baseline._composite_generator_entry


@_composite_generator(
    name="baseline",
    description=_baseline_entry.description,
    parameters=_baseline_entry.parameters,
    visualizations=_baseline_entry.visualizations,
    emitters=_baseline_entry.emitters,
    default_n_steps=_baseline_entry.default_n_steps,
)
def _baseline_alias(core=None, **kwargs):
    """Package-level alias of :func:`v2ecoli.composites.baseline.baseline`.

    Registered as ``v2ecoli.composites.baseline`` so the studies' short
    ``baseline`` composite ref resolves. Delegates to the real builder.
    """
    return baseline.baseline(core=core, **kwargs)


# --- Package-level alias for the parca composite -----------------------------
# Same doubling as ``baseline``: ``parca.py``'s ``@composite_generator(name=
# "parca")`` registers under ``v2ecoli.composites.parca.parca`` (DOUBLED tail),
# so showcase-1-parca's short ``parca`` ref fails trailing-segment resolution.
# Re-register from THIS package module under the clean id
# ``v2ecoli.composites.parca`` so the ref resolves and the Explorer can build it.
_parca_entry = parca.parca._composite_generator_entry


@_composite_generator(
    name="parca",
    description=_parca_entry.description,
    parameters=_parca_entry.parameters,
    visualizations=_parca_entry.visualizations,
    emitters=_parca_entry.emitters,
    default_n_steps=_parca_entry.default_n_steps,
)
def _parca_alias(core=None, **kwargs):
    """Package-level alias of :func:`v2ecoli.composites.parca.parca`.

    Registered as ``v2ecoli.composites.parca`` so showcase-1-parca's short
    ``parca`` composite ref resolves. Delegates to the real builder.
    """
    return parca.parca(core=core, **kwargs)


__all__ = [
    "baseline",
    "baseline_millard",
    "baseline_population",
    "baseline_time_varying_env",
    "colony",
    "millard_fba_bridge_harness",
    "millard_pdmp_baseline",
    "parca",
    "reactor_bird_coupled",
    "reactor_bird_coupled_millard",
    "parsimony_ecoli",
]
