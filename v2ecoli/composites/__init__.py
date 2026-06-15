"""Composite generators for v2ecoli architectures.

Importing this package forces the per-architecture modules to load, which
fires their ``@composite_generator`` decorators and registers the generators
in ``pbg_superpowers.composite_generator._REGISTRY``.
"""

from pbg_superpowers.composite_generator import composite_generator as _composite_generator

from v2ecoli.composites import baseline, colony, parca  # noqa: F401
from v2ecoli.composites import millard_pdmp_baseline  # noqa: F401
from v2ecoli.composites import millard_fba_bridge_harness  # noqa: F401
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


__all__ = [
    "baseline", "colony", "parca", "millard_pdmp_baseline",
    "millard_fba_bridge_harness", "parsimony_ecoli",
]
