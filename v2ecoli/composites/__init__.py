"""Composite generators for v2ecoli architectures.

Importing this package forces the per-architecture modules to load, which
fires their ``@composite_generator`` decorators and registers the generators
in ``viva_superpowers.composite_generator._REGISTRY``.
"""

import dataclasses as _dataclasses

from viva_superpowers.composite_generator import _REGISTRY as _COMPOSITE_REGISTRY

from v2ecoli.composites import (  # noqa: F401
    baseline,
    baseline_millard,
    baseline_population,
    baseline_time_varying_env,
    colony,
    millard_fba_bridge_harness,
    parca,
    reactor_bird_coupled,
    reactor_bird_coupled_millard,
)


# --- Clean-id aliases so the dashboard resolves the studies' short refs ------
# ``@composite_generator(name="baseline")`` in module ``v2ecoli.composites.
# baseline`` ids the generator as ``{__module__}.{name}`` == the DOUBLED
# ``v2ecoli.composites.baseline.baseline`` (same for ``parca``). That id's
# after-``.composites.`` segment is ``baseline.baseline``, so the dashboard's
# trailing-segment resolver never matches the studies' short ``baseline`` ref
# and the study UI shows "composite not found in registry: baseline".
#
# Register the SAME generator entry under the clean id ``v2ecoli.composites.
# <slug>`` as well (whose trailing segment is exactly ``<slug>``). We keep the
# doubled id too — studies/seed templates reference it by full path. Both keys
# point at the identical ``GeneratorEntry.func``, so ``build_composite`` dedupes
# them by function identity rather than raising "ambiguous architecture name".
def _register_clean_alias(name, module=None):
    # The generator id is ``{__module__}.{name}``. It usually doubles (module
    # slug == composite name, e.g. baseline in baseline.py). ``module`` overrides
    # the slug for the cases where it doesn't.
    module = module or name
    doubled_id = f"v2ecoli.composites.{module}.{name}"
    clean_id = f"v2ecoli.composites.{name}"
    orig = _COMPOSITE_REGISTRY.get(doubled_id)
    if orig is not None and clean_id not in _COMPOSITE_REGISTRY:
        _COMPOSITE_REGISTRY[clean_id] = _dataclasses.replace(orig, id=clean_id)


_register_clean_alias("baseline")
_register_clean_alias("parca")


__all__ = [
    "baseline",
    "baseline_millard",
    "baseline_population",
    "baseline_time_varying_env",
    "colony",
    "millard_fba_bridge_harness",
    "parca",
    "reactor_bird_coupled",
    "reactor_bird_coupled_millard",
    "parsimony_ecoli",
]
