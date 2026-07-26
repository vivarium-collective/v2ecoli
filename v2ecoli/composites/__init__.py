"""Composite generators for v2ecoli architectures.

Importing this package forces the per-architecture modules to load, which
fires their ``@composite_generator`` decorators and registers the generators
in ``viva_superpowers.composite_generator._REGISTRY``.
"""

import dataclasses as _dataclasses

from viva_superpowers.composite_generator import _REGISTRY as _COMPOSITE_REGISTRY

from v2ecoli.composites import (  # noqa: F401
    ecoli_baseline,
    ecoli_millard,
    ecoli_structural,
    ecoli_population,
    ecoli_time_varying_env,
    ecoli_colony,
    millard_fba_bridge_harness,
    parca,
    reactor_bird_coupled,
    reactor_bird_coupled_millard,
)


# --- Alias registration -----------------------------------------------------
# A generator ids as ``{__module__}.{name}``; because module slug == composite
# name (e.g. ``ecoli_baseline`` in ``ecoli_baseline.py``), that id DOUBLES to
# ``v2ecoli.composites.ecoli_baseline.ecoli_baseline``. The dashboard resolver
# matches on the trailing segment after ``.composites.``, so studies that
# reference the short ``ecoli_baseline`` need a clean-id alias whose trailing
# segment is exactly that. All alias keys point at the identical
# ``GeneratorEntry.func``, so ``build_composite`` dedupes them by function
# identity rather than raising "ambiguous architecture name".
def _register_alias(alias_id: str, source_id: str) -> None:
    orig = _COMPOSITE_REGISTRY.get(source_id)
    if orig is not None and alias_id not in _COMPOSITE_REGISTRY:
        _COMPOSITE_REGISTRY[alias_id] = _dataclasses.replace(orig, id=alias_id)


def _register_clean_alias(name: str, module: str | None = None) -> None:
    module = module or name
    _register_alias(f"v2ecoli.composites.{name}",
                    f"v2ecoli.composites.{module}.{name}")


# The ecoli_* whole-cell family was renamed from the legacy ``baseline*`` /
# ``colony`` scheme (2026-07). Register the clean new id AND keep the old ids
# (both doubled and short) resolving to the same generator, so published
# composite-state snapshots, sms-api run registrations, and any study.yaml
# still referencing the old name keep working.
_RENAMED = {
    "baseline":                  "ecoli_baseline",
    "baseline_millard":          "ecoli_millard",
    "baseline_parsimony":        "ecoli_structural",
    "baseline_population":       "ecoli_population",
    "baseline_time_varying_env": "ecoli_time_varying_env",
    "colony":                    "ecoli_colony",
}
for _old, _new in _RENAMED.items():
    _new_doubled = f"v2ecoli.composites.{_new}.{_new}"
    _register_clean_alias(_new)                                         # clean new id
    _register_alias(f"v2ecoli.composites.{_old}.{_old}", _new_doubled)  # legacy doubled
    _register_alias(f"v2ecoli.composites.{_old}", _new_doubled)         # legacy short

_register_clean_alias("parca")


__all__ = [
    "ecoli_baseline",
    "ecoli_millard",
    "ecoli_structural",
    "ecoli_population",
    "ecoli_time_varying_env",
    "ecoli_colony",
    "millard_fba_bridge_harness",
    "parca",
    "reactor_bird_coupled",
    "reactor_bird_coupled_millard",
]
