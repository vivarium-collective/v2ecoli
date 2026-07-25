"""v2ecoli — whole-cell E. coli model on process-bigraph."""

from __future__ import annotations

from typing import Any

from process_bigraph import Composite
from viva_superpowers.composite_generator import _REGISTRY, build_generator

from v2ecoli.core import build_core
from v2ecoli import composites  # noqa: F401 — forces @composite_generator decorators to fire
from v2ecoli import visualizations  # noqa: F401 — forces Visualization Steps into link_registry


def build_composite(
    name: str,
    *,
    core: Any = None,
    **kwargs: Any,
) -> Composite:
    """Build a Composite by architecture name.

    Parameters
    ----------
    name:
        One of ``"baseline"``, ``"colony"``, ``"millard_pdmp_baseline"``.
    core:
        Optional bigraph-schema core. If omitted, a fresh one is built via
        ``v2ecoli.core.build_core()``.
    **kwargs:
        Passed through to the generator's declared parameters (currently
        ``seed`` and ``cache_dir``).

    Raises
    ------
    ValueError
        If ``name`` does not match any registered architecture, or matches more
        than one, or if ``kwargs`` contains an unknown parameter name.
    """
    if core is None:
        core = build_core()
    matches = [e for e in _REGISTRY.values() if e.name == name]
    if not matches:
        available = sorted({e.name for e in _REGISTRY.values()})
        raise ValueError(
            f"unknown composite architecture {name!r}; available: {available}"
        )
    if len(matches) > 1:
        # The same architecture may be registered under more than one id (e.g. a
        # clean-id alias added so the dashboard's ".composites.<slug>" resolver
        # finds it — see v2ecoli/composites/__init__.py). Those entries share the
        # same generator function, so dedupe by function identity and pick the
        # canonical (shortest) id. Only genuinely-distinct generators are
        # ambiguous.
        if len({e.func for e in matches}) == 1:
            matches = [min(matches, key=lambda e: len(e.id))]
        else:
            raise ValueError(
                f"ambiguous architecture name {name!r}; multiple generators registered: "
                f"{[e.id for e in matches]}"
            )
    doc = build_generator(matches[0], overrides=kwargs, core=core)
    return Composite(doc, core=core)


__all__ = ["build_composite", "build_core"]
