"""v2ecoli — whole-cell E. coli model on process-bigraph."""

from __future__ import annotations

from typing import Any

from process_bigraph import Composite
from pbg_superpowers.composite_generator import _REGISTRY, build_generator

from v2ecoli.core import build_core
from v2ecoli import composites  # noqa: F401 — forces @composite_generator decorators to fire


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
        One of ``"baseline"``, ``"departitioned"``, ``"reconciled"``.
    core:
        Optional bigraph-schema core. If omitted, a fresh one is built via
        ``v2ecoli.core.build_core()``.
    **kwargs:
        Passed through to the generator's declared parameters (currently
        ``seed`` and ``cache_dir`` for all three architectures).

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
        raise ValueError(
            f"ambiguous architecture name {name!r}; multiple generators registered: "
            f"{[e.id for e in matches]}"
        )
    doc = build_generator(matches[0], overrides=kwargs, core=core)
    return Composite(doc, core=core)


__all__ = ["build_composite", "build_core"]
