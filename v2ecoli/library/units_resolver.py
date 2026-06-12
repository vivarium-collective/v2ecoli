"""Resolve declared port-schema units into axis-label strings.

Units live on a resolved bigraph-schema type node as ``_units`` (e.g.
``core.access('quantity[float,fg]')._units == 'fg'``). Wrappers such as
``overwrite[...]`` carry the unit one level down on ``_value``; ``array[...]``
and parameterized ``float[...]`` propagate it to the top node. This module
extracts that unit per type string, builds a ``dotted-path -> unit`` index by
walking the baseline composite's declared port schemas, and formats axis
labels. No sim_data is loaded.
"""
from __future__ import annotations

from typing import Any, Optional


def _unit_from_node(node: Any, _depth: int = 0) -> Optional[str]:
    """Recover a unit string from a resolved type node, unwrapping wrappers.

    Reads ``node._units``; an empty string or ``None`` means "no unit here",
    so unwrap one level via ``_value`` (e.g. ``overwrite[...]``) up to a small
    depth. Returns the first non-empty unit found, else ``None``.
    """
    if node is None or _depth > 3:
        return None
    unit = getattr(node, "_units", None)
    if unit:  # non-empty string
        return unit
    return _unit_from_node(getattr(node, "_value", None), _depth + 1)


def unit_from_type(type_str: Any, core: Any) -> Optional[str]:
    """Return the unit string declared by a bigraph-schema type string, or None.

    ``core`` is a bigraph-schema Core (``v2ecoli.core.build_core()``). Any
    resolution failure yields ``None`` (units are best-effort decoration).
    """
    if not isinstance(type_str, str) or core is None:
        return None
    try:
        node = core.access(type_str)
    except Exception:
        return None
    return _unit_from_node(node)
