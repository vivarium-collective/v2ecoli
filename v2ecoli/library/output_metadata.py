"""
Output metadata walker for v2ecoli composites.

Mirrors vEcoli's ``ecoli.experiments.ecoli_master_sim.output_metadata()`` /
``extract_metadata()`` convention: processes annotate their ``outputs()`` port
schema with ``_properties: {'metadata': <names_list>}``, this module harvests
those annotations from a live composite state and returns a store-relative dict.

The returned dict feeds into:
  - XArrayEmitter ``output_metadata`` (coord arrays for vector leaves);
  - ParquetEmitter ``config["metadata"]["output_metadata"]`` (written as
    ``output_metadata__<path>`` columns in the configuration parquet).

This mirrors the path described in vEcoli PR #414 / ``ecoli_master_sim.py``
``:852`` / ``:1011``, grounded to v2ecoli's process-bigraph composite layout.
"""
from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# extract_metadata — mirrors vEcoli's extract_metadata() exactly
# ---------------------------------------------------------------------------


def extract_metadata(schema: dict, _properties: bool = False) -> Any:
    """Extract ``_properties.metadata`` leaves from a ports schema.

    Mirrors ``ecoli.experiments.ecoli_master_sim.extract_metadata``.
    Recursively walks ``schema``; returns a dict with the same structure as
    the schema but with only ``_properties.metadata`` values as leaves.
    Returns ``None`` if no metadata is found.

    Args:
        schema: A port schema dict (e.g. the return value of ``outputs()``).
        _properties: Internal flag — set ``True`` when recursing inside a
            ``_properties`` sub-dict.

    Returns:
        Nested dict of metadata or ``None`` if none found.
    """
    if "_properties" in schema and isinstance(schema["_properties"], dict):
        return extract_metadata(schema["_properties"], True)

    if _properties and "metadata" in schema:
        metadata = schema["metadata"]
        if isinstance(metadata, np.ndarray):
            metadata = metadata.tolist()
        return metadata

    extracted: dict = {}
    for port, subschema in schema.items():
        if isinstance(subschema, dict):
            sub = extract_metadata(subschema)
            if sub is not None:
                extracted[port] = sub

    return extracted or None


# ---------------------------------------------------------------------------
# _remap_to_store — simplified inverse_topology for v2ecoli wiring
# ---------------------------------------------------------------------------


def _remap_to_store(extracted: dict, wiring: dict) -> dict:
    """Map port-relative extracted metadata to store-relative paths.

    In v2ecoli, the ``outputs`` wiring for a step edge maps port names to
    store path lists (e.g. ``{'listeners': ['listeners']}``), as built by
    ``make_edge()`` / ``list_paths()`` in ``v2ecoli.composites._helpers``.

    For each top-level port in ``extracted``, this function follows the wiring
    path to place the port's nested value at the correct location in the
    returned store-relative dict.

    Args:
        extracted: Port-relative metadata dict from ``extract_metadata()``.
        wiring: Output wiring dict for a step edge.

    Returns:
        Store-relative dict with the same leaf values as ``extracted``.
    """
    result: dict = {}
    for port_name, value in extracted.items():
        path = wiring.get(port_name)
        if path is None:
            continue
        # Navigate / create nested structure in result following the store path.
        cursor = result
        for segment in path[:-1]:
            cursor = cursor.setdefault(segment, {})
        last = path[-1]
        if isinstance(value, dict):
            existing = cursor.get(last)
            if isinstance(existing, dict):
                _deep_merge(existing, value)
            else:
                cursor[last] = value
        else:
            cursor[last] = value
    return result


def _deep_merge(target: dict, source: dict) -> None:
    """Merge ``source`` into ``target`` in-place (shallow for non-dict leaves)."""
    for k, v in source.items():
        if isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_merge(target[k], v)
        else:
            target[k] = v


# ---------------------------------------------------------------------------
# _walk_state — yield step/process edges from the composite state tree
# ---------------------------------------------------------------------------


def _walk_state(state: dict):
    """Yield ``(edge_dict, step_name)`` for each step/process edge in ``state``.

    Recurse into nested state dicts (e.g. ``agents -> '0' -> step_name``).
    An edge is identified by having an ``'instance'`` key whose value has an
    ``outputs`` callable (i.e. a Step or Process instance).
    """
    for key, value in state.items():
        if isinstance(value, dict):
            if "instance" in value and callable(
                getattr(value["instance"], "outputs", None)
            ):
                yield value, key
            else:
                # Recurse into nested state (e.g. agents → '0' → steps)
                yield from _walk_state(value)


# ---------------------------------------------------------------------------
# output_metadata — the public walker API
# ---------------------------------------------------------------------------


def output_metadata(state: dict) -> dict:
    """Harvest ``_properties.metadata`` from step ``outputs()`` across a composite state.

    Mirrors vEcoli's ``EcoliSim.output_metadata()``. Walks all step/process
    instances in ``state`` (the raw composite state dict — e.g.
    ``composite.state``, ``doc['state']``, or ``doc['state']['agents']['0']``),
    calls each instance's ``outputs()`` schema, pulls ``_properties.metadata``
    leaves, remaps to store-relative paths via the edge's output wiring, and
    deep-merges all into one dict.

    Args:
        state: Composite state dict.  Nested state trees are handled (the walker
               recurses into non-edge sub-dicts).

    Returns:
        Store-relative dict mapping paths to name lists; e.g.::

            {'listeners': {'monomer_counts': ['MONA_[c]', 'MONB_[c]', ...]}}

        Returns ``{}`` if no process carries ``_properties.metadata`` in its
        ``outputs()``.
    """
    result: dict = {}
    for edge, _name in _walk_state(state):
        instance = edge["instance"]
        try:
            outputs_schema = instance.outputs()
        except Exception:
            continue
        if not isinstance(outputs_schema, dict):
            continue
        extracted = extract_metadata(outputs_schema)
        if not extracted:
            continue
        wiring = edge.get("outputs", {})
        if not wiring:
            continue
        remapped = _remap_to_store(extracted, wiring)
        _deep_merge(result, remapped)
    return result
