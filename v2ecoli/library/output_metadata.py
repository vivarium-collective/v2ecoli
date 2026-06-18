"""
Output metadata walker for v2ecoli composites.

Makes runs self-describing: each listener process registers a named
LabeledArray type in its ``initialize()`` carrying the element IDs for its
fixed-length vector outputs.  This walker reads those labels from the
type registry and returns a store-relative dict of names that feeds
XArrayEmitter's coord arrays.

Primary (registry) path
-----------------------
When a port type is a **string name** registered in the bigraph-schema core,
the walker calls ``core.access(type_name)`` and reads ``getattr(node,
'_labels', None)`` (a plain dataclass field that is **not** in
``_schema_keys``, so the engine ignores it per-tick).  This is the approach
used by CountsDeriver / RnapData (and any future listener).

Legacy (backward-compat) path
-------------------------------
When a port schema is a dict carrying ``_properties: {'metadata': [...]}``,
the walker uses the original ``extract_metadata()`` helper so that any code
that annotates ports the old way still works.

The returned dict feeds into:
  - XArrayEmitter ``output_metadata`` (coord arrays for vector leaves);
  - ParquetEmitter ``config["metadata"]["output_metadata"]`` (written as
    ``output_metadata__<path>`` columns in the configuration parquet).
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# extract_metadata — legacy _properties.metadata path (kept for backward compat)
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
# _extract_labels_recursive — primary + legacy walker
# ---------------------------------------------------------------------------


def _labels_from_node(node: Any) -> Any:
    """Recover ``_labels`` from a resolved type node, unwrapping wrappers.

    A bare ``LabeledArray`` carries ``_labels`` directly.  When the port is
    wrapped for apply-semantics (e.g. ``overwrite[monomer_counts_vec]`` to get
    SET instead of the Array default ACCUMULATE), the labels live on the inner
    ``_value``.  Walk one level of ``_value`` so wrapped labeled vectors still
    surface their element names.
    """
    labels = getattr(node, '_labels', None)
    if labels:
        return labels
    inner = getattr(node, '_value', None)
    if inner is not None:
        return getattr(inner, '_labels', None)
    return None


def _extract_labels_recursive(schema: Any, core: Any) -> Any:
    """Recursively extract element-name labels from a port schema.

    Handles two annotation styles:

    * **Registry (primary)**: port value is a ``str`` type name that resolves
      to a ``LabeledArray`` (or any node with ``_labels``).  Labels are read
      via ``core.access(name)._labels``.

    * **Properties (legacy)**: port dict carries ``_properties.metadata``.
      Delegates to ``extract_metadata()``.

    Args:
        schema: A port schema value — either a string type name, a typed leaf
            dict, or a nested port dict.
        core:   The bigraph-schema Core instance, or ``None`` (registry path
                skipped when core is unavailable).

    Returns:
        ``list`` of labels when a port carries them; nested ``dict`` mapping
        sub-port names to their label lists; or ``None`` when no labels found.
    """
    if isinstance(schema, str):
        # Registry path: look up the type name and check for _labels.
        if core is not None:
            try:
                node = core.access(schema)
                labels = _labels_from_node(node)
                if labels:
                    return list(labels)
            except Exception:
                pass
            # Parametric-wrapper path: a port declared as ``overwrite[<name>]``
            # (or ``maybe[...]`` etc.) wraps a labeled vector. When the inner
            # type is registered as a schema dict (``register_labeled_array``),
            # the wrapper node doesn't surface the inner ``_labels`` directly,
            # so peel one ``wrapper[inner]`` level and resolve the inner name.
            m = re.match(r'^\w+\[(.+)\]$', schema.strip())
            if m:
                return _extract_labels_recursive(m.group(1), core)
        return None

    if not isinstance(schema, dict):
        return None

    # Legacy path: dict carries _properties.metadata.
    if '_properties' in schema:
        return extract_metadata(schema)

    # Typed leaf dict (e.g. {'_type': 'overwrite[array[N,integer]]', '_default': []}):
    # check the _type string for labels (unlikely but handles edge cases).
    if '_type' in schema:
        type_str = schema.get('_type')
        if isinstance(type_str, str) and core is not None:
            try:
                node = core.access(type_str)
                labels = _labels_from_node(node)
                if labels:
                    return list(labels)
            except Exception:
                pass
        return None

    # Plain nested port dict: recurse into each sub-key.
    result: dict = {}
    for key, val in schema.items():
        sub = _extract_labels_recursive(val, core)
        if sub is not None:
            result[key] = sub
    return result or None


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
        extracted: Port-relative metadata dict from ``_extract_labels_recursive()``.
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
    """Harvest element-name labels from step ``outputs()`` schemas across a composite state.

    For each step/process instance in ``state``, calls ``outputs()`` and
    extracts element-name labels via two paths (in priority order):

    1. **Registry path** (primary): port type is a ``str`` registered in
       ``instance.core``; labels come from
       ``core.access(type_name)._labels``.  Used by CountsDeriver /
       RnapData after the listener's ``initialize()`` has registered the
       named ``LabeledArray`` type.

    2. **Properties path** (legacy): port schema dict carries
       ``_properties: {'metadata': [...]}``; harvested by
       ``extract_metadata()``.

    Both paths produce the same store-relative output dict so downstream
    consumers (XArrayEmitter, ParquetEmitter) are unchanged.

    Args:
        state: Composite state dict.  Nested state trees are handled (the
               walker recurses into non-edge sub-dicts).

    Returns:
        Store-relative dict mapping paths to name lists; e.g.::

            {'listeners': {'monomer_counts': ['MONA[c]', 'MONB[c]', ...]}}

        Returns ``{}`` if no process carries labels in its ``outputs()``.
    """
    result: dict = {}
    for edge, _name in _walk_state(state):
        instance = edge["instance"]
        core = getattr(instance, 'core', None)
        try:
            outputs_schema = instance.outputs()
        except Exception:
            continue
        if not isinstance(outputs_schema, dict):
            continue
        extracted = _extract_labels_recursive(outputs_schema, core)
        if not extracted:
            continue
        wiring = edge.get("outputs", {})
        if not wiring:
            continue
        remapped = _remap_to_store(extracted, wiring)
        _deep_merge(result, remapped)
    return result
