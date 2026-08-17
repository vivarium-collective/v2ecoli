"""LabeledArray — an Array subtype that carries static element-name labels.

``_labels`` is a plain dataclass field that is intentionally **absent** from
``_schema_keys``.  This means ``is_schema_field()`` returns ``False`` for it,
so the bigraph-schema engine treats it as static metadata:

* ignored by per-tick operations (apply / check / default / serialize)
* preserved by ``resolve_subclass`` (via ``getattr``)
* never enters per-tick state
* readable from the registry via ``core.access(type_name)._labels``

Usage pattern (register a pre-built instance by name)::

    from v2ecoli.types.labeled_array import LabeledArray
    import numpy as np

    instance = LabeledArray(
        _shape=(N,), _data=np.dtype('int64'), _labels=tuple(ids)
    )
    core.register_type('monomer_counts_vec', instance)

    # In the process's outputs():
    def outputs(self):
        return {'monomer_counts': 'monomer_counts_vec'}

    # In the generic walker:
    node = core.access('monomer_counts_vec')
    labels = getattr(node, '_labels', None)  # -> tuple of element names
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field

from bigraph_schema.schema import Array


@dataclass(kw_only=True)
class LabeledArray(Array):
    """Array subtype carrying static element-name labels.

    ``_labels`` is a dataclass field but intentionally absent from
    ``_schema_keys``.  The engine ignores it per-tick; the registry
    preserves it so a generic walker can recover element names from any
    registered labeled vector type without touching per-tick logic.
    """

    # NOT added to _schema_keys → is_schema_field() returns False.
    # Engine treats this as static metadata, not a schema property.
    _labels: typing.Tuple[str, ...] = field(default_factory=tuple)


def register_labeled_array(core, name, *, shape, data='int64', labels):
    """Register a named labeled array type that is resolvable as a parameter.

    Registers a plain **schema dict** (``{'_inherit': 'array', ...}``) carrying
    ``_labels`` as static metadata, rather than a bare ``LabeledArray``
    *instance*.  Both forms let a generic walker recover element names via
    ``core.access(name)._labels``, but only the schema-dict form can be wrapped
    by a parametric type (e.g. ``overwrite[<name>]``) and survive a
    serialize → rebuild round-trip — which the dashboard composite-runner does
    via ``build_generator`` (a bare instance makes ``core.access`` return a
    string, so ``core.access('overwrite[<name>]')`` raises "cannot resolve
    types").  Use this for any vector whose port is declared as
    ``overwrite[<name>]`` (e.g. CountsDeriver's ``monomer_counts``).

    ``_labels`` is recovered through the ``overwrite[...]`` wrapper by
    ``output_metadata._extract_labels_recursive`` (it unwraps the parametric
    type and re-resolves the inner name).

    ``_data`` is registered as a real ``np.dtype`` instance, not a string.
    ``Core.access()`` on a dict with ``_inherit`` (rather than ``_type``)
    resolves the ancestor schema and then ``dataclasses.replace()``s each
    remaining key directly onto it, WITHOUT running underscore-prefixed keys
    (including ``_data``) through ``core.access()``/``reify_schema`` — that
    normalization (string dtype name -> ``np.dtype``, e.g. via
    ``schema_dtype()``) only happens on the ``_type``-keyed / type-expression
    parsing path. A string here lands verbatim in the registered ``Array``
    schema's ``_data`` field, which ``bigraph_schema``'s own ``Array``
    dataclass declares as ``np.dtype`` — so any later serialize/render of a
    port using this type (e.g. ``overwrite[monomer_counts_vec]``) crashes in
    numpy's ``dtype_to_descr``/``drop_metadata`` with
    ``AttributeError: 'str' object has no attribute 'fields'`` (item 58).
    """
    import numpy as _np
    data_dtype = data if isinstance(data, _np.dtype) else _np.dtype(data)
    # register_type is first-wins; build_core() pre-registers a bare placeholder
    # of this name (so serialized composites resolve overwrite[<name>] before
    # the deriver runs). Evict it so this labeled definition takes effect.
    try:
        core.registry.pop(name, None)
    except Exception:
        pass
    core.register_type(name, {
        '_inherit': 'array',
        '_data': data_dtype,
        '_shape': tuple(shape),
        '_labels': tuple(labels),
    })
    return core
