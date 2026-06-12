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
