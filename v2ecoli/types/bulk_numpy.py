"""Type for bulk molecule structured arrays.

Updates are lists of (index, value) tuples that get added to the
'count' field of a structured numpy array.  This matches the
semantics of vEcoli's ``bulk_numpy_updater``.
"""

import numpy as np
from dataclasses import dataclass, field

from bigraph_schema.schema import Node
from bigraph_schema.methods import infer, set_default, serialize, realize, render, wrap_default
from bigraph_schema.methods.apply import apply


@dataclass(kw_only=True)
class BulkNumpyUpdate(Node):
    pass


@apply.dispatch
def apply(schema: BulkNumpyUpdate, state, update, path):
    if update is None or state is None:
        return state if update is None else update, []

    try:
        state.flags.writeable = True
    except ValueError:
        state = state.copy()
        state.flags.writeable = True
    for idx, value in update:
        state["count"][idx] += value
    return state, []
