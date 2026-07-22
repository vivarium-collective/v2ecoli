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
from bigraph_schema.methods.reconcile import reconcile


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


@reconcile.dispatch
def reconcile(schema: BulkNumpyUpdate, updates: list):
    """Combine multiple bulk delta-lists from one execution layer by
    CONCATENATION, so the additive ``apply`` (``count[idx] += value``) sums
    every writer's deltas.

    Without this dispatch ``BulkNumpyUpdate`` (a ``Node`` subclass) falls back
    to ``reconcile(Node, ...)``, whose non-dict branch is "last non-None wins"
    — silently discarding every bulk writer but the last one in a layer. In the
    upstream-wrapper's step-batched flow that dropped PolypeptideInitiation's
    ribosomal-subunit consumption (it lost to PolypeptideElongation's release in
    the same layer), so 30S/50S subunits accumulated unconsumed → runaway
    ribosome initiation → cell-mass explosion. Mirrors the sparse-delta branch
    of ``reconcile(Array, ...)``.
    """
    combined = []
    for update in updates:
        if update is None:
            continue
        # vEcoli bulk updates are lists of (index, delta) pairs; concatenating
        # preserves every delta for the additive apply to sum.
        combined.extend(update)
    return combined or None
