"""
Custom store types for v2ecoli.

These types define how the Composite's core.apply handles updates
for different E. coli state patterns. Each type has a custom apply
dispatch that does the right thing for that store.
"""

import copy
import numpy as np
from dataclasses import dataclass, field

from bigraph_schema.schema import Node, Overwrite
from bigraph_schema.methods.apply import apply


# ---------------------------------------------------------------------------
# InPlaceDict — dict that mutates in place on apply (preserves identity)
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class InPlaceDict(Node):
    """Dict store that applies updates by mutating in place.

    Unlike the default dict apply which creates a new dict, this
    preserves the original dict object and deeply merges updates into it.
    Keys in state but not in the update are preserved.
    Keys in the update but not in state are added.
    """
    pass


def _deep_merge_apply(state, update):
    """Recursively merge update into state in place."""
    if not isinstance(update, dict):
        return update
    if not isinstance(state, dict):
        return update
    for key, value in update.items():
        if key in state and isinstance(state[key], dict) and isinstance(value, dict):
            _deep_merge_apply(state[key], value)
        else:
            state[key] = value
    return state


@apply.dispatch
def apply(schema: InPlaceDict, state, update, path):
    if update is None:
        return state, []
    if state is None:
        return update, []
    _deep_merge_apply(state, update)
    return state, []


# ---------------------------------------------------------------------------
# SetStore — dict store with 'set' semantics (full replacement on apply)
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class SetStore(Node):
    """Dict store that replaces its value entirely on apply.

    Used for stores like 'request' and 'allocate' where the allocator
    needs to overwrite the entire contents each timestep.
    """
    pass


@apply.dispatch
def apply(schema: SetStore, state, update, path):
    if update is None:
        return state, []
    return update, []


# ---------------------------------------------------------------------------
# ListenerStore — nested dict with set semantics at leaf level
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class DerivedProperties(InPlaceDict):
    """Derived-properties store: in-place dict merge at top level, set at leaves.

    Derivers (formerly "listeners") compute derived properties of cell
    state and write them under their own sub-key. The store merges at the
    top level but replaces values at the leaf level — which is why the
    per-leaf ``overwrite[...]`` wrappers are redundant for scalar leaves
    here (the container already does leaf-replace).
    """
    pass


# Back-compat alias: external code / older study specs may still import or
# reference the "listener" name. The registry exposes both 'derived_properties'
# (primary) and 'listener_store' (alias) — see v2ecoli/types/__init__.py.
ListenerStore = DerivedProperties


@apply.dispatch
def apply(schema: DerivedProperties, state, update, path):
    if update is None:
        return state, []
    if state is None:
        return update if isinstance(update, dict) else {}, []
    if isinstance(state, dict) and isinstance(update, dict):
        _deep_merge_apply(state, update)
        return state, []
    return update, []


# ---------------------------------------------------------------------------
# ScalarStore — single value with accumulate semantics
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class AccumulateFloat(Node):
    """Float that accumulates (adds) on apply."""
    pass


@apply.dispatch
def apply(schema: AccumulateFloat, state, update, path):
    if update is None:
        return state, []
    if state is None:
        return update, []
    return state + update, []
