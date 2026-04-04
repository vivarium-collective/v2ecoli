"""Type for unique molecule structured arrays.

Updates are dicts with 'set', 'add', 'delete', and 'update' keys,
matching the semantics of vEcoli's ``UniqueNumpyUpdater``.

The updater accumulates set/add/delete operations across multiple steps
and flushes when it receives ``{'update': True}``.  A module-level
registry maps state paths to shared updater instances.
"""

import numpy as np
from dataclasses import dataclass, field

from bigraph_schema.schema import Node
from bigraph_schema.methods.apply import apply

from ecoli.library.schema import UniqueNumpyUpdater


# Module-level registry: maps state path (tuple) -> shared UniqueNumpyUpdater
_updater_registry = {}


def register_unique_updater(path, updater=None):
    """Register a shared UniqueNumpyUpdater for a state path."""
    if path not in _updater_registry:
        _updater_registry[path] = updater or UniqueNumpyUpdater()
    return _updater_registry[path]


def get_unique_updater(path):
    """Get the shared UniqueNumpyUpdater for a state path, creating if needed."""
    if path not in _updater_registry:
        _updater_registry[path] = UniqueNumpyUpdater()
    return _updater_registry[path]


def clear_updater_registry():
    """Clear all registered updaters."""
    _updater_registry.clear()


@dataclass(kw_only=True)
class UniqueNumpyUpdate(Node):
    pass


@apply.dispatch
def apply(schema: UniqueNumpyUpdate, state, update, path):
    if update is None or state is None:
        return state if update is None else update, []

    if not isinstance(update, dict) or len(update) == 0:
        return state, []

    # Ensure the array is writeable before passing to the updater
    if hasattr(state, 'flags'):
        try:
            state.flags.writeable = True
        except ValueError:
            state = state.copy()
            state.flags.writeable = True

    updater = get_unique_updater(path)
    result = updater.updater(state, update)
    return result, []
