"""
Partition utilities for v2ecoli.

Provides shared helpers used by per-process Requester/Evolver Steps:
- _protect_state(): copies bulk/unique arrays before process execution
- _SafeInvokeMixin: catches errors in update() to prevent cascade crashes
- deep_merge(): recursive dict merge
"""

import numpy as np
from process_bigraph.composite import SyncUpdate


def _protect_state(state):
    """Return a shallow copy of state with bulk/unique arrays copied.

    Processes from v1 mutate their input arrays in place. Since core.view
    returns the live state object, we must copy arrays that processes
    might modify to prevent corruption of the simulation state.
    """
    protected = dict(state)
    if 'bulk' in protected and hasattr(protected['bulk'], 'copy'):
        protected['bulk'] = protected['bulk'].copy()
        protected['bulk'].flags.writeable = True
    if 'unique' in protected and isinstance(protected['unique'], dict):
        protected['unique'] = {
            k: v.copy() if hasattr(v, 'copy') else v
            for k, v in protected['unique'].items()
        }
        for arr in protected['unique'].values():
            if hasattr(arr, 'flags'):
                arr.flags.writeable = True
    return protected


class _SafeInvokeMixin:
    """Mixin that catches and LOGS errors in update() to prevent cascade crashes."""
    def invoke(self, state, interval=None):
        try:
            update = self.update(state)
        except Exception as e:
            import warnings
            step_name = getattr(self, 'name', type(self).__name__)
            warnings.warn(
                f"Step {step_name} raised {type(e).__name__}: {e}",
                RuntimeWarning, stacklevel=2)
            update = {}
        return SyncUpdate(update)


def deep_merge(base, override):
    """Recursively merge override into base (modifies base)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base

