"""
Simulation helper functions for v2ecoli.

Simplified from vEcoli's ecoli.library.schema — contains only the
functions needed at runtime by biological processes.
"""

from typing import List, Tuple, Dict, Any
import numpy as np

RAND_MAX = 2**31 - 1


# ---------------------------------------------------------------------------
# MetadataArray — numpy subclass for unique molecules
# ---------------------------------------------------------------------------

class MetadataArray(np.ndarray):
    """Numpy array subclass that stores metadata (next unique molecule index)."""

    def __new__(cls, input_array, metadata=None):
        obj = np.asarray(input_array).view(cls)
        if "unique_index" in obj.dtype.names:
            if "_entryState" in obj.dtype.names:
                unique_indices = obj["unique_index"][obj["_entryState"].view(np.bool_)]
                if len(unique_indices) != len(set(unique_indices)):
                    raise ValueError(
                        "All elements in the 'unique_index' field must be unique.")
            else:
                raise ValueError("Input array must have an '_entryState' field.")
        else:
            raise ValueError("Input array must have a 'unique_index' field.")
        obj.metadata = metadata
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.metadata = getattr(obj, "metadata", None)

    def __array_wrap__(self, out_arr, context=None, return_scalar=False):
        if out_arr.shape == ():
            return out_arr.item()
        return super().__array_wrap__(out_arr, context)


# ---------------------------------------------------------------------------
# Bulk molecule helpers
# ---------------------------------------------------------------------------

def counts(states: np.ndarray, idx) -> np.ndarray:
    """Pull out counts at given indices from a structured array or plain array."""
    if len(states.dtype) > 1:
        return states["count"][idx]
    return states[idx].copy()


def bulk_name_to_idx(names, bulk_names) -> np.ndarray:
    """Convert molecule name(s) to index/indices in the bulk array."""
    if isinstance(names, (np.ndarray, list)):
        sorter = np.argsort(bulk_names)
        return np.take(
            sorter, np.searchsorted(bulk_names, names, sorter=sorter), mode="clip")
    return np.where(np.array(bulk_names) == names)[0][0]


# ---------------------------------------------------------------------------
# Unique molecule helpers
# ---------------------------------------------------------------------------

def attrs(states: 'MetadataArray', attributes: List[str]) -> List[np.ndarray]:
    """Pull out arrays for unique molecule attributes (active molecules only)."""
    mol_mask = states["_entryState"].view(np.bool_)
    return [np.asarray(states[attribute][mol_mask]) for attribute in attributes]


def create_unique_indices(n_indexes: int, unique_molecules: 'MetadataArray') -> np.ndarray:
    """Generate unique indices for new unique molecules."""
    next_unique_index = unique_molecules.metadata
    unique_indices = np.arange(
        next_unique_index, int(next_unique_index + n_indexes), dtype=int)
    unique_molecules.metadata += n_indexes
    return unique_indices


def get_free_indices(result: 'MetadataArray', n_objects: int):
    """Find inactive rows for new molecules and expand array if needed."""
    free_indices = np.where(result["_entryState"] == 0)[0]
    n_free_indices = free_indices.size

    if n_free_indices < n_objects:
        old_size = result.size
        n_new_entries = max(int(old_size * 0.1), n_objects - n_free_indices)
        result = MetadataArray(
            np.append(result, np.zeros(int(n_new_entries), dtype=result.dtype)),
            result.metadata)
        free_indices = np.concatenate(
            (free_indices, old_size + np.arange(n_new_entries)))

    return result, free_indices[:n_objects]


def array_from(d: dict) -> np.ndarray:
    """Make a numpy array from dictionary values."""
    return np.array(list(d.values()))


# ---------------------------------------------------------------------------
# UniqueNumpyUpdater
# ---------------------------------------------------------------------------

class UniqueNumpyUpdater:
    """Accumulates set/add/delete updates for unique molecules and flushes
    them in the correct order (set, add, delete) when signaled."""

    def __init__(self):
        self.add_updates = []
        self.set_updates = []
        self.delete_updates = []

    def updater(self, current: 'MetadataArray', update: Dict[str, Any]) -> 'MetadataArray':
        if len(update) == 0:
            return current

        for update_type, update_val in update.items():
            if update_type == "add":
                if isinstance(update_val, list):
                    self.add_updates.extend(update_val)
                elif isinstance(update_val, dict):
                    self.add_updates.append(update_val)
            elif update_type == "set":
                if isinstance(update_val, list):
                    self.set_updates.extend(update_val)
                elif isinstance(update_val, dict):
                    self.set_updates.append(update_val)
            elif update_type == "delete":
                if isinstance(update_val, list):
                    if len(update_val) == 0:
                        continue
                    elif isinstance(update_val[0], (list, np.ndarray)):
                        self.delete_updates.extend(update_val)
                    elif isinstance(update_val[0], (int, np.integer)):
                        self.delete_updates.append(update_val)
                elif isinstance(update_val, np.ndarray) and np.issubdtype(
                        update_val.dtype, np.integer):
                    self.delete_updates.append(update_val)

        if not update.get("update", False):
            return current

        result = current
        result.flags.writeable = True
        active_mask = result["_entryState"].view(np.bool_)

        if len(self.delete_updates) > 0:
            initially_active_idx = np.nonzero(active_mask)[0]

        for set_update in self.set_updates:
            for col, col_values in set_update.items():
                result[col][active_mask] = col_values

        for add_update in self.add_updates:
            n_new_molecules = len(next(iter(add_update.values())))
            result, free_indices = get_free_indices(result, n_new_molecules)
            if "unique_index" not in add_update:
                result["unique_index"][free_indices] = (
                    np.arange(n_new_molecules) + result.metadata)
                result.metadata += n_new_molecules
            for col, col_values in add_update.items():
                result[col][free_indices] = col_values
            result["_entryState"][free_indices] = 1

        for delete_indices in self.delete_updates:
            rows_to_delete = initially_active_idx[delete_indices]
            result[rows_to_delete] = np.zeros(1, dtype=result.dtype)

        self.add_updates = []
        self.delete_updates = []
        self.set_updates = []
        result.flags.writeable = False
        return result


# ---------------------------------------------------------------------------


def zero_listener(listener):
    """Create a zeroed version of a listener dictionary."""
    new_listener = {}
    for key, value in listener.items():
        if isinstance(value, dict):
            new_listener[key] = zero_listener(value)
        else:
            zeros = np.zeros_like(value)
            if zeros.shape == ():
                zeros = zeros.item()
            new_listener[key] = zeros
    return new_listener
