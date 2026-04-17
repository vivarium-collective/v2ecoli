"""Deep equality helper for comparing simulation state trees.

Used by tests that need to assert two state dicts are identical across
round-trips (JSON save/load) or repeat runs (determinism). Handles the
full vocabulary that shows up in v2ecoli states:

  - dict, list, tuple                         — recurse
  - numpy.ndarray (incl. structured dtypes)   — dtype + shape + contents
  - MetadataArray                              — ndarray + metadata dict
  - pint.Quantity (incl. array magnitudes)    — units + magnitude
  - set, bytes, None, scalars                 — direct comparison

Returns (True, '') on equality, (False, reason) with a dotted path on
mismatch so test failures localize the divergent leaf.
"""
from __future__ import annotations

import numpy as np


def _arrays_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.names:
        return all(_arrays_equal(a[n], b[n]) for n in a.dtype.names)
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except (TypeError, ValueError):
        return bool(np.array_equal(a, b))


def _is_quantity(x):
    return hasattr(x, 'magnitude') and hasattr(x, 'units')


def deep_equal(a, b, path: str = '') -> tuple[bool, str]:
    """Recursively compare two state values. Return (equal, reason)."""
    # numpy first — ndarrays are not == comparable directly.
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False, f'{path}: ndarray vs {type(b).__name__}'
        ok = _arrays_equal(a, b)
        meta_a = getattr(a, 'metadata', None)
        meta_b = getattr(b, 'metadata', None)
        if meta_a != meta_b:
            return False, f'{path}: MetadataArray.metadata differs'
        return (ok, '' if ok else f'{path}: ndarray contents differ')

    if _is_quantity(a) or _is_quantity(b):
        if not (_is_quantity(a) and _is_quantity(b)):
            return False, f'{path}: Quantity vs {type(b).__name__}'
        if str(a.units) != str(b.units):
            return False, f'{path}: units differ {a.units!s} vs {b.units!s}'
        return deep_equal(a.magnitude, b.magnitude, f'{path}.magnitude')

    if isinstance(a, dict):
        if not isinstance(b, dict):
            return False, f'{path}: dict vs {type(b).__name__}'
        if set(a) != set(b):
            only_a = set(a) - set(b)
            only_b = set(b) - set(a)
            return False, f'{path}: key diff only_a={only_a} only_b={only_b}'
        for k in a:
            ok, reason = deep_equal(a[k], b[k], f'{path}.{k}')
            if not ok:
                return False, reason
        return True, ''

    if isinstance(a, (list, tuple)):
        if type(a) is not type(b):
            return False, f'{path}: {type(a).__name__} vs {type(b).__name__}'
        if len(a) != len(b):
            return False, f'{path}: length {len(a)} vs {len(b)}'
        for i, (x, y) in enumerate(zip(a, b)):
            ok, reason = deep_equal(x, y, f'{path}[{i}]')
            if not ok:
                return False, reason
        return True, ''

    if isinstance(a, set):
        if not isinstance(b, set) or a != b:
            return False, f'{path}: set contents differ'
        return True, ''

    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)):
        if np.isnan(a) and np.isnan(b):
            return True, ''
        return (a == b, '' if a == b else f'{path}: {a!r} != {b!r}')

    ok = a == b
    return (ok, '' if ok else f'{path}: {a!r} != {b!r}')
