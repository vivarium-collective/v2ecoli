"""
Repair an out/cache/sim_data_cache.dill that was saved with pint-style
serialization tuples instead of Unum quantities, leaving the simulation
pipeline unable to call .asNumber() on what the processes expect to be
Unum objects.

Two separate issues fixed:

1. ``('__pint__', value, "gram / liter")`` tuples → ``value * units.g/units.L``
   Unum quantities. Without this, e.g. tf_binding's ``bulk_mass_data`` is
   a plain ndarray instead of a Unum, and its ``.asNumber(units.fg)`` call
   fails.

2. Empty-unit Unums ("umol/mol") → plain floats via
   ``units.strip_empty_units(q.normalize())``. Without this, e.g.
   polypeptide_elongation's ``unit_conversion`` is ``1e-06 []`` instead of
   ``1e-06``, and amino_acid_synthesis raises IncompatibleUnitsError
   trying to reinterpret a ratio as a concentration.

Optionally merges a second cache to fill in holes — e.g. if the primary
cache has non-deserializable values for ``condition_to_doubling_time``,
copy that key from a known-good cache.

Usage:
    python scripts/repair_cache.py out/cache/sim_data_cache.dill [--overlay SRC]

If the cache already looks fine, this is a no-op.
"""

import argparse
import copy
import re
import sys

import dill
from unum import Unum

from wholecell.utils import units as wc_units


_BASE_UNITS = {
    'femtogram': wc_units.fg, 'fg': wc_units.fg,
    'minute': wc_units.min, 'min': wc_units.min,
    'second': wc_units.s, 's': wc_units.s,
    'hour': wc_units.h, 'h': wc_units.h,
    'mole': wc_units.mol, 'mol': wc_units.mol,
    'millimole': wc_units.mmol, 'mmol': wc_units.mmol,
    'micromole': (wc_units.umol if hasattr(wc_units, 'umol')
                  else wc_units.mmol / 1000),
    'gram': wc_units.g, 'g': wc_units.g,
    'kilogram': wc_units.g * 1000, 'kg': wc_units.g * 1000,
    'milligram': wc_units.mg, 'mg': wc_units.mg,
    'microgram': wc_units.ug, 'ug': wc_units.ug,
    'liter': wc_units.L, 'L': wc_units.L,
    'amino_acid': wc_units.aa, 'aa': wc_units.aa,
    'nucleotide': wc_units.nt, 'nt': wc_units.nt,
    'count': wc_units.count,
    'dimensionless': 1, '1': 1,
}


def _parse_unit(s):
    tokens = re.split(r'\s*([*/])\s*', s.strip())
    result = None
    op = '*'
    for t in tokens:
        t = t.strip()
        if not t:
            continue
        if t in '*/':
            op = t
            continue
        m = re.match(r'(\w+)\s*\*\*\s*(\d+)', t)
        if m:
            base = _BASE_UNITS.get(m.group(1))
            part = base ** int(m.group(2)) if base is not None else None
        else:
            part = _BASE_UNITS.get(t)
        if part is None:
            raise KeyError(f"unknown unit fragment: {t!r}")
        if result is None:
            result = part
        elif op == '*':
            result = result * part
        else:
            result = result / part
    return result


def _deserialize_pint(obj):
    """Recursively convert ('__pint__', val, unit) tuples → Unum quantities."""
    if isinstance(obj, tuple) and len(obj) == 3 and obj[0] == '__pint__':
        _, val, unit_str = obj
        try:
            u = _parse_unit(unit_str)
        except Exception:
            return obj
        # Order matters: Unum * ndarray keeps Unum; ndarray * Unum drops it.
        return u * val
    if isinstance(obj, dict):
        return {k: _deserialize_pint(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deserialize_pint(v) for v in obj]
    return obj


def _strip_empty_unums(obj):
    """Recursively collapse Unums with no net units to plain floats/ints."""
    if isinstance(obj, Unum):
        try:
            normalized = obj.normalize()
            normalized.checkNoUnit()
            return wc_units.strip_empty_units(normalized)
        except Exception:
            return obj
    if isinstance(obj, dict):
        return {k: _strip_empty_unums(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_empty_unums(v) for v in obj]
    return obj


def _rename_legacy_keys(cache):
    """Apply rename migrations that have landed in the codebase."""
    cfgs = cache.get('configs', {})
    if 'exchange_data' in cfgs and 'metabolic_kinetics' not in cfgs:
        cfgs['metabolic_kinetics'] = cfgs.pop('exchange_data')


def repair(path, overlay_path=None):
    with open(path, 'rb') as f:
        cache = dill.load(f)

    if overlay_path:
        with open(overlay_path, 'rb') as f:
            overlay = dill.load(f)
        # Prefer overlay's mass-listener condition_to_doubling_time (Unum form)
        ml = cache['configs'].get('ecoli-mass-listener')
        ml_over = overlay['configs'].get('ecoli-mass-listener', {})
        if ml is not None and 'condition_to_doubling_time' in ml_over:
            if any(isinstance(v, tuple) for v in
                   ml.get('condition_to_doubling_time', {}).values()):
                ml['condition_to_doubling_time'] = (
                    ml_over['condition_to_doubling_time'])

    cache['configs'] = _deserialize_pint(cache['configs'])
    cache['configs'] = _strip_empty_unums(cache['configs'])
    _rename_legacy_keys(cache)

    with open(path, 'wb') as f:
        dill.dump(cache, f)
    print(f"repaired {path}")


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', default='out/cache/sim_data_cache.dill')
    ap.add_argument('--overlay', default=None,
                    help='second cache to borrow well-formed entries from')
    args = ap.parse_args()
    repair(args.path, args.overlay)


if __name__ == '__main__':
    _main()
