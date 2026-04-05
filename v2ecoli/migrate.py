"""
Migration utility for v2ecoli.

Audits all processes and generates a report showing which v1 patterns
remain and what the v2 equivalents should be.

Usage: uv run python -m v2ecoli.migrate
"""

import os
import json
import numpy as np

from bigraph_schema import allocate_core
from bigraph_schema.methods import render

from v2ecoli.types import ECOLI_TYPES
from v2ecoli.steps.base import _translate_schema


def audit_processes():
    """Audit all process files for v1 vs v2 patterns."""
    from v2ecoli.composite import make_composite

    composite = make_composite(cache_dir='out/cache')
    core = composite.core
    cell = composite.state['agents']['0']

    report = []

    for name, edge in sorted(cell.items()):
        if not isinstance(edge, dict) or 'instance' not in edge:
            continue

        inst = edge['instance']
        proc = getattr(inst, 'process', inst)  # unwrap Requester/Evolver

        entry = {
            'name': name,
            'class': type(inst).__name__,
            'inner_class': type(proc).__name__ if proc is not inst else None,
            'patterns': {},
            'config': {},
            'ports': {},
        }

        # Check v1 vs v2 patterns
        has_ports_schema = hasattr(proc, 'ports_schema') and callable(getattr(proc, 'ports_schema'))
        has_inputs = hasattr(inst, 'inputs') and callable(getattr(inst, 'inputs'))
        has_outputs = hasattr(inst, 'outputs') and callable(getattr(inst, 'outputs'))
        has_defaults = hasattr(proc, 'defaults') and isinstance(getattr(proc, 'defaults', None), dict) and proc.defaults
        has_next_update = hasattr(proc, 'next_update') and 'next_update' in type(proc).__dict__
        has_update = hasattr(inst, 'update') and 'update' in type(inst).__dict__
        has_config_schema = hasattr(inst, 'config_schema') and inst.config_schema

        entry['patterns'] = {
            'ports_schema (v1)': has_ports_schema,
            'inputs() (v2)': has_inputs,
            'outputs() (v2)': has_outputs,
            'defaults (v1)': has_defaults,
            'config_schema (v2)': bool(has_config_schema),
            'next_update (v1)': has_next_update,
            'update (v2)': has_update,
        }

        # Get config info
        params = getattr(proc, 'parameters', {})
        if isinstance(params, dict):
            config_summary = {}
            for k, v in params.items():
                if callable(v) and not isinstance(v, (type, np.ndarray)):
                    config_summary[k] = 'callable'
                elif isinstance(v, np.ndarray):
                    config_summary[k] = f'array{list(v.shape)}'
                elif isinstance(v, (list, tuple)):
                    config_summary[k] = f'list[{len(v)}]'
                elif isinstance(v, dict):
                    config_summary[k] = f'dict[{len(v)}]'
                else:
                    config_summary[k] = type(v).__name__
            entry['config'] = config_summary

        # Get port info
        if has_ports_schema:
            try:
                ps = proc.ports_schema()
                translated = _translate_schema(ps)
                entry['ports'] = {
                    'ports_schema_keys': sorted(ps.keys()),
                    'translated_types': {k: type(v).__name__ for k, v in translated.items()},
                }
            except Exception as e:
                entry['ports'] = {'error': str(e)}

        # Inferred config schema
        inferred = getattr(proc, '_inferred_config_schema', None)
        if inferred:
            try:
                rendered = render(inferred, defaults=False)
                entry['inferred_config_schema'] = {
                    k: (v if isinstance(v, str) else type(v).__name__)
                    for k, v in rendered.items()
                }
            except Exception:
                pass

        report.append(entry)

    return report


def print_report(report):
    """Print a human-readable migration report."""
    print("=" * 70)
    print("  v2ecoli Migration Audit")
    print("=" * 70)

    v1_count = 0
    v2_count = 0
    hybrid_count = 0

    for entry in report:
        p = entry['patterns']
        is_v1 = p['ports_schema (v1)'] and not p.get('config_schema (v2)')
        is_v2 = p['inputs() (v2)'] and p.get('config_schema (v2)')
        is_hybrid = p['inputs() (v2)'] and p['ports_schema (v1)']

        if is_v2:
            status = "V2"
            v2_count += 1
        elif is_hybrid:
            status = "HYBRID"
            hybrid_count += 1
        else:
            status = "V1"
            v1_count += 1

        inner = f" ({entry['inner_class']})" if entry['inner_class'] else ""
        print(f"\n  [{status}] {entry['name']}")
        print(f"    Class: {entry['class']}{inner}")

        for pattern, has_it in p.items():
            marker = "✓" if has_it else "✗"
            print(f"    {marker} {pattern}")

        if entry.get('config'):
            n_callable = sum(1 for v in entry['config'].values() if v == 'callable')
            n_array = sum(1 for v in entry['config'].values() if v.startswith('array'))
            n_scalar = len(entry['config']) - n_callable - n_array
            print(f"    Config: {len(entry['config'])} keys "
                  f"({n_scalar} scalar, {n_array} array, {n_callable} callable)")

    print(f"\n{'=' * 70}")
    print(f"  Summary: {v1_count} V1, {hybrid_count} HYBRID, {v2_count} V2")
    print(f"  Total: {len(report)} steps")
    print(f"{'=' * 70}")

    return {'v1': v1_count, 'hybrid': hybrid_count, 'v2': v2_count}


def save_report(report, path='out/migration_audit.json'):
    """Save the migration report as JSON."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # Simplify for JSON
    simple = []
    for entry in report:
        simple.append({
            'name': entry['name'],
            'class': entry['class'],
            'inner_class': entry['inner_class'],
            'patterns': entry['patterns'],
            'config_keys': list(entry.get('config', {}).keys()),
            'port_keys': entry.get('ports', {}).get('ports_schema_keys', []),
        })
    from v2ecoli.cache import NumpyJSONEncoder
    with open(path, 'w') as f:
        json.dump(simple, f, indent=2, cls=NumpyJSONEncoder)
    print(f"Saved migration audit to {path}")


if __name__ == '__main__':
    report = audit_processes()
    counts = print_report(report)
    save_report(report)
