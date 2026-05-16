"""
Write merged config_schema into each process file and remove defaults dicts.

For each process:
1. Generates Python source for the merged config_schema
2. Replaces the existing config_schema block in the file
3. Removes the defaults = {…} block
"""

import os
import re
import dill
import numpy as np


PROCESS_FILES = {
    'ecoli-equilibrium': 'v2ecoli/processes/equilibrium.py',
    'ecoli-two-component-system': 'v2ecoli/processes/two_component_system.py',
    'ecoli-rna-maturation': 'v2ecoli/processes/rna_maturation.py',
    'ecoli-complexation': 'v2ecoli/processes/complexation.py',
    'ecoli-protein-degradation': 'v2ecoli/processes/protein_degradation.py',
    'ecoli-rna-degradation': 'v2ecoli/processes/rna_degradation.py',
    'ecoli-transcript-initiation': 'v2ecoli/processes/transcript_initiation.py',
    'ecoli-transcript-elongation': 'v2ecoli/processes/transcript_elongation.py',
    'ecoli-polypeptide-initiation': 'v2ecoli/processes/polypeptide_initiation.py',
    'ecoli-polypeptide-elongation': 'v2ecoli/processes/polypeptide_elongation.py',
    'ecoli-chromosome-replication': 'v2ecoli/processes/chromosome_replication.py',
    'ecoli-metabolism': 'v2ecoli/processes/metabolism.py',
    'ecoli-chromosome-structure': 'v2ecoli/processes/chromosome_structure.py',
    'ecoli-tf-binding': 'v2ecoli/processes/tf_binding.py',
    'ecoli-tf-unbinding': 'v2ecoli/processes/tf_unbinding.py',
}


def value_to_source(value):
    """Convert a default value to Python source."""
    if value is None:
        return 'None'
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (abs(value) > 1e15 or (0 < abs(value) < 1e-10)):
            return f'{value:.6e}'
        return repr(value)
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            dt = {
                'i': 'int', 'f': 'float', 'b': 'bool',
                'U': 'str', 'S': 'str',
            }.get(value.dtype.kind, 'float')
            return f'np.array([], dtype={dt})'
        if value.ndim == 2 and value.shape[1] == 0:
            return 'np.array([[]])'
        if value.ndim == 1 and value.size <= 10:
            return f'np.array({value.tolist()})'
        return f'np.zeros({list(value.shape) if value.ndim > 1 else value.shape[0]}, dtype=np.{value.dtype.name})'
    if isinstance(value, list):
        if len(value) == 0:
            return '[]'
        if len(value) <= 5 and all(isinstance(v, (str, int, float, bool)) for v in value):
            return repr(value)
        return '[]'
    if isinstance(value, dict):
        if len(value) == 0:
            return '{}'
        # Try to repr small simple dicts
        if len(value) <= 3 and all(isinstance(k, str) and isinstance(v, (str, int, float, bool, type(None))) for k, v in value.items()):
            return repr(value)
        return '{}'
    if isinstance(value, set):
        return repr(value) if len(value) <= 5 else 'set()'
    if callable(value):
        return 'None'
    if hasattr(value, 'asNumber'):
        return repr(float(value.asNumber()))
    if hasattr(value, 'magnitude'):
        return repr(float(value.magnitude))
    return repr(value)


def generate_config_schema_source(merged_schema, indent='    '):
    """Generate Python source for config_schema dict."""
    lines = [f'{indent}config_schema = {{']
    for key in sorted(merged_schema.keys()):
        spec = merged_schema[key]
        if isinstance(spec, dict):
            t = spec.get('_type', 'any')
            d = spec.get('_default')
            if d is not None:
                ds = value_to_source(d)
                lines.append(f"{indent}    '{key}': {{'_type': '{t}', '_default': {ds}}},")
            else:
                lines.append(f"{indent}    '{key}': '{t}',")
        else:
            lines.append(f"{indent}    '{key}': '{spec}',")
    lines.append(f'{indent}}}')
    return '\n'.join(lines)


def replace_block(content, pattern, replacement):
    """Replace a class-level dict block with new content."""
    lines = content.split('\n')
    result = []
    in_block = False
    brace_depth = 0
    replaced = False

    for line in lines:
        if not in_block and re.match(r'\s+' + re.escape(pattern), line):
            in_block = True
            brace_depth = line.count('{') - line.count('}')
            if not replaced:
                result.append(replacement)
                replaced = True
            if brace_depth <= 0:
                in_block = False
            continue
        if in_block:
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0:
                in_block = False
            continue
        result.append(line)

    return '\n'.join(result)


def remove_block(content, pattern):
    """Remove a class-level dict block."""
    return replace_block(content, pattern, '')


def main():
    with open('scripts/extracted_schemas.pickle', 'rb') as f:
        schemas = dill.load(f)

    for proc_name, filepath in sorted(PROCESS_FILES.items()):
        if not os.path.exists(filepath):
            print(f"  SKIP {proc_name}: file not found")
            continue

        schema = schemas.get(proc_name)
        if not schema:
            print(f"  SKIP {proc_name}: not in schemas")
            continue

        with open(filepath) as f:
            content = f.read()

        original = content
        merged = schema['merged_config_schema']

        # Generate new config_schema source
        new_cs = generate_config_schema_source(merged)

        # Replace config_schema block
        if 'config_schema = {' in content:
            content = replace_block(content, 'config_schema = {', new_cs)

        # Remove defaults block
        if 'defaults = {' in content:
            content = remove_block(content, 'defaults = {')

        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            removed = original.count('\n') - content.count('\n')
            print(f"  {proc_name:40s} updated config_schema ({len(merged)} keys), removed defaults ({removed} lines)")
        else:
            print(f"  {proc_name:40s} no changes")

    print("\nDone. Run benchmark to verify.")


if __name__ == '__main__':
    main()
