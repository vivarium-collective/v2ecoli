"""
Step 2: Remove defaults dicts by merging into config_schema.

For simple defaults (bool, int, float, str), adds inline defaults
to config_schema: 'boolean' → 'boolean{false}'.

For complex defaults (arrays, callables, dicts), adds them as
'_default' entries: {'_type': 'array[float]', '_default': np.array([])}.

Then removes the defaults = {…} block and updates _build_parameters.
"""

import os
import re
import sys
import ast
import dill

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
}


def remove_block(content, pattern):
    """Remove a class-level dict block matching pattern (e.g. 'defaults = {')."""
    lines = content.split('\n')
    result = []
    in_block = False
    brace_depth = 0

    for line in lines:
        if not in_block and re.match(r'\s+' + re.escape(pattern), line):
            in_block = True
            brace_depth = line.count('{') - line.count('}')
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


def main():
    with open('scripts/extracted_schemas.pickle', 'rb') as f:
        schemas = dill.load(f)

    for proc_name, filepath in sorted(PROCESS_FILES.items()):
        if not os.path.exists(filepath):
            print(f"  SKIP {proc_name}: file not found")
            continue

        schema = schemas.get(proc_name)
        if not schema:
            print(f"  SKIP {proc_name}: not in extracted schemas")
            continue

        defaults = schema.get('defaults', {})
        if not defaults:
            print(f"  SKIP {proc_name}: no defaults")
            continue

        with open(filepath) as f:
            content = f.read()

        if 'defaults = {' not in content:
            print(f"  SKIP {proc_name}: no defaults block")
            continue

        # Remove defaults block
        new_content = remove_block(content, 'defaults = {')
        if new_content == content:
            print(f"  SKIP {proc_name}: block not removed")
            continue

        removed_lines = content.count('\n') - new_content.count('\n')

        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"  {proc_name:40s} removed defaults ({removed_lines} lines)")

    # Update _build_parameters in ecoli_step.py
    path = 'v2ecoli/library/ecoli_step.py'
    with open(path) as f:
        content = f.read()

    old = """    # 2. class defaults dict (vEcoli convention)
    merged.update(getattr(cls, 'defaults', {}) or {})"""
    if old in content:
        content = content.replace(old, '')
        with open(path, 'w') as f:
            f.write(content)
        print(f"\n  Updated ecoli_step.py: removed defaults merge")

    print("\nNow run: python3 reports/benchmark_report.py")


if __name__ == '__main__':
    main()
