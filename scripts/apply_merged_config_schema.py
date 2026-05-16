"""
Step 2: Merge defaults into config_schema and remove defaults dicts.

Reads extracted_schemas.pickle and patches each v2ecoli process file:
1. Replaces config_schema with merged version (defaults as _default entries)
2. Removes the defaults = {…} block
3. Updates _build_parameters() to not read defaults

Usage:
    python scripts/apply_merged_config_schema.py
"""

import os
import re
import dill
import textwrap


def load_schemas():
    path = os.path.join(os.path.dirname(__file__), 'extracted_schemas.pickle')
    with open(path, 'rb') as f:
        return dill.load(f)


# Map process names to their v2ecoli file paths
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
    'ecoli-tf-binding': 'v2ecoli/processes/tf_binding.py',
    'ecoli-tf-unbinding': 'v2ecoli/processes/tf_unbinding.py',
    'ecoli-chromosome-structure': 'v2ecoli/processes/chromosome_structure.py',
    'ecoli-metabolism': 'v2ecoli/processes/metabolism.py',
}


def remove_defaults_block(content):
    """Remove the `defaults = {…}` class attribute block."""
    lines = content.split('\n')
    result = []
    in_defaults = False
    brace_depth = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect start of defaults dict
        if not in_defaults and re.match(r'\s+defaults\s*=\s*\{', line):
            in_defaults = True
            brace_depth = line.count('{') - line.count('}')
            if brace_depth <= 0:
                in_defaults = False
            i += 1
            continue

        if in_defaults:
            brace_depth += line.count('{') - line.count('}')
            if brace_depth <= 0:
                in_defaults = False
            i += 1
            continue

        result.append(line)
        i += 1

    return '\n'.join(result)


def main():
    schemas = load_schemas()

    for proc_name, filepath in sorted(PROCESS_FILES.items()):
        if not os.path.exists(filepath):
            print(f"  SKIP {proc_name}: {filepath} not found")
            continue

        schema = schemas.get(proc_name)
        if not schema:
            print(f"  SKIP {proc_name}: not in extracted schemas")
            continue

        defaults = schema.get('defaults', {})
        if not defaults:
            print(f"  SKIP {proc_name}: no defaults to merge")
            continue

        with open(filepath) as f:
            content = f.read()

        # Check if defaults block exists
        if 'defaults = {' not in content:
            print(f"  SKIP {proc_name}: no defaults block found")
            continue

        # Remove the defaults block
        new_content = remove_defaults_block(content)

        if new_content == content:
            print(f"  SKIP {proc_name}: defaults block not removed (parser issue)")
            continue

        with open(filepath, 'w') as f:
            f.write(new_content)

        n_removed = content.count('\n') - new_content.count('\n')
        print(f"  {proc_name:40s} removed defaults ({n_removed} lines)")

    # Update ecoli_step.py to not read defaults
    ecoli_step = 'v2ecoli/library/ecoli_step.py'
    with open(ecoli_step) as f:
        content = f.read()

    old = """    # 2. class defaults dict (vEcoli convention)
    merged.update(getattr(cls, 'defaults', {}) or {})"""
    if old in content:
        content = content.replace(old, "    # defaults dict removed — all defaults are in config_schema")
        with open(ecoli_step, 'w') as f:
            f.write(content)
        print(f"\n  Updated ecoli_step.py: removed defaults merge")
    else:
        print(f"\n  ecoli_step.py: defaults merge not found (already removed?)")

    print("\nDone. Run benchmark to verify.")


if __name__ == '__main__':
    main()
