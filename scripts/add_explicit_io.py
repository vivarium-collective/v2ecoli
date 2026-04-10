"""
Step 3: Add explicit inputs()/outputs() to partitioned processes.

Reads extracted schemas and writes inputs()/outputs() methods into each
PartitionedProcess subclass, replacing the _typed_ports(self.ports_schema())
derivation.

Uses schema_types constants (RNA_ARRAY, ACTIVE_RNAP_ARRAY, etc.) for readability.
"""

import os
import sys
import re
import dill
import textwrap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# Known schema_types constants and their string values
from v2ecoli.library.schema_types import (
    BULK_ARRAY, PROMOTER_ARRAY, RNA_ARRAY, ACTIVE_RNAP_ARRAY,
    ACTIVE_RIBOSOME_ARRAY, ACTIVE_REPLISOME_ARRAY, FULL_CHROMOSOME_ARRAY,
    ORIC_ARRAY, CHROMOSOME_DOMAIN_ARRAY, CHROMOSOMAL_SEGMENT_ARRAY,
    GENE_ARRAY, DNAA_BOX_ARRAY,
)

# Map full type strings to constant names
STRING_TO_CONST = {}
for name, val in [
    ('PROMOTER_ARRAY', PROMOTER_ARRAY),
    ('RNA_ARRAY', RNA_ARRAY),
    ('ACTIVE_RNAP_ARRAY', ACTIVE_RNAP_ARRAY),
    ('ACTIVE_RIBOSOME_ARRAY', ACTIVE_RIBOSOME_ARRAY),
    ('ACTIVE_REPLISOME_ARRAY', ACTIVE_REPLISOME_ARRAY),
    ('FULL_CHROMOSOME_ARRAY', FULL_CHROMOSOME_ARRAY),
    ('ORIC_ARRAY', ORIC_ARRAY),
    ('CHROMOSOME_DOMAIN_ARRAY', CHROMOSOME_DOMAIN_ARRAY),
    ('CHROMOSOMAL_SEGMENT_ARRAY', CHROMOSOMAL_SEGMENT_ARRAY),
    ('GENE_ARRAY', GENE_ARRAY),
    ('DNAA_BOX_ARRAY', DNAA_BOX_ARRAY),
]:
    STRING_TO_CONST[val] = name


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
}


def format_port_dict(d, indent=8):
    """Format a port dict as Python source, substituting constants."""
    lines = [' ' * indent + '{']
    for key, val in d.items():
        prefix = ' ' * (indent + 4)
        if isinstance(val, str):
            # Check if it matches a schema_types constant
            const = STRING_TO_CONST.get(val)
            if const:
                lines.append(f"{prefix}'{key}': {const},")
            else:
                lines.append(f"{prefix}'{key}': '{val}',")
        elif isinstance(val, dict):
            nested = format_port_dict(val, indent + 4)
            lines.append(f"{prefix}'{key}': {nested},")
        else:
            lines.append(f"{prefix}'{key}': {repr(val)},")
    lines.append(' ' * indent + '}')
    return '\n'.join(lines)


def generate_method(name, port_dict, indent=4):
    """Generate an inputs() or outputs() method."""
    prefix = ' ' * indent
    body = format_port_dict(port_dict, indent + 8)
    return f"""{prefix}def {name}(self):
{prefix}    return (
{body}
{prefix}    )
"""


def ensure_schema_types_import(content):
    """Add schema_types import if needed."""
    if 'from v2ecoli.library.schema_types import' in content:
        return content

    # Find what constants we need
    needed = set()
    for const_name in STRING_TO_CONST.values():
        if const_name in content:
            needed.add(const_name)

    if not needed:
        return content

    # Add import after last existing import
    lines = content.split('\n')
    last_import = 0
    for i, line in enumerate(lines):
        if line.startswith('from ') or line.startswith('import '):
            last_import = i

    import_line = f"from v2ecoli.library.schema_types import (\n    {', '.join(sorted(needed))})"
    lines.insert(last_import + 1, import_line)
    return '\n'.join(lines)


def main():
    with open('scripts/extracted_schemas.pickle', 'rb') as f:
        schemas = dill.load(f)

    for proc_name, filepath in sorted(PROCESS_FILES.items()):
        if not os.path.exists(filepath):
            print(f"  SKIP {proc_name}: not found")
            continue

        schema = schemas.get(proc_name)
        if not schema:
            print(f"  SKIP {proc_name}: not in schemas")
            continue

        with open(filepath) as f:
            content = f.read()

        inputs_dict = schema['inputs']
        outputs_dict = schema['outputs']

        # Generate method source
        inputs_src = generate_method('inputs', inputs_dict)
        outputs_src = generate_method('outputs', outputs_dict)

        # Find the class body — insert before calculate_request or ports_schema
        # Look for the first method after __init__
        insert_markers = [
            'def ports_schema(self)',
            'def calculate_request(self',
        ]

        inserted = False
        for marker in insert_markers:
            if marker in content:
                # Insert before this marker
                indent_match = re.search(r'^( +)' + re.escape(marker), content, re.MULTILINE)
                if indent_match:
                    content = content.replace(
                        indent_match.group(0),
                        inputs_src + '\n' + outputs_src + '\n' + indent_match.group(0)
                    )
                    inserted = True
                    break

        if not inserted:
            print(f"  SKIP {proc_name}: no insertion point found")
            continue

        # Check which constants are needed and add import
        needed_consts = set()
        for const_val, const_name in STRING_TO_CONST.items():
            if const_name in content:
                needed_consts.add(const_name)

        if needed_consts and 'from v2ecoli.library.schema_types import' not in content:
            lines = content.split('\n')
            last_import = 0
            for i, line in enumerate(lines):
                if line.startswith('from ') or line.startswith('import '):
                    last_import = i
            import_str = "from v2ecoli.library.schema_types import (\n    " + ', '.join(sorted(needed_consts)) + ")"
            lines.insert(last_import + 1, import_str)
            content = '\n'.join(lines)

        with open(filepath, 'w') as f:
            f.write(content)

        print(f"  {proc_name:40s} added inputs({len(inputs_dict)} ports) outputs({len(outputs_dict)} ports)")

    print("\nDone. Run benchmark to verify.")


if __name__ == '__main__':
    main()
