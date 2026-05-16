"""
Step 4: Add port_defaults() methods and update _seed_state_from_ports.

For each process, adds a port_defaults() method that returns the _default
values from ports_schema (used for initial state seeding).

Then updates generate.py to call port_defaults() instead of ports_schema().
"""

import os
import sys
import re
import dill

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

ALL_FILES = {
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
    'ecoli-mass-listener': 'v2ecoli/steps/listeners/mass_listener.py',
    'post-division-mass-listener': 'v2ecoli/steps/listeners/mass_listener.py',
    'RNA_counts_listener': 'v2ecoli/steps/listeners/rna_counts.py',
    'monomer_counts_listener': 'v2ecoli/steps/listeners/monomer_counts.py',
    'rna_synth_prob_listener': 'v2ecoli/steps/listeners/rna_synth_prob.py',
    'dna_supercoiling_listener': 'v2ecoli/steps/listeners/dna_supercoiling.py',
    'replication_data_listener': 'v2ecoli/steps/listeners/replication_data.py',
    'rnap_data_listener': 'v2ecoli/steps/listeners/rnap_data.py',
    'ribosome_data_listener': 'v2ecoli/steps/listeners/ribosome_data.py',
    'unique_molecule_counts': 'v2ecoli/steps/listeners/unique_molecule_counts.py',
}


def format_defaults_dict(d, indent=8):
    """Format a port_defaults dict as Python source."""
    import numpy as np

    def fmt(value, depth=0):
        prefix = ' ' * (indent + depth * 4)
        if isinstance(value, dict):
            if len(value) == 0:
                return '{}'
            lines = ['{']
            for k, v in value.items():
                formatted = fmt(v, depth + 1)
                lines.append(f"{prefix}    '{k}': {formatted},")
            lines.append(f"{prefix}}}")
            return '\n'.join(lines)
        if isinstance(value, (int, float, bool)):
            return repr(value)
        if isinstance(value, str):
            return repr(value)
        if isinstance(value, list):
            return repr(value)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return 'np.array([])'
            if value.size <= 5:
                return f'np.array({value.tolist()})'
            return f'np.zeros({value.shape[0]})'
        return repr(value)

    return fmt(d)


def generate_port_defaults_method(port_defaults, indent=4):
    """Generate port_defaults() method source."""
    prefix = ' ' * indent
    body = format_defaults_dict(port_defaults, indent + 4)
    return f"""{prefix}def port_defaults(self):
{prefix}    \"\"\"Default values for ports that need pre-population.\"\"\"
{prefix}    return {body}
"""


def main():
    with open('scripts/extracted_schemas.pickle', 'rb') as f:
        schemas = dill.load(f)

    # Track which files we've already modified (some share files)
    modified = set()

    for proc_name, filepath in sorted(ALL_FILES.items()):
        if not os.path.exists(filepath):
            continue

        schema = schemas.get(proc_name)
        if not schema:
            continue

        pd = schema.get('port_defaults', {})
        if not pd:
            continue

        if filepath in modified:
            # Already added to this file (e.g. mass_listener has both classes)
            continue

        with open(filepath) as f:
            content = f.read()

        # Skip if already has port_defaults
        if 'def port_defaults(self)' in content:
            print(f"  SKIP {proc_name}: already has port_defaults")
            continue

        # Insert before ports_schema or before calculate_request or at end of class
        markers = ['def ports_schema(self)', 'def calculate_request(self',
                   'def update(self,', 'def next_update(self,']
        inserted = False
        for marker in markers:
            if marker in content:
                # Find with indentation
                match = re.search(r'^( +)' + re.escape(marker), content, re.MULTILINE)
                if match:
                    method_src = generate_port_defaults_method(pd)
                    content = content.replace(
                        match.group(0),
                        method_src + '\n' + match.group(0)
                    )
                    inserted = True
                    break

        if inserted:
            with open(filepath, 'w') as f:
                f.write(content)
            modified.add(filepath)
            print(f"  {proc_name:40s} added port_defaults({len(pd)} keys)")
        else:
            print(f"  SKIP {proc_name}: no insertion point")

    # Update generate.py: _seed_state_from_ports -> _seed_state_from_defaults
    gpath = 'v2ecoli/generate.py'
    with open(gpath) as f:
        content = f.read()

    old_fn = """def _seed_state_from_ports(cell_state):
    \"\"\"Walk each edge's ports_schema and inject _default values.

    Ported from vEcoli ecoli_composite.py. Fills empty/missing slots
    so that listeners and other steps don't KeyError on first tick.
    \"\"\"
    for edge in list(cell_state.values()):
        if not (isinstance(edge, dict) and 'instance' in edge):
            continue
        instance = edge['instance']
        try:
            ports = instance.ports_schema()
        except (AttributeError, Exception):
            continue
        for port_name, wire_path in edge.get('inputs', {}).items():
            port = ports.get(port_name)
            if isinstance(port, dict) and isinstance(wire_path, list):
                _inject_port_default(cell_state, wire_path, port)"""

    new_fn = """def _seed_state_from_defaults(cell_state):
    \"\"\"Walk each edge's port_defaults and inject values into cell_state.

    Replaces ports_schema-based seeding with explicit port_defaults().
    \"\"\"
    for edge in list(cell_state.values()):
        if not (isinstance(edge, dict) and 'instance' in edge):
            continue
        instance = edge['instance']
        try:
            defaults = instance.port_defaults()
        except (AttributeError, Exception):
            continue
        for port_name, wire_path in edge.get('inputs', {}).items():
            default = defaults.get(port_name)
            if default is not None and isinstance(wire_path, list):
                if isinstance(default, dict):
                    _inject_port_default(cell_state, wire_path, default)
                else:
                    _inject_port_default(cell_state, wire_path, {'_default': default})"""

    if old_fn in content:
        content = content.replace(old_fn, new_fn)
        content = content.replace('_seed_state_from_ports(cell_state)', '_seed_state_from_defaults(cell_state)')
        with open(gpath, 'w') as f:
            f.write(content)
        print(f"\n  Updated generate.py: _seed_state_from_ports -> _seed_state_from_defaults")
    else:
        print(f"\n  generate.py: function not found for replacement")

    print("\nDone. Run benchmark to verify.")


if __name__ == '__main__':
    main()
