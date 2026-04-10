"""
Extract process schemas from the running vEcoli composite.

Instantiates every process via vEcoli's EcoliSim + build_composite_native,
then extracts config_schema, defaults, inputs(), outputs(), ports_schema(),
and topology from each process instance. Saves as a pickle for use by the
migration scripts.

Usage:
    cd /path/to/v2ecoli
    python scripts/extract_schemas.py
"""

import os
import sys
import dill as pickle
import copy
import warnings

import numpy as np

# Must run from the v2ecoli directory but import from vEcoli
VECOLI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'vEcoli')
os.chdir(VECOLI_DIR)
sys.path.insert(0, VECOLI_DIR)
warnings.filterwarnings('ignore')

from ecoli.experiments.ecoli_master_sim import EcoliSim
from ecoli.composites.ecoli_composite import build_composite_native
from ecoli.library.bigraph_types import ECOLI_TYPES
from bigraph_schema import allocate_core


def extract_port_defaults(instance):
    """Extract _default values from ports_schema, recursively."""
    try:
        ps = instance.ports_schema()
    except Exception:
        return {}

    def walk(schema):
        result = {}
        for key, val in schema.items():
            if key.startswith('_'):
                continue
            if isinstance(val, dict):
                if '_default' in val:
                    result[key] = val['_default']
                else:
                    sub = walk(val)
                    if sub:
                        result[key] = sub
        return result

    return walk(ps)


def infer_type(value):
    """Infer a config_schema type string from a Python value."""
    if isinstance(value, bool):
        return 'boolean'
    if isinstance(value, int):
        return 'integer'
    if isinstance(value, float):
        return 'float'
    if isinstance(value, str):
        return 'string'
    if isinstance(value, np.ndarray):
        if value.dtype.kind == 'i':
            return 'array[integer]'
        elif value.dtype.kind == 'f':
            return 'array[float]'
        elif value.dtype.kind == 'b':
            return 'array[boolean]'
        elif value.dtype.kind == 'U' or value.dtype.kind == 'S':
            return 'array[string]'
        return 'array'
    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            return 'list[string]'
        if all(isinstance(v, int) for v in value):
            return 'list[integer]'
        return 'list'
    if isinstance(value, dict):
        return 'map'
    if callable(value):
        return 'method'
    if hasattr(value, 'asNumber'):  # Unum
        return 'unum'
    if hasattr(value, 'magnitude'):  # pint
        return 'quantity'
    if hasattr(value, 'toarray'):  # sparse matrix
        return 'csr_matrix'
    return 'any'


def merge_defaults_into_schema(config_schema, defaults):
    """Merge defaults into config_schema, producing {'_type': ..., '_default': ...} entries."""
    merged = {}
    all_keys = sorted(set(list(config_schema.keys()) + list(defaults.keys())))

    for key in all_keys:
        cs_entry = config_schema.get(key)
        default_val = defaults.get(key)

        if cs_entry is not None:
            if isinstance(cs_entry, dict) and '_type' in cs_entry:
                entry = dict(cs_entry)
                if default_val is not None and '_default' not in entry:
                    entry['_default'] = default_val
                merged[key] = entry
            elif isinstance(cs_entry, str):
                if default_val is not None:
                    merged[key] = {'_type': cs_entry, '_default': default_val}
                else:
                    merged[key] = cs_entry
            else:
                merged[key] = cs_entry
        elif default_val is not None:
            inferred = infer_type(default_val)
            merged[key] = {'_type': inferred, '_default': default_val}

    return merged


def safe_repr_inputs(d):
    """Convert inputs/outputs dict to a serializable representation."""
    result = {}
    for key, val in d.items():
        if isinstance(val, str):
            result[key] = val
        elif isinstance(val, dict):
            result[key] = safe_repr_inputs(val)
        else:
            result[key] = repr(val)
    return result


def main():
    print("Building vEcoli composite...")
    sim = EcoliSim.from_cli()
    sim.processes = sim._retrieve_processes(
        sim.processes, sim.add_processes, sim.exclude_processes, sim.swap_processes)
    sim.topology = sim._retrieve_topology(
        sim.topology, sim.processes, sim.swap_processes, sim.log_updates)
    sim.process_configs = sim._retrieve_process_configs(
        sim.process_configs, sim.processes)

    core = allocate_core()
    core.register_types(ECOLI_TYPES)
    state = build_composite_native(core, sim.config)
    cell = state['agents']['0']

    schemas = {}
    seen = set()

    for name, val in sorted(cell.items()):
        if not (isinstance(val, dict) and 'instance' in val):
            continue

        inst = val['instance']
        cls_name = type(inst).__name__

        # For Requester/Evolver, extract from wrapped PartitionedProcess
        if cls_name in ('Requester', 'Evolver'):
            wrapped = inst.parameters.get('process')
            if not wrapped:
                continue
            proc_name = getattr(wrapped, 'name', '')
            if proc_name in seen:
                continue
            seen.add(proc_name)
            target = wrapped
        else:
            proc_name = name
            if proc_name in seen:
                continue
            seen.add(proc_name)
            target = inst

        target_cls = type(target)
        cs = dict(getattr(target_cls, 'config_schema', {}) or {})
        defaults = dict(getattr(target_cls, 'defaults', {}) or {})
        merged_cs = merge_defaults_into_schema(cs, defaults)

        try:
            inputs = safe_repr_inputs(target.inputs())
        except Exception as e:
            inputs = {'_error': str(e)}

        try:
            outputs = safe_repr_inputs(target.outputs())
        except Exception as e:
            outputs = {'_error': str(e)}

        port_defaults = extract_port_defaults(target)
        topology = dict(getattr(target, 'topology', {}) or {})

        schemas[proc_name] = {
            'class_name': target_cls.__name__,
            'module': target_cls.__module__,
            'config_schema': cs,
            'defaults': defaults,
            'merged_config_schema': merged_cs,
            'inputs': inputs,
            'outputs': outputs,
            'port_defaults': port_defaults,
            'topology': topology,
        }

    # Save
    outdir = os.path.join(os.path.dirname(__file__), '..', 'scripts')
    outpath = os.path.join(outdir, 'extracted_schemas.pickle')
    with open(outpath, 'wb') as f:
        pickle.dump(schemas, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nExtracted {len(schemas)} process schemas to {outpath}")
    for proc, info in sorted(schemas.items()):
        cs_keys = len(info['merged_config_schema'])
        in_keys = len(info['inputs'])
        out_keys = len(info['outputs'])
        pd_keys = len(info['port_defaults'])
        print(f"  {proc:40s} {info['class_name']:25s} "
              f"cs={cs_keys:2d} in={in_keys:2d} out={out_keys:2d} pd={pd_keys:2d}")


if __name__ == '__main__':
    main()
