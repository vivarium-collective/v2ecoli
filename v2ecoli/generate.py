"""
Document generation for v2ecoli.

This module requires vEcoli to be installed. It uses EcoliSim to build
the initial state and process instances, then saves them as a pickle
document for the self-contained runtime in composite.py.

This is the ONLY module that imports from vEcoli/wholecell.
"""

import os
import copy

import dill

from contextlib import chdir

from bigraph_schema import deep_merge

from wholecell.utils.filepath import ROOT_PATH
from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH


def list_paths(path):
    if isinstance(path, tuple):
        return list(path)
    elif isinstance(path, dict):
        return {key: list_paths(subpath) for key, subpath in path.items()}


def extract_flow_priorities(flow):
    order = list(flow.keys())
    n = len(order)
    return {step_name: float(n - i) for i, step_name in enumerate(order)}


def inject_flow_dependencies(cell_state, flow):
    order = list(flow.keys())
    for i, step_name in enumerate(order):
        edge = cell_state.get(step_name)
        if not isinstance(edge, dict):
            continue
        if i == 0:
            edge.setdefault('inputs', {}).setdefault('global_time', ['global_time'])
        if i > 0:
            edge.setdefault('inputs', {})[f'_flow_in_{i}'] = [f'_flow_token_{i-1}']
        if i < len(order) - 1:
            edge.setdefault('outputs', {})[f'_flow_out_{i}'] = [f'_flow_token_{i}']


def translate_processes(tree, topology=None, edge_type=None):
    """Translate v1 process/step instances into edge dicts."""
    from vivarium.core.process import Process as VivariumProcess, Step as VivariumStep
    from bigraph_schema import Edge as BigraphEdge

    if isinstance(tree, (VivariumProcess, VivariumStep, BigraphEdge)):
        instance_topology = getattr(tree, 'topology', None)
        if instance_topology:
            topology = instance_topology
        elif topology is None:
            topology = {}
        wires = list_paths(topology)

        if edge_type == 'process':
            state = {'interval': 1.0}
        else:
            state = {'priority': 1.0}

        state.update({
            '_type': 'step' if edge_type != 'process' else 'process',
            'instance': tree,
            'inputs': copy.deepcopy(wires),
            'outputs': copy.deepcopy(wires),
        })
        return state
    elif isinstance(tree, dict):
        return {key: translate_processes(subtree,
                    topology[key] if topology else None,
                    edge_type=edge_type)
                for key, subtree in tree.items()}
    else:
        return tree


def migrate_composite(sim):
    """Build the composite state from EcoliSim."""
    processes = translate_processes(
        sim.ecoli.processes, sim.ecoli.topology, edge_type='process')
    steps = translate_processes(
        sim.ecoli.steps, sim.ecoli.topology, edge_type='step')

    state = deep_merge(processes, steps)
    state = deep_merge(state, sim.generated_initial_state)

    flow = sim.ecoli.flow
    for path_key, substates in state.items():
        if isinstance(substates, dict):
            subflow = flow.get(path_key, {})
            for subkey, subsubstates in substates.items():
                if isinstance(subsubstates, dict):
                    inner_flow = subflow.get(subkey, {})
                    if inner_flow:
                        priorities = extract_flow_priorities(inner_flow)
                        for step_name, priority in priorities.items():
                            if isinstance(subsubstates.get(step_name), dict):
                                subsubstates[step_name]['priority'] = priority
                        inject_flow_dependencies(subsubstates, inner_flow)
    return state


def generate_document(outpath='out/ecoli.pickle'):
    """Build the E. coli composite from EcoliSim and save as a document.

    This is the only function that requires vEcoli to be installed.

    Args:
        outpath: Path for the output file.

    Returns:
        The path to the saved file.
    """
    with chdir(ROOT_PATH):
        sim = EcoliSim.from_file(CONFIG_DIR_PATH + "default.json")
        sim.build_ecoli()

    state = migrate_composite(sim)

    # Extract flow order
    flow = sim.ecoli.flow
    flow_order = []
    for path_key in state:
        if isinstance(state[path_key], dict):
            subflow = flow.get(path_key, {})
            for subkey in state[path_key]:
                inner_flow = subflow.get(subkey, {})
                if inner_flow:
                    flow_order = list(inner_flow.keys())
                    break

    document = {'state': state, 'flow_order': flow_order}

    os.makedirs(os.path.dirname(outpath) or '.', exist_ok=True)
    with open(outpath, 'wb') as f:
        dill.dump(document, f)

    print(f"Saved document to {outpath}")
    return outpath
