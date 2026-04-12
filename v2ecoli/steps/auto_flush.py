"""
Auto-flush helper for unique molecule updates.

Instead of requiring separate UniqueUpdate steps after each execution
layer, steps can inject {"update": True} into their unique molecule
port updates to trigger the flush within the same apply cycle.
"""

from v2ecoli.library.ecoli_step import EcoliStep as Step

# All unique molecule port names used by processes (v1 plural names)
ALL_UNIQUE_PORTS = [
    'full_chromosomes', 'chromosome_domains', 'active_replisomes',
    'oriCs', 'promoters', 'chromosomal_segments', 'DnaA_boxes',
    'active_RNAPs', 'RNAs', 'genes', 'active_ribosome',
]


def inject_unique_flush(update, unique_ports=None, flush_all=False):
    """Append {"update": True} to unique molecule ports in the update dict.

    When flush_all=True (used by Evolver/DepartitionedStep/ReconciledStep
    which have all unique ports wired), adds flush for ALL unique ports.
    When flush_all=False (used by standalone steps), only adds flush to
    unique ports that already have pending updates in the dict.

    Args:
        update: The update dict returned by a step's update() method.
        unique_ports: List of unique port names to flush. If None, uses
            ALL_UNIQUE_PORTS.
        flush_all: If True, flush all ports. If False, only flush ports
            that already have data in the update dict.

    Returns:
        The update dict with flush signals injected.
    """
    if unique_ports is None:
        unique_ports = ALL_UNIQUE_PORTS

    for port in unique_ports:
        if port in update:
            if isinstance(update[port], dict):
                update[port]["update"] = True
        elif flush_all:
            update[port] = {"update": True}

    return update


class AutoFlushStep(Step):
    """Wraps a standalone step and auto-flushes unique molecule updates.

    Used for steps like TfBinding, TfUnbinding, ChromosomeStructure that
    modify unique molecules but aren't wrapped by Evolver/DepartitionedStep.
    """

    config_schema = {
        'step': 'node',
    }

    def inputs(self):
        step = self.parameters.get('step')
        ports = step.inputs() if step else {}
        # Add unique molecule ports for flush
        for port in ALL_UNIQUE_PORTS:
            if port not in ports:
                ports[port] = 'node'
        return ports

    def outputs(self):
        step = self.parameters.get('step')
        ports = step.outputs() if step else {}
        for port in ALL_UNIQUE_PORTS:
            if port not in ports:
                ports[port] = 'node'
        return ports

    def initialize(self, config):
        step = self.parameters.get('step')
        if step:
            self.parameters['name'] = getattr(step, 'name', 'auto_flush')

    def port_defaults(self):
        step = self.parameters.get('step')
        if step and hasattr(step, 'port_defaults'):
            return step.port_defaults()
        return {}

    def update_condition(self, timestep, states):
        step = self.parameters.get('step')
        if step and hasattr(step, 'update_condition'):
            return step.update_condition(timestep, states)
        return True

    def update(self, states, interval=None):
        step = self.parameters.get('step')
        if step is None:
            return {}
        if hasattr(step, 'next_update'):
            update = step.next_update(interval or 1.0, states)
        else:
            update = step.update(states, interval)
        inject_unique_flush(update)
        return update
