"""
Listener Bundle — runs all listeners as a single step.

Replaces 8 individual listener steps with one bundled step that
calls each listener's update() and merges listener outputs.
"""

from bigraph_schema import deep_merge
from v2ecoli.library.ecoli_step import EcoliStep as Step


class ListenerBundle(Step):
    """Runs all listener instances in sequence within a single step.

    Listeners are read-only with respect to bulk/unique state — they
    only write to the 'listeners' store. Bundling them eliminates
    per-step framework overhead (state gathering, update application).
    """

    config_schema = {
        'listeners': 'node',
    }

    def inputs(self):
        # Union of all listener inputs
        ports = {}
        for listener in self.parameters.get('listeners', []):
            try:
                li = listener.inputs()
            except Exception:
                li = {}
            for key, schema in li.items():
                if key not in ports:
                    ports[key] = schema
                elif isinstance(ports[key], dict) and isinstance(schema, dict):
                    ports[key] = deep_merge(ports[key], schema)
        ports.setdefault('global_time', 'float')
        ports.setdefault('timestep', 'float')
        return ports

    def outputs(self):
        # Union of all listener outputs
        ports = {}
        for listener in self.parameters.get('listeners', []):
            try:
                lo = listener.outputs()
            except Exception:
                lo = {}
            for key, schema in lo.items():
                if key not in ports:
                    ports[key] = schema
                elif isinstance(ports[key], dict) and isinstance(schema, dict):
                    ports[key] = deep_merge(ports[key], schema)
        return ports

    def initialize(self, config):
        self.parameters['name'] = 'listener_bundle'
        self.listener_instances = self.parameters.get('listeners', [])

    def port_defaults(self):
        merged = {}
        for listener in self.listener_instances:
            if hasattr(listener, 'port_defaults'):
                defaults = listener.port_defaults()
                if defaults:
                    merged = deep_merge(merged, defaults)
        return merged

    def update(self, states, interval=None):
        combined_listeners = {}
        for listener in self.listener_instances:
            try:
                if hasattr(listener, 'next_update'):
                    result = listener.next_update(
                        interval or 1.0, states)
                else:
                    result = listener.update(states, interval)
            except Exception as e:
                # Don't let one listener crash the whole bundle
                continue

            if result and 'listeners' in result:
                combined_listeners = deep_merge(
                    combined_listeners, result['listeners'])

        if combined_listeners:
            return {'listeners': combined_listeners}
        return {}
