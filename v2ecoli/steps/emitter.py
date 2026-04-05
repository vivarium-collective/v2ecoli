"""
RAMEmitter step for v2ecoli.

Collects simulation state at each timestep into an in-memory history.
Wired as the last step in the flow so it sees the fully updated state.
"""

import copy


class RAMEmitter:
    """Collects specified ports into an in-memory history list."""

    name = "emitter"

    def __init__(self, parameters=None, config=None, **kwargs):
        params = parameters or config or {}
        self.parameters = params
        self.emit_keys = params.get('emit_keys', {})
        self.history = []

    def ports_schema(self):
        ports = {}
        for key, port_config in self.emit_keys.items():
            if isinstance(port_config, dict):
                ports[key] = port_config
            else:
                ports[key] = {'_default': port_config}
        return ports

    def next_update(self, timestep, states):
        snapshot = {}
        for key in self.emit_keys:
            val = states.get(key)
            if val is not None:
                snapshot[key] = copy.deepcopy(val)
        self.history.append(snapshot)
        return {}
