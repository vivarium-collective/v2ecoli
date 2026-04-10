"""
Lightweight EcoliStep/EcoliProcess adapters for pure process-bigraph.

These bridge the vEcoli class signature (parameters=None) with
process-bigraph's Step/Process (config, core=) without importing
vivarium-core.
"""

from process_bigraph import Step, Process


# Thread-local core storage so __init__(parameters) can find it
_CURRENT_CORE = None


def _apply_defaults(cls, params):
    """Merge config_schema defaults into params.

    config_schema entries can be:
    - str: type expression (no default)
    - dict with '_default': has a default value
    - str like 'boolean{false}': inline default in braces
    """
    merged = {}
    config_schema = getattr(cls, 'config_schema', {}) or {}
    for key, spec in config_schema.items():
        if isinstance(spec, dict) and '_default' in spec:
            merged[key] = spec['_default']
        elif isinstance(spec, str) and '{' in spec:
            brace_start = spec.index('{')
            default_str = spec[brace_start + 1:spec.rindex('}')]
            type_name = spec[:brace_start]
            if type_name == 'boolean':
                merged[key] = default_str.lower() == 'true'
            elif type_name in ('integer', 'int'):
                merged[key] = int(default_str)
            elif type_name in ('float',):
                merged[key] = float(default_str)
            elif type_name == 'string':
                merged[key] = default_str
            else:
                merged[key] = default_str
    if params:
        merged.update(params)
    return merged


class EcoliStep(Step):
    """Adapter: accepts vEcoli's ``parameters`` kwarg, delegates to PBG Step.

    vEcoli processes have complex config_schema with unum/method types that
    don't round-trip through PBG's realize. We skip PBG config processing
    and just store parameters directly — the instances are created in
    generate.py and used by Composite via their interface() method.
    """

    config_schema = {}

    def __init__(self, parameters=None, config=None, core=None, **kwargs):
        global _CURRENT_CORE
        params = parameters if parameters is not None else config
        if params is None:
            params = {}
        if core is None:
            core = _CURRENT_CORE
        self.parameters = _apply_defaults(self.__class__, params)
        self.core = core

    def next_update(self, timestep, states):
        """vEcoli-style entry point — delegates to update()."""
        return self.update(states, interval=timestep)

    def update(self, state, interval=None):
        return {}

    def inputs(self):
        return {}

    def outputs(self):
        return {}

    def interface(self):
        return {'inputs': self.inputs(), 'outputs': self.outputs()}


class EcoliProcess(Process):
    """Adapter: accepts vEcoli's ``parameters`` kwarg, delegates to PBG Process."""

    config_schema = {}

    def __init__(self, parameters=None, config=None, core=None, **kwargs):
        global _CURRENT_CORE
        params = parameters if parameters is not None else config
        if params is None:
            params = {}
        if core is None:
            core = _CURRENT_CORE
        self.parameters = _apply_defaults(self.__class__, params)
        self.core = core

    def next_update(self, timestep, states):
        return self.update(states, interval=timestep)

    def update(self, state, interval=None):
        return {}

    def inputs(self):
        return {}

    def outputs(self):
        return {}

    def interface(self):
        return {'inputs': self.inputs(), 'outputs': self.outputs()}


def set_current_core(core):
    """Set the core used by EcoliStep/EcoliProcess when core is not passed."""
    global _CURRENT_CORE
    _CURRENT_CORE = core
