"""
Lightweight EcoliStep/EcoliProcess adapters for pure process-bigraph.

These bridge the vEcoli class signature (parameters=None) with
process-bigraph's Step/Process (config, core=) without importing
vivarium-core.
"""

from process_bigraph import Step, Process


# Thread-local core storage so __init__(parameters) can find it
_CURRENT_CORE = None


def _defaults_from_schema(config_schema):
    """Derive defaults dict from config_schema inline defaults."""
    result = {}
    for key, spec in (config_schema or {}).items():
        if isinstance(spec, dict) and '_default' in spec:
            result[key] = spec['_default']
        elif isinstance(spec, str) and '{' in spec:
            type_str, _, default_str = spec.partition('{')
            default_str = default_str.rstrip('}')
            type_str = type_str.strip()
            if type_str == 'float':
                result[key] = float(default_str)
            elif type_str == 'integer':
                result[key] = int(default_str)
            elif type_str == 'boolean':
                result[key] = default_str.lower() in ('true', '1', 'yes')
            elif type_str == 'string':
                result[key] = default_str
            else:
                result[key] = default_str
    return result


def _build_parameters(cls, params):
    """Build parameters dict, mimicking vivarium's defaults merge.

    Priority: params > class.defaults > config_schema inline defaults
    """
    merged = {}
    # 1. config_schema inline defaults
    merged.update(_defaults_from_schema(getattr(cls, 'config_schema', {})))
    # 2. class defaults dict (vEcoli convention)
    merged.update(getattr(cls, 'defaults', {}) or {})
    # 3. provided params
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
        self.parameters = _build_parameters(self.__class__, params)
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

    def invoke(self, state, interval=None):
        from process_bigraph.composite import SyncUpdate
        update = self.update(state, interval)
        return SyncUpdate(update)


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
        self.parameters = _build_parameters(self.__class__, params)
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

    def get_schema(self):
        """Vivarium-compatible: return ports_schema dict."""
        return self.ports_schema()


def set_current_core(core):
    """Set the core used by EcoliStep/EcoliProcess when core is not passed."""
    global _CURRENT_CORE
    _CURRENT_CORE = core
