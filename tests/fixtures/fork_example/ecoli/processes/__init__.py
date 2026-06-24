"""Duck-typed fake fork — no vivarium-core dependency (mirrors the converter's
duck typing). Exposes a `process_registry` like vEcoli's ecoli.processes does."""


class _Registry:
    def __init__(self):
        self._d = {}
    def register(self, name, cls):
        self._d[name] = cls
    def access(self, name):
        if name not in self._d:
            raise KeyError(name)
        return self._d[name]


process_registry = _Registry()


class ExampleSecretion:
    """A simple vivarium-1.0-style process (ports_schema + next_update)."""
    name = "example-secretion"
    defaults = {"rate": 2.0}

    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}

    def ports_schema(self):
        return {"counts": {"_default": 0, "_updater": "accumulate"}}

    def next_update(self, timestep, states):
        return {"counts": int(self.parameters["rate"] * timestep)}


class BadPartitioned:
    """A partitioned process — must be rejected by classify_process."""
    name = "bad-partitioned"

    def __init__(self, parameters=None):
        self.parameters = parameters or {}

    def ports_schema(self):
        return {"bulk": {"_default": 0}}

    def calculate_request(self, timestep, states):
        return {}

    def evolve_state(self, timestep, states):
        return {}


process_registry.register(ExampleSecretion.name, ExampleSecretion)
process_registry.register(BadPartitioned.name, BadPartitioned)
