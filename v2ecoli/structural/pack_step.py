"""EcoliPackStep — write a parsimony 3D pack of the live cell at declared
simulation times. Appended as a final execution layer to a baseline run; it
runs every tick, packs a snapshot the first time its scheduled time arrives.
"""
from __future__ import annotations

from process_bigraph import Step

from v2ecoli.structural.build import pack_from_state, bulk_to_counts


def _default_core():
    """Lightweight bigraph-schema core (base + process_bigraph types only) —
    enough to fill this Step's plain config/port schemas without paying for
    the full ``v2ecoli.core.build_core()`` (which imports the whole
    ECOLI_TYPES registry + process tree). Used only when no ``core`` is
    supplied (standalone construction / tests); when embedded in a real
    composite the framework passes its own core via ``core.register_link``.
    """
    import process_bigraph
    from bigraph_schema import allocate_core

    core = allocate_core()
    process_bigraph.register_types(core)
    return core


class EcoliPackStep(Step):
    """Pack a parsimony 3D structural snapshot of the live cell the first
    time each declared simulation time (or ``"division_time"``) arrives.

    ``config["snapshots"]`` maps a snapshot name to either a fixed sim-time
    (float, seconds) or the string ``"division_time"``, which resolves against
    the ``full_chromosomes`` port's reported division time (set once the cell
    commits to division). Each name fires at most once, within
    ``config["epsilon_s"]`` of its target time.
    """

    # NOTE: this bigraph-schema version has no registered ``any``/``tree[any]``
    # type (parsing "tree[any]" raises — "any" isn't in the type registry), so
    # loosely-shaped config/ports use ``object`` (a generic, unvalidated leaf)
    # instead, mirroring cell_shape.ShapeStep's config_schema dict-form style.
    config_schema = {
        "snapshots": "object",          # {name: float sim-time | "division_time"}
        "study": "string",
        "out_dir": "string",
        "top_n": {"_type": "integer", "_default": 40},
        "scale": {"_type": "float", "_default": 0.3},
        "epsilon_s": {"_type": "float", "_default": 1.0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core or _default_core())
        if not self.config.get("out_dir"):
            self.config["out_dir"] = f"out/pack/{self.config.get('study') or 'snapshot'}"
        self._fired = set()

    def inputs(self):
        return {"bulk": "object", "shape": "object",
                "global_time": "float", "full_chromosomes": "object"}

    def outputs(self):
        return {"pack_status": "map[float]"}

    def _due(self, name, spec, t, states):
        if name in self._fired:
            return False
        if isinstance(spec, str) and spec == "division_time":
            dt = (states.get("full_chromosomes") or {}).get("division_time")
            if not dt:                     # not scheduled yet
                return False
            return t >= float(dt) - self.config["epsilon_s"]
        return t >= float(spec)             # fixed sim-time

    def update(self, state, interval=None):
        t = float(state.get("global_time") or 0.0)
        status = {}
        for name, spec in (self.config.get("snapshots") or {}).items():
            if not self._due(name, spec, t, state):
                continue
            counts = bulk_to_counts(state.get("bulk"))
            volume_fl = float((state.get("shape") or {}).get("volume_fl") or 0.0)
            res = pack_from_state(self.config["out_dir"], name, counts, volume_fl,
                                  top_n=self.config["top_n"], scale=self.config["scale"])
            self._fired.add(name)
            status[name] = float(len((res or {}).get("placements") or []))
        return {"pack_status": status} if status else {}
