"""Tests for the automatic vivarium-1.0 -> process-bigraph converter.

Covers ``translate_ports`` (ports_schema -> typed-port inference) and
``wrap_vivarium_process`` (running an unmodified vivarium process on the
process-bigraph runtime). The fake process is duck-typed — no vivarium-core
dependency, mirroring how a real v1 process would be added to this repo.
"""

import numpy as np

from v2ecoli.core import build_core
from v2ecoli.library.ecoli_step import set_current_core
from v2ecoli.library.vivarium_bridge import (
    translate_ports,
    wrap_vivarium_process,
)


class FakeV1Process:
    """A minimal vivarium-1.0-style process (duck-typed, no vivarium import)."""

    name = "fake-v1"
    defaults = {"rate": 2.0}

    def __init__(self, parameters=None):
        self.parameters = {**self.defaults, **(parameters or {})}
        self.rate = self.parameters["rate"]

    def ports_schema(self):
        return {
            "counts": {"_default": 0, "_updater": "accumulate"},
            "flag": {"_default": True, "_updater": "set"},
            "rates": {"_default": [0.0, 0.0, 0.0]},
            "nested": {
                "x": {"_default": 1},
                "y": {"_default": "hello"},
            },
        }

    def next_update(self, timestep, states):
        return {"counts": int(self.rate * timestep)}


def test_translate_ports_infers_scalar_and_list_types():
    core = build_core()
    typed = translate_ports(core, FakeV1Process().ports_schema())

    assert typed["counts"] == {"_type": "integer", "_default": 0}
    assert typed["rates"] == {"_type": "list[float]", "_default": [0.0, 0.0, 0.0]}
    assert typed["nested"]["x"] == {"_type": "integer", "_default": 1}
    assert typed["nested"]["y"] == {"_type": "string", "_default": "hello"}


def test_translate_ports_set_updater_becomes_overwrite():
    core = build_core()
    typed = translate_ports(core, FakeV1Process().ports_schema())
    # _updater == 'set' wraps the inferred type in overwrite[...]
    assert typed["flag"] == {"_type": "overwrite[boolean]", "_default": True}


def test_translate_ports_empty_tuple_default_normalized():
    core = build_core()
    typed = translate_ports(core, {"items": {"_default": (), "_updater": "set"}})
    assert typed["items"]["_default"] == []
    assert typed["items"]["_type"].startswith("overwrite[")


def test_translate_ports_special_store_names():
    core = build_core()
    ports = {
        "bulk": {"_default": np.zeros(3), "_updater": "accumulate"},
        "RNAs": {"_default": 0, "_updater": "set"},
    }
    typed = translate_ports(core, ports)
    # Well-known stores map to registered array types, ignoring raw default.
    assert typed["bulk"] == "bulk_array"
    assert typed["RNAs"].startswith("unique_array[")


def test_wrap_vivarium_process_inputs_outputs():
    core = build_core()
    set_current_core(core)
    Wrapped = wrap_vivarium_process(FakeV1Process)
    inst = Wrapped({"rate": 3.0}, core=core)

    assert inst.name == "fake-v1"
    ins = inst.inputs()
    assert ins["counts"] == {"_type": "integer", "_default": 0}
    assert ins["flag"] == {"_type": "overwrite[boolean]", "_default": True}
    # No output_ports restriction -> outputs mirror inputs.
    assert inst.outputs() == ins


def test_wrap_vivarium_process_update_delegates_to_next_update():
    core = build_core()
    Wrapped = wrap_vivarium_process(FakeV1Process)
    inst = Wrapped({"rate": 3.0}, core=core)
    # update(state, interval) -> v1 next_update(timestep=interval, states)
    assert inst.update({"counts": 0}, interval=2) == {"counts": 6}


def test_wrap_vivarium_process_output_ports_restricts_writes():
    core = build_core()
    Wrapped = wrap_vivarium_process(FakeV1Process, output_ports=["counts"])
    inst = Wrapped({}, core=core)
    assert set(inst.outputs()) == {"counts"}
    # inputs still expose every port for reading
    assert "flag" in inst.inputs()
