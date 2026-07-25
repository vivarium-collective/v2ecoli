"""Batch B4 process-contract tests — metabolism / regulation.

Asserts each class in the batch declares a well-formed ProcessContract:
(a) it is a ProcessContract; (b) inputs and outputs are non-empty;
(c) every declared input/output key is a real port; (d) every config key
exists in the class config_schema; (e) every symbol appears in a math line
or in the class/contract description.
"""

import pytest

from bigraph_schema.contract import ProcessContract

from v2ecoli.processes.metabolism import Metabolism
from v2ecoli.processes.two_component_system import TwoComponentSystem
from v2ecoli.processes.protein_degradation import ProteinDegradation
from v2ecoli.steps.derivers.counts_deriver import CountsDeriver
from v2ecoli.steps.division import Division, MarkDPeriod


CLASSES = [
    Metabolism,
    TwoComponentSystem,
    ProteinDegradation,
    CountsDeriver,
    Division,
    MarkDPeriod,
]


def _real_ports(cls, which):
    """Real port names for `which` ('inputs'/'outputs').

    Prefer calling the bound method on a bare (uninitialized) instance — the
    batch's port methods build literal dicts and don't touch init-time state.
    If that raises (e.g. CountsDeriver.outputs references init-time shape
    attrs), fall back to the class topology declaration.
    """
    inst = object.__new__(cls)
    try:
        d = getattr(inst, which)()
        if isinstance(d, dict) and d:
            return set(d.keys())
    except Exception:
        pass
    topo = getattr(cls, "topology", {}) or {}
    keys = set(topo.keys())
    if which == "outputs":
        keys.add("agents")  # structural division output not in topology
    return keys


def _symbol_haystack(cls):
    c = cls.contract
    return (
        "\n".join(c.math)
        + "\n"
        + (getattr(cls, "description", "") or "")
        + "\n"
        + (c.description or "")
    )


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_declares_contract(cls):
    assert isinstance(getattr(cls, "contract", None), ProcessContract)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_inputs_outputs_nonempty(cls):
    c = cls.contract
    assert c.inputs, f"{cls.__name__} has empty contract.inputs"
    assert c.outputs, f"{cls.__name__} has empty contract.outputs"


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_ports_are_real(cls):
    c = cls.contract
    real_in = _real_ports(cls, "inputs")
    real_out = _real_ports(cls, "outputs")
    for port in c.inputs:
        assert port in real_in, (
            f"{cls.__name__} contract input '{port}' not a real port {sorted(real_in)}"
        )
    for port in c.outputs:
        assert port in real_out, (
            f"{cls.__name__} contract output '{port}' not a real port {sorted(real_out)}"
        )


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_config_keys_exist(cls):
    c = cls.contract
    schema = getattr(cls, "config_schema", {}) or {}
    for key in c.config:
        assert key in schema, (
            f"{cls.__name__} contract config '{key}' not in config_schema"
        )


@pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
def test_symbols_appear_in_math_or_description(cls):
    c = cls.contract
    hay = _symbol_haystack(cls)
    for sym in c.symbols:
        assert sym in hay, (
            f"{cls.__name__} symbol '{sym}' absent from math/description"
        )
