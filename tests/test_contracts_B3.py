"""Contract tests for the REPLICATION & CHROMOSOME batch (B3).

Asserts that each process in this batch declares a well-formed
``ProcessContract`` whose ports/config/symbols are grounded in the real class.
"""

import pytest

from bigraph_schema.contract import ProcessContract

from v2ecoli.processes.chromosome_replication import ChromosomeReplication
from v2ecoli.processes.chromosome_structure import ChromosomeStructure
from v2ecoli.processes.equilibrium import Equilibrium

B3_CLASSES = [ChromosomeReplication, ChromosomeStructure, Equilibrium]


def _ports(cls):
    """Return (input_ports, output_ports) without running ``initialize``.

    ``inputs()``/``outputs()`` are effectively static (they don't touch
    instance state), so we call them on an uninitialized instance to avoid
    needing a full config.
    """
    inst = cls.__new__(cls)
    return set(inst.inputs().keys()), set(inst.outputs().keys())


@pytest.mark.parametrize("cls", B3_CLASSES, ids=[c.__name__ for c in B3_CLASSES])
def test_declares_process_contract(cls):
    assert isinstance(cls.contract, ProcessContract)


@pytest.mark.parametrize("cls", B3_CLASSES, ids=[c.__name__ for c in B3_CLASSES])
def test_inputs_outputs_nonempty(cls):
    assert cls.contract.inputs, f"{cls.__name__}.contract.inputs is empty"
    assert cls.contract.outputs, f"{cls.__name__}.contract.outputs is empty"


@pytest.mark.parametrize("cls", B3_CLASSES, ids=[c.__name__ for c in B3_CLASSES])
def test_input_keys_are_real_ports(cls):
    input_ports, _ = _ports(cls)
    unknown = set(cls.contract.inputs) - input_ports
    assert not unknown, f"{cls.__name__} contract.inputs has non-ports: {unknown}"


@pytest.mark.parametrize("cls", B3_CLASSES, ids=[c.__name__ for c in B3_CLASSES])
def test_output_keys_are_real_ports(cls):
    _, output_ports = _ports(cls)
    unknown = set(cls.contract.outputs) - output_ports
    assert not unknown, f"{cls.__name__} contract.outputs has non-ports: {unknown}"


@pytest.mark.parametrize("cls", B3_CLASSES, ids=[c.__name__ for c in B3_CLASSES])
def test_config_keys_exist_in_schema(cls):
    unknown = set(cls.contract.config) - set(cls.config_schema)
    assert not unknown, f"{cls.__name__} contract.config has unknown params: {unknown}"


@pytest.mark.parametrize("cls", B3_CLASSES, ids=[c.__name__ for c in B3_CLASSES])
def test_symbols_grounded_in_math_or_description(cls):
    contract = cls.contract
    grounded_text = "\n".join(
        [cls.description or "", contract.description or "", *contract.math]
    )
    ungrounded = [s for s in contract.symbols if s not in grounded_text]
    assert not ungrounded, (
        f"{cls.__name__} contract symbols not found in description or math: {ungrounded}"
    )
