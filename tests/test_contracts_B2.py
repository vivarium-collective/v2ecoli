"""Contract-authoring tests for the TRANSLATION batch (B2).

Asserts each B2 process class declares a well-formed ``ProcessContract`` whose
ports, config keys, and symbols are all grounded in the actual class source
(topology port declarations, ``config_schema``, and the class ``description``).
"""

import pytest

from bigraph_schema.contract import ProcessContract

from v2ecoli.processes.polypeptide_initiation import PolypeptideInitiation
from v2ecoli.processes.polypeptide_elongation import (
    BasePolypeptideElongation,
    SteadyStatePolypeptideElongation,
)
from v2ecoli.processes.rna_degradation import RnaDegradation
from v2ecoli.processes.complexation import Complexation
from v2ecoli.processes.rna_maturation import RnaMaturation

B2_CLASSES = [
    PolypeptideInitiation,
    BasePolypeptideElongation,
    RnaDegradation,
    Complexation,
    RnaMaturation,
]


def _symbol_haystack(cls, contract):
    parts = [
        cls.description or "",
        contract.description or "",
        contract.summary or "",
    ]
    parts.extend(str(line) for line in contract.math)
    return "\n".join(parts)


@pytest.mark.parametrize("cls", B2_CLASSES, ids=lambda c: c.__name__)
def test_declares_process_contract(cls):
    assert isinstance(cls.contract, ProcessContract), (
        f"{cls.__name__} must declare a ProcessContract"
    )


@pytest.mark.parametrize("cls", B2_CLASSES, ids=lambda c: c.__name__)
def test_inputs_outputs_nonempty(cls):
    assert cls.contract.inputs, f"{cls.__name__}.contract.inputs is empty"
    assert cls.contract.outputs, f"{cls.__name__}.contract.outputs is empty"


@pytest.mark.parametrize("cls", B2_CLASSES, ids=lambda c: c.__name__)
def test_ports_are_real(cls):
    """Every contract input/output port must be a declared topology port."""
    real_ports = set(cls.topology.keys())
    for port in cls.contract.inputs:
        assert port in real_ports, (
            f"{cls.__name__} contract input '{port}' is not a real port "
            f"(topology ports: {sorted(real_ports)})"
        )
    for port in cls.contract.outputs:
        assert port in real_ports, (
            f"{cls.__name__} contract output '{port}' is not a real port "
            f"(topology ports: {sorted(real_ports)})"
        )


@pytest.mark.parametrize("cls", B2_CLASSES, ids=lambda c: c.__name__)
def test_config_keys_exist(cls):
    schema_keys = set(cls.config_schema.keys())
    for key in cls.contract.config:
        assert key in schema_keys, (
            f"{cls.__name__} contract config '{key}' not in config_schema"
        )


@pytest.mark.parametrize("cls", B2_CLASSES, ids=lambda c: c.__name__)
def test_symbols_grounded(cls):
    """Every symbol must appear in a math line or in the description text."""
    haystack = _symbol_haystack(cls, cls.contract)
    for sym in cls.contract.symbols:
        assert sym in haystack, (
            f"{cls.__name__} symbol '{sym}' not found in description/math"
        )


def test_elongation_contract_inherited_by_baseline_subclass():
    """The baseline uses SteadyStatePolypeptideElongation; it inherits the
    contract authored on BasePolypeptideElongation."""
    assert SteadyStatePolypeptideElongation.contract is (
        BasePolypeptideElongation.contract
    )
    assert isinstance(SteadyStatePolypeptideElongation.contract, ProcessContract)
