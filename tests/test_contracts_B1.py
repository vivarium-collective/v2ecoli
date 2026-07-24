"""Batch B1 (transcription) process-contract validation.

Asserts each transcription-cluster process declares a well-formed
``ProcessContract`` whose documented ports / config / symbols are all backed by
the real class source (topology port names, ``config_schema`` keys, and the
math/description text).
"""

import pytest

from bigraph_schema.contract import ProcessContract

from v2ecoli.processes.transcript_initiation import TranscriptInitiation
from v2ecoli.processes.transcript_elongation import TranscriptElongation
from v2ecoli.processes.tf_binding import TfBinding
from v2ecoli.processes.tf_unbinding import TfUnbinding
from v2ecoli.steps.ppgpp_initiation import PpgppInitiation


B1_CLASSES = [
    TranscriptInitiation,
    TranscriptElongation,
    TfBinding,
    TfUnbinding,
    PpgppInitiation,
]


def _ids(cls):
    return cls.__name__


@pytest.mark.parametrize("cls", B1_CLASSES, ids=_ids)
def test_declares_process_contract(cls):
    contract = cls.__dict__.get("contract")
    assert isinstance(contract, ProcessContract), (
        f"{cls.__name__} must declare a ProcessContract class attribute"
    )


@pytest.mark.parametrize("cls", B1_CLASSES, ids=_ids)
def test_inputs_outputs_nonempty(cls):
    contract = cls.contract
    assert contract.inputs, f"{cls.__name__}.contract.inputs is empty"
    assert contract.outputs, f"{cls.__name__}.contract.outputs is empty"


@pytest.mark.parametrize("cls", B1_CLASSES, ids=_ids)
def test_ports_are_real(cls):
    """Every documented input/output port must be a real topology port."""
    real_ports = set(cls.topology.keys())
    documented = set(cls.contract.inputs) | set(cls.contract.outputs)
    unknown = documented - real_ports
    assert not unknown, (
        f"{cls.__name__} contract names non-existent ports {sorted(unknown)}; "
        f"real ports are {sorted(real_ports)}"
    )


@pytest.mark.parametrize("cls", B1_CLASSES, ids=_ids)
def test_config_keys_exist(cls):
    """Every documented config key must exist in the class config_schema."""
    schema_keys = set(cls.config_schema.keys())
    unknown = set(cls.contract.config) - schema_keys
    assert not unknown, (
        f"{cls.__name__} contract documents config keys absent from "
        f"config_schema: {sorted(unknown)}"
    )


@pytest.mark.parametrize("cls", B1_CLASSES, ids=_ids)
def test_symbols_appear_in_math_or_description(cls):
    """Every symbol must appear in a math line, the summary, or the description."""
    contract = cls.contract
    haystack = " ".join(
        [
            getattr(cls, "description", "") or "",
            contract.summary or "",
            contract.description or "",
            " ".join(contract.math),
        ]
    )
    missing = [sym for sym in contract.symbols if sym not in haystack]
    assert not missing, (
        f"{cls.__name__} declares symbols not found in math/description: {missing}"
    )
