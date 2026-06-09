"""
Parity tests for the v2ecoli tRNA-charging kernel port vs the upstream Cython
``_trna_charging.pyx``.

Each test case is loaded from
``tests/fixtures/trna_charging_kernel_golden.json.gz`` (captured by
``workspace/investigations/trna-charging-final/capture_kernel_golden.py``)
and the corresponding ported function is asserted to produce byte-identical
output.

Task 2b covers the 7 deterministic functions: get_initiations, get_codon_at,
get_candidates_to_C/N, select_candidate, is_initial_state, get_codons_read.
Cases for stochastic functions (reconcile_via_*) and get_elongation_rate are
loaded in this file but skipped until 2c–2e land them.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from v2ecoli.processes.polypeptide import kinetic_charging_kernel as kernel


FIXTURE = (
    Path(__file__).parent / "fixtures" / "trna_charging_kernel_golden.json.gz"
)


def _arr(field: dict) -> np.ndarray:
    """Deserialize the {dtype, shape, data} array form used by the golden."""
    return np.asarray(field["data"], dtype=np.dtype(field["dtype"])).reshape(
        field["shape"]
    )


@pytest.fixture(scope="module")
def golden() -> list[dict]:
    with gzip.open(FIXTURE, "rb") as fh:
        return json.loads(fh.read())["cases"]


def _cases_for(golden: list[dict], function: str) -> list[dict]:
    return [c for c in golden if c["function"] == function]


# ---------- 2b: deterministic kernels ----------


def test_get_initiations_parity(golden: list[dict]) -> None:
    for case in _cases_for(golden, "get_initiations"):
        elongations = _arr(case["inputs"]["elongations"])
        lengths = _arr(case["inputs"]["lengths"])
        indexes = _arr(case["inputs"]["indexes"])
        got = kernel.get_initiations(elongations, lengths, indexes)
        assert int(got) == case["outputs"]["return"], case["name"]


def test_get_codon_at_parity(golden: list[dict]) -> None:
    for case in _cases_for(golden, "get_codon_at"):
        sequences = _arr(case["inputs"]["sequences"])
        elongations = _arr(case["inputs"]["elongations"])
        got = kernel.get_codon_at(
            sequences,
            elongations,
            case["inputs"]["ith_ribosome"],
            case["inputs"]["relative_position"],
            case["inputs"]["absolute_position"],
        )
        assert int(got) == case["outputs"]["return"], case["name"]


def test_get_candidates_to_C_parity(golden: list[dict]) -> None:
    for case in _cases_for(golden, "get_candidates_to_C"):
        sequences = _arr(case["inputs"]["sequences"])
        elongations = _arr(case["inputs"]["elongations"])
        candidates, rel = kernel.get_candidates_to_C(
            sequences, elongations, case["inputs"]["codon_id"]
        )
        assert int(candidates) == case["outputs"]["candidates"], case["name"]
        assert int(rel) == case["outputs"]["relative_position"], case["name"]


def test_get_candidates_to_N_parity(golden: list[dict]) -> None:
    for case in _cases_for(golden, "get_candidates_to_N"):
        sequences = _arr(case["inputs"]["sequences"])
        elongations = _arr(case["inputs"]["elongations"])
        candidates, rel = kernel.get_candidates_to_N(
            sequences, elongations, case["inputs"]["codon_id"]
        )
        assert int(candidates) == case["outputs"]["candidates"], case["name"]
        assert int(rel) == case["outputs"]["relative_position"], case["name"]


def test_select_candidate_parity(golden: list[dict]) -> None:
    for case in _cases_for(golden, "select_candidate"):
        sequences = _arr(case["inputs"]["sequences"])
        elongations = _arr(case["inputs"]["elongations"])
        got = kernel.select_candidate(
            sequences,
            elongations,
            case["inputs"]["relative_position"],
            case["inputs"]["codon_id"],
            case["inputs"]["r"],
        )
        assert int(got) == case["outputs"]["return"], case["name"]


def test_get_codons_read_parity(golden: list[dict]) -> None:
    for case in _cases_for(golden, "get_codons_read"):
        sequences = _arr(case["inputs"]["sequences"])
        elongations = _arr(case["inputs"]["elongations"])
        got = kernel.get_codons_read(sequences, elongations, case["inputs"]["size"])
        expected = _arr(case["outputs"]["return"])
        np.testing.assert_array_equal(got, expected, err_msg=case["name"])


def test_is_initial_state_local_cases() -> None:
    """``is_initial_state`` isn't called anywhere in the upstream kernel
    (the Cython build flags it as unused), so it has no golden case. Spot-check
    locally."""
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([1, 2, 3, 4], dtype=np.int32)
    c = np.array([1, 2, 3, 5], dtype=np.int32)
    assert kernel.is_initial_state(a, b) is True
    assert kernel.is_initial_state(a, c) is False


# ---------- 2c–2e: still stubbed; cases exist but tests skipped ----------


@pytest.mark.skip(reason="reconcile_via_ribosome_positions is implemented in Task 2c")
def test_reconcile_via_ribosome_positions_parity(golden: list[dict]) -> None:
    ...


@pytest.mark.skip(reason="reconcile_via_trna_pools is implemented in Task 2d")
def test_reconcile_via_trna_pools_parity(golden: list[dict]) -> None:
    ...


@pytest.mark.skip(reason="get_elongation_rate is implemented in Task 2e")
def test_get_elongation_rate_parity(golden: list[dict]) -> None:
    ...


# ---------- coverage check ----------


def test_2b_covers_every_relevant_golden_case(golden: list[dict]) -> None:
    """
    Belt-and-suspenders: make sure no deterministic-function golden case
    silently slips through without an assertion. If someone adds a new case to
    the fixture for a 2b function, this gates them on writing a test for it.
    """
    deterministic = {
        "get_initiations",
        "get_codon_at",
        "get_candidates_to_C",
        "get_candidates_to_N",
        "select_candidate",
        "get_codons_read",
    }
    counts = {fn: 0 for fn in deterministic}
    for case in golden:
        if case["function"] in counts:
            counts[case["function"]] += 1
    for fn, n in counts.items():
        assert n > 0, f"no golden case for {fn}; rerun capture script"
