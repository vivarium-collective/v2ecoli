"""
Scaffold test for the tRNA-charging kernel port (Task 2a).

Verifies that the parity-test infrastructure works:

1. The golden fixture deserializes and contains cases for all expected
   functions.
2. The RNG wrapper is deterministic given the same seed and independent of
   ``numpy.random`` global state.
3. The kernel module exposes the right surface area (function names +
   signatures) so 2b–2e can fill in the bodies.

Implementation correctness for individual kernel functions is gated by
``tests/test_kinetic_charging_kernel.py`` (added in 2b).
"""

from __future__ import annotations

import gzip
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from v2ecoli.processes.polypeptide import kinetic_charging_kernel as kernel


FIXTURE = (
    Path(__file__).parent / "fixtures" / "trna_charging_kernel_golden.json.gz"
)


@pytest.fixture(scope="module")
def golden() -> dict:
    with gzip.open(FIXTURE, "rb") as fh:
        return json.loads(fh.read())


def test_golden_fixture_exists_and_round_trips(golden: dict) -> None:
    assert "metadata" in golden
    assert "cases" in golden
    assert isinstance(golden["cases"], list)
    assert len(golden["cases"]) > 0


def test_golden_metadata_includes_provenance(golden: dict) -> None:
    md = golden["metadata"]
    assert "captured_at" in md
    assert "upstream_sha" in md
    assert "platform" in md
    assert "rng" in md
    # The note explicitly flags the RNG-equivalence subtlety
    assert "libc rand()" in md["note"]


def test_golden_covers_all_kernel_functions(golden: dict) -> None:
    expected = {
        "get_initiations",
        "get_codon_at",
        "get_candidates_to_C",
        "get_candidates_to_N",
        "select_candidate",
        "get_elongation_rate",
        "get_codons_read",
        "reconcile_via_ribosome_positions",
        "reconcile_via_trna_pools",
    }
    observed = {case["function"] for case in golden["cases"]}
    assert expected.issubset(observed), f"missing: {expected - observed}"


def test_golden_arrays_serialize_with_dtype_and_shape(golden: dict) -> None:
    """Spot-check the array serialization shape used throughout."""
    case = next(c for c in golden["cases"] if c["name"] == "get_initiations/basic_2_inits")
    elong = case["inputs"]["elongations"]
    assert set(elong.keys()) == {"dtype", "shape", "data"}
    arr = np.asarray(elong["data"], dtype=np.dtype(elong["dtype"])).reshape(elong["shape"])
    np.testing.assert_array_equal(arr, np.array([1, 1, 2], dtype=np.int64))


def test_rng_deterministic_given_same_seed() -> None:
    kernel.seed(42)
    a = [kernel.randint_below(100) for _ in range(20)]
    kernel.seed(42)
    b = [kernel.randint_below(100) for _ in range(20)]
    assert a == b


def test_rng_different_seeds_diverge() -> None:
    kernel.seed(42)
    a = [kernel.randint_below(100) for _ in range(20)]
    kernel.seed(43)
    b = [kernel.randint_below(100) for _ in range(20)]
    assert a != b


def test_rng_independent_of_numpy_global_state() -> None:
    """
    Calling ``np.random.seed(...)`` between kernel.seed and kernel.randint_below
    must not affect kernel output. Confirms the kernel uses its own
    RandomState rather than the global numpy.random.
    """
    kernel.seed(7)
    np.random.seed(999)  # would corrupt output if shared
    a = [kernel.randint_below(100) for _ in range(20)]
    kernel.seed(7)
    np.random.seed(123)  # different again — still shouldn't matter
    b = [kernel.randint_below(100) for _ in range(20)]
    assert a == b


def test_rng_raises_when_unseeded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(kernel, "_RNG", None)
    with pytest.raises(RuntimeError, match="seed"):
        kernel.randint_below(10)


def test_stub_functions_exist_with_expected_signatures() -> None:
    expected = {
        "get_initiations": ["elongations", "lengths", "indexes"],
        "get_codon_at": [
            "sequences", "elongations", "ith_ribosome", "relative_position",
            "absolute_position",
        ],
        "get_candidates_to_C": ["sequences", "elongations", "codon_id"],
        "get_candidates_to_N": ["sequences", "elongations", "codon_id"],
        "select_candidate": [
            "sequences", "elongations", "relative_position", "codon_id", "r",
        ],
        "is_initial_state": ["initial_state", "state"],
        "get_codons_read": ["sequences", "elongations", "size"],
        "reconcile_via_ribosome_positions": [
            "sequence_codons", "elongations", "kinetics_codons", "sequences",
            "max_attempts",
        ],
        "reconcile_via_trna_pools": [
            "sequence_codons", "kinetics_codons", "free_trnas", "charged_trnas",
            "chargings", "amino_acids_used", "codons_to_trnas_counter",
            "trnas_to_codons", "trnas_to_amino_acid_indexes",
        ],
        "get_elongation_rate": ["sequences", "col", "time", "target"],
    }
    for name, params in expected.items():
        fn = getattr(kernel, name, None)
        assert callable(fn), f"{name} missing from kernel module"
        sig = inspect.signature(fn)
        assert list(sig.parameters) == params, (
            f"{name} signature drift: got {list(sig.parameters)}, want {params}"
        )


def test_stubs_raise_not_implemented_until_filled_in() -> None:
    """All bodies should be NotImplementedError until 2b–2e land them."""
    with pytest.raises(NotImplementedError):
        kernel.get_initiations(
            np.array([1], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )
