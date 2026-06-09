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
FIXTURE_NUMPY_RS = (
    Path(__file__).parent
    / "fixtures"
    / "trna_charging_kernel_numpy_randomstate_golden.json.gz"
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


@pytest.fixture(scope="module")
def golden_numpy_rs() -> list[dict]:
    with gzip.open(FIXTURE_NUMPY_RS, "rb") as fh:
        return json.loads(fh.read())["cases"]


def _cases_for(golden: list[dict], function: str) -> list[dict]:
    return [c for c in golden if c["function"] == function]


def _build_sequence_codons(sequences: np.ndarray, elongations: np.ndarray) -> np.ndarray:
    """Reconstruct sequence_codons the same way the capture script did."""
    sc = np.zeros(int(sequences.max()) + 1, dtype=np.int64)
    for row, cols in enumerate(elongations):
        for col in range(int(cols)):
            sc[sequences[row, col]] += 1
    return sc


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


def test_reconcile_via_ribosome_positions_byte_identity_numpy_rs(
    golden_numpy_rs: list[dict],
) -> None:
    """
    Byte-identity vs the committed numpy-RandomState golden.

    Detects regressions in the v2ecoli port: any change to the algorithm or
    RNG plumbing that alters output for these seeds must be intentional and
    accompanied by a regenerated golden (run
    ``capture_numpy_randomstate_golden.py``).
    """
    cases = _cases_for(golden_numpy_rs, "reconcile_via_ribosome_positions")
    assert cases, "numpy-RandomState golden missing reconcile_via_ribosome_positions"
    for case in cases:
        inputs = case["inputs"]
        kinetics_codons = _arr(inputs["kinetics_codons_in"])
        elongations = _arr(inputs["elongations_in"]).copy()
        sequences = _arr(inputs["sequences"])
        sequence_codons = _build_sequence_codons(sequences, elongations)
        kinetics_codons_buf = kinetics_codons.copy()

        kernel.seed(case["seed"])
        kernel.reconcile_via_ribosome_positions(
            sequence_codons,
            elongations,
            kinetics_codons_buf,
            sequences,
            int(inputs["max_attempts"]),
        )
        np.testing.assert_array_equal(
            sequence_codons,
            _arr(case["outputs"]["sequence_codons_out"]),
            err_msg=f"{case['name']} sequence_codons",
        )
        np.testing.assert_array_equal(
            elongations,
            _arr(case["outputs"]["elongations_out"]),
            err_msg=f"{case['name']} elongations",
        )
        np.testing.assert_array_equal(
            kinetics_codons_buf,
            _arr(case["outputs"]["kinetics_codons_out"]),
            err_msg=f"{case['name']} kinetics_codons (should be unchanged)",
        )


def test_reconcile_via_ribosome_positions_invariants_vs_libc(
    golden: list[dict],
) -> None:
    """
    Algorithmic invariants checked against the libc-rand golden. Bytes will
    differ (different RNG sequences pick different ribosomes) but the
    invariants below must hold for any correct port:

    * kinetics_codons is never mutated.
    * sequence_codons and elongations stay non-negative.
    * Per-step conservation:
      ``delta(elongations.sum()) == delta(sequence_codons.sum())``
      because each ribosome step changes both by 1.
    * When the libc run achieved compromise=0, the v2ecoli run must too
      (the algorithm's convergence properties are RNG-independent).
    """
    cases = _cases_for(golden, "reconcile_via_ribosome_positions")
    assert cases
    for case in cases:
        inputs = case["inputs"]
        kinetics_codons_in = _arr(inputs["kinetics_codons_in"])
        elongations_in = _arr(inputs["elongations_in"])
        sequences = _arr(inputs["sequences"])
        sequence_codons_in = _build_sequence_codons(sequences, elongations_in)

        # Run v2ecoli port
        sequence_codons = sequence_codons_in.copy()
        elongations = elongations_in.copy()
        kinetics_codons = kinetics_codons_in.copy()

        kernel.seed(case["seed"])
        kernel.reconcile_via_ribosome_positions(
            sequence_codons,
            elongations,
            kinetics_codons,
            sequences,
            int(inputs["max_attempts"]),
        )

        # 1. kinetics_codons immutable
        np.testing.assert_array_equal(
            kinetics_codons, kinetics_codons_in,
            err_msg=f"{case['name']}: kinetics_codons mutated",
        )

        # 2. Non-negative
        assert (sequence_codons >= 0).all(), f"{case['name']}: sequence_codons went negative"
        assert (elongations >= 0).all(), f"{case['name']}: elongations went negative"

        # 3. Conservation
        delta_elong = int(elongations.sum() - elongations_in.sum())
        delta_seq = int(sequence_codons.sum() - sequence_codons_in.sum())
        assert delta_elong == delta_seq, (
            f"{case['name']}: conservation broken — "
            f"delta(elongations.sum())={delta_elong} != delta(sequence_codons.sum())={delta_seq}"
        )

        # 4. Convergence parity with libc run
        libc_seq = _arr(case["outputs"]["sequence_codons_out"])
        libc_compromise = int(np.abs(libc_seq - kinetics_codons_in).sum())
        v2e_compromise = int(np.abs(sequence_codons - kinetics_codons_in).sum())
        if libc_compromise == 0:
            assert v2e_compromise == 0, (
                f"{case['name']}: upstream converged but v2ecoli didn't "
                f"(seq={sequence_codons.tolist()}, target={kinetics_codons_in.tolist()})"
            )


def test_reconcile_via_trna_pools_byte_identity_numpy_rs(
    golden_numpy_rs: list[dict],
) -> None:
    """
    Byte-identity vs the committed numpy-RandomState golden for
    ``reconcile_via_trna_pools``.
    """
    cases = _cases_for(golden_numpy_rs, "reconcile_via_trna_pools")
    assert cases, "numpy-RandomState golden missing reconcile_via_trna_pools"
    for case in cases:
        inputs = case["inputs"]
        sc = _arr(inputs["sequence_codons_in"]).copy()
        kc = _arr(inputs["kinetics_codons_in"]).copy()
        ft = _arr(inputs["free_trnas_in"]).copy()
        ct = _arr(inputs["charged_trnas_in"]).copy()
        ch = _arr(inputs["chargings_in"]).copy()
        aau = _arr(inputs["amino_acids_used_in"]).copy()
        ctc = _arr(inputs["codons_to_trnas_counter_in"]).copy()
        ttc = _arr(inputs["trnas_to_codons"])
        ttai = _arr(inputs["trnas_to_amino_acid_indexes"])

        kernel.seed(case["seed"])
        kernel.reconcile_via_trna_pools(sc, kc, ft, ct, ch, aau, ctc, ttc, ttai)

        expected = case["outputs"]
        np.testing.assert_array_equal(sc, _arr(expected["sequence_codons_out"]), err_msg=f"{case['name']} sequence_codons")
        np.testing.assert_array_equal(kc, _arr(expected["kinetics_codons_out"]), err_msg=f"{case['name']} kinetics_codons")
        np.testing.assert_array_equal(ft, _arr(expected["free_trnas_out"]), err_msg=f"{case['name']} free_trnas")
        np.testing.assert_array_equal(ct, _arr(expected["charged_trnas_out"]), err_msg=f"{case['name']} charged_trnas")
        np.testing.assert_array_equal(ch, _arr(expected["chargings_out"]), err_msg=f"{case['name']} chargings")
        np.testing.assert_array_equal(aau, _arr(expected["amino_acids_used_out"]), err_msg=f"{case['name']} amino_acids_used")
        np.testing.assert_array_equal(ctc, _arr(expected["codons_to_trnas_counter_out"]), err_msg=f"{case['name']} codons_to_trnas_counter")


def test_reconcile_via_trna_pools_invariants_vs_libc(golden: list[dict]) -> None:
    """
    Algorithmic invariants for ``reconcile_via_trna_pools`` checked against the
    libc-rand golden. RNG-independent properties — must hold for any port:

    * ``sequence_codons`` is read-only — never mutated.
    * Per-tRNA total conservation: ``free_trnas[i] + charged_trnas[i]`` unchanged.
    * Post-loop, ``kinetics_codons[c] <= sequence_codons[c]`` for all ``c``.
    * Non-negativity: ``chargings``, ``amino_acids_used``, ``codons_to_trnas_counter``
      never go below zero (caller's job to provide valid input; we still gate).
    * Convergence parity vs upstream: identical input → identical
      ``kinetics_codons`` final value (both runs decrement it by the same total).
    """
    cases = _cases_for(golden, "reconcile_via_trna_pools")
    assert cases
    for case in cases:
        inputs = case["inputs"]
        sc_in = _arr(inputs["sequence_codons_in"])
        kc_in = _arr(inputs["kinetics_codons_in"])
        ft_in = _arr(inputs["free_trnas_in"])
        ct_in = _arr(inputs["charged_trnas_in"])
        ch_in = _arr(inputs["chargings_in"])
        aau_in = _arr(inputs["amino_acids_used_in"])
        ctc_in = _arr(inputs["codons_to_trnas_counter_in"])
        ttc = _arr(inputs["trnas_to_codons"])
        ttai = _arr(inputs["trnas_to_amino_acid_indexes"])

        sc = sc_in.copy()
        kc = kc_in.copy()
        ft = ft_in.copy()
        ct = ct_in.copy()
        ch = ch_in.copy()
        aau = aau_in.copy()
        ctc = ctc_in.copy()

        kernel.seed(case["seed"])
        kernel.reconcile_via_trna_pools(sc, kc, ft, ct, ch, aau, ctc, ttc, ttai)

        np.testing.assert_array_equal(
            sc, sc_in, err_msg=f"{case['name']}: sequence_codons mutated (should be read-only)"
        )
        np.testing.assert_array_equal(
            ft + ct, ft_in + ct_in,
            err_msg=f"{case['name']}: per-tRNA total conservation broken",
        )
        assert (kc <= sc_in).all(), (
            f"{case['name']}: post-loop kinetics_codons={kc.tolist()} not <= "
            f"sequence_codons={sc_in.tolist()}"
        )
        assert (ch >= 0).all(), f"{case['name']}: chargings went negative"
        assert (aau >= 0).all(), f"{case['name']}: amino_acids_used went negative"
        assert (ctc >= 0).all(), f"{case['name']}: codons_to_trnas_counter went negative"

        # kinetics_codons converges to the same final state across RNGs because
        # the loop runs until disagreements=0, decrementing kc once per pick.
        np.testing.assert_array_equal(
            kc, _arr(case["outputs"]["kinetics_codons_out"]),
            err_msg=f"{case['name']}: kinetics_codons drift from upstream",
        )


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
