"""
Capture golden inputs+outputs for the upstream Cython _trna_charging kernel.

Runs against the built upstream extension at
`/Users/arnabmutsuddy/projects/vEcoli_trna/vEcoli/wholecell/utils/_trna_charging`
(so vEcoli_trna's venv must be the interpreter) and writes the result to
`v2ecoli/tests/fixtures/trna_charging_kernel_golden.json.gz`.

Test cases are lifted directly from upstream
`wholecell/tests/utils/test_trna_charging.py`, plus a small set of additional
larger random inputs for the stochastic kernels so 2c/2d have more parity
surface to test against.

Invocation:
    /Users/arnabmutsuddy/projects/vEcoli_trna/vEcoli/.venv/bin/python \\
        workspace/investigations/trna-charging-final/capture_kernel_golden.py

The output JSON does not depend on Cython for read-back — only NumPy.
"""

from __future__ import annotations

import gzip
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Locate vEcoli_trna and add to sys.path so the built extension imports
VECOLI_TRNA = Path("/Users/arnabmutsuddy/projects/vEcoli_trna/vEcoli")
if str(VECOLI_TRNA) not in sys.path:
    sys.path.insert(0, str(VECOLI_TRNA))

from wholecell.utils._trna_charging import (  # type: ignore
    seed_rng,
    get_initiations,
    get_codon_at,
    get_candidates_to_C,
    get_candidates_to_N,
    select_candidate,
    reconcile_via_ribosome_positions,
    reconcile_via_trna_pools,
    get_elongation_rate,
    get_codons_read,
)


V2ECOLI = Path("/Users/arnabmutsuddy/projects/v2ecoli")
OUT = V2ECOLI / "tests" / "fixtures" / "trna_charging_kernel_golden.json.gz"


# ----- helpers -----

def arr(a: np.ndarray) -> dict:
    """Serialize a numpy array to a dtype + flat-list dict."""
    return {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.tolist()}


def upstream_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(VECOLI_TRNA), "rev-parse", "HEAD"], text=True
    ).strip()


# ----- deterministic cases -----

def case_get_initiations() -> list[dict]:
    elongations = np.array([1, 1, 2], dtype=np.int64)
    lengths = np.array([1, 0, 0], dtype=np.int64)
    indexes = np.array([0, 1, 2], dtype=np.int64)
    out = get_initiations(elongations, lengths, indexes)
    return [
        {
            "name": "get_initiations/basic_2_inits",
            "function": "get_initiations",
            "inputs": {
                "elongations": arr(elongations),
                "lengths": arr(lengths),
                "indexes": arr(indexes),
            },
            "outputs": {"return": int(out)},
        }
    ]


def case_get_codon_at() -> list[dict]:
    sequences = np.array([[0, 1, 2], [3, 3, 3]], dtype=np.int8)
    elongations = np.array([2, 0], dtype=np.int64)
    cases = []
    for relative, expected_desc in [
        (0, "current"),
        (1, "+1"),
        (-1, "-1"),
        (2, "beyond_C"),
        (-2, "beyond_N"),
    ]:
        out = get_codon_at(sequences, elongations, 0, relative, 0)
        cases.append(
            {
                "name": f"get_codon_at/{expected_desc}",
                "function": "get_codon_at",
                "inputs": {
                    "sequences": arr(sequences),
                    "elongations": arr(elongations),
                    "ith_ribosome": 0,
                    "relative_position": relative,
                    "absolute_position": 0,
                },
                "outputs": {"return": int(out)},
            }
        )
    return cases


def case_get_candidates_to_C() -> list[dict]:
    sequences = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 2, 1, 3]], dtype=np.int8)
    elongations = np.array([1, 1, 2], dtype=np.int64)
    cases = []
    for codon_id, name in [(1, "immediate_2_candidates"), (3, "beyond_+1"), (0, "no_candidates")]:
        candidates, rel = get_candidates_to_C(
            sequences, elongations, codon_id, 0, 0, 0, 0
        )
        cases.append(
            {
                "name": f"get_candidates_to_C/{name}",
                "function": "get_candidates_to_C",
                "inputs": {
                    "sequences": arr(sequences),
                    "elongations": arr(elongations),
                    "codon_id": int(codon_id),
                },
                "outputs": {
                    "candidates": int(candidates),
                    "relative_position": int(rel),
                },
            }
        )
    return cases


def case_get_candidates_to_N() -> list[dict]:
    sequences = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 2, 1, 3]], dtype=np.int8)
    elongations = np.array([3, 4, 4], dtype=np.int64)
    cases = []
    for codon_id, name in [(2, "current_1_candidate"), (1, "beyond_0_3_candidates")]:
        candidates, rel = get_candidates_to_N(
            sequences, elongations, codon_id, 0, 0, 0, 0
        )
        cases.append(
            {
                "name": f"get_candidates_to_N/{name}",
                "function": "get_candidates_to_N",
                "inputs": {
                    "sequences": arr(sequences),
                    "elongations": arr(elongations),
                    "codon_id": int(codon_id),
                },
                "outputs": {
                    "candidates": int(candidates),
                    "relative_position": int(rel),
                },
            }
        )
    return cases


def case_select_candidate() -> list[dict]:
    sequences = np.array([[0, 1, 2, 3], [0, 2, 1, 3], [0, 2, 1, 3]], dtype=np.int8)
    elongations = np.array([1, 1, 2], dtype=np.int64)
    cases = []
    for r, expected_desc in [(0, "first"), (1, "second")]:
        out = select_candidate(sequences, elongations, 1, 1, r, 0, 0, 0)
        cases.append(
            {
                "name": f"select_candidate/r_{r}",
                "function": "select_candidate",
                "inputs": {
                    "sequences": arr(sequences),
                    "elongations": arr(elongations),
                    "relative_position": 1,
                    "codon_id": 1,
                    "r": int(r),
                },
                "outputs": {"return": int(out)},
            }
        )
    return cases


def case_get_elongation_rate() -> list[dict]:
    sequences = np.array(
        [[0, 1, 1, -1, -1], [1, 0, 0, 1, 1]], dtype=np.int8
    )
    rate = get_elongation_rate(sequences, 3, 1.0, 4.0)
    return [
        {
            "name": "get_elongation_rate/basic",
            "function": "get_elongation_rate",
            "inputs": {
                "sequences": arr(sequences),
                "col": 3,
                "time": 1.0,
                "target": 4.0,
            },
            "outputs": {"return": int(rate)},
        }
    ]


def case_get_codons_read() -> list[dict]:
    sequences = np.array(
        [[0, 1, 1, -1, -1], [1, 0, 0, 1, 1]], dtype=np.int8
    )
    elongations = np.array([2, 4], dtype=np.int64)
    out = get_codons_read(sequences, elongations, 2)
    return [
        {
            "name": "get_codons_read/basic",
            "function": "get_codons_read",
            "inputs": {
                "sequences": arr(sequences),
                "elongations": arr(elongations),
                "size": 2,
            },
            "outputs": {"return": arr(np.asarray(out))},
        }
    ]


# ----- stochastic cases (require seed) -----
# Each case follows the upstream test pattern:
#   1. seed_rng(seed)
#   2. call the kernel which mutates buffers in place
#   3. capture all mutated buffers + any return value

def _make_sequence_codons(sequences: np.ndarray, elongations: np.ndarray) -> np.ndarray:
    sc = np.zeros(int(sequences.max()) + 1, dtype=np.int64)
    for row, cols in enumerate(elongations):
        for col in range(int(cols)):
            sc[sequences[row, col]] += 1
    return sc


def _run_reconcile_via_ribosome_positions(
    name: str,
    seed: int,
    kinetics_codons_in: np.ndarray,
    elongations_in: np.ndarray,
    sequences: np.ndarray,
    max_attempts: int,
) -> dict:
    kinetics_codons = kinetics_codons_in.copy()
    elongations = elongations_in.copy()
    sequence_codons = _make_sequence_codons(sequences, elongations)

    seed_rng(seed)
    reconcile_via_ribosome_positions(
        sequence_codons,
        elongations,
        kinetics_codons,
        sequences,
        np.byte(max_attempts),
    )
    return {
        "name": name,
        "function": "reconcile_via_ribosome_positions",
        "seed": int(seed),
        "inputs": {
            "kinetics_codons_in": arr(kinetics_codons_in),
            "elongations_in": arr(elongations_in),
            "sequences": arr(sequences),
            "max_attempts": int(max_attempts),
        },
        "outputs": {
            "sequence_codons_out": arr(sequence_codons),
            "elongations_out": arr(elongations),
            "kinetics_codons_out": arr(kinetics_codons),
        },
    }


def case_reconcile_via_ribosome_positions() -> list[dict]:
    # Upstream test inputs
    cases = []
    common_sequences = np.array(
        [[1, 0, 1, 2], [1, 2, 1, 2], [1, 1, 2, 0]], dtype=np.int8
    )
    cases.append(
        _run_reconcile_via_ribosome_positions(
            "reconcile_via_ribosome_positions/equal",
            seed=0,
            kinetics_codons_in=np.array([0, 1, 0], dtype=np.int64),
            elongations_in=np.array([1, 0, 0], dtype=np.int64),
            sequences=common_sequences,
            max_attempts=4,
        )
    )
    cases.append(
        _run_reconcile_via_ribosome_positions(
            "reconcile_via_ribosome_positions/forward",
            seed=0,
            kinetics_codons_in=np.array([0, 2, 1], dtype=np.int64),
            elongations_in=np.array([1, 1, 0], dtype=np.int64),
            sequences=common_sequences,
            max_attempts=4,
        )
    )
    cases.append(
        _run_reconcile_via_ribosome_positions(
            "reconcile_via_ribosome_positions/backward",
            seed=0,
            kinetics_codons_in=np.array([1, 6, 1], dtype=np.int64),
            elongations_in=np.array([3, 3, 3], dtype=np.int64),
            sequences=common_sequences,
            max_attempts=4,
        )
    )

    # Backward beyond +1
    seqs = np.array([[0, 0, 2, 2], [0, 2, 2, 2], [1, 3, 3, 0]], dtype=np.int8)
    cases.append(
        _run_reconcile_via_ribosome_positions(
            "reconcile_via_ribosome_positions/backward_beyond",
            seed=0,
            kinetics_codons_in=np.array([3, 0, 3, 0], dtype=np.int64),
            elongations_in=np.array([3, 3, 3], dtype=np.int64),
            sequences=seqs,
            max_attempts=4,
        )
    )

    # Attempts threshold (10 ribosomes, only 2 codons available)
    seqs10 = np.tile(np.array([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]], dtype=np.int8), (5, 1))
    cases.append(
        _run_reconcile_via_ribosome_positions(
            "reconcile_via_ribosome_positions/attempts_threshold",
            seed=0,
            kinetics_codons_in=np.array([10, 20], dtype=np.int64),
            elongations_in=3 * np.ones(10, dtype=np.int64),
            sequences=seqs10,
            max_attempts=4,
        )
    )

    # The use_free_trna and forward_undo_charging tests are stochastic-trivial
    # for ribosome_positions (kinetics_codons > total reachable). They produce
    # a sequence_codons of all zeros without consuming RNG. Capture them too.
    seqs_free = np.array([[1, 0, 1, 2], [3, 2, 1, 2], [1, 1, 2, 0]], dtype=np.int8)
    cases.append(
        _run_reconcile_via_ribosome_positions(
            "reconcile_via_ribosome_positions/use_free_trna_prelim",
            seed=0,
            kinetics_codons_in=np.array([0, 0, 1, 0], dtype=np.int64),
            elongations_in=np.array([0, 0, 0], dtype=np.int64),
            sequences=seqs_free,
            max_attempts=4,
        )
    )

    # Larger stochastic case from upstream test_reconcile_different_seeds
    big_kc = np.array(
        [6, 1, 7, 4, 2, 3, 5, 3, 8, 2, 6, 4, 6, 0, 9, 1, 5, 5, 1, 9], dtype=np.int64
    )
    big_seqs = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1],
            [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, -1, -1, -1],
            [1, 3, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [2, 4, 6, 8, 10, 12, 14, 16, 18, 0, -1, -1, -1, -1, -1],
            [19, 17, 15, 13, 11, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1],
        ],
        dtype=np.int8,
    )
    big_elong = np.array([8, 12, 5, 10, 7], dtype=np.int64)
    # Note: upstream test treats sequence_codons as if pre-supplied; here we
    # construct via _make_sequence_codons to match the kernel's contract.
    for seed in (12345, 54321):
        cases.append(
            _run_reconcile_via_ribosome_positions(
                f"reconcile_via_ribosome_positions/big_seed_{seed}",
                seed=seed,
                kinetics_codons_in=big_kc,
                elongations_in=big_elong,
                sequences=big_seqs,
                max_attempts=50,
            )
        )

    return cases


def case_reconcile_via_trna_pools() -> list[dict]:
    """Capture both upstream tests that exercise reconcile_via_trna_pools."""

    def _run(
        name: str,
        seed: int,
        sequence_codons_in: np.ndarray,
        kinetics_codons_in: np.ndarray,
        free_trnas_in: np.ndarray,
        charged_trnas_in: np.ndarray,
        chargings_in: np.ndarray,
        amino_acids_used_in: np.ndarray,
        codons_to_trnas_counter_in: np.ndarray,
        trnas_to_codons: np.ndarray,
        trnas_to_amino_acid_indexes: np.ndarray,
    ) -> dict:
        sc = sequence_codons_in.copy()
        kc = kinetics_codons_in.copy()
        ft = free_trnas_in.copy()
        ct = charged_trnas_in.copy()
        ch = chargings_in.copy()
        aau = amino_acids_used_in.copy()
        ctc = codons_to_trnas_counter_in.copy()

        seed_rng(seed)
        reconcile_via_trna_pools(
            sc, kc, ft, ct, ch, aau, ctc, trnas_to_codons, trnas_to_amino_acid_indexes
        )
        return {
            "name": name,
            "function": "reconcile_via_trna_pools",
            "seed": int(seed),
            "inputs": {
                "sequence_codons_in": arr(sequence_codons_in),
                "kinetics_codons_in": arr(kinetics_codons_in),
                "free_trnas_in": arr(free_trnas_in),
                "charged_trnas_in": arr(charged_trnas_in),
                "chargings_in": arr(chargings_in),
                "amino_acids_used_in": arr(amino_acids_used_in),
                "codons_to_trnas_counter_in": arr(codons_to_trnas_counter_in),
                "trnas_to_codons": arr(trnas_to_codons),
                "trnas_to_amino_acid_indexes": arr(trnas_to_amino_acid_indexes),
            },
            "outputs": {
                "sequence_codons_out": arr(sc),
                "kinetics_codons_out": arr(kc),
                "free_trnas_out": arr(ft),
                "charged_trnas_out": arr(ct),
                "chargings_out": arr(ch),
                "amino_acids_used_out": arr(aau),
                "codons_to_trnas_counter_out": arr(ctc),
            },
        }

    cases = []
    # use_free_trna: 1 free tRNA -> charged
    trnas_to_codons = np.array(
        [[1, 0], [1, 1], [1, 0], [1, 1]], dtype=np.int8
    )
    trnas_to_amino_acid_indexes = np.array([0, 0], dtype=np.int8)
    codons_to_trnas_counter = np.zeros((2, 4), dtype=np.int64)
    codons_to_trnas_counter[0, 2] = 1
    cases.append(
        _run(
            "reconcile_via_trna_pools/use_free_trna",
            seed=0,
            sequence_codons_in=np.zeros(4, dtype=np.int64),
            kinetics_codons_in=np.array([0, 0, 1, 0], dtype=np.int64),
            free_trnas_in=np.array([2, 0], dtype=np.int64),
            charged_trnas_in=np.array([0, 0], dtype=np.int64),
            chargings_in=np.array([1, 0], dtype=np.int64),
            amino_acids_used_in=np.array([1], dtype=np.int64),
            codons_to_trnas_counter_in=codons_to_trnas_counter,
            trnas_to_codons=trnas_to_codons,
            trnas_to_amino_acid_indexes=trnas_to_amino_acid_indexes,
        )
    )

    # forward_undo_charging: 0 free, must undo charging
    codons_to_trnas_counter2 = np.zeros((2, 4), dtype=np.int64)
    codons_to_trnas_counter2[0, 2] = 1
    cases.append(
        _run(
            "reconcile_via_trna_pools/forward_undo_charging",
            seed=0,
            sequence_codons_in=np.zeros(4, dtype=np.int64),
            kinetics_codons_in=np.array([0, 0, 1, 0], dtype=np.int64),
            free_trnas_in=np.array([0, 0], dtype=np.int64),
            charged_trnas_in=np.array([2, 0], dtype=np.int64),
            chargings_in=np.array([1, 0], dtype=np.int64),
            amino_acids_used_in=np.array([1], dtype=np.int64),
            codons_to_trnas_counter_in=codons_to_trnas_counter2,
            trnas_to_codons=trnas_to_codons,
            trnas_to_amino_acid_indexes=trnas_to_amino_acid_indexes,
        )
    )
    return cases


# ----- top-level -----

def build_golden() -> dict:
    cases: list[dict] = []
    cases += case_get_initiations()
    cases += case_get_codon_at()
    cases += case_get_candidates_to_C()
    cases += case_get_candidates_to_N()
    cases += case_select_candidate()
    cases += case_get_elongation_rate()
    cases += case_get_codons_read()
    cases += case_reconcile_via_ribosome_positions()
    cases += case_reconcile_via_trna_pools()

    return {
        "metadata": {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "upstream_sha": upstream_sha(),
            "upstream_path": str(VECOLI_TRNA),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "rng": "libc-rand-macos-arm64",
            "note": (
                "Stochastic cases (reconcile_via_*, select_candidate when r>0) "
                "depend on libc rand() which differs between glibc and macOS. "
                "The v2ecoli numpy port will use a different RNG; expected "
                "outputs for stochastic cases will be regenerated against the "
                "ported RNG in tasks 2c/2d and stored in a sibling golden file."
            ),
        },
        "cases": cases,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = build_golden()
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    with gzip.open(OUT, "wb") as fh:
        fh.write(encoded)
    print(f"Wrote {OUT} ({len(encoded):,} bytes uncompressed, {OUT.stat().st_size:,} bytes gz)")
    print(f"Captured {len(payload['cases'])} cases across 9 functions")


if __name__ == "__main__":
    main()
