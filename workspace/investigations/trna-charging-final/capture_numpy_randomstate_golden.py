"""
Capture the v2ecoli-side (numpy.random.RandomState) golden for the stochastic
kernel functions.

Reads ``tests/fixtures/trna_charging_kernel_golden.json.gz`` (the
libc-rand-macos-arm64 golden captured from upstream Cython in 2a), runs the
ported v2ecoli kernel against each case's input at the same seed, and writes
``tests/fixtures/trna_charging_kernel_numpy_randomstate_golden.json.gz``.

This golden then gates regressions in the v2ecoli port — once committed, any
change to the port must reproduce the same outputs at the same seeds, or
the developer must explicitly regenerate the golden (and justify it in the PR).

Invocation:
    .venv/bin/python workspace/investigations/trna-charging-final/capture_numpy_randomstate_golden.py
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

from v2ecoli.processes.polypeptide import kinetic_charging_kernel as kernel


V2ECOLI = Path("/Users/arnabmutsuddy/projects/v2ecoli")
SOURCE = V2ECOLI / "tests" / "fixtures" / "trna_charging_kernel_golden.json.gz"
OUT = (
    V2ECOLI
    / "tests"
    / "fixtures"
    / "trna_charging_kernel_numpy_randomstate_golden.json.gz"
)

# Functions ported in 2c/2d/2e that need a v2ecoli-side golden because their
# output depends on the (RNG-specific) sequence of picks.
STOCHASTIC = {
    "reconcile_via_ribosome_positions",
    "reconcile_via_trna_pools",
}


def arr(a: np.ndarray) -> dict:
    return {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.tolist()}


def deserialize(field: dict) -> np.ndarray:
    return np.asarray(field["data"], dtype=np.dtype(field["dtype"])).reshape(
        field["shape"]
    )


def v2ecoli_sha() -> str:
    return subprocess.check_output(
        ["git", "-C", str(V2ECOLI), "rev-parse", "HEAD"], text=True
    ).strip()


def regenerate_reconcile_via_ribosome_positions(case: dict) -> dict:
    inputs = case["inputs"]
    kinetics_codons = deserialize(inputs["kinetics_codons_in"])
    elongations = deserialize(inputs["elongations_in"]).copy()
    sequences = deserialize(inputs["sequences"])

    # Reconstruct sequence_codons the same way the capture script did
    sequence_codons = np.zeros(int(sequences.max()) + 1, dtype=np.int64)
    for row, cols in enumerate(elongations):
        for col in range(int(cols)):
            sequence_codons[sequences[row, col]] += 1

    kinetics_codons_buf = kinetics_codons.copy()

    kernel.seed(case["seed"])
    kernel.reconcile_via_ribosome_positions(
        sequence_codons,
        elongations,
        kinetics_codons_buf,
        sequences,
        int(inputs["max_attempts"]),
    )
    return {
        "sequence_codons_out": arr(sequence_codons),
        "elongations_out": arr(elongations),
        "kinetics_codons_out": arr(kinetics_codons_buf),
    }


# Placeholder for 2d's reconcile_via_trna_pools regeneration. Until 2d lands,
# we skip those cases. The capture script will just leave them out.
def regenerate_reconcile_via_trna_pools(case: dict) -> dict | None:
    try:
        kernel.reconcile_via_trna_pools  # type: ignore[attr-defined]
    except AttributeError:
        return None
    inputs = case["inputs"]
    sc = deserialize(inputs["sequence_codons_in"])
    kc = deserialize(inputs["kinetics_codons_in"])
    ft = deserialize(inputs["free_trnas_in"])
    ct = deserialize(inputs["charged_trnas_in"])
    ch = deserialize(inputs["chargings_in"])
    aau = deserialize(inputs["amino_acids_used_in"])
    ctc = deserialize(inputs["codons_to_trnas_counter_in"])
    ttc = deserialize(inputs["trnas_to_codons"])
    ttai = deserialize(inputs["trnas_to_amino_acid_indexes"])
    kernel.seed(case["seed"])
    try:
        kernel.reconcile_via_trna_pools(sc, kc, ft, ct, ch, aau, ctc, ttc, ttai)
    except NotImplementedError:
        return None
    return {
        "sequence_codons_out": arr(sc),
        "kinetics_codons_out": arr(kc),
        "free_trnas_out": arr(ft),
        "charged_trnas_out": arr(ct),
        "chargings_out": arr(ch),
        "amino_acids_used_out": arr(aau),
        "codons_to_trnas_counter_out": arr(ctc),
    }


def main() -> None:
    with gzip.open(SOURCE, "rb") as fh:
        source = json.loads(fh.read())

    regenerated: list[dict] = []
    skipped: list[str] = []
    for case in source["cases"]:
        fn = case["function"]
        if fn not in STOCHASTIC:
            continue
        if fn == "reconcile_via_ribosome_positions":
            outputs = regenerate_reconcile_via_ribosome_positions(case)
        elif fn == "reconcile_via_trna_pools":
            outputs = regenerate_reconcile_via_trna_pools(case)
            if outputs is None:
                skipped.append(case["name"])
                continue
        else:
            continue
        regenerated.append(
            {
                "name": case["name"],
                "function": case["function"],
                "seed": case["seed"],
                "inputs": case["inputs"],
                "outputs": outputs,
            }
        )

    payload = {
        "metadata": {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "v2ecoli_sha": v2ecoli_sha(),
            "source_golden": SOURCE.name,
            "source_upstream_sha": source["metadata"]["upstream_sha"],
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "rng": "numpy.random.RandomState",
            "note": (
                "Per-RNG regression golden. Reproduces the v2ecoli kernel's "
                "stochastic output for each seed+input from the libc-rand "
                "golden. Re-capture if the port intentionally changes; "
                "otherwise tests/test_kinetic_charging_kernel.py asserts "
                "byte-identity against this file."
            ),
        },
        "cases": regenerated,
        "skipped": skipped,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    with gzip.open(OUT, "wb") as fh:
        fh.write(encoded)
    print(f"Wrote {OUT}")
    print(
        f"  cases: {len(regenerated)}, skipped: {len(skipped)}, "
        f"size: {len(encoded):,} raw / {OUT.stat().st_size:,} gz"
    )
    if skipped:
        print("Skipped (stub still in place):")
        for name in skipped:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
