"""Acceptance gate for the structural-ecoli s01 study.

The reproducible-now layer (selection conservation) is asserted green here; the
end-to-end pack gate is exercised when a real pack exists on disk and otherwise
SKIPPED with the reproduction requirement (see v2ecoli/structural/acceptance.py
and SUMMARY.md) — never silently passed.
"""
import pytest

from v2ecoli.structural.acceptance import (
    check_selection_conservation, evaluate_pack_gate, DEFAULT_SNAPSHOTS,
)

S01_DIR = "workspace/studies/s01-birth-and-division"


def test_selection_layer_conserves_counts_and_caps_top_n():
    """counts -> ingredient selection conserves every selected species' count
    exactly, caps auto-expansion at top_n, and is deterministic. Runs with the
    installed deps only (no parsimony binary)."""
    counts = {f"SPECIES-{i}": 1000 + i * 37 for i in range(500)}
    # include real proteins so top-N auto-expansion actually engages
    from v2ecoli.structural.build import _proteins
    prot = _proteins()
    for i, mid in enumerate(sorted(m for m in prot if isinstance(prot[m], str)
                                   and prot[m] not in ("", "null"))[:1000]):
        counts[mid] = 2000 + i
    res = check_selection_conservation(counts, top_n=40)
    assert res["count_mismatches"] == 0, res
    assert res["deterministic"], res
    assert res["passed"], res
    # curated + 40 auto-expanded monomers + 1 lipid; auto layer capped at top_n
    assert res["n_from_counts"] >= 40


def test_pack_gate_over_artifacts():
    """End-to-end gate over s01's written pack artifacts. Skips (does NOT fail,
    does NOT falsely pass) when the artifacts are absent — the canonical env
    cannot currently produce them (missing parsimony binary + stale
    pbg_parsimony; see study.yaml reproducibility note)."""
    result = evaluate_pack_gate(S01_DIR, DEFAULT_SNAPSHOTS)
    if not result.get("available"):
        pytest.skip(f"pack artifacts absent — {result.get('reason')}. "
                    "Produce them with the pinned toolchain, then re-run.")
    failed = {n: t for n, t in result["tests"].items() if not t["passed"]}
    assert not failed, f"pack gate failures: {failed}\nsnapshots: {result['snapshots']}"
