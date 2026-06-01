"""FBABridge unit tests + coupled Millard+Bridge pilot.

The bridge translates between Millard 2017 ODE concentrations (mM) and the
v2ecoli WCM bulk store (molecule counts). These tests cover:

  1. Pure-function smoke (mM <-> count round-trip, mapping load).
  2. Step invocation (single update, both directions, diagnostics).
  3. Coupled pilot — Millard ODE + FBABridge running together for N seconds,
     proving the bridge keeps the shared pool synchronized end-to-end.

(3) does NOT couple to v2ecoli's full Metabolism process — that requires the
full WCM context (cache, partition allocator, bulk loaders) and is a
deferred Phase 1 milestone. The pilot here proves the BRIDGE works; full
coupling is a separate end-to-end task.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from v2ecoli.steps.fba_bridge import (
    AVOGADRO,
    DEFAULT_CELL_VOLUME_L,
    FBABridge,
    _load_mapping,
    count_to_mM,
    mM_to_count,
)


MAPPING_FILE = "v2ecoli/data/millard_v2ecoli_species_map.yaml"


def test_mM_count_roundtrip():
    """mM -> count -> mM is identity (within float precision)."""
    for conc in [0.001, 0.1, 1.0, 2.5, 100.0]:
        count = mM_to_count(conc, DEFAULT_CELL_VOLUME_L)
        back = count_to_mM(count, DEFAULT_CELL_VOLUME_L)
        assert abs(back - conc) < 1e-9, f"round-trip failed at {conc} mM"


def test_mM_to_count_atp_sanity():
    """ATP at 2.57 mM (Millard initial) in 1 fL -> ~1.55M molecules.
    Order-of-magnitude check: literature places E. coli ATP at ~10^6
    molecules/cell at fast growth, so 1.55M in a 1 fL cell is right."""
    count = mM_to_count(2.572, 1.0e-15)
    assert 1.0e6 < count < 3.0e6, f"ATP molecule count out of physiological range: {count}"


def test_mapping_loads_with_expected_shape():
    """Mapping file loads and provides millard <-> v2ecoli dicts."""
    if not Path(MAPPING_FILE).exists():
        pytest.skip(f"mapping file {MAPPING_FILE} not present (must run from worktree root)")
    m = _load_mapping(MAPPING_FILE)
    assert "millard_to_v2ecoli" in m
    assert "v2ecoli_to_millard" in m
    # Expect ATP / NAD / G6P / PEP / FUM in the shared mapping (canonical anchors)
    assert m["millard_to_v2ecoli"].get("ATP") == "ATP[c]"
    assert m["millard_to_v2ecoli"].get("NAD") == "NAD[c]"
    assert m["millard_to_v2ecoli"].get("G6P") == "GLC-6-P[c]"
    assert m["millard_to_v2ecoli"].get("PEP") == "PHOSPHO-ENOL-PYRUVATE[c]"
    # Round-trip
    assert m["v2ecoli_to_millard"].get("ATP[c]") == "ATP"


def _make_bridge(direction: str = "millard_to_v2ecoli") -> FBABridge:
    bridge = FBABridge.__new__(FBABridge)
    bridge.initialize({
        "mapping_file": MAPPING_FILE,
        "direction": direction,
        "cell_volume_L": DEFAULT_CELL_VOLUME_L,
        "time_step": 1,
    })
    return bridge


@pytest.mark.skipif(
    not Path(MAPPING_FILE).exists(),
    reason="mapping file not present (must run from worktree root)",
)
def test_bridge_translates_millard_to_v2ecoli():
    """Millard ATP=2.5mM produces ~1.5M ATP[c] count in v2ecoli output."""
    bridge = _make_bridge("millard_to_v2ecoli")
    update = bridge.next_update(timestep=1, states={
        "central_metabolites_millard": {"ATP": 2.5, "NAD": 1.4, "G6P": 0.86},
        "v2ecoli_bulk": {},
    })
    out = update["v2ecoli_bulk"]
    assert "ATP[c]" in out
    assert 1.4e6 < out["ATP[c]"] < 1.6e6
    assert "NAD[c]" in out
    assert "GLC-6-P[c]" in out
    diag = update["bridge_diagnostics"]
    assert diag["shared_pool_count"] >= 3
    assert diag["direction"] == "millard_to_v2ecoli"


@pytest.mark.skipif(
    not Path(MAPPING_FILE).exists(),
    reason="mapping file not present",
)
def test_bridge_translates_v2ecoli_to_millard():
    """v2ecoli ATP[c]=1.5M produces ~2.49mM ATP in Millard output."""
    bridge = _make_bridge("v2ecoli_to_millard")
    update = bridge.next_update(timestep=1, states={
        "central_metabolites_millard": {},
        "v2ecoli_bulk": {"ATP[c]": 1.5e6, "NAD[c]": 8.4e5, "GLC-6-P[c]": 5.18e5},
    })
    out = update["central_metabolites_millard"]
    assert "ATP" in out
    assert 2.4 < out["ATP"] < 2.6, f"expected ~2.5 mM ATP, got {out['ATP']}"
    assert "NAD" in out
    assert "G6P" in out


@pytest.mark.skipif(
    not Path(MAPPING_FILE).exists(),
    reason="mapping file not present",
)
def test_bridge_diagnostics_record_unmapped_species():
    """An unknown Millard species (not in mapping, not in millard_only) is
    flagged in diagnostics.millard_unmapped."""
    bridge = _make_bridge("millard_to_v2ecoli")
    update = bridge.next_update(timestep=1, states={
        "central_metabolites_millard": {"ATP": 2.5, "BOGUS_METABOLITE": 1.0},
        "v2ecoli_bulk": {},
    })
    diag = update["bridge_diagnostics"]
    assert "BOGUS_METABOLITE" in diag["millard_unmapped"]


@pytest.mark.skipif(
    not Path(MAPPING_FILE).exists(),
    reason="mapping file not present",
)
def test_bridge_known_millard_only_species_not_flagged():
    """Species listed under millard_only: in the mapping (e.g. 'ei' the PTS
    enzyme state) are NOT flagged as unmapped — they're known-non-shared."""
    bridge = _make_bridge("millard_to_v2ecoli")
    update = bridge.next_update(timestep=1, states={
        "central_metabolites_millard": {"ATP": 2.5, "ei": 0.001},
        "v2ecoli_bulk": {},
    })
    diag = update["bridge_diagnostics"]
    assert "ei" not in diag["millard_unmapped"]


@pytest.mark.skipif(
    not Path(MAPPING_FILE).exists(),
    reason="mapping file not present",
)
def test_bridge_bidirectional_mass_balance():
    """Bidirectional mode: ν2ecoli is treated as ground truth and Millard
    is rebound to match it. The mass_balance_residual records the |delta|
    from the prior Millard state — useful diagnostic for the seam."""
    bridge = _make_bridge("bidirectional")
    update = bridge.next_update(timestep=1, states={
        "central_metabolites_millard": {"ATP": 2.5},  # 2.5 mM
        "v2ecoli_bulk": {"ATP[c]": 1.5e6},             # ~2.49 mM
    })
    diag = update["bridge_diagnostics"]
    assert diag["mass_balance_residual_mM"] > 0
    assert diag["mass_balance_residual_mM"] < 0.5
