"""Unit tests for the substrate-exhaustion growth arrest (#572).

The arrest is OPT-IN (``carbon_exhaustion_arrest``, default False) so the
validated fed regime is byte-identical; when enabled and the carbon source is
not importable, metabolism stops supplying net biomass monomers so the cell
cannot build biomass from phantom internal carbon.

These test the two pure helpers the process delegates to. The end-to-end
behavior (dry_mass grows +14 fg on zero carbon UNFIXED vs declines with the
arrest) is validated by a real multi-generation run against the ParCa cache —
see the PR description; not run in CI because it needs the cache + minutes.
"""

import numpy as np
import pytest

from v2ecoli.processes.metabolism import (
    arrest_monomer_supply,
    is_carbon_starved,
)


# --- is_carbon_starved ------------------------------------------------------


@pytest.mark.fast
def test_not_starved_when_disabled():
    # Opt-in off (the default) -> never starved, even with no carbon importable.
    assert is_carbon_starved(False, {"GLC[p]"}, {"OXYGEN-MOLECULE[p]"}) is False


@pytest.mark.fast
def test_not_starved_when_no_carbon_source_configured():
    # Enabled but no carbon source ids -> never triggers (avoids misfiring on
    # media whose carbon source we were not told about).
    assert is_carbon_starved(True, set(), {"OXYGEN-MOLECULE[p]"}) is False
    assert is_carbon_starved(True, [], {"GLC[p]"}) is False


@pytest.mark.fast
def test_not_starved_while_carbon_importable():
    # Glucose in the gate -> fed -> not starved (arrest inert during growth).
    assert is_carbon_starved(True, {"GLC[p]"}, {"GLC[p]", "OXYGEN-MOLECULE[p]"}) is False


@pytest.mark.fast
def test_starved_when_carbon_source_absent_from_gate():
    # Enabled + glucose gone from the gate (only O2/ions importable) -> starved.
    assert is_carbon_starved(True, {"GLC[p]"}, {"OXYGEN-MOLECULE[p]", "PI[p]"}) is True


@pytest.mark.fast
def test_starved_needs_all_carbon_sources_gone():
    # With two configured sources, importing EITHER means not starved.
    srcs = {"GLC[p]", "ACET[p]"}
    assert is_carbon_starved(True, srcs, {"ACET[p]"}) is False
    assert is_carbon_starved(True, srcs, {"OXYGEN-MOLECULE[p]"}) is True


# --- arrest_monomer_supply --------------------------------------------------


@pytest.mark.fast
def test_zeros_only_positive_monomer_deltas():
    # delta:      [ +5,  -3,  +7,  +9 ]
    # mask (mono):[  T,   T,   F,   T ]
    # -> zero the +5 and +9 (monomer SUPPLY); keep -3 (monomer consumption)
    #    and +7 (non-monomer production).
    delta = np.array([5, -3, 7, 9], dtype=np.int64)
    mask = np.array([True, True, False, True])
    out = arrest_monomer_supply(delta, mask)
    assert out.tolist() == [0, -3, 7, 0]


@pytest.mark.fast
def test_leaves_monomer_consumption_and_nonmonomers_untouched():
    delta = np.array([-4, -8, 2], dtype=np.int64)   # no positive monomer supply
    mask = np.array([True, False, False])
    out = arrest_monomer_supply(delta, mask)
    # Nothing to clamp -> returns the SAME object (callers use identity as no-op).
    assert out is delta


@pytest.mark.fast
def test_noop_when_no_monomers_produced():
    delta = np.array([10, 20, 30], dtype=np.int64)
    mask = np.zeros(3, dtype=bool)                   # no monomers at all
    assert arrest_monomer_supply(delta, mask) is delta


@pytest.mark.fast
def test_does_not_mutate_input():
    delta = np.array([5, 6], dtype=np.int64)
    mask = np.array([True, True])
    out = arrest_monomer_supply(delta, mask)
    assert delta.tolist() == [5, 6]      # input untouched
    assert out.tolist() == [0, 0]
