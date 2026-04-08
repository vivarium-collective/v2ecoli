"""Test that RNAP count at initialization matches expected value.

This test catches the ppGpp unit conversion bug where .asNumber()
without target units (mol/L) vs .asNumber(umol/L) caused the RNAP
active fraction to be wrong, leading to 2x too many active RNAPs.
"""

import pytest
import numpy as np
from tests.conftest import skip_no_cache


@skip_no_cache
class TestRnapCount:

    EXPECTED_INIT_RNAP = 717  # From initial_state.json
    MAX_INIT_DEVIATION = 10  # Allow small deviation from init steps

    def test_rnap_count_at_init(self, dep_composite):
        """RNAP count after Composite init should be close to initial state."""
        cell = dep_composite.state.get('agents', {}).get('0', {})
        unique = cell.get('unique', {})
        rnap = unique.get('active_RNAP')
        if rnap is not None and hasattr(rnap, 'dtype') and '_entryState' in rnap.dtype.names:
            n_rnap = int(rnap['_entryState'].view(np.bool_).sum())
        else:
            pytest.skip("No active_RNAP unique molecules")

        assert abs(n_rnap - self.EXPECTED_INIT_RNAP) <= self.MAX_INIT_DEVIATION, (
            f"RNAP count {n_rnap} deviates too much from expected {self.EXPECTED_INIT_RNAP}. "
            f"This may indicate a ppGpp unit conversion bug in the function registry "
            f"(check .asNumber() calls use umol/L target units).")

    def test_rnap_count_after_60s(self, dep_composite_fresh):
        """RNAP count after 60s should be in reasonable range (600-900)."""
        comp = dep_composite_fresh
        comp.run(60.0)

        cell = comp.state.get('agents', {}).get('0', {})
        unique = cell.get('unique', {})
        rnap = unique.get('active_RNAP')
        if rnap is not None and hasattr(rnap, 'dtype') and '_entryState' in rnap.dtype.names:
            n_rnap = int(rnap['_entryState'].view(np.bool_).sum())
        else:
            pytest.skip("No active_RNAP unique molecules")

        assert 600 <= n_rnap <= 900, (
            f"RNAP count {n_rnap} is outside expected range [600, 900] after 60s. "
            f"Too high suggests ppGpp regulation is not working correctly.")
