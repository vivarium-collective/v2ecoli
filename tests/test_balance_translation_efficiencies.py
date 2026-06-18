"""Regression test for the r-protein translation-efficiency balancing bug.

balanced_translation_efficiencies groups use BARE monomer IDs, but
monomer_data['id'] carries a compartment tag ('...-MONOMER[c]'). The balancing
must strip the tag before matching (mirroring vEcoli's monomer['id'][:-3]);
otherwise no group matches and ribosomal-protein efficiencies are left at their
raw per-gene values — the root cause of the vEcoli<->v2ecoli condition
divergences.
"""
import numpy as np

from v2ecoli.processes.parca.steps.step_02_input_adjustments import (
    balance_translation_efficiencies,
)


def test_balance_strips_compartment_tag_and_averages_group():
    # monomer_ids carry a compartment tag ([c]/[i]); group IDs are bare.
    monomer_ids = np.array(
        ["A-MONOMER[c]", "B-MONOMER[c]", "C-MONOMER[i]", "D-MONOMER[c]"])
    eff = np.array([0.9, 0.7, 0.5, 2.0])
    groups = [["A-MONOMER", "B-MONOMER", "C-MONOMER"]]  # bare ids, mixed compartments

    out = balance_translation_efficiencies(monomer_ids, eff.copy(), groups)

    mean = np.mean([0.9, 0.7, 0.5])
    assert np.allclose(out[:3], mean), \
        "grouped monomers must be balanced to their mean despite [c]/[i] tags"
    assert out[3] == 2.0, "a non-group monomer must be untouched"


def test_balance_is_noop_when_group_absent():
    monomer_ids = np.array(["X-MONOMER[c]"])
    eff = np.array([1.23])
    out = balance_translation_efficiencies(monomer_ids, eff.copy(), [["Z-MONOMER"]])
    assert out[0] == 1.23
