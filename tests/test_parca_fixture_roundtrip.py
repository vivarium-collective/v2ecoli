"""Fixture-roundtrip tests for the shipped ParCa state.

Loads ``models/parca/parca_state.pkl.gz`` via
``v2ecoli.processes.parca.data_loader.load_parca_state`` and asserts the
structural invariants downstream consumers depend on: top-level keyset,
subsystem class identities, and presence of the array-valued fields that
the online simulation reads from.

Fast tests — no ParCa execution, just a pickle load + attribute checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from v2ecoli.processes.parca.data_loader import load_parca_state


EXPECTED_TOP_LEVEL_KEYS = {
    'adjustments', 'cell_specs', 'condition', 'condition_active_tfs',
    'condition_inactive_tfs', 'condition_to_doubling_time', 'conditions',
    'constants', 'expected_dry_mass_increase_dict', 'external_state',
    'getter', 'growth_rate_parameters', 'internal_state', 'mass',
    'molecule_groups', 'molecule_ids', 'pPromoterBound', 'process',
    'relation', 'sim_data_root', 'tf_to_active_inactive_conditions',
    'tf_to_direction', 'tf_to_fold_change', 'translation_supply_rate',
}

EXPECTED_SUBSYSTEMS = {
    'complexation', 'equilibrium', 'metabolism', 'replication', 'rna_decay',
    'transcription', 'transcription_regulation', 'translation',
    'two_component_system',
}


@pytest.fixture(scope='module')
def state():
    return load_parca_state()


def test_top_level_keys(state):
    assert set(state.keys()) == EXPECTED_TOP_LEVEL_KEYS


def test_process_subsystems(state):
    assert set(state['process'].keys()) == EXPECTED_SUBSYSTEMS


def test_subsystems_are_real_classes(state):
    # Not dicts, not MagicMocks — the pickle legacy-alias machinery must
    # resolve subsystem classes back to their dataclass modules.
    base = 'v2ecoli.processes.parca.reconstruction.ecoli.dataclasses'
    for name, obj in state['process'].items():
        mod = type(obj).__module__
        assert mod.startswith(base), (
            f'subsystem {name} is {type(obj).__module__}.{type(obj).__name__}, '
            f'expected module under {base}')


def test_mass_and_constants(state):
    from v2ecoli.processes.parca.reconstruction.ecoli.dataclasses.growth_rate_dependent_parameters import Mass
    from v2ecoli.processes.parca.reconstruction.ecoli.dataclasses.constants import Constants
    assert isinstance(state['mass'], Mass)
    assert isinstance(state['constants'], Constants)


def test_transcription_array_fields(state):
    tx = state['process']['transcription']
    for field in ('cistron_data', 'rna_data', 'mature_rna_data'):
        assert hasattr(tx, field), f'Transcription missing {field}'
        value = getattr(tx, field)
        assert value is not None
    # rna_data['deg_rate'] is a Unum quantity; asNumber() unwraps to ndarray
    deg = tx.rna_data['deg_rate']
    arr = deg.asNumber() if hasattr(deg, 'asNumber') else np.asarray(deg)
    assert arr.size > 0


def test_cell_specs_has_basal(state):
    cs = state['cell_specs']
    assert 'basal' in cs
    basal = cs['basal']
    for field in ('expression', 'synthProb', 'doubling_time'):
        assert field in basal, f'cell_specs["basal"] missing {field}'


def test_condition_to_doubling_time_shape(state):
    d = state['condition_to_doubling_time']
    assert isinstance(d, dict)
    assert len(d) > 0
    # Values should be unit-carrying Quantity-likes with an ``asNumber()``.
    some = next(iter(d.values()))
    assert hasattr(some, 'asNumber') or isinstance(some, (int, float, np.number))
