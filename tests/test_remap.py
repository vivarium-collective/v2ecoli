import numpy as np
from v2ecoli.composites._remap import remap_path, remap_cell_state


def test_bulk_relocates_to_cell_molecules():
    assert remap_path(['bulk']) == ['cell', 'molecules']

def test_bulk_subpath_preserves_tail():
    assert remap_path(['bulk', 'count']) == ['cell', 'molecules', 'count']

def test_unique_rnap_relocates_and_renames():
    assert remap_path(['unique', 'active_RNAP']) == ['cell', 'transcription', 'rna_polymerases']

def test_unique_chromosome_groups_under_chromosome():
    assert remap_path(['unique', 'full_chromosome']) == ['cell', 'chromosome', 'full_chromosome']

def test_listeners_subleaf_relocates():
    assert remap_path(['listeners', 'mass']) == ['cell', 'observables', 'mass']

def test_global_time_relocates_to_clock():
    assert remap_path(['global_time']) == ['clock', 'global_time']

def test_coordination_store_relocates_to_machinery():
    assert remap_path(['process_state', 'polypeptide_elongation']) == \
        ['machinery', 'process_state', 'polypeptide_elongation']

def test_flow_token_is_left_untouched():
    assert remap_path(['_layer_token_3']) == ['_layer_token_3']

def test_unknown_head_is_left_untouched():
    assert remap_path(['agents', '0']) == ['agents', '0']

def test_empty_path_is_noop():
    assert remap_path([]) == []


def _fake_edge():
    # Mimics make_edge output: wires are {port: [path, segments]} lists.
    return {
        '_type': 'step',
        'priority': 1.0,
        'instance': object(),
        '_inputs': {}, '_outputs': {},
        'inputs': {
            'bulk': ['bulk'],
            'active_RNAP': ['unique', 'active_RNAP'],
            'mass': ['listeners', 'mass'],
            'global_time': ['global_time'],
            '_layer_in_1': ['_layer_token_0'],
        },
        'outputs': {
            'bulk': ['bulk'],
            'next': ['next_update_time', 'metabolism'],
        },
    }


def _fake_cell_state():
    return {
        'bulk': np.array([1, 2, 3]),
        'unique': {'active_RNAP': np.array([10, 11]),
                   'full_chromosome': np.array([7])},
        'listeners': {'mass': {'cell_mass': 4.0}},
        'global_time': 0.0,
        'process_state': {'polypeptide_elongation': {'gtp_to_hydrolyze': 0}},
        'ecoli-metabolism': _fake_edge(),
    }


def test_data_stores_move_to_biological_paths():
    out = remap_cell_state(_fake_cell_state())
    assert np.array_equal(out['cell']['molecules'], np.array([1, 2, 3]))
    assert np.array_equal(out['cell']['transcription']['rna_polymerases'], np.array([10, 11]))
    assert np.array_equal(out['cell']['chromosome']['full_chromosome'], np.array([7]))
    assert out['cell']['observables']['mass'] == {'cell_mass': 4.0}
    assert out['clock']['global_time'] == 0.0
    assert out['machinery']['process_state'] == {'polypeptide_elongation': {'gtp_to_hydrolyze': 0}}


def test_old_top_level_keys_are_gone():
    out = remap_cell_state(_fake_cell_state())
    for old in ('bulk', 'unique', 'listeners', 'global_time', 'process_state'):
        assert old not in out


def test_edge_stays_at_root_and_wires_rewritten():
    out = remap_cell_state(_fake_cell_state())
    edge = out['ecoli-metabolism']
    assert edge['_type'] == 'step'
    assert edge['inputs']['bulk'] == ['cell', 'molecules']
    assert edge['inputs']['active_RNAP'] == ['cell', 'transcription', 'rna_polymerases']
    assert edge['inputs']['mass'] == ['cell', 'observables', 'mass']
    assert edge['inputs']['global_time'] == ['clock', 'global_time']
    assert edge['inputs']['_layer_in_1'] == ['_layer_token_0']      # untouched
    assert edge['outputs']['next'] == ['machinery', 'next_update_time', 'metabolism']


def test_input_is_not_mutated():
    src = _fake_cell_state()
    remap_cell_state(src)
    assert 'bulk' in src and 'cell' not in src    # original untouched
