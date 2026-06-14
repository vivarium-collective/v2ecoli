import numpy as np
from v2ecoli.composites._remap import remap_path, remap_cell_state


def test_bulk_relocates_to_cell_molecules():
    assert remap_path(['bulk']) == ['cell', 'molecules']

def test_bulk_subpath_preserves_tail():
    assert remap_path(['bulk', 'count']) == ['cell', 'molecules', 'count']

def test_unique_relocates_whole_under_cell():
    # Phase 1: `unique` relocates WHOLE (like `bulk`), not split per-molecule —
    # division + the mass listeners consume the whole `unique` store through a
    # single map-typed port, so molecules must stay co-located. The per-molecule
    # biological split is deferred to Phase 2 (see UNIQUE_REMAP_PHASE2).
    assert remap_path(['unique']) == ['cell', 'unique_molecules']
    assert remap_path(['unique', 'active_RNAP']) == ['cell', 'unique_molecules', 'active_RNAP']

def test_unique_chromosome_molecule_stays_co_located():
    assert remap_path(['unique', 'full_chromosome']) == ['cell', 'unique_molecules', 'full_chromosome']

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
    assert np.array_equal(out['cell']['unique_molecules']['active_RNAP'], np.array([10, 11]))
    assert np.array_equal(out['cell']['unique_molecules']['full_chromosome'], np.array([7]))
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
    assert edge['inputs']['active_RNAP'] == ['cell', 'unique_molecules', 'active_RNAP']
    assert edge['inputs']['mass'] == ['cell', 'observables', 'mass']
    assert edge['inputs']['global_time'] == ['clock', 'global_time']
    assert edge['inputs']['_layer_in_1'] == ['_layer_token_0']      # untouched
    assert edge['outputs']['next'] == ['machinery', 'next_update_time', 'metabolism']


def test_input_is_not_mutated():
    src = _fake_cell_state()
    remap_cell_state(src)
    assert 'bulk' in src and 'cell' not in src    # original untouched


def test_biological_wraps_baseline_and_remaps(monkeypatch):
    import v2ecoli.composites.biological as biomod

    def fake_baseline(**kwargs):
        return {
            'state': {
                'agents': {'0': {
                    'bulk': [1],
                    'unique': {'active_RNAP': [2]},
                    'listeners': {'mass': {}},
                    'global_time': 0.0,
                    'emitter': {'_type': 'step', 'inputs': {'b': ['bulk']},
                                'outputs': {}},
                }},
                'global_time': 0.0,
            },
            'skip_initial_steps': True,
            'sequential_steps': False,
            'flow_order': ['emitter'],
        }

    monkeypatch.setattr(biomod, 'baseline', fake_baseline)
    doc = biomod.biological(seed=0)
    agent = doc['state']['agents']['0']
    assert set(agent) >= {'cell', 'clock', 'emitter'}
    assert 'bulk' not in agent and 'unique' not in agent and 'listeners' not in agent
    assert agent['emitter']['inputs']['b'] == ['cell', 'molecules']
    # The outer document scaffolding is preserved verbatim.
    assert doc['skip_initial_steps'] is True
    assert doc['flow_order'] == ['emitter']


def test_edge_instance_not_deepcopied_and_survives_unpicklable():
    """Regression: remap must SHALLOW-copy edges. An edge holds a live process
    instance (e.g. ParquetEmitter, which owns a _queue.SimpleQueue) that is
    unpicklable and must stay the SAME shared object — deep-copying it both
    crashes (cannot pickle SimpleQueue) and would wrongly clone the process."""
    import queue
    from v2ecoli.composites._remap import remap_cell_state

    class _Unpicklable:
        def __init__(self):
            self.q = queue.SimpleQueue()  # not deep-copyable / picklable

    inst = _Unpicklable()
    cell_state = {
        'bulk': np.array([1]),
        'emitter': {'_type': 'step', 'instance': inst,
                    'inputs': {'b': ['bulk']}, 'outputs': {}},
    }
    out = remap_cell_state(cell_state)              # must not raise
    assert out['emitter']['instance'] is inst       # shared, not cloned
    assert out['emitter']['inputs']['b'] == ['cell', 'molecules']
    # original edge's wires untouched (no-mutation contract)
    assert cell_state['emitter']['inputs']['b'] == ['bulk']
