from v2ecoli.composites._remap import remap_path


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
