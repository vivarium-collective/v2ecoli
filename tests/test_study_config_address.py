"""#602: study_config()'s content address must not omit result-changing knobs.

The address decides whether a previously computed result still describes the
current declaration. It used to be a WHITELIST that omitted five keys that each
change what is computed — so `change a knob and re-grade`, the normal working
loop, served the stale result. It is now a fail-safe denylist.
"""
import pytest

from v2ecoli.library.comparison_composite import study_config


@pytest.mark.fast
def test_the_five_formerly_omitted_knobs_now_enter_the_address():
    """from_vecoli_config / inject_processes / exchange_flux_basis /
    exchange_fluxes / observable_bulk_ids each change what is computed."""
    declaration = {
        'condition': 'basal',
        'comparison': {
            'seeds': 4, 'generations': 8,
            'from_vecoli_config': 'configs/test_violacein_with_metabolism.json',
            'inject_processes': ['metabolite_counts_listener'],
            'exchange_flux_basis': 'gdcw',
            'exchange_fluxes': {'violacein_exchange': 'VIOLACEIN[c]'},
            'observable_bulk_ids': ['VIOLACEIN[c]'],
        }}
    addr = study_config(declaration)
    for key in ('from_vecoli_config', 'inject_processes', 'exchange_flux_basis',
                'exchange_fluxes', 'observable_bulk_ids'):
        assert key in addr, f"{key} omitted from the content address (#602)"
    # And the shape that motivated the fix actually moves the address now.
    other = dict(declaration, comparison=dict(declaration['comparison'],
                                              exchange_flux_basis='counts'))
    assert study_config(declaration) != study_config(other)


@pytest.mark.fast
def test_engine_identity_keys_are_exempt_addressed_via_the_input_edges():
    declaration = {
        'condition': 'basal',
        'comparison': {
            'seeds': 4,
            'candidate': 'v2ecoli',
            'reference': {'repo': 'env:V2E_VECOLI_DIR', 'kind': 'vecoli'}}}
    addr = study_config(declaration)
    assert 'candidate' not in addr and 'reference' not in addr
    assert addr == {'condition': 'basal', 'seeds': 4}


@pytest.mark.fast
def test_it_is_fail_safe_a_brand_new_comparison_knob_is_addressed_by_default():
    """The structural point: a key nobody taught this function about is addressed
    unless deliberately exempted — the reverse of the whitelist that omitted every
    new key silently."""
    base = {'condition': 'basal', 'comparison': {'seeds': 4}}
    with_knob = {'condition': 'basal',
                 'comparison': {'seeds': 4, 'a_future_knob': 'changes-the-result'}}
    assert 'a_future_knob' in study_config(with_knob)
    assert study_config(base) != study_config(with_knob)


@pytest.mark.fast
def test_prose_status_and_outcomes_still_never_reach_the_address():
    """They live at the top level, not under comparison:, so they are untouched."""
    declaration = {
        'condition': 'acetate',
        'comparison': {'seeds': 4, 'generations': 4},
        'title': 'anything', 'status': 'evaluated', 'runs': [{'result': 'PASS'}]}
    assert study_config(declaration) == {
        'condition': 'acetate', 'seeds': 4, 'generations': 4}


@pytest.mark.fast
def test_list_values_are_copied_not_referenced():
    comparison = {'inject_processes': ['a', 'b']}
    addr = study_config({'condition': 'basal', 'comparison': comparison})
    comparison['inject_processes'].append('c')
    assert addr['inject_processes'] == ['a', 'b']
