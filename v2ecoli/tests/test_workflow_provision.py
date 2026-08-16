import os, pytest
from process_bigraph import allocate_core, Composite
from process_bigraph.composite_generator import _REGISTRY, apply_core_extensions, build_generator

CACHE = '/Users/eranagmon/code/v2ecoli/out/cache'


@pytest.mark.skipif(not os.path.isdir(CACHE), reason='no ParCa cache')
def test_baseline_on_bare_core_via_core_extensions():
    entry = next(e for e in _REGISTRY.values() if e.name == 'ecoli_baseline')
    assert entry.core_extensions
    core = apply_core_extensions(entry, allocate_core())        # NO build_core()
    comp = Composite(build_generator(entry, overrides={'seed': 0, 'cache_dir': CACHE}, core=core), core=core)
    comp.run(5.0)
    assert comp.state.get('global_time') == 5.0


def test_register_types_hook():
    import v2ecoli
    assert v2ecoli.register_types is v2ecoli.core.register_ecoli_core
