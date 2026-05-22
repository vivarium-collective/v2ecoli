import json, sys, traceback
try:
    from v2ecoli.core import build_core
    from process_bigraph import Composite, gather_emitter_results
    from process_bigraph.emitter import SQLiteEmitter
    from pbg_superpowers.composite_generator import (
        _REGISTRY, build_generator, discover_generators,
        apply_core_extensions,
    )
    from vivarium_dashboard.lib import composite_runs as cr
    from bigraph_schema.json_codec import BigraphJSONEncoder as _BJE
    _payload = {'spec_id': 'v2ecoli.composites.baseline_recipes.dnaa_02_with_intrinsic_hydrolysis', 'overrides': {'seed': 0, 'cache_dir': 'out/cache', 'mechanism': 'rida-v0', 'rida_rate_per_min': 40.0, 'stage1_dnaa_expression': True, 'enable_data': True, 'data_rate_per_min': 12.0, 'enable_dars': True, 'dars1_rate_per_min': 5.0, 'dars2_rate_per_min': 10.0}, 'run_id': 'v2ecoli.composites.baseline_recipes.dnaa_02_with_intrinsic_hydrolysis__1779422054__761c36', 'db_file': '/Users/eranagmon/code/v2ecoli/.claude/worktrees/dnaa-mock-investigation-start/studies/dnaa-06-extrinsic-regulation/runs.db', 'steps': 3600, 'emit_paths': ['agents/0/listeners/dnaA_cycle/atp_fraction', 'listeners/dnaA_cycle/atp_fraction']}
    if not _REGISTRY: discover_generators()
    entry = _REGISTRY[_payload['spec_id']]
    core = build_core()
    core.register_link('SQLiteEmitter', SQLiteEmitter)
    # v2ecoli friction #16: register types/processes the composite
    # needs from packages build_core() doesn't know about (declared
    # via @composite_generator(core_extensions=[...])).
    core = apply_core_extensions(entry, core)
    doc = build_generator(entry, overrides=_payload['overrides'])
    state = doc.get('state', doc) if isinstance(doc, dict) else doc
    if _payload.get('emit_paths'):
        state = cr.inject_emitter_for_declared_paths(state, _payload['emit_paths'])
    state = cr.inject_sqlite_emitter(
        state, run_id=_payload['run_id'], db_file=_payload['db_file'])
    composite = Composite({'state': state}, core=core)
    cr.run_with_division(composite, _payload['steps'])
    results = gather_emitter_results(composite)

    # Flatten tuple keys to JSON-friendly dotted strings
    out = {}
    for path_tuple, entries in results.items():
        key = '.'.join(str(p) for p in path_tuple)
        out[key] = entries
    # Gather rendered viz HTML, if pbg_superpowers is importable.
    viz_html = {}
    try:
        from pbg_superpowers.visualization import render_results
        rendered = render_results(composite)
        for path_tuple, payload in rendered.items():
            key = '.'.join(str(p) for p in path_tuple)
            viz_html[key] = payload
    except Exception:
        viz_html = {}
    from bigraph_schema.json_codec import BigraphJSONEncoder as _BJE
    print('@@@RESULTS@@@')
    print(json.dumps({'results': out, 'viz_html': viz_html}, cls=_BJE))
except Exception as e:
    print('@@@ERROR@@@')
    print(traceback.format_exc())
