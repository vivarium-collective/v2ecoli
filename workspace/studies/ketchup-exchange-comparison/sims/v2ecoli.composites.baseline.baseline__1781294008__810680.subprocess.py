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
    _payload = {'spec_id': 'v2ecoli.composites.baseline.baseline', 'overrides': {'cache_dir': 'out/cache'}, 'run_id': 'v2ecoli.composites.baseline.baseline__1781294008__810680', 'db_file': '/Users/eranagmon/code/v2e-ketchup-compare/studies/ketchup-exchange-comparison/runs.db', 'steps': 4, 'emit_paths': [], 'default_emitter': 'parquet', 'max_generations': 8, 'single_daughters': False, 'zarr_store': '/Users/eranagmon/code/v2e-ketchup-compare/studies/ketchup-exchange-comparison/runs.v2ecoli.composites.baseline.baseline__1781294008__810680.zarr'}
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
    _use_xarray = _payload.get('default_emitter') == 'xarray'
    _view = []
    if _use_xarray:
        # Auto-view from the study's declared observables. v0 of
        # view_from_emit_paths is scalar-only — vector observables
        # (monomer_counts, fork_coordinates, RNAP_coordinates, …)
        # are skipped. If a study declares ONLY vector observables
        # (e.g. dnaa-01 emits only listeners.monomer_counts), the
        # auto-view is empty and the XArrayEmitter constructor
        # would crash. In that case, fall back to SQLite for this
        # run so the study isn't blocked.
        from v2ecoli.library.xarray_run import (
            run_multigen_xarray, view_from_emit_paths,
        )
        _view = view_from_emit_paths(_payload.get('emit_paths') or [])
        if not _view:
            print('[xarray-run] auto-view is empty (all declared '
                  'observables are vector / non-listeners-rooted); '
                  'falling back to SQLite emitter for this run.',
                  file=sys.stderr)
            _use_xarray = False
    if _use_xarray:
        # XArray multi-gen path: drive the composite externally past
        # divisions, per-generation emitter swap; results land in a
        # partitioned zarr store. See v2ecoli plan
        # 2026-05-12-migrate-emitters.md task 7.x.
        composite = Composite({'state': state}, core=core)
        _md = {
            'experiment_id': _payload['run_id'],
            'variant': 0,
            'lineage_seed': 0,
            'time_step': 1.0,
            'max_duration': float(_payload['steps']),
        }
        _xarr = run_multigen_xarray(
            composite,
            store_path=_payload['zarr_store'],
            view=_view,
            metadata_base=_md,
            max_steps=_payload['steps'],
            max_generations=_payload['max_generations'],
        )
        results = {'zarr_store': _xarr['store'],
                   'generations': _xarr['generations'],
                   'steps': _xarr['steps']}
    else:
        _mg = int(_payload.get('max_generations') or 1)
        if _mg > 1:
            # Multi-gen: workspace-side runner drives the
            # SQLiteEmitter externally (mirrors how the
            # xarray branch drives XArrayEmitter). The
            # composite does NOT get an injected emitter —
            # the static `agents/0/...` wiring would write
            # empty rows after division. The runner extracts
            # the followed agent's state each chunk and
            # calls `emitter.update` with it; on division it
            # switches to the daughter agent_id.
            composite = Composite({'state': state}, core=core)
            from v2ecoli.library.sqlite_run import run_multigen_sqlite
            _sq = run_multigen_sqlite(
                composite,
                run_id=_payload['run_id'],
                db_file=_payload['db_file'],
                emit_paths=_payload.get('emit_paths') or [],
                max_steps=_payload['steps'],
                max_generations=_mg,
                single_daughters=bool(_payload.get('single_daughters')),
                core=core,
            )
            results = {'steps': _sq['steps'],
                       'generations': _sq['generations']}
        else:
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
