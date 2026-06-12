import json, sys, traceback
try:
    from v2ecoli.core import build_core
    from process_bigraph import Composite, gather_emitter_results
    from process_bigraph.emitter import SQLiteEmitter
    from bigraph_schema.json_codec import bigraph_json_hook
    from vivarium_dashboard.lib import composite_runs as cr
    core = build_core()
    core.register_link('SQLiteEmitter', SQLiteEmitter)
    with open('/var/folders/vy/vr0_ytms6m95qrnk7xnh0bth0000gp/T/vivarium-run-sa2rj8mt.state.json') as _sf:
        _state = json.load(_sf, object_hook=bigraph_json_hook)
    composite = Composite({'state': _state}, core=core)
    cr.run_with_division(composite, 3)
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
