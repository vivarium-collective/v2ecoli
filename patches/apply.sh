#!/usr/bin/env bash
# Re-apply the workbench patches (edge relations + authored confidence)
# after a reinstall.
set -euo pipefail
cd "$(dirname "$0")/.."
P=.venv/lib/python3.12/site-packages
python3 - <<'PY'
import pathlib
P=pathlib.Path(".venv/lib/python3.12/site-packages")
f=P/"vivarium_workbench/lib/investigation_graph_views.py"; s=f.read_text()
if '"relation": pre.get("relation")' not in s:
    s=s.replace('''                                   "rel": "prerequisite",
                                   "condition": explicit.get(pre["study"], "")})''',
'''                                   "rel": "prerequisite",
                                   "relation": pre.get("relation"),
                                   "outputs_used": pre.get("outputs_used"),
                                   "condition": explicit.get(pre["study"], "")})''')
    f.write_text(s); print("server patched")
f=P/"vivarium_workbench/static/walkthrough.js"; s=f.read_text()
if "e.relation ||" not in s:
    s=s.replace("""        relation: e.artifact ? 'model-input' : 'leads-to',""",
                """        relation: e.relation || (e.artifact ? 'model-input' : 'leads-to'),""")
    f.write_text(s); print("client patched")

f=P/"vivarium_workbench/lib/investigation_graph_views.py"; s=f.read_text()
if '_authored = str(study_spec.get("confidence")' not in s:
    s=s.replace('''            _node["confidence"] = _dc(study_spec)''',
'''            _authored = str(study_spec.get("confidence") or "").strip()
            _valid = {"Accepted", "Investigating", "Refuted", "Planned"}
            _node["confidence"] = _authored if _authored in _valid else _dc(study_spec)''')
    f.write_text(s); print("confidence patched")

# --- rail dot: colour by confidence, not raw gate status ---
f=P/"vivarium_workbench/lib/investigations_index.py"; s=f.read_text()
if "_display_confidence" not in s:
    s=s.replace('            "conclusions_excerpt": _conclusions_excerpt(spec),',
                '            "conclusions_excerpt": _conclusions_excerpt(spec),\n'
                '            "confidence": _display_confidence(spec),')
    s=s.replace("def _conclusions_excerpt(spec: dict, limit: int = 240) -> str:",
        'def _display_confidence(spec: dict):\n'
        '    authored = str(spec.get("confidence") or "").strip()\n'
        '    if authored in {"Accepted", "Investigating", "Refuted", "Planned"}:\n'
        '        return authored\n'
        '    try:\n'
        '        from viva_superpowers.study_verdict import derive_confidence\n'
        '        return derive_confidence(spec)\n'
        '    except Exception:\n'
        '        return None\n\n\n'
        "def _conclusions_excerpt(spec: dict, limit: int = 240) -> str:")
    f.write_text(s); print("rail server patched")

f=P/"vivarium_workbench/static/walkthrough.js"; s=f.read_text()
if "_railConfidenceColor" not in s:
    s=s.replace("    var color = _railStatusColor(status);",
                "    var color = s.confidence ? _railConfidenceColor(s.confidence) : _railStatusColor(status);")
    s=s.replace("""    var tip = _esc(s.name) + ' — ' + _esc(status) + (s.blocked ? ' (blocked)' : '');""",
                """    var tip = _esc(s.name) + ' — ' + (s.confidence ? _esc(s.confidence) + ' (gate: ' + _esc(status) + ')' : _esc(status)) + (s.blocked ? ' (blocked)' : '');""")
    s=s.replace("  function _railStatusColor(status) {",
        "  function _railConfidenceColor(confidence) {\n"
        "    var c = String(confidence || '').toLowerCase();\n"
        "    if (c === 'accepted') return '#16a34a';\n"
        "    if (c === 'investigating') return '#f59e0b';\n"
        "    if (c === 'refuted') return '#ef4444';\n"
        "    return '#9ca3af';\n"
        "  }\n\n"
        "  function _railStatusColor(status) {")
    f.write_text(s); print("rail client patched")
PY
