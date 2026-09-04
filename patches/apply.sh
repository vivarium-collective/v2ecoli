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
PY
