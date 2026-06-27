#!/usr/bin/env python3
"""Assemble the LOCAL two-state viewer from the RAW (object-format) packs.

The raw packs keep BUILD-relative mesh URLs (out/ecoli3d/meshes/X), so the
local view needs nested `out/ecoli3d[-div]/meshes` symlinks in addition to the
plain `meshes` symlink. Assembles out/ecoli3d/_view with a model dropdown, then
print the serve command. See REPRODUCE.md §7.

    /Users/eranagmon/code/v2ecoli/.venv/bin/python v2ecoli/structural/publish/03_assemble_local_view.py
    # then:  python -m http.server 8799 --bind 127.0.0.1   (from out/ecoli3d/_view)
"""
import os, shutil, sys
from pathlib import Path

ROOT = Path(os.environ.get("BUILD_ROOT", ".")).resolve()
VSRC = Path(os.environ.get("VIEWER_SRC",
            "/Users/eranagmon/code/pbg-parsimony/pbg_parsimony/viewer"))
VIEW = ROOT / "out/ecoli3d/_view"
STATES = {"birth": ROOT / "out/ecoli3d", "div": ROOT / "out/ecoli3d-div"}
MODELS = [("Newborn (birth)", "data/birth/ecoli_3d.pack.json"),
          ("Pre-division",    "data/div/ecoli_3d.pack.json")]

if VIEW.exists():
    shutil.rmtree(VIEW)
VIEW.mkdir(parents=True)
for f in ("viewer.js", "obj-worker.js", "vr.js"):
    shutil.copy2(VSRC / f, VIEW / f)

html = (VSRC / "index.html").read_text()
models_js = ",".join(f'{{"name":{n!r},"file":{p!r}}}'.replace("'", '"') for n, p in MODELS)
anchor = '<script type="module" src="./viewer.js'
if anchor not in html:
    sys.exit("viewer.js script tag not found in index.html")
html = html.replace(anchor, f'  <script>window.PARSIMONY_MODELS=[{models_js}];</script>\n  ' + anchor, 1)
html = html.replace("./viewer.js?v=50", "./viewer.js?v=local1")
(VIEW / "index.html").write_text(html)

for key, outdir in STATES.items():
    pack = outdir / "ecoli_3d.pack.json"
    meta = outdir / "ecoli_3d.meta.json"
    meshes = outdir / "meshes"
    if not pack.exists() or not meshes.is_dir():
        sys.exit(f"MISSING raw build at {outdir} (run 01_build_states.sh first)")
    dd = VIEW / "data" / key
    dd.mkdir(parents=True)
    os.symlink(pack, dd / "ecoli_3d.pack.json")
    os.symlink(meta, dd / "ecoli_3d.meta.json")
    os.symlink(meshes, dd / "meshes")
    # nested symlink so the build-relative urls (out/<dir>/meshes/X) resolve
    nested = dd / "out" / outdir.name
    nested.mkdir(parents=True)
    os.symlink(meshes, nested / "meshes")
    print(f"{key}: linked {pack.name} + meshes ({len(list(meshes.glob('*.obj')))} objs)")

print(f"\n_view at {VIEW}")
print(f"serve:  (cd {VIEW} && /Users/eranagmon/code/v2ecoli/.venv/bin/python -m http.server 8799 --bind 127.0.0.1)")
print("open:   http://127.0.0.1:8799/")
