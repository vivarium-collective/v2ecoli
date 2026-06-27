#!/bin/zsh
# Publish the compacted two-state model to Cloudflare R2 (bucket vivarium-3d).
# Uploads meshes + packs + metas, writes models.json + the model-wired
# index.html, all stamped with VERSION. See REPRODUCE.md §7-§8.
# Prereq: 02_stage_compact.py has populated out/_publish/.
set -e

# ── config ───────────────────────────────────────────────────────────────
VERSION=${VERSION:-4}                       # bump on every republish (?v=N cache-bust)
BUILD_ROOT=${BUILD_ROOT:-$PWD}
CREDS=${CREDS:-/Users/eranagmon/code/r2-deploy/creds.env}
PY=${PY:-/Users/eranagmon/code/v2ecoli/.venv/bin/python}
VIEWER_SRC=${VIEWER_SRC:-/Users/eranagmon/code/pbg-parsimony/pbg_parsimony/viewer}
PUB_BASE="https://pub-eb913fbbdc584bd7add047c823570b13.r2.dev"
BIRTH_NAME="Birth — 1.8 µm rod"             # edit counts in the python block below
DIV_NAME="Pre-division — 3.06 µm rod"

set -a; source "$CREDS"; set +a
export AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=us-east-1
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required   # else R2 rejects checksums
EP="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
DST="s3://vivarium-3d/ecoli-3d/viz/3d"
STAGE="$BUILD_ROOT/out/_publish"

# ── models.json + index.html (absolute URLs, ?v=VERSION, PARSIMONY_MODELS) ──
$PY - "$VERSION" "$PUB_BASE" "$STAGE" "$VIEWER_SRC" "$BIRTH_NAME" "$DIV_NAME" <<'PY'
import json, sys
v, pub, stage, vsrc, birth, div = sys.argv[1:7]
base = f"{pub}/ecoli-3d/viz/3d"
# auto-fill counts from the staged compacted packs
def stats(p):
    d = json.load(open(f"{stage}/{p}"))
    return len(d["ingredients"]), len(d["placements"])
bsp, bpl = stats("ecoli_3d.pack.json")
dsp, dpl = stats("ecoli_3d_division.pack.json")
models = [
    {"name": f"{birth} ({bsp} species · {bpl/1e6:.2f}M molecules)",
     "file": f"{base}/ecoli_3d.pack.json?v={v}"},
    {"name": f"{div} ({dsp} species · {dpl/1e6:.2f}M molecules)",
     "file": f"{base}/ecoli_3d_division.pack.json?v={v}"},
]
open(f"{stage}/models.json", "w").write(json.dumps(models, ensure_ascii=True))
# inject window.PARSIMONY_MODELS into a COPY of the repo viewer index.html and
# bump the viewer.js cache-bust stamp (so the bare /viewer/index.html works)
html = open(f"{vsrc}/index.html").read()
inject = '  <script>window.PARSIMONY_MODELS=%s;</script>\n  ' % json.dumps(models, ensure_ascii=False)
anchor = '<script type="module" src="./viewer.js'
html = html.replace(anchor, inject + anchor, 1)
import re
html = re.sub(r'\./viewer\.js\?v=\d+', f'./viewer.js?v={v}', html)
open(f"{stage}/index.html", "w").write(html)
print(f"models.json + index.html staged (v={v}); birth {bsp}sp/{bpl}, div {dsp}sp/{dpl}")
PY

# ── upload ───────────────────────────────────────────────────────────────
echo "==== sync meshes (both states → shared meshes/) ===="; date
aws s3 sync "$BUILD_ROOT/out/ecoli3d/meshes"     "$DST/meshes" --endpoint-url "$EP" --size-only --content-type text/plain --exclude "*.bin" | tail -1
aws s3 sync "$BUILD_ROOT/out/ecoli3d-div/meshes" "$DST/meshes" --endpoint-url "$EP" --size-only --content-type text/plain --exclude "*.bin" | tail -1

echo "==== packs + metas + models.json ===="; date
for f in ecoli_3d.pack.json ecoli_3d.meta.json ecoli_3d_division.pack.json ecoli_3d_division.meta.json models.json; do
  aws s3 cp "$STAGE/$f" "$DST/$f" --endpoint-url "$EP" --content-type application/json
done

echo "==== viewer.js + index.html ===="; date
aws s3 cp "$VIEWER_SRC/viewer.js" s3://vivarium-3d/viewer/viewer.js --endpoint-url "$EP" --content-type application/javascript
aws s3 cp "$STAGE/index.html"     s3://vivarium-3d/viewer/index.html --endpoint-url "$EP" --content-type text/html

echo "PUBLISHED v=$VERSION → $PUB_BASE/viewer/index.html"; date
