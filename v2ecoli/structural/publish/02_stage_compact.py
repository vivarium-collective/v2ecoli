#!/usr/bin/env python3
"""Stage compacted array8 packs + metas for R2 publish.

Reads the raw object-format packs from out/ecoli3d{,-div}/ and writes
compacted (array8, pack-relative mesh URLs) copies to out/_publish/, renaming
the division pack to the published `ecoli_3d_division` name. See REPRODUCE.md §6.

Run with the v2ecoli venv python from the worktree root:
    /Users/eranagmon/code/v2ecoli/.venv/bin/python v2ecoli/structural/publish/02_stage_compact.py
"""
import json, os, shutil, sys
from pathlib import Path

ROOT = Path(os.environ.get("BUILD_ROOT", ".")).resolve()
STAGE = ROOT / "out/_publish"
from v2ecoli.structural.build import compact_to_array8  # noqa: E402

JOBS = [
    (ROOT / "out/ecoli3d",     "ecoli_3d"),           # birth
    (ROOT / "out/ecoli3d-div", "ecoli_3d_division"),  # pre-division
]

if STAGE.exists():
    shutil.rmtree(STAGE)
STAGE.mkdir(parents=True)

for outdir, name in JOBS:
    src_pack, src_meta = outdir / "ecoli_3d.pack.json", outdir / "ecoli_3d.meta.json"
    dst_pack, dst_meta = STAGE / f"{name}.pack.json", STAGE / f"{name}.meta.json"
    if not src_pack.exists():
        sys.exit(f"MISSING raw pack: {src_pack} (run 01_build_states.sh first)")
    shutil.copy2(src_pack, dst_pack)
    shutil.copy2(src_meta, dst_meta)
    raw_mb = dst_pack.stat().st_size / 1e6
    compact_to_array8(dst_pack)
    d = json.loads(dst_pack.read_text())
    sample = next((l["url"] for ing in d["ingredients"]
                   for l in ing.get("shape", {}).get("lods", [])), None)
    print(f"{name}: {raw_mb:.0f}MB -> {dst_pack.stat().st_size/1e6:.0f}MB array8 | "
          f"{len(d['placements'])} placements, {len(d['ingredients'])} species | url0={sample}")

print(f"\nstaged at {STAGE}")
