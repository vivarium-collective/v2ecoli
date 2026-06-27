# Reproducing the v2ecoli 3D transcription/translation model

This directory pins **every component** that produced the online two-state 3D
molecular model (birth + pre-division E. coli) hosted on Cloudflare R2, so the
exact build can be regenerated and republished reliably.

**Live viewer:** https://pub-eb913fbbdc584bd7add047c823570b13.r2.dev/viewer/index.html

What the model shows (true abundance, all confined inside the cell envelope):
the chromosome at real genomic geometry, every active RNAP at its real locus,
nascent RNA off each RNAP, free mRNA, active ribosomes + free 30S/50S subunits,
nascent peptide coils, enlarged replication markers (oriC/terC/replisome), and —
for the dividing cell — a septum-confined two-nucleoid pre-division state.

---

## 1. Component pins (the three repos)

| Repo | Path | Commit | What it provides |
|---|---|---|---|
| **parsimony** (Rust engine) | `~/code/parsimony` | `7d09ba2` (branch `main`) | geometry: `strand_point` (bp→3D), `generate_rna_strand`, `bubble_point`, `place_chromosome`/`place_translation`, septum-aware `CellShape::Capsule` confinement |
| **pbg-parsimony** (Python API + viewer) | `~/code/pbg-parsimony` | `a3e54f4` (base) + `217af97` (branch `feat/viewer-crowding-default`) | `Chromosome`/`Ingredient`/`build_pack`; the WebGL viewer with lowered default crowding |
| **v2ecoli** (state → ingredients) | `~/code/v2ecoli` | `f008f61b` (branch `main`, incl. PR #297) | `v2ecoli/structural/build.py` — state extraction, curated species, septum wiring, `compact_to_array8` |

`pbg-parsimony` must be **editable-installed** into the v2ecoli venv at the
pinned commit:
```bash
/Users/eranagmon/code/v2ecoli/.venv/bin/pip install -e /Users/eranagmon/code/pbg-parsimony --no-deps
```
The Rust binary must be built at the pinned parsimony commit:
```bash
cd /Users/eranagmon/code/parsimony && cargo build --release -p parsimony-cli
# → target/release/parsimony   (PARSIMONY_HOME points here)
```

## 2. Build environment

- **Interpreter:** `/Users/eranagmon/code/v2ecoli/.venv/bin/python` (has `unum`/`xarray`; bare `python` does not).
- **`PARSIMONY_HOME=/Users/eranagmon/code/parsimony`** (so the build finds the release binary).
- **ParCa cache:** the build needs `out/cache` (ParCa output). In a fresh
  worktree, symlink it to a populated checkout:
  `ln -s /Users/eranagmon/code/v2ecoli/out/cache <workdir>/out/cache`.
- **`.parsimony/cache` is recipe-keyed, NOT binary-keyed** — `rm -rf .parsimony/cache`
  after ANY parsimony Rust change or the geometry stage silently reuses the old result.
- **Stale-branch / worktree hazard:** `git worktree remove` of the build worktree
  **wipes the gitignored `out/`** (packs + meshes + `_view`). `main` is usually held
  by another worktree, so build from a fresh `git worktree add --detach <path> <main-sha>`.

## 3. Inputs

- **Cell state snapshots** (committed, in `v2ecoli/structural/data/`):
  - `v2ecoli_state.npz` — birth (`--state snapshot`)
  - `v2ecoli_state_division.npz` — pre-division (`--state division`)
  These carry the unique-molecule arrays (`active_RNAP`, `RNA`, `active_ribosome`,
  `chromosome_domain`, …) the geometry is built from.
- **Structure cache (reuse to avoid net fetch):** pre-seed `out/ecoli3d/structures`
  from `~/code/v2e-pdmp-refresh/out/ecoli3d_expanded/structures` (1074 cached
  PDB/CIF/AlphaFold files). Uncached species fetch from AlphaFold on demand.

## 4. Build parameters (THIS version)

| Param | Value | Meaning |
|---|---|---|
| `--top-n` | **400** | abundant AlphaFold protein monomers added beyond the curated set |
| `top_complexes` | 150 (build_model default) | assembled CPLX blobs from the bulk |
| `--scale` | 1.0 | true abundance (1 placement per real copy) |
| states | `snapshot` + `division` | the two models |

**`--top-n` is the species-richness lever.** It adds protein monomers *on top of*
the model without disturbing the chromosome/RNAP/RNA/ribosome structure — the
packer places curated + `pack_first` big assemblies + the chromosome stage FIRST,
so added monomers fill remaining interior space. Scaling observed:
`top_n=40` → 205/206 species (1.66M/2.86M placements); `top_n=400` → **563/564
species (1.90M/3.25M placements)**. ~3484 eligible monomers exist (the ceiling).

## 5. Pipeline (run in order, from the v2ecoli worktree root)

```bash
cd <v2ecoli-worktree>
bash v2ecoli/structural/publish/01_build_states.sh      # build birth + division packs (top_n=400)
python v2ecoli/structural/publish/02_stage_compact.py   # compact_to_array8 → out/_publish/
python v2ecoli/structural/publish/03_assemble_local_view.py  # local two-state viewer (serve :8799)
bash v2ecoli/structural/publish/04_publish_r2.sh         # upload packs+meshes+models.json+index.html to R2
```

Each script has a config block at the top (paths, `TOP_N`, R2 `VERSION`). To bump
the published version, change `VERSION` in `04_publish_r2.sh` (it stamps `?v=N`
on the pack URLs in both `models.json` and the injected `index.html`).

## 6. Compaction (why + how)

The raw packs are object-format with **build-relative** mesh URLs
(`out/ecoli3d/meshes/X`). `compact_to_array8` (in `build.py`) rewrites them to:
- **array8 placements** `[id, x,y,z, w,qx,qy,qz]` — ~¼ the size (385MB→88MB, 659MB→151MB)
- **pack-relative** mesh URLs (`meshes/X`) — resolve next to the pack wherever hosted

Compacted packs are what R2 serves; the raw object-format packs are used by the
**local** viewer (which needs nested `out/ecoli3d[-div]/meshes` symlinks because
their URLs stay build-relative).

## 7. Viewer configuration

- **Default crowding:** `pbg-parsimony` `viewer.js` `TARGET_DRAWN = 75000`
  (commit `217af97`); whole-cell packs default to ~75k drawn instances. Bump the
  `viewer.js?v=` stamp in `index.html` whenever `viewer.js` changes or browsers
  serve the cached old JS.
- **Bare-URL model wiring (critical):** the viewer loads models from
  `window.PARSIMONY_MODELS` or `?models=<url>`; with NEITHER it falls back to a
  non-existent `data/demo.pack.json` (404) and the page hangs on "loading" —
  looks like a perf stall but is NOT. `04_publish_r2.sh` injects
  `window.PARSIMONY_MODELS=[…]` (absolute pack URLs) into the deployed
  `index.html`, so the bare `/viewer/index.html` shows the dropdown directly.

## 8. R2 publish target

- Bucket `vivarium-3d`; creds `~/code/r2-deploy/creds.env`
  (`R2_ACCESS_KEY_ID/R2_SECRET_ACCESS_KEY/R2_ACCOUNT_ID/R2_BUCKET`).
- S3 endpoint `https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com`; public base
  `https://pub-eb913fbbdc584bd7add047c823570b13.r2.dev`.
- Layout: viewer assets at `s3://vivarium-3d/viewer/`; packs + metas + meshes +
  `models.json` at `s3://vivarium-3d/ecoli-3d/viz/3d/` (birth=`ecoli_3d`,
  division=`ecoli_3d_division`, shared `meshes/`).
- aws needs `AWS_REQUEST_CHECKSUM_CALCULATION=when_required` or R2 rejects
  checksums. R2 has **no 100MB file limit** (that's git/gh-pages).
- Shared-name meshes are structure-derived → identical across the two states,
  safe to union into one `meshes/`.

## 9. Output of THIS version (verification targets)

| Model | Species | Placements | Compacted pack | R2 file |
|---|---|---|---|---|
| Birth — 1.8 µm rod | 563 | 1.90M | 88 MB | `ecoli_3d.pack.json?v=4` |
| Pre-division — 3.06 µm rod | 564 | 3.25M | 151 MB | `ecoli_3d_division.pack.json?v=4` |

Septum check (division): neck (|x|<800) radial>3500 protrusions = **0**
(was 22,565 before the septum fix); max neck radial 3198 (was 4997).
