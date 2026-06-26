# 3D Transcription — BF2: division multi-chromosome routing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** In the division state (`n_chromosomes ≥ 2`) route each RNAP/RNA to its OWN chromosome (so both chromosomes are populated, not just chromosome 0), with daughter-domain entries overlaid on that chromosome's bubble — resolving the whole-branch review's Important finding.

**Architecture:** v2ecoli capture computes, per RNAP, a `chromosome_index` (which `full_chromosome` lineage contains its `domain_index`, via the `chromosome_domain` tree) and an `is_daughter` flag (its domain is not the chromosome's root domain). Those flow through the recipe to Rust, where `place_chromosome` routes each entry to chromosome `chromosome_index`'s main strand (and, if daughter, its sister bubble — reusing BF1's `bubble_point`).

**Tech Stack:** Python (numpy) `v2ecoli`; Rust (serde, nalgebra) `parsimony-core`; `pbg-parsimony` passthrough.

## Global Constraints

- **Per-chromosome routing:** each RNAP/RNA lands on chromosome `chromosome_index`'s strands; with the theta builder's grouping `[c0_main, c0_sister, c1_main, c1_sister, …]`, chromosome `k`'s main strand index is `2k` and sister is `2k+1` WHEN each chromosome has a sister; the helper must derive the actual indices from the per-chromosome strand groups (don't hardcode 2k if a chromosome lacks a sister).
- **Daughter overlay:** `is_daughter` entries also render on their chromosome's sister via `bubble_point` (BF1 behavior, per chromosome).
- **Birth unchanged:** single-chromosome behavior (BF1) is identical (`chromosome_index` defaults 0; `is_daughter` from `domain_index != 0`).
- **1:1 molecule count** preserved (daughter overlay is the documented 2× rendering).
- **Confinement + determinism** preserved.
- **Env:** worktree `/Users/eranagmon/code/v2e-3d-txn`, branch `feat/3d-transcription-translation`; interpreter `/Users/eranagmon/code/v2ecoli/.venv/bin/python` from the worktree; `PARSIMONY_HOME=/Users/eranagmon/code/parsimony`; pbg-parsimony editable; ParCa cache symlink `out/cache` present.

**Reference reading:**
- `scripts/render_chromosome_gif.py` — `_domain_children_from_dill` (~29, reads `chromosome_domain` `domain_index`+`child_domains`), the domain-tree pattern.
- `v2ecoli/visualizations/workflow.py` — `_descendant_domains` (~182, transitive children).
- `scripts/capture_structural_snapshot.py` — `_extract_snapshot` (~44): how it reads `unique["full_chromosome"]`, `active_RNAP`, `RNA`; add `chromosome_domain` + the new derived arrays here.
- `v2ecoli/structural/build.py` — `rnap_state`/`rna_state` readers; `build_model` rnap/rna dict construction.
- `parsimony/crates/parsimony-core/src/recipe.rs` — `RnapPlacement`/`RnaSpec` (add `chromosome_index`/`is_daughter`); `placer.rs` — `place_chromosome` RNAP + RNA loops, the per-chromosome `chrom_groups`/flat `strands`, `bubble_point` (BF1).

---

### Task BF2-1: capture chromosome_index + is_daughter (Python)

Add the `chromosome_domain` tree + `full_chromosome` roots to the capture; compute per-RNAP `rnap_chromosome_index` (i4) + `rnap_is_daughter` (bool), saved alongside the RNAP arrays; expose them via `rnap_state`.

**Files:**
- Modify: `v2ecoli/scripts/capture_structural_snapshot.py` (`_extract_snapshot`).
- Modify: `v2ecoli/v2ecoli/structural/build.py` (`rnap_state` returns the two new arrays).
- Test: `v2ecoli/tests/structural/test_chromosome_index.py` (create).

**Interfaces:**
- New npz keys: `rnap_chromosome_index` (i4[]), `rnap_is_daughter` (bool[]).
- `rnap_state` gains keys `chromosome_index`, `is_daughter` (empty i4/bool fallbacks).
- Classification (pure, unit-testable — factor a helper `classify_domains(domain_children, full_chromosome_domains, query_domains) -> (chromosome_index[], is_daughter[])`):
  - Build the parent→children tree from `chromosome_domain` (`domain_index` → `child_domains` ≥ 0), like `_domain_children_from_dill`.
  - For each `full_chromosome` `k` (ordered by their `domain_index`), its lineage = `{root_k} ∪ descendants(root_k)` (transitive, via `_descendant_domains`-style walk).
  - For each query `domain`: `chromosome_index` = the `k` whose lineage contains it (0 if none matched — single-chromosome / unknown); `is_daughter` = `domain != root_k` (it's on a replicated copy).

- [ ] **Step 1: Write the failing test** (pure classifier, no sim)

```python
import numpy as np
from v2ecoli.structural.build import classify_domains  # re-exported from build.py

def test_classify_domains_two_chromosomes():
    # chromosome A root=1 -> children 3,4 ; chromosome B root=2 -> children 5,6
    tree = {1: [3, 4], 2: [5, 6]}
    full_chrom_domains = [1, 2]
    q = np.array([1, 3, 4, 2, 5, 6], dtype="i4")
    ci, isd = classify_domains(tree, full_chrom_domains, q)
    assert list(ci) == [0, 0, 0, 1, 1, 1]       # 1/3/4 -> chrom0 ; 2/5/6 -> chrom1
    assert list(isd) == [False, True, True, False, True, True]  # roots not daughters
    assert isd.dtype == bool and ci.dtype == np.int32
```

- [ ] **Step 2: Run → FAIL** (`cd /Users/eranagmon/code/v2e-3d-txn && /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest tests/structural/test_chromosome_index.py -q`) — `classify_domains` missing.

- [ ] **Step 3: Implement** `classify_domains` in build.py (pure, the tree-walk + lineage match); extend `rnap_state` to read/return `rnap_chromosome_index`/`rnap_is_daughter` (empty fallbacks). In the capture script, read `unique["chromosome_domain"]` (`_entryState` mask → `domain_index` + `child_domains`), `unique["full_chromosome"]` domains, then `classify_domains(tree, fc_domains, active_rnap["domain_index"])` → save the two arrays. Guard absence (chromosome_domain missing → all zeros / not-daughter).

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Regenerate snapshots** (records the new arrays; uses the ParCa cache symlink)

Run: `cd /Users/eranagmon/code/v2e-3d-txn && PARSIMONY_HOME=/Users/eranagmon/code/parsimony /Users/eranagmon/code/v2ecoli/.venv/bin/python scripts/capture_structural_snapshot.py`
Expected: prints RNAP/RNA counts; the division npz now has `rnap_chromosome_index` spanning {0,1} (both chromosomes). If the sim is blocked, report DEFERRED — the unit test gates the task.

- [ ] **Step 6: Commit** build.py + capture script + test + regenerated npz (`git add … v2ecoli/structural/data/v2ecoli_state*.npz`).

---

### Task BF2-2: recipe + bridge fields chromosome_index + is_daughter (Rust + Python)

Carry `chromosome_index` + `is_daughter` on RNAP and RNA specs through the recipe.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/recipe.rs` — `RawRnap`/`RnapPlacement` + `RawRna`/`RnaSpec`: add `chromosome_index: i32` (`#[serde(default)]` → 0) and `is_daughter: bool` (`#[serde(default)]` → false). Map in the raw→spec conversions.
- Modify: `pbg-parsimony/api.py` — none needed (rnaps/rnas dicts pass through verbatim); confirm and note.
- Modify: `v2ecoli/v2ecoli/structural/build.py` — `build_model` adds `chromosome_index`/`is_daughter` to each rnap dict (from `rnap_state`) and each nascent rna dict (from its RNAP via the uid map).
- Test: `parsimony` recipe round-trip test; `v2ecoli` build dict test (optional).

**Interfaces:**
- `RnapPlacement.chromosome_index: i32`, `RnapPlacement.is_daughter: bool`; `RnaSpec.chromosome_index: i32`, `RnaSpec.is_daughter: bool`. Recipe JSON keys `chromosome_index`, `is_daughter` on each rnap/rna entry.

- [ ] **Step 1: Write the failing Rust test** — parse a recipe whose chromosome `rnaps` entry has `"chromosome_index": 1, "is_daughter": true`; assert the resolved `RnapPlacement` carries them. (Base the JSON on the existing `parses_explicit_rnaps_from_recipe_json`.) Similarly assert defaults (absent → 0/false) for a plain entry.

- [ ] **Step 2: Run → FAIL** (`cargo test -p parsimony-core --lib`).

- [ ] **Step 3: Implement** the four field additions (RawRnap/RnapPlacement/RawRna/RnaSpec) + mappings, mirroring the existing `is_forward`/`is_free` plumbing. Then wire `build_model` to populate `chromosome_index`/`is_daughter` on each rnap dict from `rnap_state` (and on each nascent rna dict from its RNAP via `rnap_uid_to_cd`-style lookup — extend that map to also carry chromosome_index/is_daughter).

- [ ] **Step 4: Run → PASS** (Rust suite). Confirm pbg passes the new keys through (a quick `pbg-parsimony` test asserting a recipe with chromosome_index round-trips, OR rely on the verbatim passthrough already covered by test_rnaps).

- [ ] **Step 5: Commit** (separate commits per repo: parsimony recipe, v2ecoli build wiring).

---

### Task BF2-3: per-chromosome routing in place_chromosome (Rust) + checkpoint

Route each RNAP/RNA to its chromosome's strands using `chromosome_index`; overlay daughters on that chromosome's sister.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/placer.rs` — `place_chromosome` RNAP + RNA loops; a helper mapping `chromosome_index` → `(main_strand_idx, sister_strand_idx_opt)` from the per-chromosome strand groups.
- Test: `placer.rs` tests (two-chromosome routing).

**Interfaces:**
- Consumes: `RnapPlacement.chromosome_index`/`is_daughter`, `RnaSpec.*`, `bubble_point` (BF1), the per-chromosome `chrom_groups` (already built in `place_chromosome` — capture each chromosome's main + sister strand indices into a `Vec<(usize, Option<usize>)>` as groups are pushed to the flat `strands`).
- Routing: main placement uses `strand_point(&strands, <treat as strand 0 of this chromosome's main>, coord)` — i.e. call `strand_point` on a single-element slice `&[main_strand.clone()]` OR add a `strand_point_on(&strand, coord, glen)` variant that maps a coordinate onto ONE given strand. (Read `strand_point`/`domain_index_to_strand`; the cleanest is a small `point_on_strand(strand: &[Point3], frac)` core that both `strand_point` and this reuse.) Daughter overlay uses `bubble_point(sister, coord, fork_fraction, glen)` on that chromosome's sister.

- [ ] **Step 1: Write the failing test** — a 2-chromosome recipe (`n_chromosomes: 2`, fork_fraction 0.45) with two RNAPs: one `chromosome_index: 0` and one `chromosome_index: 1`. Assert each RNAP is placed near its OWN chromosome's main strand (use the captured `chrom_groups`/strand extents) — specifically that the chromosome-1 RNAP is closer to chromosome 1's strand group centroid than chromosome 0's. (Before the fix both land on `strands[0]`.)

- [ ] **Step 2: Run → FAIL** (both on chromosome 0).

- [ ] **Step 3: Implement** the per-chromosome strand-group capture + routing for both the RNAP and RNA loops; daughter overlay on the chromosome's own sister. Keep the single-chromosome path (chromosome_index 0) identical to BF1.

- [ ] **Step 4: Run full suite → PASS.** `cargo build --release -p parsimony-cli`.

- [ ] **Step 5: Division-state checkpoint**

```bash
cd /Users/eranagmon/code/v2e-3d-txn && rm -rf .parsimony/cache
PARSIMONY_HOME=/Users/eranagmon/code/parsimony /Users/eranagmon/code/v2ecoli/.venv/bin/python -m v2ecoli.structural.build --out out/ecoli3d-div --state division
/Users/eranagmon/code/v2ecoli/.venv/bin/python -c "import json,numpy as np; p=json.load(open('out/ecoli3d-div/ecoli_3d.pack.json')); n2i={i['name']:i['id'] for i in p['ingredients']}; pos=np.array([q['position'] for q in p['placements']]); ids=np.array([q['ingredient'] for q in p['placements']]); r=pos[ids==n2i['rna_polymerase']]; print('RNAP n=',len(r),'x range',r[:,0].min().round(),r[:,0].max().round())"
```
Expected: RNAPs span BOTH chromosome regions (a wide x-range covering both poles), not clustered on one. Optionally view with the `out/ecoli3d-div` pack. Report the spread.

- [ ] **Step 6: Commit** placer.rs.

---

## Self-Review

**Spec coverage (BF2 rows):**
- Capture chromosome_domain tree + classify per-RNAP chromosome_index/is_daughter → BF2-1. ✓
- Recipe/bridge fields → BF2-2. ✓
- Per-chromosome routing + daughter overlay → BF2-3. ✓
- Birth unchanged (chromosome_index 0) → BF2-3 keeps single-chromosome path. ✓
- 1:1 molecule count + confinement + determinism → reuse BF1 overlay; no new drop. ✓

**Placeholder scan:** the BF2-3 routing note offers two concrete implementation options (single-element slice vs a `point_on_strand` core) — the implementer picks one; both are real. No TODOs.

**Type consistency:** `chromosome_index: i32` / `is_daughter: bool` consistent across capture (i4/bool) → recipe (i32/bool) → spec; `classify_domains` signature consistent BF2-1; reuses `bubble_point` (BF1) + `rnap_uid_to_cd` map (B1).
