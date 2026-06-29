# 3D Transcription/Translation — Phase B2: free cytoplasmic mRNAs

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render every fully-transcribed free mRNA (an `RNA` whose `RNAP_index` doesn't match a captured RNAP) as an extended thin strand seeded at a confined random point in the cytoplasm, so the full RNA population (nascent + free) is shown 1:1.

**Architecture:** Extends Phase B1. `RnaSpec` gains an `is_free` flag; in `place_chromosome` a free spec seeds its strand at a confined random interior point (rejection-sampled inside the envelope) instead of at a chromosome `strand_point`. `v2ecoli/build_model` emits a free spec for every non-nascent RNA. `pbg-parsimony` needs no change (it passes the RNA dicts through verbatim).

**Tech Stack:** Rust (nalgebra, serde) `parsimony-core`; Python (numpy) `v2ecoli/structural`.

## Global Constraints

- **Free classification:** a free mRNA is an `RNA` whose `int(RNAP_index)` is NOT a key in `rnap_uid_to_cd` (covers `-1` and any orphaned uid). Every captured RNA is therefore rendered as either nascent (B1) or free (B2) — total count 1:1, never dropped.
- **Geometry:** identical extended-strand shape as B1 — contour length = `length_nt × rna_angstrom_per_nt`, confined via surface-pull (never `inset.medial` collapse).
- **Free root:** a point inside `shape.inset(rna_bead_radius)`, rejection-sampled deterministically from the placer's `rng` (fall back to the cell center after a bounded number of tries). Free strands are NOT rooted on the chromosome.
- **Determinism:** bit-for-bit reproducible for a fixed seed; no new global RNG state.
- **Cache gotcha:** after any Rust change, `rm -rf /Users/eranagmon/code/v2e-3d-txn/.parsimony/cache` (worktree root) before a build; regenerate the mesh bundle after a rebuild.
- **Repos & env:** Rust `/Users/eranagmon/code/parsimony` (`cargo test -p parsimony-core --lib`; `cargo build --release -p parsimony-cli`). Python in worktree `/Users/eranagmon/code/v2e-3d-txn` (branch `feat/3d-transcription-translation`); interpreter `/Users/eranagmon/code/v2ecoli/.venv/bin/python` run from the worktree; `PARSIMONY_HOME=/Users/eranagmon/code/parsimony`.

**Reference reading:**
- `parsimony/crates/parsimony-core/src/recipe.rs` — `RnaSpec` (~365) + `RawRna` (mirror the existing fields; add `is_free`).
- `parsimony/crates/parsimony-core/src/placer.rs` — the rna loop in `place_chromosome` (~1046); `random_roll`/`random_unit` helpers (~45) and any interior-sampling helper; `recipe_with_chromosome_and_rnas` test fixture (~2173).
- `parsimony/crates/parsimony-core/src/fiber.rs` — `generate_rna_strand`, `CellShape` (`inset`/`contains`/`reach`).
- `v2ecoli/v2ecoli/structural/build.py` — the B1 nascent-RNA block in `build_model` (the `rnap_uid_to_cd` map + `rnas` list) to extend with free specs.
- `v2ecoli/tests/structural/test_build_rnas.py` — extend with a free-mRNA case.

---

### Task B2-1: free RNA root — `is_free` spec + confined interior seeding (Rust)

Add an `is_free` flag to the RNA spec and seed free strands at a confined random interior point.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/recipe.rs` (`RawRna` + `RnaSpec`: add `is_free`).
- Modify: `parsimony/crates/parsimony-core/src/placer.rs` (the rna loop in `place_chromosome`, ~1046).
- Test: `placer.rs` `#[cfg(test)]`.

**Interfaces:**
- Produces: `RnaSpec.is_free: bool` (recipe key `is_free`, `#[serde(default)]` → `false`). When `true`, the strand root is a confined random interior point (rejection-sample a point and accept when `shape.inset(rna_bead_radius).contains(&p)`; up to ~64 tries; fall back to `Point3::origin()` — the center — if none accepted); when `false`, the existing `strand_point` rooting (B1) is unchanged.
- Sampling: draw a point in the shape's bounding region using the existing `random_roll`/`random_unit` helpers (e.g. sample within `shape.reach()` of the center along random directions, reject until inside the inset) — keep it deterministic from `rng`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn free_rna_seeds_in_interior_not_at_strand_point() {
    // one nascent (is_free=false) at a coordinate, one free (is_free=true)
    let recipe = recipe_with_chromosome_and_rnas_freeflag(&[(100000_i64, 600_i64, false), (0, 600, true)]);
    let out = GreedyRandomPlacer::new(&recipe, PlacerConfig::default()).pack(7);
    assert_eq!(out.snapshot.rna_strands.len(), 2);
    let (center, shape) = first_capsule_cell(&recipe);
    let inset = shape.inset(4.0);
    for rs in &out.snapshot.rna_strands {
        for p in &rs.points { assert!(inset.contains(&(center + p.coords)) || inset.contains(p), "RNA bead outside envelope"); }
    }
    // the free strand's root must NOT coincide with the nascent strand's chromosome-rooted start
    let nascent_root = out.snapshot.rna_strands[0].points[0];
    let free_root = out.snapshot.rna_strands[1].points[0];
    assert!((nascent_root - free_root).norm() > 1.0, "free strand should not root at the same chromosome point");
}
```
(Add `recipe_with_chromosome_and_rnas_freeflag` mirroring `recipe_with_chromosome_and_rnas` but emitting an `is_free` key per spec — read the existing fixture at ~2173 and add the third tuple field.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p parsimony-core --lib free_rna_seeds_in_interior`
Expected: FAIL — `is_free` unknown field / fixture helper missing.

- [ ] **Step 3: Implement** — add `is_free` to `RawRna`/`RnaSpec` (serde default false) + the raw→spec mapping; in the rna loop, branch on `rna.is_free`: if true, compute `root` via the confined rejection-sample helper; else keep the existing `strand_point` rooting. Everything downstream (bead_count, `generate_rna_strand`, push `RnaStrand`) unchanged.

- [ ] **Step 4: Run tests**

Run: `cargo test -p parsimony-core --lib`
Expected: PASS (new + existing).

- [ ] **Step 5: Build + commit**

```bash
cd /Users/eranagmon/code/parsimony
cargo build --release -p parsimony-cli
git add crates/parsimony-core/src/recipe.rs crates/parsimony-core/src/placer.rs
git commit -m "feat(placer): free RNA strands seed at a confined interior point (is_free)"
```

---

### Task B2-2: emit free mRNAs in the build (Python) + checkpoint

Extend `build_model` to also emit a free spec for every non-nascent RNA; final viewer checkpoint shows nascent + free.

**Files:**
- Modify: `v2ecoli/v2ecoli/structural/build.py` — the B1 nascent block in `build_model`.
- Test: `v2ecoli/tests/structural/test_build_rnas.py` (add a free-mRNA case).

**Interfaces:**
- Consumes: `RnaSpec.is_free` (B2-1); the existing `rna_state` / `rnap_uid_to_cd` (B1).
- For each RNA `i`: if `int(rna_state["RNAP_index"][i])` is in `rnap_uid_to_cd` → nascent spec (B1, `is_free` omitted/False, rooted). Else → free spec: `{"root_coordinate": 0, "root_domain": 0, "length_nt": int(transcript_length[i]), "is_mRNA": bool(is_mRNA[i]), "is_free": True}`. Append both kinds to the same `rnas` list. Log nascent vs free counts.
- Total rendered RNA strands == number of captured RNAs (1:1).

- [ ] **Step 1: Write the failing test**

```python
import json, numpy as np, pytest
from pathlib import Path
from v2ecoli.structural import build

@pytest.mark.slow
def test_build_renders_free_mrnas(tmp_path, monkeypatch):
    if not _STRUCT_CACHE.exists():  # reuse the module's guard
        pytest.skip("structure cache not available")
    np.savez(tmp_path / "v2ecoli_state.npz",
        ids=np.array(["EG10893-MONOMER[c]"]), counts=np.array([100]), volume=np.array(1.0),
        n_chromosomes=np.array(1), fork_fraction=np.array(0.45), division_progress=np.array(0.0),
        rnap_coordinates=np.array([0], "i8"), rnap_domain_index=np.array([0], "i4"),
        rnap_is_forward=np.array([True]), rnap_unique_index=np.array([7], "i8"),
        # rna 20 = nascent (RNAP 7); rnas 21,22 = free (RNAP_index -1)
        rna_unique_index=np.array([20,21,22], "i8"), rna_RNAP_index=np.array([7,-1,-1], "i8"),
        rna_transcript_length=np.array([600,600,600], "i8"), rna_is_mRNA=np.array([True,True,True]),
        rna_is_full_transcript=np.array([False,True,True]), rna_TU_index=np.array([1,2,3], "i8"))
    monkeypatch.setattr(build, "DATA", tmp_path)
    res = build.build_model(str(tmp_path/"pack"), state_source="snapshot", top_n=5)
    pack = json.loads(Path(res["pack_path"]).read_text())
    # 3 RNAs (1 nascent + 2 free) all rendered → rna_segment placements present
    assert pack_count_of(pack, "rna_segment") > 0
```
(Reuse `_STRUCT_CACHE`, `pack_count_of`, and the structures-cache pre-seed from the existing `test_build_rnas.py`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2e-3d-txn && PARSIMONY_HOME=/Users/eranagmon/code/parsimony /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest tests/structural/test_build_rnas.py::test_build_renders_free_mrnas -q`
Expected: FAIL initially only if free RNAs are dropped — since B1 currently `continue`s on non-map RNAs, the 2 free RNAs are skipped; with only 1 nascent the count may still be >0, so make the assertion stronger: capture the nascent-only `rna_segment` count first, then assert the with-free build is larger. Concretely, also assert via a build-log or a second build: build once treating all as nascent-skipped vs with-free. SIMPLER: assert the build's reported free-count log == 2 (expose nascent/free counts in the return dict or assert the rnas list length). Implement the assertion against whatever `build_model` exposes; the binding requirement is that the 2 free RNAs ARE rendered (not skipped).

- [ ] **Step 3: Implement** — in the nascent loop, replace the `continue` (skip non-map RNAs) with: build a free spec (`is_free=True`, root 0/0) and append it. Keep nascent specs unchanged. Log `nascent=<n> free=<m>`.

- [ ] **Step 4: Run test**

Run: same as Step 2
Expected: PASS.

- [ ] **Step 5: Full build + viewer checkpoint**

```bash
cd /Users/eranagmon/code/v2e-3d-txn
rm -rf .parsimony/cache
PARSIMONY_HOME=/Users/eranagmon/code/parsimony /Users/eranagmon/code/v2ecoli/.venv/bin/python -m v2ecoli.structural.build --out out/ecoli3d --state snapshot
/Users/eranagmon/code/v2ecoli/.venv/bin/python out/ecoli3d/_view/make_local_bundle.py out/ecoli3d/ecoli_3d.pack.json out/ecoli3d/meshes
```
Expected: build logs nascent + free counts (free should be the bulk of the ~3180 RNAs); free mRNA strands fill the cytoplasm, nascent ones still emanate from RNAPs; confirm in the viewer; verify no strands outside the envelope.

- [ ] **Step 6: Commit**

```bash
cd /Users/eranagmon/code/v2e-3d-txn
git add v2ecoli/structural/build.py tests/structural/test_build_rnas.py
git commit -m "feat(structural): render free cytoplasmic mRNAs (Phase B2)"
```

---

## Self-Review

**Spec coverage (Phase B / B2 rows):**
- Free mRNAs (`RNAP_index == -1` / non-nascent) rendered as confined cytoplasmic strands → B2-1 (interior root) + B2-2 (emit). ✓
- Same extended-strand geometry + surface-pull confinement → reuses B1 `generate_rna_strand`. ✓
- 1:1 (nascent + free = all RNAs, never dropped) → B2-2 replaces the `continue` with a free spec. ✓
- Determinism → B2-1 samples from the placer rng, fixed-seed reproducible. ✓
- Free strands NOT chromosome-rooted → B2-1 test asserts free root ≠ nascent root. ✓

**Placeholder scan:** the B2-2 Step-2 note describes how to make the assertion non-vacuous against whatever `build_model` exposes — the implementer picks the concrete assertion (rnas-list length or logged counts); the binding requirement (free RNAs rendered, not skipped) is explicit. No TODOs.

**Type consistency:** `RnaSpec.is_free: bool` (recipe key `is_free`) consistent B2-1→B2-2; free spec dict keys `root_coordinate/root_domain/length_nt/is_mRNA/is_free` match the Rust `RawRna` fields; reuses B1 `rnap_uid_to_cd`, `pack_count_of`, `_STRUCT_CACHE`.
