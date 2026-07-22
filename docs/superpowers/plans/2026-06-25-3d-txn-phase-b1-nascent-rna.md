# 3D Transcription/Translation — Phase B1: nascent RNA strands

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render every nascent transcript (`RNA` with `RNAP_index >= 0`) as an extended thin RNA strand rooted at its RNA polymerase on the chromosome, contour length ∝ `transcript_length`, confined inside the cell envelope.

**Architecture:** Rust engine grows a confined self-avoiding RNA strand from each RNAP's 3D position (`generate_rna_strand`), stores the per-RNA bead paths on the snapshot, and tiles an `rna_segment` mesh along each (mirroring how the chromosome's `dna_segment` is tiled in `output.rs`). `pbg-parsimony` passes the RNA specs through the recipe; `v2ecoli` captures the `RNA` arrays + RNAP `unique_index` and roots each nascent strand at the matching RNAP.

**Tech Stack:** Rust (nalgebra, serde) `parsimony-core`; Python (numpy) `pbg-parsimony` + `v2ecoli/structural`.

## Global Constraints

- **Connectivity (exact):** a nascent strand roots at its RNAP — link via `RNA.RNAP_index == active_RNAP.unique_index`, then the RNAP's `coordinates`/`domain_index` → `strand_point`. The strand's first bead is within `bead_radius` of that RNAP's 3D position.
- **Extended-strand length:** strand contour length = `transcript_length × Å_PER_NT`, **linear** (relative lengths preserved). `Å_PER_NT` is a tunable constant; default `2.0`.
- **True abundance (1:1):** one strand per nascent RNA; never drop or subsample.
- **Confinement:** strands stay inside the envelope via **surface-pull** (move an out-of-bounds bead to the envelope wall along its radial/inward normal) — **never** `inset.medial()` collapse-to-centerline (that caused the chromosome sheaf).
- **Determinism:** bit-for-bit reproducible for a fixed seed.
- **Cache gotcha:** after ANY Rust change, `rm -rf /Users/eranagmon/code/v2e-3d-txn/.parsimony/cache` (worktree root — NOT out/.parsimony) before re-running a build; regenerate the mesh bundle after a rebuild.
- **Repos & env:** Rust `/Users/eranagmon/code/parsimony` (`cargo test -p parsimony-core --lib`; `cargo build --release -p parsimony-cli` → `target/release/parsimony`). Python in worktree `/Users/eranagmon/code/v2e-3d-txn` (branch `feat/3d-transcription-translation`); interpreter `/Users/eranagmon/code/v2ecoli/.venv/bin/python` run from the worktree dir (CWD shadow); `PARSIMONY_HOME=/Users/eranagmon/code/parsimony`; `pbg-parsimony` is editable-installed in that venv.

**Reference reading (open before the relevant task):**
- `parsimony/crates/parsimony-core/src/fiber.rs` — `generate_fiber` (114, seeds at `Point3::origin()`; the SAW to base the RNA strand on), `CellShape` (27: `inset`/`contains`/`medial`/`outward`/`cap_radius`), `dna_segment_transforms` (564).
- `parsimony/crates/parsimony-core/src/placer.rs` — `place_chromosome` (665), `strand_point` (Phase A), `confine_center` reuse (fiber_pack.rs).
- `parsimony/crates/parsimony-core/src/output.rs` — chromosome segment-tiling block (~222–250); mirror it for RNA.
- `parsimony/crates/parsimony-core/src/placement.rs` — `Snapshot` (54), `Chromosome` (34, `strands`).
- `parsimony/crates/parsimony-core/src/recipe.rs` — chromosome raw struct + `ChromosomeSpec` + the Phase A `rnaps`/`rnap_marker` fields to mirror.
- `pbg-parsimony/api.py` — `Chromosome` dataclass + `build_pack` chrom_block.
- `v2ecoli/v2ecoli/structural/build.py` — `rnap_state`/`chromosome_state` (Phase A, ~37–80), `build_model` (~905+), the marker/`Chromosome(...)` block.
- `v2ecoli/scripts/capture_structural_snapshot.py` — extend for RNA arrays + RNAP unique_index.
- `v2ecoli/v2ecoli/bridge.py` (~152) — the `_entryState`-mask unique-array access pattern.

---

### Task B1-1: `generate_rna_strand` — confined strand from a root (Rust)

A self-avoiding worm-like walk seeded at an arbitrary `root` (not the origin), confined to the envelope, returning `bead_count` points. Basis for every RNA strand.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/fiber.rs` (add `generate_rna_strand`; reuse the `generate_fiber` walk body — consider extracting a shared `walk_from(start, …)` if clean, else a focused copy).
- Test: `fiber.rs` `#[cfg(test)] mod tests`.

**Interfaces:**
- Produces: `pub fn generate_rna_strand<R: Rng>(root: Point3<f32>, bead_count: usize, step: f32, bead_radius: f32, shape: CellShape, rng: &mut R) -> Vec<Point3<f32>>` — first point is `root` (clamped inside if needed), each subsequent a worm-like step kept inside `shape.inset(bead_radius)`; when a step can't be placed inside, it surface-pulls (never medial-collapses). Returns however many it placed (≥1).

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn rna_strand_roots_at_given_point_and_stays_inside() {
    use crate::fiber::CellShape;
    let shape = CellShape::Capsule { half_len: 400.0, radius: 120.0, axis: Vector3::x() };
    let root = Point3::new(-200.0, 50.0, 0.0);
    let mut rng = rng_from(3); // use the module's existing seeding helper name
    let strand = generate_rna_strand(root, 60, 18.0, 4.0, shape, &mut rng);
    assert!(strand.len() >= 30, "expected a substantial strand, got {}", strand.len());
    assert!((strand[0] - root).norm() < 8.0, "strand must root at the RNAP point");
    let inset = shape.inset(4.0);
    for p in &strand { assert!(inset.contains(p), "RNA bead outside envelope: {:?}", p); }
    // longer request → longer strand (monotone in bead_count)
    let longer = generate_rna_strand(root, 120, 18.0, 4.0, shape, &mut rng_from(3));
    assert!(longer.len() >= strand.len());
}
```
(Match `rng_from`/`Point3`/`Vector3` to the helpers/imports already in the fiber.rs test module.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p parsimony-core --lib rna_strand_roots_at_given_point`
Expected: FAIL — `generate_rna_strand` not found.

- [ ] **Step 3: Implement** — copy the `generate_fiber` worm-like-chain walk but `pts.push(root_clamped)` as the seed (clamp root into `shape.inset(bead_radius)` via surface-pull if it's marginally outside). On a step that can't find an in-bounds candidate after the retry budget, stop early (like `generate_fiber`). No medial-collapse.

- [ ] **Step 4: Run test**

Run: `cargo test -p parsimony-core --lib rna_strand_roots_at_given_point`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/eranagmon/code/parsimony
git add crates/parsimony-core/src/fiber.rs
git commit -m "feat(fiber): generate_rna_strand — confined strand from a root point"
```

---

### Task B1-2: recipe `rnas` schema + `rna_segment` (Rust)

Add an optional `rnas` list and an `rna_segment` ingredient name + `rna_aa_per_nt`-style scale to the chromosome recipe block, resolved into `ChromosomeSpec`. Mirror the Phase A `rnaps`/`rnap_marker` handling.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/recipe.rs` (raw chromosome struct, `ChromosomeSpec`, build mapping).
- Test: recipe round-trip in `recipe.rs` tests.

**Interfaces:**
- Produces: `pub struct RnaSpec { pub root_coordinate: i64, pub root_domain: i32, pub length_nt: i64, pub is_mRNA: bool }`; `ChromosomeSpec.rnas: Vec<RnaSpec>`; `ChromosomeSpec.rna_segment: Option<String>`; `ChromosomeSpec.rna_aa_per_nt: f32` (the Å-per-nt scale, recipe key `rna_angstrom_per_nt`, default `2.0`).
- JSON (chromosome block):
```json
"rna_segment": "rna_segment",
"rna_angstrom_per_nt": 2.0,
"rnas": [{"root_coordinate": 123456, "root_domain": 0, "length_nt": 850, "is_mRNA": true}]
```

- [ ] **Step 1: Write the failing test** — parse a recipe with `rnas` + `rna_segment` + `rna_angstrom_per_nt`; assert the resolved `ChromosomeSpec` fields. Use the real loader (`Recipe::from_json_str`, needs `bounding_box`/`objects`/`composition.space` — read the Phase A `parses_explicit_rnaps` test for the exact minimal JSON and copy its shape).

```rust
#[test]
fn parses_rnas_from_recipe_json() {
    // (start from the minimal valid recipe JSON used by parses_explicit_rnaps,
    //  add to the chromosome block:)
    //   "rna_segment": "rna_segment", "rna_angstrom_per_nt": 2.0,
    //   "rnas": [{"root_coordinate": 100000, "root_domain": 0, "length_nt": 850, "is_mRNA": true}]
    let recipe = Recipe::from_json_str(JSON).unwrap();
    let chr = recipe.chromosome.as_ref().unwrap();
    assert_eq!(chr.rnas.len(), 1);
    assert_eq!(chr.rnas[0].length_nt, 850);
    assert!(chr.rnas[0].is_mRNA);
    assert_eq!(chr.rna_segment.as_deref(), Some("rna_segment"));
    assert!((chr.rna_angstrom_per_nt - 2.0).abs() < 1e-6);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p parsimony-core --lib parses_rnas_from_recipe_json`
Expected: FAIL — unknown field / missing `rnas`.

- [ ] **Step 3: Implement** — `#[serde(default)]` raw fields `rnas: Vec<RawRna>`, `rna_segment: Option<String>`, `rna_angstrom_per_nt: Option<f32>`; define `RawRna`/`RnaSpec`; map into `ChromosomeSpec` (default `rna_angstrom_per_nt` → 2.0). Mirror the `rnaps`/`RawRnap` pattern exactly.

- [ ] **Step 4: Run test** → PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/eranagmon/code/parsimony
git add crates/parsimony-core/src/recipe.rs
git commit -m "feat(recipe): nascent RNA specs (rnas) + rna_segment on the chromosome block"
```

---

### Task B1-3: grow + store RNA strands in the snapshot (Rust)

Add `rna_strands` to the snapshot and, in `place_chromosome`, grow one confined strand per `RnaSpec` rooted at its RNAP's 3D position.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/placement.rs` (`Snapshot`: add `pub rna_strands: Vec<RnaStrand>` where `pub struct RnaStrand { pub points: Vec<Point3<f32>>, pub is_mrna: bool }`, both `Debug, Clone, Default`).
- Modify: `parsimony/crates/parsimony-core/src/placer.rs` `place_chromosome` (after the strands + RNAP seating, ~836).
- Test: `placer.rs` tests.

**Interfaces:**
- Consumes: `generate_rna_strand` (B1-1), `chr.rnas`/`chr.rna_segment`/`chr.rna_angstrom_per_nt` (B1-2), `strand_point` (Phase A), the chromosome `strands` + `center`, `CellShape` (`shape`).
- Produces: `snapshot.rna_strands` populated — one `RnaStrand` per `RnaSpec`. Strand points are stored in the SAME frame as `chr.center`-relative chromosome strands (so output.rs adds `center` consistently), i.e. compute the root via `strand_point(strands, root_domain, root_coordinate, GENOME_BP_DEFAULT)` (center-relative) and grow the strand in that frame, confining against `shape`.
- Bead count per strand: `((length_nt as f32 * chr.rna_angstrom_per_nt) / rna_step).round()`, with `rna_step = chr.spacing.min(60.0)` (or a fixed ~40.0 Å — pick one, document it), clamped to ≥ 2.

- [ ] **Step 1: Write the failing test** — a capsule recipe with a chromosome + N rnas at spread root coordinates; assert `out.snapshot.rna_strands.len() == N`, each strand's first point is near its RNAP root (`strand_point` of the same coordinate), each strand longer for larger `length_nt`, all beads inside the envelope.

```rust
#[test]
fn grows_one_confined_strand_per_rna_rooted_at_rnap() {
    let recipe = recipe_with_chromosome_and_rnas(&[(100000_i64, 400_i64), (-50000, 1200)]); // (root_coordinate, length_nt)
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(9);
    assert_eq!(out.snapshot.rna_strands.len(), 2);
    let (center, shape) = first_capsule_cell(&recipe);
    let strands = &out.snapshot.chromosome.as_ref().unwrap().strands;
    let inset = shape.inset(4.0);
    for (rs, &(coord, _len)) in out.snapshot.rna_strands.iter().zip(&[(100000_i64,400_i64),(-50000,1200)]) {
        let (root, _t) = strand_point(strands, 0, coord, 4_641_652).unwrap();
        assert!((rs.points[0] - root).norm() < 30.0, "strand not rooted at its RNAP");
        for p in &rs.points { assert!(inset.contains(&(center + p.coords)) || inset.contains(p), "RNA bead outside envelope"); }
    }
    // longer length_nt → more beads
    assert!(out.snapshot.rna_strands[1].points.len() > out.snapshot.rna_strands[0].points.len());
}
```
(Write `recipe_with_chromosome_and_rnas` mirroring the Phase A `recipe_with_chromosome_and_rnaps` fixture; confirm the center-relative vs world frame against how the RNAP test asserted containment, and assert in the frame the code actually uses.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p parsimony-core --lib grows_one_confined_strand_per_rna`
Expected: FAIL — `rna_strands` empty / field missing.

- [ ] **Step 3: Implement** — add the `Snapshot.rna_strands` field + `RnaStrand`; in `place_chromosome`, after the chromosome strands are built and RNAPs seated, loop over `chr.rnas`: root via `strand_point`, bead_count from `length_nt × rna_angstrom_per_nt / rna_step`, grow with `generate_rna_strand(root, bead_count, rna_step, rna_bead_radius, shape, rng)`, push `RnaStrand { points, is_mrna }`. Never drop a spec (1:1) — if `strand_point` returns `None`, fall back to rooting at `center` so the strand still exists.

- [ ] **Step 4: Run tests**

Run: `cargo test -p parsimony-core --lib`
Expected: PASS (new + existing).

- [ ] **Step 5: Commit**

```bash
cd /Users/eranagmon/code/parsimony
git add crates/parsimony-core/src/placement.rs crates/parsimony-core/src/placer.rs
git commit -m "feat(placer): grow a confined nascent-RNA strand per RNA, rooted at its RNAP"
```

---

### Task B1-4: render RNA strands as tiled segments (Rust output)

Tile the `rna_segment` mesh along each `snapshot.rna_strands` entry, emitting `rna_segment` placements — mirroring the chromosome `dna_segment` tiling block in `output.rs`.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/output.rs` (after the chromosome segment block, ~250).
- Test: an output-level test in `output.rs` tests (or a placer+output integration test) asserting `rna_segment` placements are emitted, count > 0, roughly proportional to total RNA bead count.

**Interfaces:**
- Consumes: `snapshot.rna_strands` (B1-3), `recipe.chromosome.rna_segment` ingredient id, `dna_segment_transforms`.
- Produces: pack `placements` of ingredient `rna_segment`, one set of tiled segments per strand. Use `seg_step` ≈ the strand `step` (one segment per bead) and `twist = 0.0` (ssRNA, no helical twist). Add `chr.center` to strand points if they are center-relative (match B1-3's frame).

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn emits_rna_segment_placements_for_each_strand() {
    let recipe = recipe_with_chromosome_and_rnas(&[(100000, 800), (-50000, 800)]);
    let out = GreedyRandomPlacer::new(&recipe, PlacerConfig::default()).pack(5);
    let pack = crate::output::to_json(&out.snapshot, &recipe); // use the real output entry point name
    let seg_id = recipe.ingredients.get_index_of("rna_segment").unwrap();
    let n = pack["placements"].as_array().unwrap().iter()
        .filter(|p| p["ingredient"].as_u64() == Some(seg_id as u64)).count();
    assert!(n > 20, "expected many tiled rna_segment placements, got {n}");
}
```
(Use the actual `output.rs` serialization entry point + JSON shape — read the chromosome block to match `to_json`/`build_output`'s real name and the `"ingredient"`/`"placements"` keys.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p parsimony-core --lib emits_rna_segment_placements`
Expected: FAIL — 0 rna_segment placements.

- [ ] **Step 3: Implement** — after the chromosome `dna_segment` block in `output.rs`, if `recipe.chromosome.rna_segment` resolves to an ingredient id, for each `RnaStrand` build the world path (`center + p.coords` if center-relative) and `for (pos, rot) in dna_segment_transforms(&world, seg_step, 0.0)` push an `rna_segment` placement (same json shape as the dna_segment block).

- [ ] **Step 4: Run tests** → PASS. Then `cargo build --release -p parsimony-cli`.

- [ ] **Step 5: Commit**

```bash
cd /Users/eranagmon/code/parsimony
git add crates/parsimony-core/src/output.rs
git commit -m "feat(output): tile rna_segment mesh along each nascent-RNA strand"
```

---

### Task B1-5: pbg-parsimony — pass RNA specs through (Python)

Extend `Chromosome` + `build_pack` to serialize the RNA block into the recipe.

**Files:**
- Modify: `pbg-parsimony/api.py` — `Chromosome` dataclass + `build_pack` chrom_block.
- Test: `pbg-parsimony/tests/test_rnas.py` (create).

**Interfaces:**
- Produces: `Chromosome.rnas: list = field(default_factory=list)` (items `{"root_coordinate": int, "root_domain": int, "length_nt": int, "is_mRNA": bool}`), `Chromosome.rna_segment: str | None = None`, `Chromosome.rna_angstrom_per_nt: float = 2.0`. `build_pack` writes `chrom_block["rnas"]`, `chrom_block["rna_segment"]`, `chrom_block["rna_angstrom_per_nt"]` — each guarded so a default `Chromosome` leaves the recipe unchanged.

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path
from pbg_parsimony import Chromosome, Capsule, Ingredient, build_pack

def test_rnas_serialized_into_recipe(tmp_path):
    chrom = Chromosome(beads=1000, rna_segment="rna_segment", rna_angstrom_per_nt=2.0,
                       rnas=[{"root_coordinate": 100000, "root_domain": 0, "length_nt": 850, "is_mRNA": True}])
    ing = [Ingredient(id="rna_segment", count=0, sphere_radius=8.0)]
    res = build_pack(ing, Capsule(half_len=400, radius=120), chrom, out_dir=tmp_path, name="t")
    cb = json.loads(Path(res["recipe_path"]).read_text())["chromosome"]
    assert cb["rna_segment"] == "rna_segment"
    assert cb["rnas"][0]["length_nt"] == 850
    assert cb["rna_angstrom_per_nt"] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/pbg-parsimony && PARSIMONY_HOME=/Users/eranagmon/code/parsimony /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest tests/test_rnas.py -q`
Expected: FAIL — `Chromosome` has no `rnas`.

- [ ] **Step 3: Implement** the three dataclass fields + guarded chrom_block assignments. Run the existing `tests/test_recipe.py` + `tests/test_rnaps.py` too (no regression).

- [ ] **Step 4: Run test** → PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/eranagmon/code/pbg-parsimony
git add api.py tests/test_rnas.py
git commit -m "feat(api): pass nascent RNA specs + rna_segment into the recipe"
```

---

### Task B1-6: capture RNA arrays + RNAP unique_index (Python)

Extend the snapshot capture + readers with the `RNA` unique-molecule arrays and `active_RNAP.unique_index`.

**Files:**
- Modify: `v2ecoli/scripts/capture_structural_snapshot.py` (add RNA extraction + rnap unique_index).
- Modify: `v2ecoli/v2ecoli/structural/build.py` — extend `rnap_state` to also return `unique_index`; add `rna_state(state_source)`.
- Test: `v2ecoli/tests/structural/test_rna_state.py` (create).

**Interfaces:**
- New npz keys: `rnap_unique_index` (i8); `rna_unique_index` (i8), `rna_RNAP_index` (i8), `rna_transcript_length` (i8), `rna_is_mRNA` (bool), `rna_is_full_transcript` (bool), `rna_TU_index` (i8).
- `rnap_state(state_source)` gains key `unique_index` (empty-array fallback).
- `rna_state(state_source) -> dict` with keys `unique_index`, `RNAP_index`, `transcript_length`, `is_mRNA`, `is_full_transcript`, `TU_index` (numpy arrays; empty fallbacks with correct dtypes when keys absent).
- Capture path: `rna = cell["unique"]["RNA"]; active = rna[rna["_entryState"].view(bool)]`; read the six fields (guard each with `in rna.dtype.names`). Also read `active_RNAP[...]["unique_index"]`.

- [ ] **Step 1: Write the failing test** (synthesize a tiny npz; no sim run)

```python
import numpy as np
from v2ecoli.structural import build

def test_rna_state_reads_arrays(tmp_path, monkeypatch):
    np.savez(tmp_path / "v2ecoli_state.npz",
        ids=np.array(["x[c]"]), counts=np.array([1]), volume=np.array(1.0),
        n_chromosomes=np.array(1), fork_fraction=np.array(0.45), division_progress=np.array(0.0),
        rnap_coordinates=np.array([100000], "i8"), rnap_domain_index=np.array([0], "i4"),
        rnap_is_forward=np.array([True]), rnap_unique_index=np.array([7], "i8"),
        rna_unique_index=np.array([20, 21], "i8"), rna_RNAP_index=np.array([7, -1], "i8"),
        rna_transcript_length=np.array([850, 1200], "i8"), rna_is_mRNA=np.array([True, True]),
        rna_is_full_transcript=np.array([False, True]), rna_TU_index=np.array([3, 4], "i8"))
    monkeypatch.setattr(build, "DATA", tmp_path)
    assert list(build.rnap_state("snapshot")["unique_index"]) == [7]
    st = build.rna_state("snapshot")
    assert list(st["RNAP_index"]) == [7, -1]
    assert st["is_mRNA"].dtype == bool and list(st["transcript_length"]) == [850, 1200]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2e-3d-txn && /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest tests/structural/test_rna_state.py -q`
Expected: FAIL — `build.rna_state` missing / `rnap_state` has no `unique_index`.

- [ ] **Step 3: Implement** `rna_state` + the `rnap_state` `unique_index` addition (mirror `rnap_state`'s file selection + empty fallbacks). Extend the capture script to save the new keys.

- [ ] **Step 4: Run test** → PASS.

- [ ] **Step 5: Regenerate snapshots** (records real RNA loci)

Run: `cd /Users/eranagmon/code/v2e-3d-txn && PARSIMONY_HOME=/Users/eranagmon/code/parsimony /Users/eranagmon/code/v2ecoli/.venv/bin/python scripts/capture_structural_snapshot.py`
Expected: prints nonzero RNA + RNAP counts; `data/v2ecoli_state*.npz` updated (uses the symlinked ParCa cache from Phase A; if absent, `mkdir -p out && ln -s /Users/eranagmon/code/v2ecoli/out/cache out/cache`).

- [ ] **Step 6: Commit**

```bash
cd /Users/eranagmon/code/v2e-3d-txn
git add scripts/capture_structural_snapshot.py v2ecoli/structural/build.py tests/structural/test_rna_state.py v2ecoli/structural/data/v2ecoli_state.npz v2ecoli/structural/data/v2ecoli_state_division.npz
git commit -m "feat(structural): capture RNA arrays + RNAP unique_index into the snapshot"
```

---

### Task B1-7: wire nascent RNA into the build + render (Python)

Build the nascent-RNA recipe specs (rooted at each RNAP) and add the `rna_segment` ingredient so strands render; final viewer checkpoint.

**Files:**
- Modify: `v2ecoli/v2ecoli/structural/build.py` — `build_model`: build the rnap-uid→(coordinate,domain) map, classify nascent RNAs, build the `rnas` list, add an `rna_segment` ingredient, pass `rnas`/`rna_segment`/`rna_angstrom_per_nt` to `Chromosome(...)`.
- Test: `v2ecoli/tests/structural/test_build_rnas.py` (create).

**Interfaces:**
- Consumes: `rna_state` + `rnap_state.unique_index` (B1-6), `Chromosome.rnas`/`rna_segment`/`rna_angstrom_per_nt` (B1-5).
- `rna_segment` ingredient: a thin curved/cylinder segment mesh OR `sphere_radius`-style thin proxy with a `segment` mesh — simplest first pass: reuse the dsDNA-style segment approach with an RNA color. Use `category="Transcription"` (or a new `"RNA"` category) + a distinct color. `count=0` (placed by the rna stage, not randomly).
- Nascent classification: `rnap_uid_to_cd = {uid:(coord,dom) for uid,coord,dom in zip(rnap unique_index, coordinates, domain_index)}`; for each RNA with `RNAP_index in rnap_uid_to_cd`, emit `{"root_coordinate": int(coord), "root_domain": int(dom), "length_nt": int(transcript_length), "is_mRNA": bool(is_mRNA)}`.

- [ ] **Step 1: Write the failing test** — synthesize a snapshot (monkeypatch `DATA`) with 1 RNAP (unique_index 7 at a coordinate) + 3 nascent RNAs (RNAP_index 7) of increasing length; build a small model; assert the pack contains `rna_segment` placements (>0) and that strand-segment count grows with total transcript length (compare against a second build with longer RNAs).

```python
import json, numpy as np
from pathlib import Path
from v2ecoli.structural import build

def test_build_renders_nascent_rna(tmp_path, monkeypatch):
    np.savez(tmp_path / "v2ecoli_state.npz",
        ids=np.array(["EG10893-MONOMER[c]"]), counts=np.array([100]), volume=np.array(1.0),
        n_chromosomes=np.array(1), fork_fraction=np.array(0.45), division_progress=np.array(0.0),
        rnap_coordinates=np.array([0], "i8"), rnap_domain_index=np.array([0], "i4"),
        rnap_is_forward=np.array([True]), rnap_unique_index=np.array([7], "i8"),
        rna_unique_index=np.array([20,21,22], "i8"), rna_RNAP_index=np.array([7,7,7], "i8"),
        rna_transcript_length=np.array([300,900,1500], "i8"), rna_is_mRNA=np.array([True,True,True]),
        rna_is_full_transcript=np.array([False,False,False]), rna_TU_index=np.array([1,2,3], "i8"))
    monkeypatch.setattr(build, "DATA", tmp_path)
    res = build.build_model(str(tmp_path/"pack"), state_source="snapshot", top_n=5)
    pack = json.loads(Path(res["pack_path"]).read_text())
    meta = json.loads(Path(res["sidecar_path"]).read_text())["ingredients"]
    assert "rna_segment" in meta
    assert pack_count_of(pack, "rna_segment") > 0  # reuse/define the Phase A helper
```
(Pre-seed the structures cache like the Phase A `test_build_rnaps.py` does; respect `PARSIMONY_HOME` via `os.environ.get`; mark `@pytest.mark.slow`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2e-3d-txn && PARSIMONY_HOME=/Users/eranagmon/code/parsimony /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest tests/structural/test_build_rnas.py -q`
Expected: FAIL — no `rna_segment` / 0 placements.

- [ ] **Step 3: Implement** the rnap-uid map, nascent `rnas` list, the `rna_segment` ingredient, and the `Chromosome(...)` wiring in `build_model`.

- [ ] **Step 4: Run test** → PASS.

- [ ] **Step 5: Full build + viewer checkpoint**

```bash
cd /Users/eranagmon/code/v2e-3d-txn
rm -rf .parsimony/cache
PARSIMONY_HOME=/Users/eranagmon/code/parsimony /Users/eranagmon/code/v2ecoli/.venv/bin/python -m v2ecoli.structural.build --out out/ecoli3d --state snapshot
/Users/eranagmon/code/v2ecoli/.venv/bin/python out/ecoli3d/_view/make_local_bundle.py out/ecoli3d/ecoli_3d.pack.json out/ecoli3d/meshes
```
Expected: build completes; nascent RNA strands emanate from RNAPs on the centered chromosome; confirm in the viewer (`?file=…`, bump nothing — same server). Verify no strands outside the envelope.

- [ ] **Step 6: Commit**

```bash
cd /Users/eranagmon/code/v2e-3d-txn
git add v2ecoli/structural/build.py tests/structural/test_build_rnas.py
git commit -m "feat(structural): render nascent RNA strands rooted at their RNAPs"
```

---

## Self-Review

**Spec coverage (B1):**
- Capture RNA arrays + RNAP unique_index → B1-6. ✓
- Connectivity (RNA→RNAP via unique_index → strand_point root) → B1-7 (map) + B1-3 (root via strand_point). ✓
- Extended strand, contour ∝ transcript_length, Å_PER_NT default 2.0 → B1-2 (scale field) + B1-3 (bead count) + B1-1 (strand). ✓
- True abundance 1:1 (one strand per nascent RNA, never drop) → B1-3 (fallback-to-center, no drop). ✓
- Confinement surface-pull, no medial-collapse → B1-1 + B1-3. ✓
- Render as RNA category/color, instanced segments → B1-4 (tiling) + B1-7 (ingredient/category). ✓
- Determinism → B1-1/B1-3 tests use fixed seeds; no new global RNG. ✓
- B2 (free mRNAs) intentionally OUT → separate plan after B1 review. ✓

**Placeholder scan:** test bodies + signatures are concrete; the few "match the real loader/output entry point name" notes point at real symbols the implementer reads (same pattern Phase A used successfully), not deferred work.

**Type consistency:** `RnaSpec{root_coordinate:i64, root_domain:i32, length_nt:i64, is_mRNA:bool}` consistent B1-2→B1-3→B1-5→B1-7; `RnaStrand{points, is_mrna}` B1-3→B1-4; npz keys `rna_*`/`rnap_unique_index` consistent B1-6→B1-7; `generate_rna_strand` signature B1-1→B1-3; `rna_angstrom_per_nt` default 2.0 consistent B1-2/B1-5/B1-7.
