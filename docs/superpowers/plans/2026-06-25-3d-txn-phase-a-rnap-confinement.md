# 3D Transcription/Translation — Phase A: Markers + Precise RNAP + Confinement

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Place every `active_RNAP` at its exact genomic locus on the rendered nucleoid, enlarge the replication markers, and confine all fiber-bound entities inside the cell envelope (fixing the currently-protruding RNAPs).

**Architecture:** Rust engine (`parsimony-core`) gains a genomic-coordinate→3D-strand-point mapping and seats explicit RNAPs from a new recipe field; the fiber-pack path takes the cell shape so no bound molecule exits the envelope. `pbg-parsimony` passes RNAP placements through the recipe. `v2ecoli` captures `active_RNAP` arrays into the snapshot and feeds them to the build, and enlarges the markers.

**Tech Stack:** Rust (nalgebra, serde) for `parsimony-core`; Python (numpy, dataclasses) for `pbg-parsimony` + `v2ecoli/structural`.

## Global Constraints

- **Confinement invariant:** every placed entity center satisfies `cell.inset(proxy_radius).contains(center)` — zero protrusions in the final pack.
- **True abundance (1:1):** RNAP count rendered == v2ecoli `active_RNAP` count (no subsampling).
- **Determinism:** packing stays bit-for-bit reproducible for a fixed seed.
- **Genomic-axis fidelity:** RNAP 3D position derived only from `coordinates` (bp) + `domain_index` + `is_forward`, mapped via `Genome::fraction`; oriC at the strand midpoint.
- **Cache gotcha:** after ANY Rust change, `rm -rf /path/to/out/.parsimony/cache` (recipe-keyed, not binary-keyed) before re-running a build, or edits silently no-op.
- **Repos & build:** Rust in `/Users/eranagmon/code/parsimony` (`cargo build --release -p parsimony-cli` → `target/release/parsimony`; tests `cargo test -p parsimony-core --lib`). Python work in the worktree `/Users/eranagmon/code/v2e-3d-txn` (branch `feat/3d-transcription-translation`); run via `.venv/bin/python`. `pbg-parsimony` is installed non-editable in that venv — for live edits do `uv pip install -e /Users/eranagmon/code/pbg-parsimony --no-deps` first, OR edit the installed copy and mirror to the repo before committing (the plan assumes editable install).

**Reference reading (implementers should open these before their task):**
- `parsimony/crates/parsimony-core/src/placer.rs` — `place_chromosome` (665), `seat_markers` (636), `chromosome_cell` (1035).
- `parsimony/crates/parsimony-core/src/fiber_pack.rs` — `pack_on_fiber` (43), `pack_on_fiber_at` (155).
- `parsimony/crates/parsimony-core/src/fiber.rs` — `CellShape` (27): `inset`, `contains`, `outward`, `medial`, `cap_radius`.
- `parsimony/crates/parsimony-core/src/genome.rs` — `Genome::fraction(bp)` (154), `from_csv` (53).
- `parsimony/crates/parsimony-core/src/recipe.rs` — chromosome raw struct (~60–95), `ChromosomeSpec` (282), build mapping (~648–668).
- `pbg-parsimony/api.py` — `Chromosome` (60), `build_pack` (75, chrom_block at 135).
- `v2ecoli/v2ecoli/structural/build.py` — `load_state` (291), markers + `Chromosome(...)` (915–955).
- `v2ecoli/v2ecoli/internal_state.py` analogue: `processes/parca/reconstruction/ecoli/dataclasses/state/internal_state.py` (84–110) for `active_RNAP` attributes.

---

### Task A1: Confine fiber-bound proteins inside the envelope (Rust)

Root cause of the published protrusion: `pack_on_fiber`/`pack_on_fiber_at` offset a bound protein radially outward from the strand and never receive the cell shape, so a strand bead near the wall pushes it through. Give both functions the `CellShape` and reject/redirect any candidate that leaves `shape.inset(proxy_radius)`.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/fiber_pack.rs` (`pack_on_fiber` 43, `pack_on_fiber_at` 155, add a shared confinement helper)
- Modify callers: `parsimony/crates/parsimony-core/src/placer.rs:809,817`; `parsimony/crates/parsimony-core/src/pipeline.rs:575,587`
- Test: in `fiber_pack.rs` `#[cfg(test)] mod tests`

**Interfaces:**
- Produces: `pack_on_fiber(fiber, proteins, obstacles, fiber_radius, shape: CellShape, rng)` and `pack_on_fiber_at(fiber, at, obstacles, fiber_radius, shape: CellShape, rng)` — new trailing `shape` param before `rng`. `FiberBinding` unchanged.
- Consumes: `CellShape` from `crate::fiber`.

- [ ] **Step 1: Write the failing test** — a fiber that runs along the capsule wall; every binding must sit inside the inset envelope.

```rust
#[test]
fn bound_proteins_stay_inside_the_cell_envelope() {
    use crate::fiber::CellShape;
    let shape = CellShape::Capsule { half_len: 400.0, radius: 120.0, axis: Vector3::x() };
    // A fiber hugging the wall (y ~ +radius), where a naive outward offset escapes.
    let fiber: Vec<Point3<f32>> = (0..40)
        .map(|i| Point3::new(-300.0 + i as f32 * 15.0, 115.0, 0.0))
        .collect();
    let ing = sphere_ingredient(25.0); // proxy radius 25
    let mut rng = rng_from(7);
    let binds = pack_on_fiber(&fiber, &[(0, &ing, 30)], &[], 12.0, shape, &mut rng);
    assert!(!binds.is_empty());
    let inset = shape.inset(25.0);
    for b in &binds {
        assert!(inset.contains(&b.position), "binding outside envelope: {:?}", b.position);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p parsimony-core --lib bound_proteins_stay_inside_the_cell_envelope`
Expected: FAIL — compile error (arity) then, once arity fixed in the test only to reproduce, assertion failure on an escaped binding. (Write the test against the NEW signature; it fails to compile until Step 3 adds the param — that is the failing state.)

- [ ] **Step 3: Implement confinement**

Add a `shape: CellShape` parameter to both functions. Where the candidate center is computed as `strand_point + offset_dir * radius`, wrap it: if `!shape.inset(proxy_radius).contains(&cand)`, rotate `offset_dir` around the local strand tangent in fixed increments (e.g. 8 steps of `TAU/8`) and take the first orientation whose candidate is inside; if none fit, fall back to `shape.inset(proxy_radius).medial(&cand)`-pull-inward (project toward the medial axis until contained). Use the existing tangent computation already in the function. Keep the obstacle/overlap check after confinement.

- [ ] **Step 4: Update the 4 call sites** to pass the cell shape. In `place_chromosome` the shape is the `shape` already bound from `chromosome_cell`; thread it into the `pack_on_fiber*` calls at placer.rs:809 and 817. In `pipeline.rs:575,587` the chromosome shape is `chrom`-derived — pass the same `CellShape` used to build the strand (read the surrounding code; it has the compartment shape in scope).

- [ ] **Step 5: Run tests**

Run: `cargo test -p parsimony-core --lib`
Expected: PASS — new test passes; existing `fiber_pack` and `placer` tests still pass.

- [ ] **Step 6: Commit**

```bash
cd /Users/eranagmon/code/parsimony
git add crates/parsimony-core/src/fiber_pack.rs crates/parsimony-core/src/placer.rs crates/parsimony-core/src/pipeline.rs
git commit -m "fix(fiber): confine fiber-bound proteins inside the cell envelope"
```

---

### Task A2: Genomic coordinate → 3D strand point (Rust)

Pure mapping function: a v2ecoli `coordinates` (bp, signed across replichores; oriC=0) → a point + unit tangent on the correct strand. Reuses `Genome::fraction`. The strands vector is per-chromosome (main, sister…) as produced in `place_chromosome`; `domain_index` selects the strand.

**Files:**
- Create helper in `parsimony/crates/parsimony-core/src/placer.rs` (free fn near `subdivide`) or a small new module `genome_map.rs` re-exported from `lib.rs`.
- Test: `#[cfg(test)]` in the same file.

**Interfaces:**
- Produces: `fn strand_point(strands: &[Vec<Point3<f32>>], domain_index: i32, coordinate: i64, genome_len_bp: u32) -> Option<(Point3<f32>, Vector3<f32>)>` — returns the 3D point and a unit tangent (local strand direction). `None` if `strands` empty.
- Mapping: `frac = (0.5 + coordinate / genome_len_bp).rem_euclid(1.0)` (oriC at midpoint); `strand = strands[domain_index_to_strand(domain_index, strands.len())]`; `idx = (frac * (strand.len()-1)) rounded`; tangent = normalized `strand[idx+1]-strand[idx]` (or previous segment at the end).
- `domain_index_to_strand`: for Phase A map `domain_index == 0 → strand 0 (main)`, any `> 0 → last strand (sister)` when present, clamped to `strands.len()-1`. (Documented simplification; refined when replication topology is revisited.)

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn strand_point_maps_origin_to_midpoint_and_is_on_strand() {
    // A simple straight strand of 101 beads along x.
    let strand: Vec<Point3<f32>> = (0..101)
        .map(|i| Point3::new(-500.0 + i as f32 * 10.0, 0.0, 0.0))
        .collect();
    let strands = vec![strand.clone()];
    let glen = 4_641_652u32;
    // coordinate 0 (oriC) -> fraction 0.5 -> bead 50 -> x = 0.0
    let (p0, t0) = strand_point(&strands, 0, 0, glen).unwrap();
    assert!((p0.x - 0.0).abs() < 1e-3, "oriC not at midpoint: {}", p0.x);
    assert!((t0.norm() - 1.0).abs() < 1e-4);
    // a positive coordinate moves toward the high-index end (downstream of oriC)
    let (p1, _) = strand_point(&strands, 0, (glen / 4) as i64, glen).unwrap();
    assert!(p1.x > p0.x);
    // every mapped point is an actual bead-interpolated point on the strand
    assert!(strand.iter().map(|b| (b - p1).norm()).fold(f32::MAX, f32::min) < 11.0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p parsimony-core --lib strand_point_maps_origin`
Expected: FAIL with "cannot find function `strand_point`".

- [ ] **Step 3: Implement `strand_point` + `domain_index_to_strand`** per the mapping above. Guard empty strands / single-bead strands (return `strand[0]` with `Vector3::x` tangent).

- [ ] **Step 4: Run test**

Run: `cargo test -p parsimony-core --lib strand_point_maps_origin`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/eranagmon/code/parsimony
git add crates/parsimony-core/src/placer.rs crates/parsimony-core/src/lib.rs
git commit -m "feat(placer): map genomic coordinate (bp) to a 3D strand point + tangent"
```

---

### Task A3: Recipe schema — explicit RNAP placements (Rust)

Add an optional `rnaps` list to the chromosome recipe block and resolve it into `ChromosomeSpec`. Each entry carries the v2ecoli attributes needed to place it.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/recipe.rs` — raw chromosome struct (~60–95), `ChromosomeSpec` (282), build mapping (~648–668).
- Test: a JSON round-trip test in `recipe.rs` tests.

**Interfaces:**
- Produces: `pub struct RnapPlacement { pub coordinates: i64, pub domain_index: i32, pub is_forward: bool }` and `ChromosomeSpec.rnaps: Vec<RnapPlacement>` (also the marker ingredient name `rnap_marker: Option<String>` for which ingredient to instance at each RNAP).
- JSON shape (in the chromosome block):
```json
"rnaps": [{"coordinates": 123456, "domain_index": 0, "is_forward": true}],
"rnap_marker": "rna_polymerase"
```

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn parses_explicit_rnaps_from_recipe_json() {
    let json = r#"{
      "name": "t", "compartments": {"cell": {"capsule": {"a":[-400,0,0],"b":[400,0,0],"radius":120}}},
      "objects": {}, "directives": {"interior": [], "surface": []},
      "chromosome": {"beads": 1000, "spacing": 135, "bead_radius": 12, "compartment": "cell",
        "rnap_marker": "rna_polymerase",
        "rnaps": [{"coordinates": 100000, "domain_index": 0, "is_forward": true},
                  {"coordinates": -50000, "domain_index": 0, "is_forward": false}]}
    }"#;
    let recipe = Recipe::from_json_str(json).unwrap();   // use the crate's real loader entry point
    let chr = recipe.chromosome.as_ref().unwrap();
    assert_eq!(chr.rnaps.len(), 2);
    assert_eq!(chr.rnaps[0].coordinates, 100000);
    assert!(!chr.rnaps[1].is_forward);
    assert_eq!(chr.rnap_marker.as_deref(), Some("rna_polymerase"));
}
```
(Adjust the loader call + minimal JSON to match the crate's actual `Recipe` deserialization entry point — read the existing recipe tests for the exact constructor and required fields.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p parsimony-core --lib parses_explicit_rnaps`
Expected: FAIL — unknown field / missing `rnaps` on `ChromosomeSpec`.

- [ ] **Step 3: Implement** — add `#[serde(default)] rnaps: Vec<RawRnap>` + `#[serde(default)] rnap_marker: Option<String>` to the raw chromosome struct; define `RawRnap`/`RnapPlacement`; add `rnaps`/`rnap_marker` to `ChromosomeSpec`; map them in the build block (~648–668).

- [ ] **Step 4: Run test**

Run: `cargo test -p parsimony-core --lib parses_explicit_rnaps`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/eranagmon/code/parsimony
git add crates/parsimony-core/src/recipe.rs
git commit -m "feat(recipe): explicit RNAP placements on the chromosome block"
```

---

### Task A4: Seat explicit RNAPs in `place_chromosome` (Rust)

When `chr.rnaps` is non-empty, place each RNAP at `strand_point(...)`, oriented along the tangent (flipped if `!is_forward`), confined to the envelope, instancing `chr.rnap_marker`. This supersedes the count-based random RNAP packing for RNAP specifically (other fiber proteins keep the old path).

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/placer.rs` `place_chromosome` (after strands built, ~753; before/after `seat_markers` at 836).
- Test: `placer.rs` tests.

**Interfaces:**
- Consumes: `strand_point` (A2), `chr.rnaps`/`chr.rnap_marker` (A3), `CellShape` (in scope as `shape`), `Genome` length (`chr.genome` via `Genome::from_csv(...).length_bp`, else `GENOME_BP` const — define `const GENOME_BP_DEFAULT: u32 = 4_641_652;`).
- Orientation: build `UnitQuaternion` rotating `+x` (or the ingredient principal axis) onto `±tangent`.

- [ ] **Step 1: Write the failing test** — a capsule recipe with a chromosome + 50 RNAPs at spread coordinates; assert all 50 placed, each inside the envelope, near a strand bead.

```rust
#[test]
fn seats_every_rnap_on_strand_inside_envelope() {
    let recipe = recipe_with_chromosome_and_rnaps(50); // helper builds a capsule + 1000-bead chrom + 50 rnaps
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(11);
    let (center, shape) = first_capsule_cell(&recipe);
    let rnap_id = recipe.ingredients.get_full("rna_polymerase").unwrap().0 as u32;
    let rnaps: Vec<_> = out.snapshot.placements.iter().filter(|p| p.ingredient_id == rnap_id).collect();
    assert_eq!(rnaps.len(), 50);
    let inset = shape.inset(20.0);
    for p in &rnaps {
        assert!(inset.contains(&p.position), "rnap outside envelope: {:?}", p.position);
    }
}
```
(Write `recipe_with_chromosome_and_rnaps` + `first_capsule_cell` test helpers in the test module, mirroring existing placer test fixtures.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p parsimony-core --lib seats_every_rnap_on_strand`
Expected: FAIL — RNAP count 0 (not yet placed) or helper-not-found.

- [ ] **Step 3: Implement** the explicit-RNAP loop in `place_chromosome`: resolve `rnap_marker` ingredient id; for each `RnapPlacement`, `strand_point(&per_chrom_strands_or_flat, domain_index, coordinates, glen)`, confine via `shape.inset(proxy).contains` (pull inward if needed, reuse the A1 helper), compute rotation, push a `Placement`. Skip the old count-based RNAP packing when explicit rnaps are present (leave other fiber proteins intact).

- [ ] **Step 4: Run tests**

Run: `cargo test -p parsimony-core --lib`
Expected: PASS — new test + all existing tests.

- [ ] **Step 5: Build the release binary** (downstream Python uses it)

Run: `cargo build --release -p parsimony-cli`
Expected: `Finished release` and `target/release/parsimony` exists.

- [ ] **Step 6: Commit**

```bash
cd /Users/eranagmon/code/parsimony
git add crates/parsimony-core/src/placer.rs
git commit -m "feat(placer): seat explicit RNAPs at genomic loci, confined + oriented"
```

---

### Task A5: pbg-parsimony API — pass RNAP placements through (Python)

Extend the `Chromosome` dataclass and `build_pack` to serialize RNAP placements into the recipe chromosome block.

**Files:**
- Modify: `pbg-parsimony/api.py` — `Chromosome` (60), `build_pack` chrom_block (135).
- Test: `pbg-parsimony/tests/test_rnaps.py` (create; mirror existing tests dir).

**Interfaces:**
- Produces: `Chromosome.rnaps: list = field(default_factory=list)` where each item is `{"coordinates": int, "domain_index": int, "is_forward": bool}`, and `Chromosome.rnap_marker: str | None = None`. `build_pack` writes `chrom_block["rnaps"] = chromosome.rnaps` and `chrom_block["rnap_marker"] = chromosome.rnap_marker` when set.

- [ ] **Step 1: Write the failing test**

```python
import json
from pathlib import Path
from pbg_parsimony import Chromosome, Capsule, Ingredient, build_pack

def test_rnaps_serialized_into_recipe(tmp_path):
    chrom = Chromosome(beads=1000, rnap_marker="rna_polymerase",
                       rnaps=[{"coordinates": 100000, "domain_index": 0, "is_forward": True}])
    ing = [Ingredient(id="rna_polymerase", count=0, sphere_radius=30.0, region="fiber")]
    res = build_pack(ing, Capsule(half_len=400, radius=120), chrom,
                     out_dir=tmp_path, name="t")
    recipe = json.loads(Path(res["recipe_path"]).read_text())
    chrom_block = recipe["chromosome"]
    assert chrom_block["rnap_marker"] == "rna_polymerase"
    assert chrom_block["rnaps"][0]["coordinates"] == 100000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/pbg-parsimony && /Users/eranagmon/code/v2e-3d-txn/.venv/bin/python -m pytest tests/test_rnaps.py -q`
Expected: FAIL — `Chromosome` has no `rnaps` field / KeyError on recipe block.

- [ ] **Step 3: Implement** the two dataclass fields + the two `chrom_block` assignments (guard with `if chromosome.rnaps:` / `if chromosome.rnap_marker:`).

- [ ] **Step 4: Run test**

Run: same as Step 2
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/eranagmon/code/pbg-parsimony
git add api.py tests/test_rnaps.py
git commit -m "feat(api): pass explicit RNAP placements into the recipe chromosome block"
```

---

### Task A6: Capture `active_RNAP` into the v2ecoli snapshot (Python)

Add a reproducible snapshot generator that runs the baseline composite and saves the unique `active_RNAP` arrays alongside the existing aggregates; extend the loader to read them.

**Files:**
- Create: `v2ecoli/scripts/capture_structural_snapshot.py` (worktree).
- Modify: `v2ecoli/v2ecoli/structural/build.py` — add `rnap_state(state_source)` reader near `chromosome_state` (37).
- Test: `v2ecoli/tests/structural/test_rnap_state.py` (create).

**Interfaces:**
- Produces (npz keys, appended to existing): `rnap_coordinates` (i8[]), `rnap_domain_index` (i4[]), `rnap_is_forward` (bool[]).
- Produces: `rnap_state(state_source="snapshot") -> dict` with keys `coordinates`, `domain_index`, `is_forward` (numpy arrays; empty arrays when absent).
- Capture path: `comp = v2ecoli.build_composite("ecoli_baseline", seed, cache_dir="out/cache"); comp.run(advance_s); cell = comp.state["agents"]["0"]; rnap = cell["unique"]["active_RNAP"]; active = rnap[rnap["_entryState"].view(bool)]` then read `active["coordinates"]`, `active["domain_index"]`, `active["is_forward"]` (pattern proven in `v2ecoli/bridge.py:152`).

- [ ] **Step 1: Write the failing test** (uses the existing committed snapshot once regenerated; until then, synthesize a tiny npz so the reader is unit-tested without a sim run)

```python
import numpy as np
from v2ecoli.structural import build

def test_rnap_state_reads_arrays(tmp_path, monkeypatch):
    npz = tmp_path / "v2ecoli_state.npz"
    np.savez(npz, ids=np.array(["x[c]"]), counts=np.array([1]), volume=np.array(1.0),
             n_chromosomes=np.array(1), fork_fraction=np.array(0.0), division_progress=np.array(0.0),
             rnap_coordinates=np.array([100000, -50000], dtype="i8"),
             rnap_domain_index=np.array([0, 0], dtype="i4"),
             rnap_is_forward=np.array([True, False]))
    monkeypatch.setattr(build, "DATA", tmp_path)
    st = build.rnap_state("snapshot")
    assert list(st["coordinates"]) == [100000, -50000]
    assert st["is_forward"].dtype == bool and len(st["domain_index"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2e-3d-txn && .venv/bin/python -m pytest tests/structural/test_rnap_state.py -q`
Expected: FAIL — `build.rnap_state` does not exist.

- [ ] **Step 3: Implement `rnap_state`** — load the npz for the given `state_source` (mirror `chromosome_state`'s file selection), return the three arrays with empty-array fallbacks when keys absent.

- [ ] **Step 4: Implement the capture script** — `capture_structural_snapshot.py` builds baseline, runs `advance_s` (default 2.0), extracts bulk aggregates (as today) **plus** the RNAP arrays, and `np.savez` to `v2ecoli/structural/data/v2ecoli_state.npz` (and `_division.npz` for the 2-gen end state). Print the RNAP count captured.

- [ ] **Step 5: Run test**

Run: same as Step 2
Expected: PASS.

- [ ] **Step 6: Regenerate the committed snapshots** (records real RNAP loci)

Run: `cd /Users/eranagmon/code/v2e-3d-txn && .venv/bin/python scripts/capture_structural_snapshot.py`
Expected: prints a nonzero RNAP count (~1–2k); `data/v2ecoli_state*.npz` updated. (Needs the v2ecoli sim env + ParCa cache; if the cache is absent, see `reference_v2ecoli_worktree_cache_symlink` — symlink `out/cache` to the main checkout's.)

- [ ] **Step 7: Commit**

```bash
cd /Users/eranagmon/code/v2e-3d-txn
git add scripts/capture_structural_snapshot.py v2ecoli/structural/build.py tests/structural/test_rnap_state.py v2ecoli/structural/data/v2ecoli_state.npz v2ecoli/structural/data/v2ecoli_state_division.npz
git commit -m "feat(structural): capture active_RNAP loci into the snapshot + reader"
```

---

### Task A7: Wire RNAPs into the build + enlarge markers (Python)

Feed the captured RNAP arrays into the `Chromosome`, switch RNAP to explicit placement, and enlarge the replication markers. Final whole-pack confinement gate.

**Files:**
- Modify: `v2ecoli/v2ecoli/structural/build.py` — markers (920–931), the `Chromosome(...)` construction (945–951), and the `rna_polymerase` ingredient (104–105) to `region="fiber"`, `count=0` (placed explicitly now).
- Test: `v2ecoli/tests/structural/test_build_rnaps.py` (create) — a small end-to-end build assertion.

**Interfaces:**
- Consumes: `rnap_state` (A6), `Chromosome.rnaps`/`rnap_marker` (A5).
- Marker sizes (enlarge): `oriC` and `terminus` `sphere_radius` 70 → 130; `replisome` `proxy_voxel_size` 14 → 22 (coarser proxy → larger rendered footprint). Keep colors/categories.

- [ ] **Step 1: Write the failing test** — build a model from a synthesized snapshot (monkeypatch `DATA`) with 20 RNAPs; assert the pack contains 20 `rna_polymerase` placements and zero protrusions.

```python
import json, numpy as np
from pathlib import Path
from v2ecoli.structural import build

def test_build_places_rnaps_and_confines(tmp_path, monkeypatch):
    # minimal snapshot with a handful of RNAPs + a couple of bulk species
    np.savez(tmp_path / "v2ecoli_state.npz",
             ids=np.array(["EG10893-MONOMER[c]"]), counts=np.array([100]),
             volume=np.array(1.0), n_chromosomes=np.array(1), fork_fraction=np.array(0.0),
             division_progress=np.array(0.0),
             rnap_coordinates=(np.linspace(-2.2e6, 2.2e6, 20)).astype("i8"),
             rnap_domain_index=np.zeros(20, "i4"), rnap_is_forward=np.ones(20, bool))
    monkeypatch.setattr(build, "DATA", tmp_path)
    out = tmp_path / "pack"
    res = build.build_model(str(out), state_source="snapshot", top_n=5)
    pack = json.loads(Path(res["pack_path"]).read_text())
    meta = json.loads(Path(res["sidecar_path"]).read_text())["ingredients"]
    rid = [k for k in meta if k == "rna_polymerase"][0]
    # count placements of rna_polymerase via ingredient index mapping in the pack
    assert pack_count_of(pack, "rna_polymerase") == 20  # helper resolves name->id and counts
    assert pack_protrusions(pack, res) == 0             # helper: every placement inside the envelope
```
(Write `pack_count_of` + `pack_protrusions` helpers in the test file; `pack_protrusions` reads the capsule from the recipe and checks each placement center against the inset envelope. Keep `top_n` small so the build is fast.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/eranagmon/code/v2e-3d-txn && .venv/bin/python -m pytest tests/structural/test_build_rnaps.py -q`
Expected: FAIL — RNAP count 0 (still random-packed) or no rnaps wired.

- [ ] **Step 3: Implement** — in `build_model`: `rs = rnap_state(state_source)`; build `rnaps = [{"coordinates": int(c), "domain_index": int(d), "is_forward": bool(f)} for c,d,f in zip(rs["coordinates"], rs["domain_index"], rs["is_forward"])]`; pass `rnaps=rnaps, rnap_marker="rna_polymerase"` to `Chromosome(...)`. Set the `rna_polymerase` ingredient `region="fiber"`, `count=0`. Enlarge the three markers per the interface.

- [ ] **Step 4: Run test**

Run: same as Step 2
Expected: PASS.

- [ ] **Step 5: Full build smoke + manual viewer review** (the Phase A checkpoint)

```bash
cd /Users/eranagmon/code/v2e-3d-txn
rm -rf out/.parsimony/cache
PARSIMONY_HOME=/Users/eranagmon/code/parsimony .venv/bin/python -m v2ecoli.structural.build --out out/ecoli3d --state snapshot
```
Expected: build completes; RNAPs at real loci; markers visibly larger; no RNAP outside the envelope. Open the viewer (bump `?v=`), confirm visually.

- [ ] **Step 6: Commit**

```bash
cd /Users/eranagmon/code/v2e-3d-txn
git add v2ecoli/structural/build.py tests/structural/test_build_rnaps.py
git commit -m "feat(structural): place RNAPs at real loci + enlarge replication markers"
```

---

## Self-Review

**Spec coverage (Phase A rows of the spec):**
- Enlarged replication markers → A7. ✓
- Precise RNAP placement (coordinate/domain/is_forward) → A2 (mapping) + A4 (seating) + A6 (capture) + A7 (wire). ✓
- Extend snapshot npz + live path for `active_RNAP` → A6. ✓
- Confinement fix (root cause) + whole-pack zero-protrusion gate → A1 (fix) + A4 (RNAP confinement) + A7 Step 1 (`pack_protrusions == 0`). ✓
- Gates: every RNAP on-strand & inside envelope (A4); n=1 vs n=2 differ (existing `densify_packs...`/subdivide tests already cover n-dependence; A2 asserts midpoint mapping); determinism (existing determinism tests + A4 uses fixed seed). ✓
- Phases B/C (RNA, ribosomes, peptides) → intentionally OUT of this plan; separate plans after A lands. ✓

**Placeholder scan:** test bodies and signatures are concrete; the two intentional "read the surrounding code to match the exact constructor" notes (A3 loader entry, pipeline.rs shape in scope) are pointers to real symbols, not deferred work. Acceptable.

**Type consistency:** `RnapPlacement{coordinates:i64, domain_index:i32, is_forward:bool}` consistent A3→A4; Python dict keys `coordinates/domain_index/is_forward` consistent A5/A6/A7; `strand_point` signature consistent A2→A4; npz keys `rnap_coordinates/rnap_domain_index/rnap_is_forward` consistent A6→A7. ✓
