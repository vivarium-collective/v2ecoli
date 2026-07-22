# 3D Transcription — BF1: birth replication-bubble overlay

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For a single replicating chromosome, render daughter-domain RNAPs (and their nascent RNA) a *second* time on the sister/bubble strand — the daughter copy — matching the v2ecoli `_draw_chromosome` reference.

**Architecture:** Pure `parsimony` (Rust) change. A daughter is `domain_index != 0` (single chromosome); the sister strand is the last strand. A new `bubble_point` helper maps a genomic coordinate to a **bubble-relative** position on the sister; `place_chromosome` overlays daughter RNAPs/RNA there in addition to their main-contour placement. No recipe/pbg/build changes (BF2 adds those).

**Tech Stack:** Rust (nalgebra) `parsimony-core`.

## Global Constraints

- **Reference overlay:** daughter-domain entries (`domain_index != 0` for RNAP / `root_domain != 0` for nascent RNA) appear on BOTH the main contour (existing) AND the sister strand (new overlay). domain-0 entries appear once (main only). Free mRNAs (`is_free`) are NOT overlaid.
- **Bubble-relative mapping:** sister position uses `frac = ((coord + fork_bp) / (2·fork_bp)).clamp(0,1)`, `fork_bp = chr.fork_fraction × (genome_len_bp / 2)` — NOT the genome-relative `strand_point` mapping.
- **Documented over-render:** the molecule count (active_RNAP) is unchanged; daughter *placements* are 2× by design (two physical copies). Not a 1:1 violation.
- **Confinement** (surface-pull via `confine_center`, never medial-collapse) and **determinism** (fixed seed; no new RNG draws — overlay is deterministic) preserved.
- **Build/test:** `/Users/eranagmon/code/parsimony` — `cargo test -p parsimony-core --lib`; `cargo build --release -p parsimony-cli` → `target/release/parsimony`.

**Reference reading:**
- `placer.rs` — `place_chromosome` RNAP loop (~974–1030, the `strand_point(&strands, 0, …)` placement + `confine_center` + `orient_x_onto` + push), the nascent-RNA loop (~1034+, `is_free` branch + `strand_point(&strands, 0, …)` root + `generate_rna_strand`), `strand_point` (~123), `GENOME_BP_DEFAULT`.
- `fiber.rs` — `generate_rna_strand`, `CellShape`.
- Test fixtures: `recipe_with_chromosome_and_rnaps` (~2044, domain 0, no fork), `recipe_replicating_with_one_rnap` (~2092, fork_fraction 0.45, parametric domain), `first_capsule_cell`. Existing tests to UPDATE: `rnap_placed_on_main_contour_regardless_of_domain`, `nascent_rna_roots_on_main_contour_like_its_rnap` (these assert a count of 1 for a domain-2 entry — BF1 makes it 2).

---

### Task BF1-1: `bubble_point` — bubble-relative coordinate → sister position (Rust)

A pure helper mapping a genomic coordinate to a point + unit tangent on the sister (bubble) strand, parametrized over the replicated region `[-fork_bp, +fork_bp]`.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/placer.rs` (free fn near `strand_point`).
- Test: `placer.rs` `#[cfg(test)]`.

**Interfaces:**
- Produces: `fn bubble_point(sister: &[Point3<f32>], coordinate: i64, fork_fraction: f32, genome_len_bp: u32) -> Option<(Point3<f32>, Vector3<f32>)>` — `frac = ((coordinate as f32 + fork_bp) / (2.0 * fork_bp)).clamp(0.0, 1.0)` with `fork_bp = fork_fraction * (genome_len_bp as f32 / 2.0)`; `idx = (frac * (sister.len()-1) as f32).round()`; tangent = normalized forward segment (or previous at the end). `None` if `sister.len() < 2` or `fork_bp <= 0`.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn bubble_point_maps_forks_to_ends_and_oric_to_middle() {
    // sister of 101 beads along x in [-500, 500]
    let sister: Vec<Point3<f32>> = (0..101).map(|i| Point3::new(-500.0 + i as f32 * 10.0, 0.0, 0.0)).collect();
    let glen = 4_641_652u32;
    let ff = 0.45_f32;
    let fork_bp = (ff * glen as f32 / 2.0) as i64; // bubble half-width in bp
    // oriC (coord 0) → frac 0.5 → middle (x≈0)
    let (mid, _) = bubble_point(&sister, 0, ff, glen).unwrap();
    assert!(mid.x.abs() < 6.0, "oriC should map near the sister middle, got {}", mid.x);
    // +fork → frac 1.0 → last bead (x≈+500)
    let (hi, _) = bubble_point(&sister, fork_bp, ff, glen).unwrap();
    assert!(hi.x > 480.0, "+fork should map near the sister far end, got {}", hi.x);
    // -fork → frac 0.0 → first bead (x≈-500)
    let (lo, _) = bubble_point(&sister, -fork_bp, ff, glen).unwrap();
    assert!(lo.x < -480.0, "-fork should map near the sister near end, got {}", lo.x);
    // coordinate beyond the bubble clamps to an end (does not panic / wrap)
    let (clamped, _) = bubble_point(&sister, glen as i64, ff, glen).unwrap();
    assert!(clamped.x > 480.0);
}
```

- [ ] **Step 2: Run test → FAIL** (`cargo test -p parsimony-core --lib bubble_point_maps_forks`) — `bubble_point` not found.

- [ ] **Step 3: Implement** `bubble_point` per the interface (guard `sister.len() < 2` and `fork_bp <= 0.0` → `None`). Mirror `strand_point`'s tangent computation.

- [ ] **Step 4: Run test → PASS.**

- [ ] **Step 5: Commit**

```bash
cd /Users/eranagmon/code/parsimony
git add crates/parsimony-core/src/placer.rs
git commit -m "feat(placer): bubble_point — bubble-relative coordinate mapping on the sister strand"
```

---

### Task BF1-2: overlay daughter RNAPs on the sister (Rust)

In the RNAP loop, after the existing main-contour placement, place a SECOND copy of each daughter RNAP (`domain_index != 0`) on the sister strand at its bubble-relative position.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/placer.rs` — RNAP loop (~974–1030).
- Test: update `rnap_placed_on_main_contour_regardless_of_domain` + add `daughter_rnap_overlaid_on_bubble`.

**Interfaces:**
- Consumes: `bubble_point` (BF1-1), `chr.fork_fraction`, `glen`, the `strands` list, `confine_center`, `orient_x_onto`.
- Behavior: for each RNAP, keep the existing main placement (`strand_point(&strands, 0, …)`). THEN, if `rnap.domain_index != 0 && strands.len() > 1`, compute `bubble_point(&strands[strands.len()-1], rnap.coordinates, chr.fork_fraction, glen)`, convert to world (`center + p.coords`), confine (same `confine_center` pattern), orient by `is_forward`, and push a second `Placement` of the same ingredient. Reuse the existing confine+orient+push code (factor a small local closure to avoid duplication).

- [ ] **Step 1: Update the existing test + write the new one**

Update `rnap_placed_on_main_contour_regardless_of_domain`: the domain-2 RNAP now yields TWO placements (main + sister overlay). Change `assert_eq!(rnaps.len(), 1, …)` → `assert_eq!(rnaps.len(), 2, …)`, and assert that ONE placement is near the main locus (`strand_point(strands,0,coord)`) and ONE is near the bubble locus (`bubble_point(sister,coord,ff,glen)`):

```rust
// after collecting `rnaps` (now expect 2):
assert_eq!(rnaps.len(), 2, "daughter RNAP renders on main + bubble");
let ff = 0.45_f32;
let sister = strands.last().unwrap();
let main_w = center + strand_point(strands, 0, coord, GENOME_BP_DEFAULT).unwrap().0.coords;
let bub_w = center + bubble_point(sister, coord, ff, GENOME_BP_DEFAULT).unwrap().0.coords;
let near_main = rnaps.iter().any(|p| (p.position - main_w).norm() < 30.0);
let near_bubble = rnaps.iter().any(|p| (p.position - bub_w).norm() < 30.0);
assert!(near_main && near_bubble, "expected one RNAP near main and one near the bubble");
```

Add `daughter_rnap_overlaid_on_bubble` (domain 2, coord 0) asserting count 2 and confinement of both; and confirm a domain-0 RNAP (use `recipe_replicating_with_one_rnap(0, 0)`) yields exactly ONE placement (no overlay).

- [ ] **Step 2: Run → FAIL** (count is still 1).

- [ ] **Step 3: Implement** the overlay after the main push. Factor the confine+orient+push into a local closure `place_at(world_pt, tangent)` used for both main and sister so the logic isn't duplicated.

- [ ] **Step 4: Run full suite → PASS** (`cargo test -p parsimony-core --lib`). Existing `seats_every_rnap_on_strand_inside_envelope` (domain-0 fixture) is unaffected (still N).

- [ ] **Step 5: Commit**

```bash
cd /Users/eranagmon/code/parsimony
git add crates/parsimony-core/src/placer.rs
git commit -m "feat(placer): overlay daughter RNAPs on the replication bubble (sister strand)"
```

---

### Task BF1-3: overlay daughter nascent RNA on the sister (Rust) + checkpoint

For each nascent daughter RNA (`root_domain != 0`, not `is_free`), grow a SECOND strand rooted at the sister bubble position, so the daughter copy carries its transcript too.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/placer.rs` — the nascent-RNA loop (~1034+).
- Test: update `nascent_rna_roots_on_main_contour_like_its_rnap` + add `daughter_rna_overlaid_on_bubble`.

**Interfaces:**
- Consumes: `bubble_point` (BF1-1), `chr.fork_fraction`, `glen`, `generate_rna_strand`, `RnaStrand`.
- Behavior: keep the existing nascent strand (rooted via `strand_point(&strands, 0, …)`, pushed with `is_free:false`). THEN, if `!rna.is_free && rna.root_domain != 0 && strands.len() > 1`, root a second strand at `bubble_point(&strands[strands.len()-1], rna.root_coordinate, chr.fork_fraction, glen)`, grow with `generate_rna_strand` (same bead_count/step/radius), and push a second `RnaStrand { points, is_mrna: rna.is_mRNA, is_free: false }`. Free strands are NOT overlaid.

- [ ] **Step 1: Update the existing test + write the new one**

Update `nascent_rna_roots_on_main_contour_like_its_rnap` (domain-2 RNA): now `rna_strands.len() == 2`; assert one strand's root is near the main locus and one near the bubble locus (mirror the BF1-2 main/bubble proximity assertions). Add `daughter_rna_overlaid_on_bubble` asserting count 2 for a domain-2 nascent RNA, and that a domain-0 nascent RNA yields ONE strand, and a FREE RNA yields ONE strand (no overlay).

- [ ] **Step 2: Run → FAIL** (count is 1).

- [ ] **Step 3: Implement** the RNA overlay after the main strand push.

- [ ] **Step 4: Run full suite → PASS.** Then `cargo build --release -p parsimony-cli`.

- [ ] **Step 5: Full build + viewer checkpoint**

```bash
cd /Users/eranagmon/code/v2e-3d-txn
rm -rf .parsimony/cache
PARSIMONY_HOME=/Users/eranagmon/code/parsimony /Users/eranagmon/code/v2ecoli/.venv/bin/python -m v2ecoli.structural.build --out out/ecoli3d --state snapshot
/Users/eranagmon/code/v2ecoli/.venv/bin/python out/ecoli3d/_view/make_local_bundle.py out/ecoli3d/ecoli_3d.pack.json out/ecoli3d/meshes
```
Expected: rna_polymerase placement count rises above 734 (734 main + the ~586 daughter overlays ≈ 1320); the replication bubble now shows daughter RNAPs + their RNA (second copy); confirm in the viewer; no protrusions.

- [ ] **Step 6: Commit**

```bash
cd /Users/eranagmon/code/parsimony
git add crates/parsimony-core/src/placer.rs
git commit -m "feat(placer): overlay daughter nascent RNA on the replication bubble"
```

---

## Self-Review

**Spec coverage (BF1 rows):**
- Daughter RNAP overlay on the bubble → BF1-2. ✓
- Daughter nascent RNA overlay on the bubble → BF1-3. ✓
- Bubble-relative mapping → BF1-1 (`bubble_point`). ✓
- domain-0 single placement; free RNA not overlaid → BF1-2/BF1-3 tests. ✓
- Documented over-render (count 2× for daughters, molecule count unchanged) → BF1-3 checkpoint verifies ~1320 RNAP placements. ✓
- Confinement + determinism → reuse `confine_center`; no new RNG. ✓
- BF2 (capture + per-chromosome routing) intentionally OUT → separate plan after BF1 review. ✓

**Placeholder scan:** all steps have concrete code/commands; the "factor a local closure" guidance points at the real confine+orient+push block in the diff.

**Type consistency:** `bubble_point` signature consistent BF1-1→BF1-2→BF1-3; uses `GENOME_BP_DEFAULT`, `chr.fork_fraction`, `strands.last()` (the sister) consistently; `RnaStrand { points, is_mrna, is_free }` matches the current struct.
