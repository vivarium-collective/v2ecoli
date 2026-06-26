# 3D Transcription — C2: nascent peptides

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow an extended nascent-peptide coil from each placed ribosome, contour length ∝ `peptide_length`, rendered as a distinct `peptide_segment` — completing the transcription→translation chain.

**Architecture:** In `place_chromosome`'s ribosome loop (C1-3), after each ribosome is placed, grow a confined coil from its position (reusing `generate_rna_strand`), store it on the snapshot (`peptide_strands`), and tile a `peptide_segment` mesh along each in `output.rs` (mirroring the RNA-strand tiling). `pbg-parsimony` passes the peptide ingredient through; `v2ecoli` adds the ingredient + scale.

**Tech Stack:** Rust (nalgebra) `parsimony-core`; Python `v2ecoli`; `pbg-parsimony` passthrough.

## Global Constraints

- **One coil per placed ribosome with `peptide_length > 0`:** contour length = `peptide_length × peptide_angstrom_per_aa` (default 3.0 Å/aa), rooted at the ribosome, confined (surface-pull, never medial-collapse). `peptide_length == 0` (just-initiated) → no coil.
- **Count:** peptide coils == placed ribosomes with peptide_length>0 (no extra; dropped ribosomes have no peptide).
- **Confinement + determinism** preserved (the coil consumes rng deterministically per seed; grow AFTER the ribosome is placed).
- **Build/test:** Rust `/Users/eranagmon/code/parsimony` (`cargo test -p parsimony-core --lib`; `cargo build --release -p parsimony-cli`). Python in worktree `/Users/eranagmon/code/v2e-3d-txn`; interpreter `/Users/eranagmon/code/v2ecoli/.venv/bin/python` from the worktree; `PARSIMONY_HOME=/Users/eranagmon/code/parsimony`; pbg-parsimony editable.

**Reference reading:**
- `parsimony/crates/parsimony-core/src/placer.rs` — the ribosome loop in `place_chromosome` (C1-3: places each ribosome at `pos`, has `ribo_r`/`inset`/`center`); `generate_rna_strand` usage in the RNA loop (root → coil).
- `parsimony/crates/parsimony-core/src/placement.rs` — `Snapshot` (add `peptide_strands`); `RnaStrand`/`rna_strands` pattern.
- `parsimony/crates/parsimony-core/src/output.rs` — the `rna_strands` → `rna_segment` tiling block (mirror for `peptide_strands` → `peptide_segment`).
- `parsimony/crates/parsimony-core/src/recipe.rs` — `ChromosomeSpec` rna fields (`rna_segment`/`rna_angstrom_per_nt`) — mirror for `peptide_segment`/`peptide_angstrom_per_aa`.
- `parsimony/crates/parsimony-core/src/fiber.rs` — `generate_rna_strand`.
- `v2ecoli/structural/build.py` — the `rna_segment` ingredient + the `Chromosome(...)` call.

---

### Task C2-1: grow + render peptide coils (Rust + pbg)

Recipe peptide fields, snapshot `peptide_strands`, grow a coil per ribosome, tile `peptide_segment`.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/recipe.rs` — `RawChromosome`/`ChromosomeSpec`: add `peptide_segment: Option<String>` (`#[serde(default)]`) + `peptide_angstrom_per_aa: f32` (recipe key `peptide_angstrom_per_aa`, default 3.0).
- Modify: `parsimony/crates/parsimony-core/src/placement.rs` — `Snapshot.peptide_strands: Vec<Vec<Point3<f32>>>` (`#[serde(default)]`, center-relative).
- Modify: `parsimony/crates/parsimony-core/src/placer.rs` — in the ribosome loop, after pushing the ribosome `Placement`, if `ribo.peptide_length > 0` grow a coil and push to `snapshot.peptide_strands`.
- Modify: `parsimony/crates/parsimony-core/src/output.rs` — tile `recipe.chromosome.peptide_segment` along each `peptide_strands` entry (mirror the rna_segment block).
- Modify: `parsimony/crates/parsimony-core/src/pipeline.rs` — carry `peptide_strands` through the multi-stage merge (like `rna_strands` — find that line and add the analogous `merged.peptide_strands.extend(...)`).
- Modify: `pbg-parsimony/api.py` — `Chromosome.peptide_segment`/`peptide_angstrom_per_aa` passthrough.
- Test: `placer.rs` (a ribosome with peptide_length>0 → a peptide strand; ==0 → none) + `output.rs` (peptide_segment placements emitted).

**Interfaces:**
- `ChromosomeSpec.peptide_segment: Option<String>`, `ChromosomeSpec.peptide_angstrom_per_aa: f32` (default 3.0).
- `Snapshot.peptide_strands: Vec<Vec<Point3<f32>>>` (center-relative).
- Coil growth: `pep_step = 30.0` (Å/bead), `pep_bead_radius = 3.0`; `bead_count = ((ribo.peptide_length as f32 * chr.peptide_angstrom_per_aa) / pep_step).round().max(2)`; `root_rel = pos - center` (the ribosome world `pos` minus `center` → center-relative); `points = generate_rna_strand(root_rel, bead_count, pep_step, pep_bead_radius, shape, rng)`; push `points`.

- [ ] **Step 1: Write the failing tests** — (a) placer: a recipe (free mRNA unique_index 20 length 600, ribosome_marker, `ribosomes: [{mRNA_index:20, pos_on_mRNA:300, peptide_length:200}, {mRNA_index:20, pos_on_mRNA:0, peptide_length:0}]`, `peptide_segment: "peptide_segment"`, `peptide_angstrom_per_aa: 3.0`, + a peptide_segment object); pack; assert `out.snapshot.peptide_strands.len() == 1` (the peptide_length>0 ribosome only), and a longer peptide_length yields more beads. (b) output: assert `peptide_segment` placements emitted (>0).

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** the recipe fields, `Snapshot.peptide_strands` (+ in `Snapshot::new`), the coil growth in the ribosome loop, the output tiling, the pipeline merge carry, and the pbg passthrough.

- [ ] **Step 4: Run full suite → PASS.** `cargo build --release -p parsimony-cli`. Run the pbg tests.

- [ ] **Step 5: Commit** (parsimony recipe+placement+placer+output+pipeline; pbg api).

---

### Task C2-2: build wiring + peptide ingredient (Python) + checkpoint

Add the `peptide_segment` ingredient + scale, pass through, viewer checkpoint.

**Files:**
- Modify: `v2ecoli/v2ecoli/structural/build.py` — add a `peptide_segment` ingredient + pass `peptide_segment`/`peptide_angstrom_per_aa` to `Chromosome(...)`.
- Test: `v2ecoli/tests/structural/test_build_peptides.py` (create).

**Interfaces:**
- Add `Ingredient(id="peptide_segment", count=0, structure=StructureRef("pdb","1BNA") OR a thin proxy, color=<distinct Translation shade, e.g. orange-red (0.95,0.45,0.3)>, category="Translation", display_name="Nascent peptide")`. (Reuse the dsDNA segment mesh as the tiled unit, like rna_segment, OR a thin cylinder; simplest is the same segment-mesh approach with a distinct color.)
- `Chromosome(..., peptide_segment="peptide_segment", peptide_angstrom_per_aa=3.0)`.

- [ ] **Step 1: Write the failing test** — synthesize a snapshot with one mRNA (unique_index 20, length 600) + one ribosome (mRNA 20, pos 300, peptide_length 200); build a small model (top_n 5, slow/skip-guard/PARSIMONY_HOME pattern); assert `peptide_segment` placements > 0 and `peptide_segment` in the sidecar meta.

- [ ] **Step 2: Run → FAIL** (no peptide_segment).

- [ ] **Step 3: Implement** the ingredient + wiring.

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Full build + viewer checkpoint**

```bash
cd /Users/eranagmon/code/v2e-3d-txn && rm -rf .parsimony/cache
PARSIMONY_HOME=/Users/eranagmon/code/parsimony /Users/eranagmon/code/v2ecoli/.venv/bin/python -m v2ecoli.structural.build --out out/ecoli3d --state snapshot
/Users/eranagmon/code/v2ecoli/.venv/bin/python out/ecoli3d/_view/make_local_bundle.py out/ecoli3d/ecoli_3d.pack.json out/ecoli3d/meshes
```
Report: the `peptide_segment` placement count (>0; coils from ribosomes with peptide_length>0), a protrusion check, total placements (perf — heaviest yet). Confirm in the viewer peptides trail from ribosomes on the mRNAs.

- [ ] **Step 6: Commit** build.py + test.

---

## Self-Review

**Spec coverage (C2 rows):**
- Peptide coil per ribosome ∝ peptide_length → C2-1 (growth) + C2-2 (ingredient). ✓
- peptide_length 0 → no coil → C2-1 test. ✓
- Distinct Translation shade, tiled segment → C2-1 (output) + C2-2 (ingredient color). ✓
- Confinement + determinism → reuse generate_rna_strand; coil after ribosome push. ✓
- Count == ribosomes with peptide>0 → C2-1. ✓

**Placeholder scan:** the peptide ingredient mesh choice gives a concrete default (reuse the segment mesh + distinct color) with the cylinder alternative noted; pep_step/radius/scale are concrete numbers. No TODOs.

**Type consistency:** `peptide_segment`/`peptide_angstrom_per_aa` consistent C2-1→C2-2; `Snapshot.peptide_strands: Vec<Vec<Point3<f32>>>` consistent placer→output→pipeline; reuses `RibosomeSpec.peptide_length` (C1-2) + `generate_rna_strand` (B1).
