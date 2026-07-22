# 3D Transcription — C1: ribosomes on mRNA (+ corrected ribosome state)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Place each `active_ribosome` (assembled 70S) on its mRNA at `pos_on_mRNA`, and correct the ribosome representation — replace the fabricated curated `70S_ribosome=20000` with active 70S on mRNAs + real free 30S/50S subunit pools.

**Architecture:** v2ecoli captures `active_ribosome` arrays and threads each RNA's `unique_index` onto its strand spec; `parsimony` carries `unique_index`+`length_nt` on `RnaStrand`, then `place_translation` maps `mRNA_index → RnaStrand` and seats each ribosome at the `pos_on_mRNA` contour fraction, offset outward. The corrected subunit ingredients come from real bulk counts (`CPLX0-3953`/`CPLX0-3962`).

**Tech Stack:** Python (numpy) `v2ecoli`; Rust (serde, nalgebra) `parsimony-core`; `pbg-parsimony` passthrough.

## Global Constraints

- **Correct state:** no `70S_ribosome` curated count of 20000; instead active 70S on mRNAs (count 0 curated, placed explicitly) + free 30S (`CPLX0-3953`) + free 50S (`CPLX0-3962`) at their real bulk counts.
- **On the mRNA:** each ribosome at `frac = pos_on_mRNA / length_nt` along its mRNA strand (`mRNA_index == RNA.unique_index`), offset outward from the strand by ~the ribosome radius so it sits ON the mRNA.
- **True abundance (1:1):** ribosome count rendered == active_ribosome count, EXCEPT ribosomes whose `mRNA_index` matches no rendered strand are dropped-with-logged-count (surfaced, not hidden).
- **Confinement** (surface-pull) + **determinism** (fixed seed) preserved.
- **Build/test:** Rust `/Users/eranagmon/code/parsimony` (`cargo test -p parsimony-core --lib`; `cargo build --release -p parsimony-cli`). Python in worktree `/Users/eranagmon/code/v2e-3d-txn`; interpreter `/Users/eranagmon/code/v2ecoli/.venv/bin/python` from the worktree; `PARSIMONY_HOME=/Users/eranagmon/code/parsimony`; pbg-parsimony editable; ParCa cache symlink `out/cache` present.

**Reference reading:**
- `v2ecoli/scripts/capture_structural_snapshot.py` `_extract_snapshot` — how `active_RNAP`/`RNA`/`full_chromosome` are read (mirror for `active_ribosome`).
- `v2ecoli/structural/build.py` — `rna_state`/`rnap_state` readers; `CURATED` (~265); `select_ingredients` (~625); `build_model` rna/rnap dict construction.
- `parsimony/crates/parsimony-core/src/recipe.rs` — `RnaSpec` (~397, fields: root_coordinate/root_domain/length_nt/is_mRNA/is_free/chromosome_index/is_daughter); the `rnaps`/`RnapPlacement` block (mirror for a `ribosomes` block).
- `parsimony/crates/parsimony-core/src/placement.rs` — `RnaStrand{points,is_mrna,is_free}` (~14).
- `parsimony/crates/parsimony-core/src/placer.rs` — `place_chromosome` RNA loop (where `RnaStrand` is pushed); `confine_center`, `CellShape::outward`.

---

### Task C1-1: capture active_ribosome (Python)

Capture the `active_ribosome` unique-molecule arrays + a reader.

**Files:**
- Modify: `v2ecoli/scripts/capture_structural_snapshot.py` (`_extract_snapshot`).
- Modify: `v2ecoli/v2ecoli/structural/build.py` (add `ribosome_state`).
- Test: `v2ecoli/tests/structural/test_ribosome_state.py` (create).

**Interfaces:**
- New npz keys: `ribo_mRNA_index` (i8), `ribo_pos_on_mRNA` (i8), `ribo_peptide_length` (i8), `ribo_protein_index` (i8).
- `ribosome_state(state_source) -> dict` with keys `mRNA_index`, `pos_on_mRNA`, `peptide_length`, `protein_index` (numpy arrays; empty i8 fallbacks). Mirror `rna_state`'s file selection.
- Capture: `rib = cell["unique"]["active_ribosome"]; active = rib[rib["_entryState"].view(bool)]`; read the four fields (guard each with `in rib.dtype.names`). Preserve all existing keys.

- [ ] **Step 1: Write the failing test** (synthesized npz, no sim)

```python
import numpy as np
from v2ecoli.structural import build

def test_ribosome_state_reads_arrays(tmp_path, monkeypatch):
    np.savez(tmp_path / "v2ecoli_state.npz",
        ids=np.array(["x[c]"]), counts=np.array([1]), volume=np.array(1.0),
        n_chromosomes=np.array(1), fork_fraction=np.array(0.0), division_progress=np.array(0.0),
        ribo_mRNA_index=np.array([20, 21], "i8"), ribo_pos_on_mRNA=np.array([0, 300], "i8"),
        ribo_peptide_length=np.array([0, 100], "i8"), ribo_protein_index=np.array([5, 6], "i8"))
    monkeypatch.setattr(build, "DATA", tmp_path)
    st = build.ribosome_state("snapshot")
    assert list(st["mRNA_index"]) == [20, 21]
    assert list(st["pos_on_mRNA"]) == [0, 300] and st["peptide_length"].dtype == np.int64
```

- [ ] **Step 2: Run → FAIL** (`cd /Users/eranagmon/code/v2e-3d-txn && /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest tests/structural/test_ribosome_state.py -q`).

- [ ] **Step 3: Implement** `ribosome_state` (mirror `rna_state`) + the capture-script extraction (mirror the `RNA` block). Empty fallbacks dtype i8.

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Regenerate snapshots** — `PARSIMONY_HOME=/Users/eranagmon/code/parsimony /Users/eranagmon/code/v2ecoli/.venv/bin/python scripts/capture_structural_snapshot.py`. Verify the birth npz `ribo_mRNA_index` is nonzero-length (active ribosome count). If blocked, report DEFERRED (unit test gates).

- [ ] **Step 6: Commit** capture script + build.py + test + regenerated npz.

---

### Task C1-2: identity plumbing + recipe ribosomes block (Rust + pbg)

Carry the RNA's `unique_index`+`length_nt` onto `RnaStrand`, and add a recipe `ribosomes` block.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/recipe.rs` — `RawRna`/`RnaSpec` add `unique_index: i64` (`#[serde(default)]` → 0); add `RawRibosome`/`RibosomeSpec` + a chromosome `ribosomes: Vec<RibosomeSpec>` + `ribosome_marker: Option<String>`.
- Modify: `parsimony/crates/parsimony-core/src/placement.rs` — `RnaStrand` add `unique_index: i64` + `length_nt: i64`.
- Modify: `parsimony/crates/parsimony-core/src/placer.rs` — the `RnaStrand` push sets `unique_index`/`length_nt` from the `RnaSpec` (BOTH the main strand and the BF1 bubble overlay copy carry the same unique_index/length_nt).
- Modify: `pbg-parsimony/api.py` — confirm `rnas` passthrough carries `unique_index`; add `Chromosome.ribosomes`/`ribosome_marker` passthrough.
- Test: `parsimony` recipe round-trip + a placer test that `RnaStrand.unique_index` is set.

**Interfaces:**
- `RnaSpec.unique_index: i64`; `RnaStrand.unique_index: i64`, `RnaStrand.length_nt: i64`.
- `RibosomeSpec { mRNA_index: i64, pos_on_mRNA: i64, peptide_length: i64 }`; `ChromosomeSpec.ribosomes: Vec<RibosomeSpec>`; `ChromosomeSpec.ribosome_marker: Option<String>`.
- Recipe JSON chromosome block: `"ribosome_marker": "70S_ribosome"`, `"ribosomes": [{"mRNA_index": 20, "pos_on_mRNA": 0, "peptide_length": 0}]`; rnas entries gain `"unique_index"`.

- [ ] **Step 1: Write the failing tests** — (a) recipe round-trip: a rnas entry with `"unique_index": 20` resolves; a chromosome `ribosomes` entry resolves with `ribosome_marker`. (b) placer: build a recipe with one nascent rna (`unique_index` 20, length_nt 400); pack; assert `out.snapshot.rna_strands[0].unique_index == 20 && .length_nt == 400`.

- [ ] **Step 2: Run → FAIL** (`cargo test -p parsimony-core --lib`).

- [ ] **Step 3: Implement** — add `RnaSpec.unique_index` (mirror `is_free`); add `RawRibosome`/`RibosomeSpec` + the chromosome fields (mirror `rnas`/`RnaSpec`); add `RnaStrand.unique_index`/`length_nt` (derive Default; serde) and set them at BOTH `RnaStrand` push sites (main + BF1 bubble overlay) from `rna.unique_index`/`rna.length_nt`. In pbg api.py, add `Chromosome.ribosomes`/`ribosome_marker` + guarded chrom_block emission, and ensure rnas dicts already carry `unique_index` (verbatim passthrough — confirm).

- [ ] **Step 4: Run → PASS** (Rust + a quick pbg test if added).

- [ ] **Step 5: Commit** (parsimony recipe+placement+placer; pbg api).

---

### Task C1-3: place_translation — ribosomes on their mRNA (Rust)

Place each ribosome at its `pos_on_mRNA` on the matching mRNA strand, offset outward.

**Files:**
- Modify: `parsimony/crates/parsimony-core/src/placer.rs` — in `place_chromosome`, after the RNA loop, add a ribosome loop (or a `place_translation` helper called there).
- Test: `placer.rs` tests.

**Interfaces:**
- Consumes: `chr.ribosomes`/`chr.ribosome_marker` (C1-2), `snapshot.rna_strands` (each with `unique_index`/`length_nt`/`points`), `center`, `shape`, `confine_center`, `CellShape::outward`.
- Behavior: build a map `{unique_index → &RnaStrand}` from `snapshot.rna_strands`. For each `RibosomeSpec`: look up `mRNA_index`; if no match, increment a dropped counter and `continue` (no crash). Else `frac = (pos_on_mRNA as f32 / length_nt.max(1) as f32).clamp(0,1)`; `idx = (frac*(points.len()-1)).round()`; `base = center + points[idx].coords`; offset outward: `pos = base + shape.outward(&base) * ribosome_offset` (ribosome_offset ≈ ribosome enclosing radius, e.g. read from the `ribosome_marker` ingredient's `enclosing_radius`, fallback ~120.0); confine via `confine_center` (surface-pull, same pattern as RNAP); push a `Placement` of the `ribosome_marker` ingredient (identity rotation or orient along the strand tangent). Increment uid.

- [ ] **Step 1: Write the failing test** — a recipe with a chromosome + one free mRNA (`unique_index` 20, length_nt 600) + a `ribosomes` entry `{mRNA_index: 20, pos_on_mRNA: 300, peptide_length: 0}` + `ribosome_marker` = a sphere ingredient (radius 120). Pack; assert one ribosome placement exists, it is near the mRNA strand's midpoint bead (frac 0.5) within `offset + tolerance`, and inside the envelope. Also assert a ribosome with `mRNA_index` 999 (no strand) is NOT placed (dropped).

- [ ] **Step 2: Run → FAIL** (no ribosome placements).

- [ ] **Step 3: Implement** the ribosome loop. `log`/return the dropped count (a `log!`/eprintln or a stat field — keep simple: count + a comment; the build will surface it).

- [ ] **Step 4: Run full suite → PASS.** `cargo build --release -p parsimony-cli`.

- [ ] **Step 5: Commit** placer.rs.

---

### Task C1-4: build wiring + corrected subunits (Python) + checkpoint

Remove the fabricated 70S, add 30S/50S, thread RNA unique_index, build the ribosomes list, wire the marker; viewer checkpoint.

**Files:**
- Modify: `v2ecoli/v2ecoli/structural/build.py` — `CURATED` + `build_model`.
- Test: `v2ecoli/tests/structural/test_build_ribosomes.py` (create).

**Interfaces:**
- Consumes: `ribosome_state` (C1-1), `rna_state` `unique_index` (Phase B), `Chromosome.ribosomes`/`ribosome_marker` (C1-2).
- `CURATED`: REMOVE `("70S_ribosome", …, 20000, "interior")`; ADD:
  - `("70S_ribosome", None, "Translation", ("cif","4YBB"), 0, "interior")` — count 0, placed by place_translation as the ribosome_marker.
  - `("30S_subunit", None, "Translation", ("pdb","2AVY"), "CPLX0-3953", "interior")` — count from state.
  - `("50S_subunit", None, "Translation", ("pdb","2AW4"), "CPLX0-3962", "interior")` — count from state.
  (If `2AVY`/`2AW4` fail to fetch/mesh, the build logs a skip — try alternates like `("pdb","4V4Q")` split or note it; do NOT block on a perfect subunit mesh, but the 30S/50S COUNTS must come from `CPLX0-3953`/`CPLX0-3962`.)
- `build_model`: thread each RNA's `unique_index` (from `rna_state`) onto its rnas dict; build `ribosomes = [{"mRNA_index": int(m), "pos_on_mRNA": int(p), "peptide_length": int(l)} for m,p,l in zip(ribosome_state[...])]`; pass `ribosomes=ribosomes, ribosome_marker="70S_ribosome"` to `Chromosome(...)`.

- [ ] **Step 1: Write the failing test** — synthesize a snapshot with one RNA (`rna_unique_index` 20, is_mRNA, length 600) and one ribosome (`ribo_mRNA_index` 20, pos 300); build a small model (top_n 5, pre-seed structures cache, `@pytest.mark.slow`, `_STRUCT_CACHE` skip-guard, `PARSIMONY_HOME` via os.environ.get); assert (a) the pack contains a `70S_ribosome` placement (>0) on the mRNA, (b) `70S_ribosome` curated count is 0 (no 20000 fabrication — assert the meta count or that no random 70S interior placements exist beyond the active one), (c) `30S_subunit`/`50S_subunit` appear in the sidecar meta.

- [ ] **Step 2: Run → FAIL** (no 70S on mRNA / still 20000).

- [ ] **Step 3: Implement** the CURATED edit + build_model wiring.

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Full build + viewer checkpoint** (the C1 perf checkpoint — heaviest layer)

```bash
cd /Users/eranagmon/code/v2e-3d-txn && rm -rf .parsimony/cache
PARSIMONY_HOME=/Users/eranagmon/code/parsimony /Users/eranagmon/code/v2ecoli/.venv/bin/python -m v2ecoli.structural.build --out out/ecoli3d --state snapshot
/Users/eranagmon/code/v2ecoli/.venv/bin/python out/ecoli3d/_view/make_local_bundle.py out/ecoli3d/ecoli_3d.pack.json out/ecoli3d/meshes
```
Report: the `70S_ribosome` placement count (== active_ribosome count, minus any dropped — report the dropped count), the `30S_subunit`/`50S_subunit` counts (≈2622 each), confirm NO 20000-fabrication, a protrusion check over 70S positions, and the total placement count (perf). Confirm in the viewer ribosomes sit on the mRNAs (polysomes) + free subunits in cytoplasm.

- [ ] **Step 6: Commit** build.py + test.

---

## Self-Review

**Spec coverage (C1 rows):**
- Active ribosomes on mRNA at pos_on_mRNA → C1-3 (place) + C1-1 (capture) + C1-2 (identity) + C1-4 (wire). ✓
- Corrected inactive: free 30S/50S real counts, remove 70S=20000 → C1-4. ✓
- mRNA identity (unique_index on strand) → C1-2. ✓
- True abundance + dropped-with-log for unmatched mRNA_index → C1-3 + C1-4 checkpoint. ✓
- Confinement + determinism → reuse confine_center; no uncontrolled RNG. ✓
- C2 (peptides) intentionally OUT → separate plan after C1 review. ✓

**Placeholder scan:** the 30S/50S structure IDs (2AVY/2AW4) are concrete with a documented fallback; the dropped-count surfacing is concrete (counter + checkpoint report). No TODOs.

**Type consistency:** `RnaSpec.unique_index`/`RnaStrand.unique_index` (i64) consistent C1-2→C1-3; `RibosomeSpec{mRNA_index:i64, pos_on_mRNA:i64, peptide_length:i64}` consistent C1-2→C1-3→C1-4; npz keys `ribo_*` consistent C1-1→C1-4; `ribosome_marker`/`ribosomes` consistent across.
