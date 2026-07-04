# 3D structural model: full transcription / translation state

**Date:** 2026-06-25
**Status:** Design approved, pending spec review
**Repos touched:** `v2ecoli` (state extraction + `structural/build.py`), `pbg-parsimony` (recipe API), `parsimony` (Rust geometry engine), viewer.

## Summary

Make the v2ecoli 3D structural model render the cell's transcription/translation
state **exactly as v2ecoli tracks it**: every active RNAP at its real genomic
locus, the nascent RNA being transcribed off each RNAP, and — for mRNAs — the
ribosomes actively translating them into nascent polypeptide chains. Replication
markers (replisome / oriC / terC) are enlarged for legibility. Every placed
entity stays inside the cell envelope.

This replaces today's behaviour, where RNAP is placed as ~2000 fiber-packed
copies at random / generic genome-binding sites with no transcript, no
ribosomes, and no envelope confinement (some RNAPs protrude in the published
build).

## Goals

- **Precise RNAP placement** from v2ecoli `active_RNAP` state: exact genomic
  coordinate, correct chromosome domain/strand, orientation from `is_forward`.
- **Nascent RNA** rooted at each transcribing RNAP, contour length proportional
  to `transcript_length`, visually connected to the chromosome.
- **Ribosomes + nascent peptides**: ribosomes seated on each mRNA at
  `pos_on_mRNA`, each with a nascent polypeptide blob proportional to
  `peptide_length`. Polysomes (multiple ribosomes per mRNA) rendered at true
  count.
- **True abundance (1:1)** with v2ecoli counts for all of the above.
- **Confinement invariant:** every placed entity lies inside the cell envelope.
- **Enlarged replication markers** (replisome / oriC / terC).

## Non-goals

- Real 3D chromosome coordinates. v2ecoli has none — positions are 1D base
  pairs. We map exactly along the genomic axis onto our stylized nucleoid curve.
  Absolute 3D position is our model's curve, not a v2ecoli measurement.
- Physically accurate RNA/peptide folding. Nascent strands are stylized
  self-avoiding fibers whose *length* (not secondary structure) is faithful.
- Time dynamics / animation. This is a single static state, as today.

## Source data: v2ecoli unique-molecule state

From `internal_state.py` (`_build_unique_molecules`); accessed on the live
composite as `cell["unique"][<name>]` — structured numpy arrays masked by
`_entryState` (pattern already used in `bridge.py`).

| Molecule | Attributes used | Role |
|---|---|---|
| `active_RNAP` | `coordinates` (i8, bp from origin), `domain_index` (i4), `is_forward` (?) | RNAP genomic locus + strand + direction |
| `RNA` | `RNAP_index` (i8), `transcript_length` (i8), `is_mRNA` (?), `is_full_transcript` (?), `TU_index` (i8) | nascent transcript; `RNAP_index` links it to its RNAP |
| `active_ribosome` | `mRNA_index` (i8), `pos_on_mRNA` (i8), `peptide_length` (i8), `protein_index` (i8) | ribosome on an mRNA + nascent peptide; `mRNA_index` links it to its RNA |
| `active_replisome` | `coordinates`, `domain_index` | fork markers (already partly read) |
| `full_chromosome`, `chromosome_domain`, `oriC` | counts / `domain_index` | chromosome topology, already summarized as `n_chromosomes` / `fork_fraction` |

Connectivity chain (all recoverable):
```
active_RNAP[coordinates, domain_index, is_forward]
   └─ RNA[RNAP_index → rnap uid, transcript_length, is_mRNA]   (partial = attached)
        └─ active_ribosome[mRNA_index → rna uid, pos_on_mRNA, peptide_length]
```
`RNA` rows with `is_full_transcript == False` are attached to their RNAP via
`RNAP_index`; fully transcribed mRNAs (`RNAP_index == -1`) float free and may
still carry ribosomes.

## Architecture

Rust-native connectivity geometry. Python extracts state and hands the engine an
explicit per-molecule spec; the Rust engine maps genomic coordinate → 3D point
and generates the RNA / ribosome / peptide geometry, inheriting `CellShape`
confinement and instancing for true-abundance counts.

```
v2ecoli unique state ──extract──> TranscriptionState (Python)
   rnaps:     [(coord, domain_index, is_forward, rnap_uid)]
   rnas:      [(rnap_uid|-1, length_nt, is_mRNA, rna_uid)]
   ribosomes: [(rna_uid, pos_on_mRNA, peptide_len_aa, protein_idx)]
        │  serialized into the recipe as a new "transcription" block
        ▼
pbg_parsimony.build_pack(..., transcription=TranscriptionState)
        │
        ▼
parsimony-core (Rust):
   coordinate(bp) → strand point  + domain_index → main/sister strand   [placer.rs]
   generate_rna_strand(root, length, shape)                            [fiber.rs]
   seat_translation(rna_strand, ribosomes, shape)                      [placer.rs]
```

### Genomic coordinate → 3D point

The rendered strand is a `Vec<Point3>` of beads along a self-avoiding contour;
the theta builder already places oriC at the strand midpoint. Define bead `i`'s
genomic coordinate as `(i/beads)·GENOME_BP` with oriC at the midpoint, so a
v2ecoli `coordinates` value (bp from origin, signed across replichores) maps to
`frac = 0.5 + coordinates/GENOME_BP → bead index → 3D point`. `domain_index`
selects which strand (main vs sister for a replicating chromosome); `is_forward`
orients the RNAP along the local strand tangent. Reuse the existing
`genome`/`binding_sites` contour parametrization where it already exists.

### Confinement invariant (applies to ALL entities, incl. existing RNAPs)

Root cause of today's protrusion: `pack_on_fiber` / `pack_on_fiber_at`
(`fiber_pack.rs`) place a bound protein at `strand_point + outward_normal ×
offset` and never receive the cell shape, so a strand bead near the wall pushes
the protein through it.

Fix: pass `CellShape` into `pack_on_fiber*` and the new generators. Invariant —
every entity center satisfies `cell.inset(proxy_radius).contains(center)`. When
the outward radial offset would exit the envelope, rotate the offset around the
strand tangent (and/or shrink it inward) until it fits, mirroring the existing
`clamp_inside` / bulge-shrink patterns in `fiber.rs`. A whole-pack test scans
every placement for protrusion (must be zero).

## Phased delivery (review checkpoint after each)

### Phase A — markers + precise RNAP + confinement fix
- Enlarge replisome / oriC / terC markers.
- Implement `coordinate → 3D point` + `domain_index → strand` + `is_forward`
  orientation; place every `active_RNAP` at its real locus.
- Extend snapshot npz (+ live path) to capture `active_RNAP` arrays.
- Add `CellShape` confinement to `pack_on_fiber*`; fix protruding RNAPs.
- **Gates:** every RNAP on-strand and inside the envelope; oriC-relative
  ordering preserved; n=1 vs n=2 differ; whole-pack protrusion count == 0.

### Phase B — nascent RNA
- `generate_rna_strand`: thin self-avoiding fiber rooted at each transcribing
  RNAP, contour length ∝ `transcript_length`, confined by `CellShape`, distinct
  RNA category/color. Partial transcripts (`RNAP_index ≥ 0`, attached) vs full
  mRNAs (free) handled via `is_full_transcript` / `RNAP_index`.
- Extend snapshot/live extraction + recipe with `RNA` rows.
- **Gates:** each strand root within `bead_radius` of its RNAP; length monotone
  in `transcript_length`; confined; deterministic for a seed.

### Phase C — ribosomes + peptides
- Seat ribosomes (reuse `70S_ribosome` structure) along each mRNA strand at
  `pos_on_mRNA`; nascent peptide as a short coil/blob ∝ `peptide_length` from
  each ribosome. Polysomes at true count.
- Extend extraction + recipe with `active_ribosome` rows.
- **Gates:** ribosome count == v2ecoli; each ribosome on its mRNA; peptide
  length monotone in `peptide_length`; confined; viewer instancing holds at true
  abundance.

## Testing

- **Rust** (`cargo test -p parsimony-core --lib`): confinement (no protrusion),
  connectivity (RNA→RNAP, ribosome→RNA), genomic-coordinate ordering, length
  monotonicity, determinism per seed. Build with
  `cargo build --release -p parsimony-cli` (binary `target/release/parsimony`).
- **Python**: extraction round-trips the unique arrays; rendered counts match
  v2ecoli counts at true abundance.
- **Viewer**: new size-filter categories (RNA / Ribosome / Peptide);
  connectivity reads visually; after any Rust change, `rm -rf .parsimony/cache`
  (recipe-keyed, not binary-keyed) and bump the viewer `?v=` stamp.

## Risks / notes

- **Scale:** ~10–20k ribosomes × peptides plus ~1–4k RNA strands is heavy;
  relies on instancing + LOD and the octree backend. Phase C is where viewer
  perf is validated; if it stalls, fall back to per-category LOD/size-filter
  defaults (no count reduction — true abundance is a requirement).
- **`.parsimony/cache` gotcha:** keyed by recipe content, not Rust version —
  must clear after every geometry change or edits silently no-op.
- **Stale-branch hazard:** `/code/v2ecoli` sits on `work/main-current`; do
  geometry work from a fresh worktree off current `origin/main` (handled in the
  implementation plan, not here).
- **Domain→strand mapping** for replicating chromosomes is the subtlest
  fidelity point; Phase A validates it before RNA/ribosomes build on top.
