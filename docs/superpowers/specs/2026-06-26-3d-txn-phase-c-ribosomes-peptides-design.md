# 3D structural model — Phase C: ribosomes + nascent peptides

**Date:** 2026-06-26
**Status:** Design approved, pending spec review
**Predecessors:** Phase A (RNAP), Phase B (nascent + free RNA), bubble fidelity.
**Repos touched:** `parsimony` (placement + peptide coil), `pbg-parsimony` (passthrough), `v2ecoli` (capture + build + corrected ribosome representation), viewer.

## Summary

Render the cell's translation machinery exactly as v2ecoli tracks it: each
`active_ribosome` (assembled, translating 70S) placed **on its mRNA** at
`pos_on_mRNA`, with an extended nascent-peptide coil whose length is proportional
to `peptide_length`. As part of this, **correct the ribosome representation** —
replace the fabricated curated `70S_ribosome = 20000` with the real state: active
70S on mRNAs + the free 30S/50S subunit pools in the cytoplasm.

Delivered in two sub-phases with a viewer checkpoint between:
- **C1 — ribosomes on mRNA** (+ the corrected free-subunit representation).
- **C2 — nascent peptides** (extended coil per ribosome).

## The correct ribosome state (v2ecoli)

`n_total_ribosomes = n_active_70S + min(n_free_30S, n_free_50S)`:
- **Active 70S** = the `active_ribosome` unique molecule (assembled, translating,
  bound to an mRNA). Attributes: `protein_index` (i8), `peptide_length` (i8, aa),
  `mRNA_index` (i8 → `RNA.unique_index`), `pos_on_mRNA` (i8, bases from the 5′ end).
- **Free 30S** = bulk `CPLX0-3953[c]` (≈2622 at birth); **free 50S** = bulk
  `CPLX0-3962[c]` (≈2622). There is NO bulk 70S — an assembled 70S exists only as
  an `active_ribosome`.

The current build's `CURATED` entry `("70S_ribosome", …, 20000, "interior")` is a
fabrication (wrong count, free in the cytoplasm) and is removed.

## Goals

- **Active ribosomes on mRNA:** every `active_ribosome` placed at its
  `pos_on_mRNA` on the matching mRNA strand (`mRNA_index == RNA.unique_index`),
  offset outward so it sits on the strand; 70S = the existing 4YBB mesh.
- **Correct inactive representation:** free 30S (`CPLX0-3953`) + free 50S
  (`CPLX0-3962`) as distinct cytoplasmic ingredients at their real bulk counts;
  the curated `70S_ribosome = 20000` removed.
- **Nascent peptides:** an extended thin coil from each ribosome, contour length
  ∝ `peptide_length`, distinct Translation shade, confined.
- **True abundance (1:1):** every active_ribosome rendered (no cap/subsample).
- **mRNA identity:** `RnaSpec`/`RnaStrand` carry the RNA's `unique_index` (+
  `length_nt`) so ribosomes can find and address their strand by `pos_on_mRNA`.
- **Confinement + determinism** preserved (surface-pull; fixed-seed reproducible).

## Non-goals

- Folded/atomically-accurate peptide structure (stylized confined coil; only
  length is faithful).
- Distinguishing inactive 70S vs free subunits beyond the 30S/50S bulk pools.
- tRNA rendering; ribosome–tRNA detail.
- Modeling ribosome footprint collisions precisely (some polysome overlap is
  accepted — see Risks).

## Source data

- `active_ribosome` unique molecule (capture like `active_RNAP`/`RNA`):
  `protein_index`, `peptide_length`, `mRNA_index`, `pos_on_mRNA`.
- `RNA.unique_index` (already captured, Phase B) — threaded into `RnaSpec` so the
  rendered strand is identifiable; `length_nt` (already in `RnaSpec`).
- Bulk `CPLX0-3953[c]` / `CPLX0-3962[c]` counts (already in the snapshot bulk).

## Architecture

Rust-native placement, mirroring Phase B.

```
v2ecoli capture: + active_ribosome[mRNA_index, pos_on_mRNA, peptide_length, protein_index]
   build.py: + remove curated 70S=20000; add curated 30S(CPLX0-3953)+50S(CPLX0-3962)
             + thread RNA unique_index onto each rnas dict
             + ribosomes[] dict list keyed by mRNA_index → recipe "ribosomes" block
        ▼  pbg-parsimony passthrough → recipe
parsimony place_translation (placer.rs):
   RnaStrand gains unique_index + length_nt (carried from RnaSpec)
   map {rna unique_index → RnaStrand}
   for each ribosome:
     strand = map[mRNA_index]; frac = pos_on_mRNA / strand.length_nt
     bead = strand bead at frac; pos = bead + outward*offset (sits ON the mRNA)
     place 70S (ribosome_marker ingredient) confined
     + (C2) grow a peptide coil from `pos`, contour ∝ peptide_length, confined  [peptide_segment]
```

### Identity plumbing (C1 prerequisite)
- `RnaSpec` gains `unique_index: i64` (default 0) + already has `length_nt`.
- `RnaStrand` gains `unique_index: i64` + `length_nt: i64` (carried from the spec)
  so the ribosome loop can address it.
- `build_model` threads each RNA's `unique_index` (from `rna_state`) onto its rnas
  dict; free + nascent both carry it.

### Recipe additions
- `RnaSpec.unique_index`.
- A new chromosome (or top-level) `ribosomes` block: list of
  `{mRNA_index: i64, pos_on_mRNA: i64, peptide_length: i64}`, plus
  `ribosome_marker: Option<String>` (the 70S ingredient) and (C2)
  `peptide_segment: Option<String>` + `peptide_angstrom_per_aa: f32` (default ~3.0).

## C1 — ribosomes on mRNA (+ corrected subunits)

1. **Capture:** add `ribo_mRNA_index`, `ribo_pos_on_mRNA`, `ribo_peptide_length`,
   `ribo_protein_index` (i8) from `active_ribosome`; add `ribosome_state` reader.
2. **Build:** remove curated 70S=20000; add curated 30S (`CPLX0-3953`, mesh) + 50S
   (`CPLX0-3962`, mesh), region interior, real counts; thread RNA `unique_index`
   onto rnas dicts; build `ribosomes` list `{mRNA_index, pos_on_mRNA, peptide_length}`.
3. **Recipe/Rust:** `RnaSpec`/`RnaStrand` `unique_index` + `length_nt`; recipe
   `ribosomes`/`ribosome_marker`; `place_translation` places each ribosome on its
   mRNA at `pos_on_mRNA` (outward offset, confined). 70S = 4YBB.

**Gates (C1):** ribosome count == active_ribosome count; each ribosome within
`offset+tolerance` of its mRNA strand at the `pos_on_mRNA` contour fraction; a
ribosome whose `mRNA_index` has no matching strand is dropped-with-log (not
crashed) — note the count caveat; confined; deterministic; 30S/50S present at real
counts; no `70S_ribosome` ingredient remains.

## C2 — nascent peptides

A peptide coil per ribosome, rooted at the ribosome, contour length =
`peptide_length × peptide_angstrom_per_aa` (default ~3.0 Å/aa, tunable), confined
via `generate_rna_strand`-style surface-pull, rendered as instanced
`peptide_segment` (distinct Translation shade). Reuses the strand/segment
machinery.

**Gates (C2):** peptide count == ribosome count; contour monotone in
`peptide_length`; rooted at its ribosome; confined; deterministic.

## Testing

- **Rust** (`cargo test -p parsimony-core --lib`): ribosome placed on its mRNA at
  the right contour fraction + outward offset; unmatched mRNA_index dropped (not
  panicked); peptide length-monotone + rooted; confinement; determinism.
  `cargo build --release -p parsimony-cli`.
- **Python**: capture round-trips the active_ribosome arrays; `select_ingredients`
  yields 30S/50S (real counts) and no `70S_ribosome`; build wires ribosomes.
- **Viewer**: after C1, ribosomes sit on the mRNAs (polysomes) + free 30S/50S in
  the cytoplasm; after C2, peptides trail from each ribosome. Clear
  `.parsimony/cache` + regenerate the bundle after Rust changes. C1 is the
  perf checkpoint (true-abundance ribosomes are the heaviest layer).

## Risks / notes

- **Perf:** active ribosomes (~10-15k) × 4YBB mesh + peptides is the heaviest
  layer; relies on instancing + LOD. C1 checkpoint validates; if it stalls, tune
  LOD/size-filter defaults (NOT count — true abundance required) and `log` nothing
  dropped.
- **Polysome overlap (scale floor):** a 70S is ~200 Å; ribosomes ~80 nt apart on
  an mRNA at `rna_angstrom_per_nt=2` map to ~160 Å → some overlap. Accepted
  (true `pos_on_mRNA`); `rna_angstrom_per_nt` stays tunable if spacing is wanted
  (raising it lengthens all mRNAs — a global trade).
- **Subunit meshes:** source PDB meshes for 30S (`CPLX0-3953`) and 50S
  (`CPLX0-3962`) — standard structures (e.g. 30S ≈ 2AVY/the 4YBB small subunit,
  50S ≈ the 4YBB large subunit); splitting 4YBB into its two subunits is an
  acceptable source. Document the chosen structures.
- **mRNA_index without a strand:** if a ribosome's `mRNA_index` doesn't match a
  rendered RnaStrand (e.g. its mRNA wasn't a unique RNA), drop it with a logged
  count rather than crash; this is a true-abundance caveat to surface, not hide.
- **Stale-branch/env:** continue on `feat/3d-transcription-translation` (worktree
  `v2e-3d-txn`); interpreter `/code/v2ecoli/.venv/bin/python`;
  `PARSIMONY_HOME=/code/parsimony`; pbg-parsimony editable.
