# 3D structural model — Phase B: nascent RNA strands

**Date:** 2026-06-25
**Status:** Design approved, pending spec review
**Predecessor:** Phase A (precise RNAP placement + markers + confinement) — done. See `2026-06-25-3d-transcription-translation-state-design.md`.
**Repos touched:** `v2ecoli` (state capture + build), `pbg-parsimony` (recipe API), `parsimony` (Rust geometry), viewer.

## Summary

Render the cell's RNA exactly as v2ecoli tracks it: every transcript as an
**extended thin strand** whose contour length is proportional to its
`transcript_length`. Nascent (still-being-transcribed) RNAs are **rooted at and
emanate from their RNA polymerase** on the chromosome; fully-transcribed free
mRNAs snake from a point in the cytoplasm. This is the layer the user asked for —
"the RNA that is coming from the RNAP as it is transcribed, connected to the
chromosome."

Delivered in two sub-phases with a viewer checkpoint between:
- **B1 — nascent strands** rooted on RNAPs (this spec's primary focus).
- **B2 — free cytoplasmic mRNAs** (`RNAP_index == -1`).

## Goals

- **Capture the RNA state** by extending the snapshot: per-RNA `unique_index`,
  `RNAP_index`, `transcript_length`, `is_mRNA`, `is_full_transcript`, `TU_index`;
  and add `unique_index` to the captured `active_RNAP` arrays so RNAs can find
  their RNAP.
- **Connectivity (exact):** a nascent transcript is rooted at the 3D position of
  its RNAP (`RNA.RNAP_index → active_RNAP.unique_index → that RNAP's coordinate →
  strand_point`), so the strand visibly emerges from the polymerase on the DNA.
- **Extended-strand geometry:** each transcript is a thin self-avoiding fiber
  whose contour length = `transcript_length × Å_PER_NT` (linear, so relative
  lengths are preserved), confined inside the cell envelope.
- **True abundance (1:1):** one strand per nascent transcript (B1) and per free
  mRNA (B2); no subsampling.
- **Confinement:** strands stay inside the envelope using the surface-pull
  confinement adopted for the chromosome — never the medial-collapse that caused
  the centerline sheaf.

## Non-goals

- Ribosomes and nascent peptides on mRNAs — that is Phase C.
- tRNA / rRNA unique molecules — out of scope (short/structural; not the
  transcription story). Only the `RNA` unique molecule (mRNAs + partial
  transcripts) is rendered.
- Nucleoid-avoidance / obstacle-aware routing of the strand — a strand may
  visually overlap the nucleoid in v1; confinement is to the cell envelope only.
- Biophysically exact RNA folding/secondary structure. The strand is a stylized
  confined walk; only its *length* is faithful (scaled by a tunable constant).

## Source data: the `RNA` unique molecule

From `internal_state.py` (`_build_unique_molecules`), accessed on the live
composite as `cell["unique"]["RNA"]`, masked by `_entryState` (same pattern as
`active_RNAP` in `bridge.py` / Phase A's capture):

| Field | Type | Use |
|---|---|---|
| `unique_index` | i8 | identity; ribosomes link here in Phase C |
| `RNAP_index` | i8 | links to `active_RNAP.unique_index`; `-1` = free mRNA |
| `transcript_length` | i8 | current length in nt → strand contour length |
| `is_mRNA` | ? | mRNA vs other RNA (shade / Phase C eligibility) |
| `is_full_transcript` | ? | released (free) vs still attached |
| `TU_index` | i8 | transcription unit identity (carried, for future labelling) |

`active_RNAP` also exposes `unique_index` (i8) — Phase A captured `coordinates`,
`domain_index`, `is_forward` but **not** `unique_index`; B1 adds it.

A nascent transcript = `RNAP_index >= 0`. A free mRNA = `RNAP_index == -1` (B2).

## Architecture

Rust-native geometry, matching Phase A. Python captures + classifies and hands
the engine explicit RNA specs; the Rust engine grows each strand from its root,
confined, and renders it as instanced segments.

```
v2ecoli unique state ──capture──> snapshot npz
  active_RNAP: + rnap_unique_index
  RNA[]: rna_unique_index, rna_RNAP_index, rna_transcript_length, rna_is_mRNA, rna_is_full_transcript, rna_TU_index
        │
        ▼  build.py: rnap_uid→(coordinate,domain) map; for each nascent RNA, root = its RNAP's (coordinate,domain)
recipe "rnas" block: [{root_coordinate, root_domain, length_nt, is_mRNA}]   (B1: nascent only)
        │  pbg_parsimony.Chromosome.rnas
        ▼
parsimony-core (Rust):
  generate_rna_strand(root_point, contour_len, shape, rng)   [fiber.rs]
    = a thin self-avoiding fiber seeded at root_point, confined (surface-pull)
  place_chromosome / rna stage: for each rna spec,
    root = strand_point(strands, root_domain, root_coordinate, GENOME_BP)   (B1)
    emit instanced `rna_segment` placements along the strand
```

### `Å_PER_NT` length scale

Contour length = `transcript_length × Å_PER_NT`, linear so relative lengths are
preserved. `Å_PER_NT` is a tunable constant (recipe/build parameter) defaulted so
a typical mRNA reads as a short snake rather than filling the cell (physical
extended ssRNA ≈ 5.9 Å/nt; the default starts lower, e.g. ~2–3 Å/nt, and is
tuned against the viewer). Bead count per strand = `contour_len / rna_step`.

### Rendering

A distinct **RNA** category and color, separate from Nucleoid. Strands render as
instanced thin segments (the proven dna_segment-style path, which already scales
to ~149k DNA segments). mRNA may use a distinct shade from non-mRNA RNA.

## B1 — nascent strands (primary)

1. **Capture:** extend the capture script + `rnap_state` to also save
   `rnap_unique_index`; add `rna_state(state_source)` returning the RNA arrays.
2. **Build:** in `build_model`, build `{rnap_unique_index: (coordinate, domain)}`;
   for each RNA with `RNAP_index >= 0` present in that map, emit a recipe rna spec
   rooted at that RNAP's `(coordinate, domain)` with `length_nt = transcript_length`.
3. **Recipe + API:** `Chromosome.rnas` (list of `{root_coordinate, root_domain,
   length_nt, is_mRNA}`) → recipe `rnas` block → Rust `RnaSpec`.
4. **Rust:** `generate_rna_strand` (confined self-avoiding fiber from a root) +
   an rna stage in `place_chromosome` that roots each strand via `strand_point`
   on the RNAP's coordinate and emits `rna_segment` placements.

**Gates (B1):**
- each nascent strand's first bead is within `bead_radius` of its RNAP's 3D
  position (rooted at the polymerase);
- strand contour length is monotone in `transcript_length`;
- every strand bead is inside the envelope (zero protrusion);
- deterministic for a fixed seed;
- rendered nascent-strand count == number of nascent RNAs (1:1).

## B2 — free mRNAs (after B1 review)

Free mRNAs (`RNAP_index == -1`, `is_full_transcript`) get a recipe rna spec with
no root; the Rust rna stage seeds them at a confined random interior point and
grows the same confined strand. Same gates minus the rooted-at-RNAP one;
count == number of free mRNAs.

## Testing

- **Rust** (`cargo test -p parsimony-core --lib`): rooted-at-root, length
  monotonicity, confinement (no protrusion, no centerline-collapse), determinism.
  Build `cargo build --release -p parsimony-cli`.
- **Python:** capture round-trips the RNA arrays; nascent classification
  (`RNAP_index >= 0` and present in the RNAP map) is correct; rendered counts match.
- **Viewer:** new RNA category reads correctly; nascent strands visibly emanate
  from RNAPs; clear `out/.parsimony/cache` (recipe-keyed) + regenerate the mesh
  bundle after a Rust change.

## Risks / notes

- **Scale:** thousands of extended strands × tens of beads each is comparable to
  the DNA segment count; instancing handles it. If visually cluttered, tune
  `Å_PER_NT` down (preserve relative lengths) — do NOT drop molecules (true
  abundance is required). `log()` the strand + total-segment count at build.
- **Confinement:** reuse the chromosome's surface-pull (never `inset.medial`
  collapse) so long strands hug the wall rather than collapsing to the axis.
- **RNAP without a matching captured unique_index:** if an RNA's `RNAP_index`
  isn't in the captured RNAP map (shouldn't happen for a consistent snapshot),
  the RNA is treated as free (B2) rather than dropped — preserves 1:1.
- **Stale-branch / env:** continue on `feat/3d-transcription-translation`
  (worktree `v2e-3d-txn`); interpreter `/code/v2ecoli/.venv/bin/python` run from
  the worktree; `PARSIMONY_HOME=/code/parsimony`; pbg-parsimony editable.
