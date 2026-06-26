# 3D structural model — replication-bubble fidelity (reference overlay)

**Date:** 2026-06-26
**Status:** Design approved, pending spec review
**Predecessors:** Phase A (RNAP placement), Phase B (nascent + free RNA). The
recent fix places every RNAP/RNA on the main genome contour by coordinate.
**Repos touched:** `parsimony` (overlay placement + bubble mapping), `pbg-parsimony`
(passthrough), `v2ecoli` (BF2 capture + classification), viewer.

## Summary

Render the chromosome's replication structure as the v2ecoli `_draw_chromosome`
reference does (`v2ecoli/visualizations/workflow.py`): every RNAP/RNA sits on the
main genome contour by coordinate (already implemented), **and** daughter-domain
RNAPs + their nascent RNA are overlaid a *second* time on the replication bubble
(the newly-synthesized daughter copy). In the multi-chromosome division state,
each RNAP/RNA is routed to *its own* chromosome so both chromosomes are
populated.

Delivered in two sub-phases with a viewer checkpoint between:
- **BF1 — birth bubble overlay** (single chromosome): daughter-domain RNAPs/RNA
  also appear on the sister/bubble strand.
- **BF2 — division multi-chromosome**: capture the chromosome-domain tree,
  classify each RNAP by chromosome, route per chromosome.

## Goals

- **Reference-faithful overlay:** daughter-domain RNAPs (and their nascent RNA)
  appear on BOTH the main contour (by coordinate) and the replication bubble
  (the second daughter copy) — matching `_draw_chromosome` ("Rim RNAPs: ALL of
  them, regardless of domain" + daughters plotted again on the bubble arc).
- **Correct bubble coordinate mapping:** a daughter overlaid on the bubble is
  positioned by a *bubble-relative* fraction, not the genome-relative one.
- **Multi-chromosome routing (BF2):** in the division state (`n_chromosomes ≥ 2`)
  every RNAP/RNA lands on its OWN chromosome's strands — no chromosome left bare.
- **True abundance preserved for the molecule count:** there are still N
  `active_RNAP` molecules; the bubble overlay is a deliberate *second rendering*
  of the daughter copies (the replicated region has two physical copies), exactly
  as the reference shows — documented as such, not an accidental double-count.
- **Confinement + determinism** unchanged (surface-pull, fixed-seed reproducible).

## Non-goals

- True single-copy theta topology (mother's middle replaced by two daughter
  strands, each RNAP placed exactly once). Explicitly rejected in favor of the
  reference overlay.
- Ribosomes / peptides (Phase C).
- Capturing or rendering more than the active replication round's daughters.

## Background: the reference

`_draw_chromosome` / `_draw_replication_bubbles` (workflow.py):
- **Rim:** every RNAP plotted at `_coord_to_angle(coord)` regardless of domain.
- **Bubble:** for each fork pair, the daughter domains = transitive descendants
  of the replicating parent domain (`_descendant_domains` over the
  `domain_children` tree); their RNAPs are plotted again at `_coord_to_angle` on
  the bubble arc (inset radius). The bubble arc spans fork→oriC→fork.

Our 3D theta builder already produces, per chromosome, a main strand + a sister
strand (the bubble) when `fork_fraction > 0`. BF overlays the daughters onto the
sister.

## BF1 — birth bubble overlay (single chromosome)

For the single replicating chromosome, daughter-domain entries (`domain_index !=
0`) are placed a second time on the sister strand:

1. **RNAP overlay:** after the existing main-contour placement, for each RNAP
   with `domain_index != 0`, also seat a copy on the sister strand at the
   bubble-relative position of its coordinate.
2. **RNA overlay:** the nascent RNA for a daughter RNAP also grows a strand
   rooted at that sister-strand position (so the second copy has its transcript
   too). Free mRNAs are unaffected.
3. **Bubble-relative mapping (new Rust helper):** the sister strand spans the
   bubble `[-fork_bp, +fork_bp]` with `fork_bp = fork_fraction × (GENOME_BP/2)`.
   A daughter at genomic coordinate `c` maps to sister fraction
   `frac = ((c + fork_bp) / (2·fork_bp)).clamp(0,1)` → bead index along the
   sister. (Genome-relative `strand_point` would mis-place it, since the sister
   covers only the bubble, not the whole genome.)

No new state capture — `domain_index` is already captured.

**Gates (BF1):** a daughter-domain RNAP appears on BOTH the main contour and the
sister (two placements); its bubble position matches the bubble-relative frac of
its coordinate; nascent RNA for the daughter has a strand on the sister too;
domain-0 RNAPs appear once (main only); all confined; deterministic.

## BF2 — division multi-chromosome routing

1. **Capture extension (v2ecoli):** add the `chromosome_domain` tree and
   `full_chromosome` roots. In the capture script (reusing the
   `render_chromosome_gif.py` `_domain_children` + `_descendant_domains` logic),
   precompute and save per-RNAP `rnap_chromosome_index` (i4) and
   `rnap_is_daughter` (bool); likewise per-RNA via its RNAP. `chromosome_index`
   is the index (0..n_chromosomes-1) of the `full_chromosome` whose domain
   lineage contains the entry's `domain_index`.
2. **Routing (build + Rust):** `RnapPlacement`/`RnaSpec` gain `chromosome_index`
   (i4, default 0) + `is_daughter` (bool, default false). `place_chromosome`
   routes each entry to chromosome `chromosome_index`'s MAIN strand (by
   coordinate), and — if `is_daughter` — also its SISTER strand (bubble-relative).
   The flat `strands` list is already grouped per chromosome (`[c0_main, c0_sister,
   c1_main, c1_sister, …]`); a helper maps `chromosome_index` → that chromosome's
   main + sister strand indices.

**Gates (BF2):** in a 2-chromosome recipe, RNAPs route to both chromosomes' main
strands (neither bare); a daughter on chromosome 1 overlays chromosome 1's sister
(not chromosome 0's); birth (single-chromosome) behavior unchanged; 1:1 molecule
count preserved (overlays are the documented second copy).

## Architecture

```
v2ecoli capture (BF2): + chromosome_domain tree, full_chromosome roots
   → per-RNAP rnap_chromosome_index (i4), rnap_is_daughter (bool)   (BF2)
        │  build.py: rnaps/rnas dicts gain chromosome_index + is_daughter
        ▼  pbg-parsimony passthrough → recipe
parsimony place_chromosome:
   per entry: strands[chrom_main] @ strand_point(coord)             (always)
              + if is_daughter: strands[chrom_sister] @ bubble_point(coord, fork)  (overlay)
   bubble_point(coord, fork_fraction) = bubble-relative frac → bead   [new helper, fiber.rs/placer.rs]
```

`is_daughter` is `domain_index != 0` in BF1 (single chromosome); in BF2 it's the
captured flag (descends from a replicating parent domain). `chromosome_index` is
0 in BF1, the captured value in BF2.

## Testing

- **Rust** (`cargo test -p parsimony-core --lib`): daughter RNAP on both strands;
  bubble-relative position; division per-chromosome routing; domain-0 single
  placement; confinement; determinism. `cargo build --release -p parsimony-cli`.
- **Python**: BF2 capture round-trips the tree-derived `chromosome_index` /
  `is_daughter`; classification matches `_descendant_domains` on a known tree.
- **Viewer**: after BF1, the birth bubble shows daughter RNAPs/RNA (second copy);
  after BF2, the division state shows both chromosomes populated. Clear
  `.parsimony/cache` + regenerate the bundle after Rust changes.

## Risks / notes

- **Documented over-render:** daughter molecules render twice (main + bubble).
  This is intentional (two physical copies) and matches the reference; note it in
  the viewer/ingredient text so it isn't read as a count bug. The 1:1 *molecule*
  count (active_RNAP) is unchanged; placement count for daughters is 2×.
- **Bubble mapping clamp:** coordinates outside `[-fork_bp, +fork_bp]` shouldn't
  reach the overlay (only daughters, which are inside the bubble), but the
  `clamp(0,1)` guards mis-tagged inputs.
- **BF2 capture dependency:** the `chromosome_domain` tree (`child_domains`) must
  be available on the live composite (it is, per `render_chromosome_gif.py`);
  the birth snapshot's single round makes BF1 testable without it.
- **Stale-branch/env:** continue on `feat/3d-transcription-translation`
  (worktree `v2e-3d-txn`); interpreter `/code/v2ecoli/.venv/bin/python`;
  `PARSIMONY_HOME=/code/parsimony`; pbg-parsimony editable.
