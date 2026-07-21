# Design: genotype perturbations (KO / KD / OE of native genes) in v2ecoli

**Date:** 2026-07-21
**Author:** Chris (with Claude)
**Status:** Draft — for review by Eran, Riley
**Prior art:** RFC-007 Appendix A (strain modification in the WCM);
[`2026-06-06-ecoli-sources-bundle-integration-design.md`](./2026-06-06-ecoli-sources-bundle-integration-design.md)
(this document is the deferred "PR 2" spec from that design, extended to cover
the perturbation layer that sits on top of it).

## Goal

Bring the full suite of native-gene genotype perturbations — knockout,
knockdown, overexpression — into v2ecoli as first-class, declarable variants,
so that genotype panels can be run and graded the same way condition and
parameter sweeps already are.

The work divides along a seam worth naming up front:

- **Mechanical** (this spec) — the variant machinery, the multi-ParCa runner it
  needs, and the tests that establish the implementation does what it says.
- **Scientific** (separate, see
  `workspace/investigations/genotype-perturbation-response/`) — whether the
  model's *response* to those perturbations matches measured biology.

These are separated deliberately. The mechanical work can be reviewed and
merged on its own terms; the scientific work is a research question that only
becomes answerable once the machinery exists. Conflating them has historically
meant the second half never gets scheduled.

## Background: what exists, and where

RFC-007 Appendix A decomposes strain modification into four variant sets. Their
current state:

| Set | What | State | Where |
|---|---|---|---|
| 1 | New-gene insertion, variable expression per inserted TU | Built | vEcoli-private PR #18 |
| 2 | **Translation-level** KO / KD / OE of native genes | Built | public `CovertLab/vEcoli` branch `strain-modification-variants` |
| 3 | **ParCa-level** KO (chromosome deletion) | Built, not stress-tested | vEcoli-private (`knowledge_base_raw.py`) |
| 3 | **ParCa-level** KD | Not implemented | — |
| 4 | Transcription-level OE of native genes | Not implemented | — |

Two facts about this table drive the design.

**Set 2 is already public and already generic.** The
`strain-modification-variants` branch carries
`ecoli/variants/native_translation_perturbation.py` (a per-monomer multiplier on
`translation_efficiencies_by_monomer`: 0 = knockout, 0–1 = knockdown, >1 =
overexpression), `ecoli/variants/strain_design.py` (composes that with the
condition and new-gene-shift variants), and roughly 3,800 lines of supporting
analysis. A diff against public master shows no pathway-specific or otherwise
non-public content — it is native-gene machinery throughout. It can be ported
into public v2ecoli directly.

**Set 3's chromosome surgery is generic too, and can be upstreamed.** The
deletion code is currently private only, but it is genome-surgery mechanism
rather than payload. Confirmed with Riley (2026-07-21) that it can be
upstreamed to public v2ecoli. That keeps the whole suite in one repo, which is
the difference between one coherent variant surface and two half-suites in
different places.

## Current v2ecoli state

What is already in place:

- **ecoli-sources bundle resolution.** `SourceBundle`
  (`v2ecoli/processes/parca/reconstruction/ecoli/sources.py`) resolves ~135
  canonical keys to flat files, with a `parca_overrides.tsv` layer on top for
  v2ecoli's divergent biology. `v2ecoli-parca --bundle-manifest-path` already
  accepts a wholly alternate bundle (`v2ecoli/cli/parca.py:65-68`, `:95-101`).
- **A variant grammar.** `v2ecoli/workflow/variants.py` expands
  `value` / `linspace` blocks under `op: prod|zip|add` into a variant × seed
  branch grid.
- **Two working sim_data-level perturbation precedents.**
  `sim_data.genetic_perturbations` (a fixed transcription-initiation
  probability per TU, consumed at `v2ecoli/processes/transcript_initiation.py:305-320`
  and already used as a variant in `showcase-4-variant-comparison`), and
  `scripts/build_condition_cache.py`, which hydrates the ParCa fixture, applies
  a named patch, and re-caches with a provenance manifest.
- **Cheap ParCa.** A full 51-condition ParCa is ~2.4 min. A 20-genotype panel
  is under an hour of ParCa — this is a plumbing problem, not a compute problem.

What is missing:

- **Multi-ParCa.** The 2026-06-06 design split into PR 1 (bundle integration,
  landed) and PR 2 (multi-ParCa runner, sketched, deferred). PR 2 has not
  landed. Every branch in `workflow/meta_composite.py:33` receives the same
  top-level `cache_dir`, and variant overrides patch process-config dicts only —
  they never reach sim_data. `scripts/sweep_report.py:232` states the
  consequence directly: *"Deferred: nested grammar and sim_data-recomputing
  variants such as gene knockouts."*
- **A named variant registry.** There is exactly one variant mechanism
  (`<process>.<config-key>` override); `nested` raises `NotImplementedError`
  (`variants.py:56`).
- **A per-variant `cache_dir` in the runtime grammar.** `study.yaml` already
  expresses one (`showcase-4-variant-comparison`), but `meta_composite.py`
  cannot consume it, so those runs were done out of band.

## Design

### Tranche A — translation-level suite (variant set 2)

Port `native_translation_perturbation` and `strain_design` onto v2ecoli's
variant grammar. Requires no multi-ParCa: translation-efficiency edits are a
sim_data patch, and `build_condition_cache.py` already demonstrates
hydrate → patch → re-cache with a manifest. The work is generalizing that
script's single-entry `PATCHES` registry into a named-variant registry that the
workflow can address declaratively.

Delivers KD, OE, and translational KO on real rails, independent of everything
below.

### Tranche B — multi-ParCa runner

Implement the PR-2 sketch from the 2026-06-06 design as written:

- Top-level `parca_variants: []`; each entry overrides `parca_options`, with
  `bundle_manifest_path` as the natural v2ecoli override — **each genotype is a
  different ecoli-sources bundle**.
- Run ParCa once per variant into its own cache bundle.
- Two-level index (`global = parca_idx * pickles_per_parca + variant_idx`) so
  caches and outputs never collide; wire into the existing
  `meta_composite.py` variant × seed branches.
- Merge per-ParCa metadata for the existing comparison/analysis layer.
- Isolate a failed ParCa variant; the rest proceed.

This also closes the per-variant `cache_dir` gap that `study.yaml` already
assumes, and unblocks the sim_data-level injection requested by
`workspace/investigations/parameter-uq/investigation.yaml:42-44`.

### Tranche C — ParCa-level KO and KD (variant set 3)

With B in place, a genotype becomes a generated bundle. Deletion logic moves
from the consumer to the supplier: rather than a `gene_deletions` parca-option
that mutates KB tables in-process, an `ecoli-sources` generator materializes the
perturbed flat files and emits a complete manifest. The manifest hash is then
the genotype identity — reproducible, campaign-pinnable, and cacheable, so a
genotype seen before need not re-run ParCa.

Two paths, one per perturbation type, chosen to match the biology:

- **KO — chromosome deletion.** Multi-key coupled transform (sequence, genes,
  transcription_units, DNA sites). Deleting actual sequence matters: chromosome
  replication timing is computed from total chromosome length. It also matches
  how the deletions are made experimentally (homology arms).
- **KD — expression scaling.** Single-key swap on the expression tables. There
  is no natural "partial chromosome," and this composes trivially with the
  existing expression-data generators.

Retire `gene_deletions` as a parca-option once the generator path is canonical;
keeping both re-introduces the duplication this is meant to remove.

## Normalization and ordering semantics

RFC-007 Appendix A's technical note on RNAP/ribosome allocation is a design
constraint, not just a caveat. Both transcription and translation initiation are
competitive allocation problems over a finite pool, and the two levels behave
differently:

- **Translation-level edits are order-independent.**
  `translation_efficiencies_by_monomer` is a vector of rate-like weights, not a
  probability distribution; nothing in sim_data renormalizes it. Final state
  depends only on the last write per monomer. Competition is enforced at runtime
  as `normalize(cistron_counts × translation_efficiencies)`, so ordering is not
  a correctness concern — but the realized effect is weighted by mRNA abundance,
  and a translation-level OE of a gene whose mRNA was not also raised captures
  little ribosome capacity.
- **Transcription- and ParCa-level edits are order-dependent.** The probability
  vector must sum to 1, so perturbing one gene reweights every other. Knocking
  down A 10× then overexpressing B 10× does not equal the reverse: B may end up
  more than 10× up because renormalization redistributes the capacity freed by
  A. Further, `basal_prob` and `delta_prob` feed the runtime
  `synth_prob_from_ppgpp` reconstruction, so they must be edited in lockstep
  with `rna_synth_prob` or the regulation logic reintroduces the un-edited
  values at simulation time.

**Design consequences.** Tranche C's KD must define a canonical application
order and assert it, rather than leaving it to dict iteration order. Tranche A
inherits order-independence for free and should have a regression test pinning
it (the upstream branch already carries
`test_strain_design_order_independence_of_native_perturbations`). And the two
knockout kinds are not interchangeable: a translation-level KO still transcribes
mRNA and consumes RNAP capacity, where a ParCa-level KO removes the gene
entirely. Both are legitimate; the card and the configs should name which is in
use.

## Coordinate handling — defects found on review

Riley flagged transcription-unit coordinate edge cases as the key trap in the
ParCa-level KO path, and noted the implementation was never stress-tested.
Reviewing `_update_global_coordinates_for_deletion`, three defects reproduce by
calling the function directly. All three are in branches the existing fixtures
in `test_knowledge_base_operations.py` do not exercise — that suite uses only
well-separated features, so every case below falls outside it.

1. **A feature overlapping the deletion's left edge is silently skipped.** The
   "before deletion" guard tests `right < del_right_pos` where it needs
   `right < del_left_pos`. A feature spanning 450–550 against a deletion of
   500–600 returns unchanged at 450–550; it should be truncated to 450–499. No
   error is raised, so the coordinates are quietly wrong.
2. **The containment branch raises `UnboundLocalError`.** After `data.remove(row)`
   it falls through to `row.update({"left_end_pos": updated_left})` without
   `continue`, and `updated_left` was never assigned on that path. It crashes on
   the first fully-contained feature — or, worse, silently writes the *previous*
   iteration's coordinates if `updated_left` happens to be bound from an earlier
   row.
3. **`data.remove(row)` mutates the list being iterated,** so the element
   following a removed one is skipped. Two adjacent features inside one deletion
   leave the second in place.

The code's own fallback branch prints *"this is a deletion case that has not
been considered,"* which is consistent with the case analysis being known-incomplete.

None of this contradicts the deletions performed to date: for well-separated
genes the clean before/after paths are correct, which is why the existing tests
pass. The defects bite on nested and overlapping features — common enough in
E. coli that a genome-wide panel would hit them. Fixing the case analysis and
extending the fixtures is a precondition for Tranche C, not a follow-up.

## Test obligations

Porting the existing tests is the cheapest quality win available here: PR #18's
`test_knowledge_base_operations.py` is 521 lines covering multiple deletions,
deletion-then-insertion, sequence content, and a single-base-genome edge case.
Those port directly.

On top of that:

- **Coordinate case matrix** — every relative position of a feature against a
  deletion (before, after, contained, overlapping each edge, spanning), for
  genes, TUs, and DNA sites, plus adjacent-contained features. Covers the three
  defects above and the incomplete case analysis.
- **Chromosome length and replication timing** — assert deleted sequence
  changes total length and that replication-initiation timing responds.
- **Order independence (translation) and canonical order (ParCa)** — per the
  section above.
- **Composition** — multiple KO + KD + OE together, and combined with variable
  new-gene expression, per RFC-007's validation section.
- **Amino-acid parameter-fitting interaction** — see below.

## Known risk: amino-acid supply parameter fitting

RFC-007 carries "ParCa level changes will interact with the mechanistic amino
acid supply parameter fitting" as its own heading, and Riley identified this as
the specific un-stress-tested area: knockouts of trp-related genes may change
how kcats are recalculated for mechanistic AA production, in the same failure
family as the alternate-RNAseq issues.

This is the highest-uncertainty part of the work and it is not addressable by
unit tests — it needs a run. It is therefore scoped into the scientific
investigation rather than this spec, as an explicitly anticipated failure mode
with a dedicated study. Flagging it here so the mechanical work does not claim
more confidence than it has: passing the test matrix above establishes that
coordinates and expression vectors are edited correctly, not that the resulting
sim_data is physiologically sound.

## Sequencing

A → B → C. Tranche A is independently deliverable and gives a working KD/OE/KO
surface with no new infrastructure. B is the enabling rail for C and is already
designed. C carries the coordinate work and the AA-fitting risk.

The scientific investigation is scaffolded now but runs after C, since the
essentiality panel and the AA-pathway stress test both need ParCa-level KO.

## Out of scope

- Variant set 4 (transcription-level OE of native genes). RFC-007 leaves open
  whether it gains much over the translation-level implementation; that question
  is better answered with A in hand.
- Pathway- or product-specific configs and data. This spec covers native-gene
  machinery only.
- Grading criteria for the genotype report card — see the investigation.
