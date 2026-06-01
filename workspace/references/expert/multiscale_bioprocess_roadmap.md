# Multiscale bioprocess — upstream roadmap reference

The `multiscale-bioprocess` investigation in this workspace is INSPIRED BY
(but diverges from) the roadmap at:

  https://github.com/vivarium-collective/multiscale-bioprocess
  branch: chris/initial-scaffolding
  path:   docs/roadmap.md

**Divergence point.** The upstream roadmap couples a PBG-native well-mixed
reactor to a **dynamic-FBA (iML1515)** cell side as Phases 1–5, with
v2ecoli / v2coli substitution deferred to a "longer-horizon" stage. The
v2ecoli-workspace investigation skips the dFBA detour: it puts the v2ecoli
whole-cell composite directly into the reactor under the same substitution
contract. Phases (studies) mbp-01 (reactor scaffold + placeholder Monod),
mbp-02 (kLa gas-liquid transport), and the fed-batch / Palsson phases stay
analogous; mbp-03 replaces Monod with v2ecoli.

**What lives upstream that this investigation does NOT duplicate.**

- The biological-BDD methodology framing (`docs/concepts/biological-bdd.md`,
  `docs/concepts/phase-based-development.md`) is the operating discipline
  for the upstream repo's phase artifacts. The v2ecoli workspace runs the
  same study-discipline through `studies/<slug>/study.yaml` and pipeline
  gates — same shape, different framing terminology.
- The Palsson 2025 ingestion (raw PDF + structured extraction) is upstream
  scope (the "palsson-ingestion plan"). When that work lands, the artefacts
  should be mirrored or referenced from `references/papers/palsson-2025/`
  and `references/expert/palsson_2025_fed_batch.md` here.
- Prior-art reactor physics — kLa correlations, Higbie penetration,
  Henry's law, Wilke–Chang diffusivity — are lifted (with attribution)
  from `pbg-bioreactordesign` and BiRD (NREL OpenFOAM toolbox).

**Cross-link.** This document is the v2ecoli-workspace pointer; the real
plan content lives upstream. Re-read upstream when restructuring the
investigation; do not let this file drift into a paraphrase.

## Maintainers

- Chris Long (Cell Systems Logic / UConn) — bioprocess expert
- Eran Agmon (UConn) — PBG + WCM expert

## Status

Open IP class. SMS / DARPA project. Upstream repo is private as of
2026-05-22.
