# Responses to Maya Abdalla's review — flagella-cascade (2026-07-24)

Point-by-point response to all 27 inline annotations
(`feedback/2026-07-24T15-27-27Z.yaml`). Each item lists Maya's comment → the action
taken. "study.yaml" edits change what the investigation report renders.

## References / overall

1. **"Like the appendices + testing. Design: more detail / bigger font to
   differentiate the studies."**
   → Logged as `open_decisions: report-study-differentiation` (a report-template
   change for the vivarium-workbench maintainers). Interim fix: each study title +
   description now leads with its Aim (2A/2B) and a one-line scope so the sections
   read as distinct. Added a `glossary` + `testing_standard` to the investigation
   so reviewers have shared definitions.

## Study 01 — overexpression baseline

2. **"What is features=[]?"** → Defined in the investigation `glossary` and in the
   study `notes`: `[]` = regulation OFF (stock WCM); `['flagella_regulation']` =
   SUM-gate ON.
3. **"What does stale mean — outdated?"** → Yes. Added to the `glossary`; chart notes
   that say "stale" now mean "outdated cache/run, superseded."
4. **"Baseline with no edits — why a regulation-ON plot? Were two runs done (OFF and
   ON)?"** → Exactly right, two runs. Chart-01 description rewritten to say so
   up-front: the study both characterizes the OFF baseline AND shows ON for
   reference. Same clarification added to chart 02.
5. **"If this is a true overexpression baseline I expected no regulation-on; if we
   keep it, define regulation on."** → "Regulation ON/OFF" now defined in the
   glossary and inline in the chart-02 caption.
6. **"on/off fraction in the title means what — 0%, 25%, 0%?"** → Real label bug: the
   chart-04 title said "ON below OFF at every generation" while the data show the
   opposite. Fixed in `run_studies_ensemble.py` — the title now reads "regulation ON
   sits ABOVE OFF" and the numbers are labelled "seeds where ON < OFF: gen1=0%,
   gen2=25%, gen3=0% → ON ≥ OFF for the rest."
7. **"Don't understand: is the port byte-neutral when the feature is off?"** →
   Plain-language definition added to the study `purpose` and the glossary:
   feature-OFF output is bit-for-bit identical to pre-port main.
8. **"Is the claims editable? Are tests standard or unique to each study?"** →
   Answered in the study `notes`: yes, every test/claim is hand-authored YAML you
   can edit; tests are a mix of STANDARD categories (see `testing_standard`) and
   study-unique claims.

## Study 02 — SUM-gate cascade

9. **"Study 1 had 4 tests, study 2 has 2 — make a 'standards' testing criteria."** →
   Added `testing_standard` to the investigation (categories: model-runs,
   model-mechanism, biology-correct, robustness) and `success_criteria: {model,
   biology}` to every study so acceptance is comparable. (Tests stay hand-authored —
   no auto-generated filler.)
10. **"In vEcoli my config script was named flagella_regulation — maybe that's where
    the feature name came from?"** → Confirmed yes. Provenance note added to the
    study `notes` and glossary.
11. **"It says not run but the visualizations show plots — how did it not run?"** →
    The figures always came from real runs; the status field lagged. The study now
    has a completed run persisted to the Simulations DB
    (`flagella-02-sumgate-cascade__regON__seed0__600s`), and `simulation_status` is
    `ran` with a note explaining the earlier contradiction.
12. **"How many generations? All sims 600 s — is that enough?"** → Chart-01 caption
    now states: single generation, 600 s < one doubling (~40 min), so it shows the
    within-generation ordering only; the cross-generation view is the multigen +
    seed-band figures.
13. **"Is this 2 or 3 generations? … 2, based on the shading."** → Correct — caption
    confirms 2 generations.
14. **"Class II higher than Class III is exactly right, nice separation."** → Noted,
    thank you; recorded as the study's biology-success criterion.
15. **"Bursts at ~14 and ~60 min — division? SUM-gate calc issue? stochastic
    artifact?"** → Most likely a division re-initialization transient (they coincide
    with the dashed division lines; pools halve and the normalized gate re-settles
    over a few ticks). Written into the chart-02 caption and added as an explicit
    open test `bursts-are-division-transients` (status: untested) to confirm they
    track division timing exactly.
16. **"Phase portrait: x-axis not time, seems arbitrary."** → The axes are the two
    normalized 0–1 activities the SUM-gate integrates (not time — time is the
    color). Plot updated to pin both axes to [0,1] with clearer labels; caption
    rewritten to say so.
17. **"Like this question/mechanism/expected-outcome format (clearer than the
    overview)."** → Kept that format; added a plainer glossary + testing_standard so
    the investigation level reads as clearly as the study level.

## Study 03 — FlgM / FliA feedback

18. **"Says no variants run, but there's the feature — contradicts."** → Clarified:
    the flagella_regulation feature is part of this study's BASELINE, not a variant;
    "no variants" just means no perturbation sweep yet. Note added; candidate
    variants listed under `success_criteria.biology`.
19. **"My overrides are {CPLX0-7452:4, MOTOR:0, FliA:500, FlgM:800}. I don't
    understand the FlgM jump 1500→200 unless it's running without my overrides; it's
    all in the first 100 s then flat; and 31.25 flagella on the right axis?"** →
    Chart-01 uses the DEFAULT already-flagellated cache (NOT your overrides — those
    drive the separate low-IC figure). ~30 flagella at t=0 means FlgM export runs at
    full rate immediately, depleting FlgM in ~100 s, then flat — exactly the "jump
    then flat." Caption rewritten to make the initial condition explicit. The
    "31.25 flagella" was a matplotlib autolocator artifact; the right axis is now
    forced to integer ticks in `run_studies.py`.
20. **"Left plot looks like the one above but over more time. What does
    02_lowIC_gated_cascade mean?"** → "lowIC" = LOW Initial Condition; this IS the
    run with your overrides. Caption rewritten to contrast it with the default-cache
    figure (default = steady-state magnitudes; low-IC = the transient onset).
21. **"Technical context box is empty — a list of what was run would help."** →
    Filled in the chart-02 caption: composite, features, seed, cache, generations,
    lineage, division threshold.

## Study 04 — NFsim assembly

22. **"I have not worked on this fully yet."** → Acknowledged in the study `notes`;
    this is the least-mature study (standalone example; WCM-coupling unstarted) and
    reviews should read it that way.
23. **"Should we make NFsim its own investigation?"** → Logged as
    `open_decisions: nfsim-own-investigation`. Recommendation: keep it here as
    Aim 2B while standalone; split once coupled into the WCM. Maintainers' call.
24. **"Right-plot scale seems unfair — many FlgE, can't see complete flagella."** →
    Fixed: the monomer-vs-flagella subplot in `run_nfsim_assembly.py` now uses a
    symlog y-axis so the handful of complete flagella stay visible against the large
    free-monomer pools.
25. **"NFsim run on what baseline WCM — with regulation or without?"** → Neither —
    it's a STANDALONE NFsim + monomer-production composite, not coupled to the WCM,
    so regulation on/off doesn't apply. Caption now states this explicitly.

## Cross-cutting actions

- Added `success_criteria: {model, biology}` to all four studies (your two-tier idea).
- Added investigation-level `glossary`, `testing_standard`, `open_decisions`,
  `review_log`.
- Fixed four chart-label/scale issues at the source (ensemble title, FlgM integer
  axis, phase-portrait axes, NFsim symlog) and regenerated the affected figures.
- Persisted the three WCM studies as tracked runs in the Simulations DB.
