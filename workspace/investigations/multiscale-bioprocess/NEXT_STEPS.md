# multiscale-bioprocess — continuation (resume after PR #69)

PR #69 landed the investigation on main: 7 studies, the 3-arm Beulig comparison
(WCM-FBA · Millard-kinetic · iML1515-genome-scale) with report cards, mbp-06's
15-gap synthesis, the Millard ATP fix, and the 7→5 composite consolidation. The
main result — a ~3-order batch-scale gap holding across seeds — is established.

> ⚠ **Updated 2026-08-20 — the Millard arm is deferred.** mbp-07 is now framed as
> the investigation's *extensibility demonstration* rather than a Millard study,
> and the Millard arm's five criteria are withdrawn from the acceptance set
> (three of them grade against Beulig; the other two are a growth-recovery band
> and a head-to-head tracking summary) (the drop-in itself stands: tFBA Metabolism is replaced by the
> kinetic ODE across the whole-cell model's full process set and coupled to the reactor). The
> Beulig comparison the investigation still **gates** on is therefore
> **two-armed** — WCM-FBA (mbp-05, all five axis groups) and iML1515 (mbp-07) —
> not three. **Items 4 and 5 below are consequently DEFERRED, not queued.**
> Reopening either means first settling whether a coupled reactor should drive
> the Millard model's external glucose (it currently must not, and `v2ecoli#550`
> — merged 2026-08-21 — enforces that with an explicit exclusion rather than
> leaving it to chance), and then re-calibrating and regenerating this study's
> figures and the `millard_vs_beulig` card.

This PR resumes the documented follow-ups (priority order):

1. **Both-daughters population runner** — the path past the single-lineage
   plateau to Beulig-scale biomass; the one item that would move the main
   result. (mbp-06 axis A, `batch-scale-accumulation-architectural`.)
2. **Full multi-gen production runs** — fix the `run_multigen_sqlite` dir bug
   (`.pbg/parquet-runs/default/history/` not created → WCM truncates at gen 2),
   then complete the mini multi-seed sweep for variance-band report cards.
3. **Ungraded comparison axes** — add sim concentration stores for byproducts +
   dissolved-O2 so those overlays grade (Beulig lacks acetate/DO ref columns —
   scope decision). (mbp-06 axis B.)
4. **O2-limited Millard kinetics** — ⚠ DEFERRED 2026-08-20 (see the note above).
   Close the reactor→Millard dissolved-O2 feedback (Millard currently treats O2
   as a fixed species).
5. **Millard model completeness** — ⚠ DEFERRED 2026-08-20 (see the note above).
   Add H2O/CO2 efflux to the SBML so the
   Millard-cell mass balance closes; investigate why Millard growth is too slow
   to reach division (gen 1 at 9k steps).
6. **pbg-bioreactordesign#2** — upstream O2-saturation temperature-sign fix
   (biases dissolved-O2 at 310 K).
7. **Housekeeping** — refresh stale study `phase` fields (mbp-03/04/05 lag the
   actual built/run state); flip the report-card acceptance criteria from
   in-progress to graded verdicts once studies are evaluated.

Full gap inventory with evidence + resolution routing: `mbp-06-gap-analysis`.
