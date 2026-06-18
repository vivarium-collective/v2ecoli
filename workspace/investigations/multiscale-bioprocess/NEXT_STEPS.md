# multiscale-bioprocess — continuation (resume after PR #69)

PR #69 landed the investigation on main: 7 studies, the 3-arm Beulig comparison
(WCM-FBA · Millard-kinetic · iML1515-genome-scale) with report cards, mbp-06's
15-gap synthesis, the Millard ATP fix, and the 7→5 composite consolidation. The
headline result — a ~3-order batch-scale gap, robust across seeds — is established.

This PR resumes the work on the documented follow-ups (priority order):

1. **Both-daughters population runner** — the real path past the single-lineage
   plateau to Beulig-scale biomass. The one item that would move the headline
   result. (mbp-06 axis A, `batch-scale-accumulation-architectural`.)
2. **Full multi-gen production runs** — fix the `run_multigen_sqlite` dir bug
   (`.pbg/parquet-runs/default/history/` not created → WCM truncates at gen 2),
   then complete the mini multi-seed sweep for variance-band report cards.
3. **Ungraded comparison axes** — add sim concentration stores for byproducts +
   dissolved-O2 so those overlays grade (Beulig lacks acetate/DO ref columns —
   scope decision). (mbp-06 axis B.)
4. **O2-limited Millard kinetics** — close the reactor→Millard dissolved-O2
   feedback (Millard currently treats O2 as a fixed species).
5. **Millard model completeness** — add H2O/CO2 efflux to the SBML so the
   Millard-cell mass balance closes; investigate why Millard growth is too slow
   to reach division (gen 1 at 9k steps).
6. **pbg-bioreactordesign#2** — upstream O2-saturation temperature-sign fix
   (biases dissolved-O2 at 310 K).
7. **Housekeeping** — refresh stale study `phase` fields (mbp-03/04/05 lag the
   actual built/run state); flip the report-card acceptance criteria from
   in-progress to graded verdicts once studies are evaluated.

Full gap inventory with evidence + resolution routing: `mbp-06-gap-analysis`.
