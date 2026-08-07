# raw/ — local source staging (gitignored)

Dump raw source files here for ingest/sorting. **Everything in this directory is
gitignored** (see `.gitignore`) — it never gets committed, so copyrighted PDFs and
bulky supplements stay local.

Drop here, in any structure:
- the paper PDF (e.g. `basan2015.pdf`)
- supplement spreadsheets / data tables (xlsx, csv)
- figure screen-grabs for digitization (e.g. the acetate-vs-growth panel)
- scratch extraction work

Then **sort → extract the clean, ingest-ready curve** into the tracked sibling:

```
../basan2015_acetate_vs_growth.csv      # columns: growth_rate_per_h, acetate_secretion_mmol_gDCW_h, sigma
```

Record the digitization provenance (which figure/table, units, any conversion) in
`../README.md` or `../../notes/`. Only the clean CSV + its provenance are committed;
the raw inputs stay here.
