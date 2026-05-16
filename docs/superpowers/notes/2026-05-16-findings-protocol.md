# 2026-05-16 — Findings as first-class study output

Notes for the pbg-superpowers listening session about adding a
**findings protocol** to the /pbg-study lifecycle.

## The gap

Studies currently surface OUTCOMES (`runs[].outcomes` per behavior_test:
PASS/FAIL plus a value). Outcomes tell you *what happened in this run*
but not *what we learned*. A biology expert evaluating the HTML report
needs to see, for each study:

  - what biological claim does this evidence support / contradict / extend?
  - which expert reference (paper, expert PDF, prior knowledge) is the
    comparator?
  - is this a real biological mismatch with v2ecoli, or a wiring /
    calibration / tooling issue?
  - what should the next person actually do about it?

Today this lives in `conclusion:` as freeform prose. Hard to scan,
hard to cross-reference between studies, easy to lose.

## Proposal — `findings:` block on study.yaml

A new top-level field. Each entry is a structured finding:

```yaml
findings:
- id: F-NN
  kind: biological | computational | methodological
  status: confirms | partial | contradicts | novel
  statement: |
    One-paragraph English claim. The headline a reader scans.
  evidence:
    from_run: <runs[].simulation>          # which sim produced it
    from_test: <behavior_tests[].name>     # which test confirmed it
    observed: <value>                      # the measurement
    units: <string>
    window: <string>                       # optional reduction context
    reduction: <string>
    smoking_gun: |                         # optional — drop-down detail
      Specific log/trace/diff that made the cause obvious.
    discovered_during: <study | run name>  # if uncovered as a side-effect
  expected:
    cites: [bib_keys, ...]                 # papers.bib references
    range: [low, high]                     # OR threshold: <num>
    summary: |                             # what reference says, in plain English
      One paragraph quoting the literature claim being compared against.
  expert_reference:                        # optional — links to expert PDFs
    doc: <expert_doc_id>                   # from workspace.yaml.expert_docs
    section: "§N.N"
    quote: |
      Verbatim excerpt from the PDF, indented in the report.
    note: |
      Curator's gloss when no clean quote is available.
  explanation: |
    Why we believe the finding — mechanism, sources of error, alternatives ruled out.
  next_action: |
    Concrete actionable line (often "Seed follow-up X" or "Re-run with Y").
```

## Required adds to pbg-superpowers

### A. `/pbg-study findings` subcommand

After a run + evaluate, walk `runs[].outcomes` and propose draft
findings. Each PASS/FAIL outcome becomes a candidate finding:
  - PASS → "confirms" status, statement = "v2ecoli reproduces X within tolerance"
  - FAIL → ask the user: biological (contradicts) vs computational (novel)
  - If FAIL and a prior session has already pinpointed root cause, draft
    the explanation pointing at the prior finding's id.

The subcommand opens an interactive walk:
  1. Show the outcome row.
  2. Suggest kind + status from a heuristic.
  3. Ask the user (or LLM agent) to fill statement / expected / next_action.
  4. Search references/notes/*.md and references/papers.bib for
     keyword-matching cites; offer them.
  5. Search references/expert/*.pdf for keyword-matching quotes; offer
     verbatim excerpts. (Use pypdf — confirmed working in v2ecoli.)
  6. Write to study.yaml.findings[].

### B. Expert-PDF search helper

A library function `search_expert_docs(workspace, terms: list[str]) ->
list[dict]` that returns hits as `{doc, page, snippet}`. Used by the
findings subcommand to surface quote candidates.

Implementation hint: `pypdf.PdfReader(path).pages[i].extract_text()`,
keep an in-memory page cache to avoid re-parsing per term.

### C. Findings linter

`scripts/lint-workspace.py` extension: any study with `phase: Decide`
should have at least one finding. Studies with `status: ran` should have
at least one finding tied to each `behavior_tests[]` outcome.

### D. Cross-study findings report

`/pbg-report` extension: a workspace-level "Findings index" page
aggregating findings across all studies, grouped by kind, sortable by
status. Lets a reader skim "all biological contradictions across the
workspace" in one view.

### E. Findings-aware follow-up seeding

The existing seed-from-followup flow can take a `finding_id` parameter
and pre-populate the new study's `purpose.question` + `purpose.mechanism`
from the finding's `explanation` + `next_action`. Keeps the lineage
finding→follow-up→child-study legible.

## Seed example — dnaa-01

The dashboard side of this proposal is implemented today (commit
[pending] on vivarium-dashboard's feat/studies-with-tests-and-investigations).
dnaa-01 carries five findings (3 biological, 2 computational) as a
worked example for the lint rule and the cross-study report.

Two findings cite the v2ecoli expert PDFs directly:
  - F-01 (DnaA count shortfall) cites `chromosome_replication_plan §2.1`
  - F-02 (autorepression missing) cites `replication_initiation_molecular_info §4`

The quote for F-02 was extracted manually for this session; the
`/pbg-study findings cite` subcommand should automate that.

## Open questions

- Should `findings` be append-only across runs, or replaceable on re-run?
  (Provisional: append-only, with an `obsoleted_by: F-NN` field to chain.)
- How do findings flow upstream into investigation.yaml? Probably as
  aggregates ("X of Y studies confirm Z, A contradict, B novel").
- Heuristic for confirms-vs-novel when a FAIL is on a test that has no
  literature comparator: probably default `novel` + flag for review.
