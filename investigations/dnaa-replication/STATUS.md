# DnaA / Replication Initiation — Investigation Status

> Bird's-eye view of where each study stands. Auto-discoverable from the
> dashboard via `investigations/dnaa-replication/investigation.yaml`.
> Hand-maintained today; future revision: have `/pbg-study fill-overview`
> or a status job regenerate this from the per-study `status:` +
> `tests.last_results.summary` fields.

| # | Study | Phase | Status | Tests (pass/total) | Behaviors implemented | Key gaps | Blocks |
|---|---|---|---|---|---|---|---|
| 1 | [dnaa-01-expression-dynamics](../../studies/dnaa-01-expression-dynamics/study.yaml) | Expression baseline | planned | — / 6 | 5 implemented · 1 stub | gap-1 listener · gap-2 hook | dnaa-02 |
| 2 | [dnaa-02-atp-hydrolysis](../../studies/dnaa-02-atp-hydrolysis/study.yaml) | Nucleotide cycle | planned | — / 5 | 1 implemented · 4 gated | upstream: DnaA-ATP/ADP/apo split | dnaa-03, dnaa-05 |
| 3 | [dnaa-03-box-binding](../../studies/dnaa-03-box-binding/study.yaml) | DnaA-box binding model | planned | — / 6 | 0 implemented · 6 gated | upstream: 307 chromosomal + 11 oriC + 4 dnaAp box binding | dnaa-04 |
| 4 | [dnaa-04-initiation-mechanism](../../studies/dnaa-04-initiation-mechanism/study.yaml) | Mechanism-driven initiation | planned | — / 7 | 0 implemented · 7 gated | upstream: DnaA-ATP filament + DUE opening | dnaa-05, dnaa-06 |
| 5 | [dnaa-05-rida-ddah-dars](../../studies/dnaa-05-rida-ddah-dars/study.yaml) | Extrinsic conversion | planned | — / 5 | 0 implemented · 5 gated | upstream: RIDA + DDAH + DARS1 + DARS2 processes | — |
| 6 | [dnaa-06-seqa-sequestration](../../studies/dnaa-06-seqa-sequestration/study.yaml) | Post-initiation sequestration | planned | — / 5 | 0 implemented · 5 gated | upstream: SeqA + Dam GATC methylation | — |

**Totals:** 6 studies · 34 behavioral tests (6 implemented today, 28 gated on upstream processes) · 21 interventions · 26 references in `papers.bib`.

## Dependency DAG

```
dnaa-01-expression-dynamics       (root — implemented baselines, ready to run)
  └── dnaa-02-atp-hydrolysis      (DnaA-ATP/ADP/apo split — first hard upstream)
        ├── dnaa-03-box-binding   (307 chromosomal + 11 oriC + 4 dnaAp boxes)
        │     └── dnaa-04-initiation-mechanism   (replace mass-threshold heuristic)
        │           ├── dnaa-05-rida-ddah-dars   (RIDA + DDAH + DARS1 + DARS2)
        │           └── dnaa-06-seqa-sequestration   (SeqA + GATC methylation)
        └── dnaa-05-rida-ddah-dars   (also depends on dnaa-02 directly)
```

Open the DAG live in the dashboard: **Investigations** → **DnaA / Replication Initiation**.

## What's runnable today (no new code)

Only **dnaa-01** has implemented behavioral tests against the existing baseline. Run with:

```
pytest studies/dnaa-01-expression-dynamics/tests/test_with_synthetic_history.py -v
```

5 tests pass against synthetic in-memory history (BT-01..BT-04). The same evaluator will run against a real `runs.db` once the baseline is executed via the dashboard.

## What needs to land first (recommended order)

1. **gap-1** — `DnaaConcentrationListener` Step (XS effort). Unblocks dnaa-01's `derived-needed` observables.
2. **gap-2** — `translation_efficiency_override` composite-level hook (S). Unblocks dnaa-01's `stop-dnaA-synthesis` variant + its 2 behavioral tests.
3. **dnaa-02 upstream** — split DnaA bulk into ATP / ADP / apo species + intrinsic hydrolysis Step (M). Unblocks 4 of 5 dnaa-02 behaviors.
4. **dnaa-03 upstream** — DnaA-box catalog + binding kinetics (L). The biggest single block in the chain; unblocks all of dnaa-03 + much of dnaa-04.

Each gap's full implementation plan is in the corresponding study.yaml under `gaps:`.

## Acceptance criteria (when does this investigation conclude?)

Listed in `investigation.yaml.acceptance_criteria:`. Summary: 8 specific behavioral tests across 5 studies must all pass.

| Study | Behavior | Why this gates investigation completion |
|---|---|---|
| dnaa-01 | dnaa-count-in-mass-spec-range | Foundational DnaA pool is right |
| dnaa-01 | dnaa-concentration-stable-within-10pct | Foundational pool is stable |
| dnaa-02 | dnaa-atp-fraction-in-physiological-range | DnaA-ATP fraction in [0.2, 0.5] (Boesen 2024) |
| dnaa-03 | oric-stays-unoccupied-until-chromosome-saturates | Titration is real |
| dnaa-04 | one-initiation-per-generation | Stable cell cycle |
| dnaa-04 | initiation-mass-mean-matches-heuristic | Mechanism reproduces the heuristic baseline |
| dnaa-05 | inter-initiation-cv-narrows-vs-intrinsic-only | Extrinsic conversion improves homeostasis |
| dnaa-06 | wildtype-zero-reinitiation | SeqA prevents reinitiation |

## Open questions

Per-study `expert_questions:` lists the active questions for domain experts. Highlights:

- dnaa-01: ask Katayama / Lobner-Olesen labs on dnaa autorepression strength + each DnaA box in dnaAp.
- dnaa-02: apo-DnaA pool size assumption + ATP/ADP binding kinetics resolution.
- dnaa-03: 307-box equivalence vs context-dependent cooperativity.
- dnaa-04: cooperativity of DnaA-ATP filament at oriC + DnaA box dissociation mechanism on fork passage.
- dnaa-05: quantitative scaling of RIDA per replication fork.
- dnaa-06: SeqA-multimer cooperativity vs Hill-shaped on/off.

## How to use this file

- **Daily**: glance at the table to see what's blocked / runnable.
- **PR-by-PR**: update the status column when a phase moves planned → running → ran → complete.
- **Cross-study triage**: the Blocks column tells you the priority order — landing dnaa-01 first unblocks dnaa-02, etc.

The dashboard auto-discovers this file as the investigation's README equivalent (planned dashboard feature: render this in the Investigations tab as a "Status" subtab).
