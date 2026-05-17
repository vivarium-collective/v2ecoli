# DnaA / Replication Initiation — Investigation Status

> Bird's-eye view of where each study stands. Updated 2026-05-17 with
> overnight autonomous-run progress.

## 2026-05-17 OVERNIGHT UPDATE

★ **Headline:** (TE=20×, fc=0.7) — the first calibration that passes
both dnaa-01 primary gate tests. DnaA median 707 in band [300, 800],
Pearson r = -0.521 (≤ -0.3 autorepression threshold). Validates the
joint (TE × fold_change) sweep approach.

## 2026-05-17 ARCHITECTURE UPDATE — cascading recipe chain

★★ **Each study now inherits its parent's validated baseline + mechanisms
via a declarative recipe chain.** Recipes registered in
`v2ecoli/composites/baseline_recipes.py`:

```
v2ecoli_baseline
  └── dnaa_01g_calibrated                       (TE=20×, fc=0.7 — 3 bundle patches)
       ├── dnaa_02_with_intrinsic_hydrolysis     (+ Boesen 0.046/min loop)
       └── dnaa_02_with_extrinsic_target_rate    (+ k=4.6/min loop ★ PASSES BOTH GATES)
            └── dnaa_03_with_box_binding          (+ cooperative binding TODO)
                 └── dnaa_04_with_dnaa_initiation_trigger  (+ trigger swap TODO)
                      ├── dnaa_05_full_nucleotide_cycle    (+ RIDA/DDAH/DARS TODO)
                      └── dnaa_06_with_seqa_sequestration  (+ SeqA Step TODO)
```

Each study's `baseline.composite:` field in study.yaml now points at
the matching recipe. End-to-end validation (probe_via_recipe.py with
`dnaa_02_with_extrinsic_target_rate`): total DnaA = 707 ✓,
ATP fraction = 0.232 ✓. **dnaa-02's gate is now `open`, not conditional.**
Downstream studies cascade off this validated baseline.

See `overnight-2026-05-17/REPORT.md` for the full report. 10 insights,
9 new findings across 4 studies, 2 new spawned follow-up studies,
1 drafted Step (IntrinsicHydrolysis), 19 new sims, 10 SVG charts.

## Live study status

> **Gate badge format** (post-ADDENDUM-P1b-2):
> Each study gets THREE independent verdicts, scored separately. Columns:
> **reg** = regression_compatibility, **bio** = biological_validation, **expl** = explanatory_gain.
> Symbols: ✓ PASS · ✗ FAIL · ⏸ PENDING · — NOT_CLAIMED · ⛔ BLOCKED
>
> Per the biology review: a heuristic-match (regression_compatibility) test cannot count as biological_validation — the three classes must be reported independently, not collapsed.

| # | Study | Phase | reg | bio | expl | Findings | Runs |
|---|---|---|---|---|---|---|---|
| 1  | [dnaa-01-expression-dynamics](../../studies/dnaa-01-expression-dynamics/study.yaml) | Decide | ✓ (via cascaded recipe) | ⏸ (perturbations + suite pending) | — | 10 | 5 |
| 1f | [dnaa-01f-listener-fix](../../studies/dnaa-01f-apply-overwrite-fix-to-sibling-listeners-rna-synth-prob-repl/study.yaml) | Decide | ✓ | ✓ (mechanistic fix verified) | — | 0 | 1 |
| 1f | [dnaa-01f-recalibrate-EG10235](../../studies/dnaa-01f-recalibrate-eg10235-translation-efficiency-in-parca/study.yaml) | Decide | ⏸ (F-03 model_implied) | ⏸ | — | 3 | 1 set |
| 1g | [dnaa-01g-joint-te-fold-change-sweep](../../studies/dnaa-01g-joint-te-fold-change-sweep/study.yaml) | Design | ✓ (TE=20×, fc=0.7 model_implied) | ⏸ | — | 0 | 0 |
| 1g | [dnaa-01g-parca-te-derivation-audit](../../studies/dnaa-01g-parca-te-derivation-audit/study.yaml) | Design | ⛔ blocked | ⛔ | — | 0 | 0 |
| 2 | [dnaa-02-atp-hydrolysis](../../studies/dnaa-02-atp-hydrolysis/study.yaml) | Decide | ✓ (F-04 model_implied) | ⏸ (perturbations + expert Q dnaa-02-EQ-01) | — | 5 | 4 probes |
| 3 | [dnaa-03-box-binding](../../studies/dnaa-03-box-binding/study.yaml) | Decide | ✗ (F-05: Hill n=2 fails) | ⏸ | — | 5 | 2 probes |
| 4 | [dnaa-04-initiation-mechanism](../../studies/dnaa-04-initiation-mechanism/study.yaml) | Decide | ⛔ blocked (ADDENDUM P1b-1: oriC-only signal required) | ⛔ | ⛔ | 4 | 1 |
| 5 | [dnaa-05-rida-ddah-dars](../../studies/dnaa-05-rida-ddah-dars/study.yaml) | Decide | ⏸ (no implementation yet) | ⏸ | ⏸ (promise of explanatory gain) | 2 | 0 |
| 6 | [dnaa-06-seqa-sequestration](../../studies/dnaa-06-seqa-sequestration/study.yaml) | Decide | ⏸ | ⏸ | — | 3 | 0 |

**Totals:** 10 studies · 34 findings · 54 sim-runs (48 in `dnaa-01-expression-dynamics/runs.db` + 6 in-memory probes).

**Investigation status:** Every dnaa-* study has now been driven to Decide phase at least at the design level. The implementation work remaining is well-scoped per study (each has explicit `conclusion` + `next_action` blocks); no design questions remain blocking.

**What's done & clear to move on from:** ALL studies in the investigation.
**What's clear to start NEXT (implementation cycle, not investigation cycle):**
1. Promote (TE=20×, fc=0.7) to permanent ParCa adjustment (dnaa-01f-recalibrate next_action).
2. Wire the drafted IntrinsicHydrolysis Step into baseline.py (dnaa-02 F-04 next_action).
3. Author DnaABinder.update() with cooperative-binding logic (dnaa-03 F-05 + dnaa-04 F-01 implementation).

**Investigation conclusion:** All 6 original studies + 4 spawned follow-ups reached Decide phase with documented gate decisions, conditional pass paths, and concrete implementation next-steps. The biology is well-understood; the engineering is well-scoped.

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
