# Chris Long — second-pass review on `multiscale-bioprocess` (2026-05-28)

**Source.** Two reviews + nine inline comments on PR
[vivarium-collective/v2ecoli#69](https://github.com/vivarium-collective/v2ecoli/pull/69),
authored by `cplong90` (Chris Long) on 2026-05-28 between 14:58 and 16:02 UTC.
Raw GH-API export preserved at
`references/expert/raw/chris_feedback_2026_05_28.json`.

**Reviewer.** Chris Long (v2ecoli-workspace co-maintainer; pbg-bioreactordesign
upstream owner; SMS/DARPA project).

**Audience.** Loaded as cleanup constraint on the `multiscale-bioprocess`
investigation. Items below MUST be resolved (or explicitly deferred) before
mbp-03 can enter Build.

---

## How this feedback was integrated

Each item tagged with status:

- **adopted** — edit lands on this branch, target file/line/replacement specified.
- **adopted-broadened** — Chris flagged one site; the same root cause produced
  adjacent stale references that this digest also fixes.
- **validation-only** — affirmative finding, no action.

This round is materially smaller than [chris_feedback_2026_05_26.md] — Chris's
own framing is "all trivial cleanup, flagging for follow-up rather than
blocking." The cross-cutting pattern is **stale references from the §3 / §10 /
§11 reframes that didn't fully propagate through `backs[]`, `inherits_from[]`,
`unblocks[]`, `failure_modes`, and `open_questions.blocks`** — a structural
gap in how a study reshape walked its own downstream linkage. See §10 below.

---

## 1. Validation observations (review-summary level)

Chris ran two passes. Pass 1 (14:58 UTC) checked investigation.yaml + mbp-02
+ mbp-05; Pass 2 (16:02 UTC) checked mbp-01, mbp-03, mbp-04, mbp-06 + the
generated charts.

### 1.a Affirmative findings — no action

Pass 2 records three positive observations worth quoting verbatim so they
land in the investigation report as confirmed-by-reviewer:

- **`cells_per_agent` scaling validated empirically.**
  `studies/mbp-02-population-aggregation/charts/01_population-scaling.svg`
  shows perfect log-y parallelism across cpa ∈ {1, 1e6, 1e9} — 9 orders of
  magnitude scaling, reaching Beulig regime (~10^12-10^13 cells/L at
  cpa=1e9) from a single-cell-with-Division simulation.
  `00_per-cell-mass-invariant.svg` shows three overlapping traces —
  aggregator never touches per-cell state. The §2 architectural decision
  is not just adopted in YAML; it has been validated with sim output.
- **mbp-06 schema extensions fully adopted.** `mandatory_findings_to_consider`
  (4 priors including `cells_per_agent-scaling-stochastic-floor`),
  `findings_schema`, `resolution_type` / `resolution_venue` /
  `decision_authority` / `provenance`, per-axis floors A=3 D=3 B=C=E=1, three
  new behavior tests enforcing the schema, and the `framework_axis_e_gaps`
  block operationalizing 2026-05-26 Comment B as 6 concrete upstream PR
  candidates.
- **Cross-investigation hook fully wired.** `pbg-bioreactor-transport-fork`
  appears as hard prerequisite in `mbp-03.pipeline_gate.prerequisites` and
  as a pre-seeded candidate-future-study in mbp-06 with
  `decision_authority: pbg-bioreactordesign maintainer (Chris)`.

**Status.** validation-only. Add as a one-paragraph "Reviewer verification"
section to the investigation overview the next time the report is
regenerated (no source-tree edit needed — `pbg-report` reads the
investigation YAML; the verification text belongs in the investigation
narrative-spine `executive.verdict` once mbp-03 ships).

### 1.b Cleanup pattern Chris named — actioned in §2–§10 below

> "When mbp-01's test set was reshaped under the §3 plumbing-only reframe,
> the downstream `regression_tests.inherits_from` blocks didn't get updated.
> Plus a few obsolete failure modes / unblocks lists. Six items inline below
> — all trivial."

Same root cause as Pass 1's mbp-05 catches. Nine concrete items follow.

---

## 2. mbp-05 `claims.backs[]` references old test names

**File.** `studies/mbp-05-palsson-benchmark/study.yaml:399, 401`
**Status.** adopted-broadened (Chris flagged 399; 401, 415, 424 share the root
cause and are the same fix).

**Concern.** The §10 reframe renamed `*-matches-published` → `*-comparison-rendered`
for every behavior test (lines 139–145 confirm). The claim-FK `backs[]`
linkages and the `blocks[]` lists in `open_questions:` still reference the
pre-reframe names.

**Action.**

| Line | Current | Replace with |
|---|---|---|
| 399 | `backs: [end-of-run-od-matches-published]` | `backs: [end-of-run-od-comparison-rendered]` |
| 401 | `backs: [end-of-run-od-matches-published, time-to-target-od-matches-published]` | `backs: [end-of-run-od-comparison-rendered, time-to-target-od-comparison-rendered]` |
| 415 | `blocks: [palsson-matched-batch duration_min, time-to-target-od-matches-published threshold]` | `blocks: [palsson-matched-batch duration_min, time-to-target-od-comparison-rendered threshold]` |
| 424 | `blocks: [end-of-run-od-matches-published tolerance band]` | `blocks: [end-of-run-od-comparison-rendered tolerance band]` |

Line 493 (`limitations:` quoting `end-of-run-od-matches-published` in a
sentence explaining the §10 reframe) is intentional historical reference —
leave it.

---

## 3. mbp-05 `regression_tests.inherits_from` references removed mbp-01 test

**File.** `studies/mbp-05-palsson-benchmark/study.yaml:484-488`
**Status.** adopted. Chris explicitly listed the replacement test set.

**Concern.** `uptake-flux-tracks-external-glucose` was removed from mbp-01 per
the §3 plumbing-only reframe. The current inherit block:

```yaml
- study: mbp-01-time-varying-environment
  tests:
    - uptake-flux-tracks-external-glucose
    - cumulative-mass-balance-closes
    - static-env-baseline-unchanged
```

**Action.** Replace with the post-reframe mbp-01 test set (verified to exist
in mbp-01 study.yaml lines 242–298):

```yaml
- study: mbp-01-time-varying-environment
  tests:
    - external_glucose_updates_propagate_within_one_step
    - cumulative-mass-balance-closes
    - zero-substrate-blocks-uptake
    - saturating-substrate-respects-vmax
    - plateau-across-saturating-range
    - static-env-baseline-unchanged
```

---

## 4. mbp-05 `od_to_gdw` default disagrees with mbp-02

**File.** `studies/mbp-05-palsson-benchmark/study.yaml:279`
**Status.** adopted.

**Concern.** Cross-study inconsistency. mbp-05 says `od_to_gdw, default 0.33`;
mbp-02:212 sets `0.34` (Beulig 2025 value, per the resolved
`req-1-palsson-ingestion` ingestion note).

**Action.** Update mbp-05 line 279 to `0.34` and add the same `# Beulig 2025
value (was textbook 0.33)` parenthetical mbp-02 carries, so the next reader
sees both numbers and the resolution.

---

## 5. mbp-03 `regression_tests.inherits_from` references removed mbp-01 test

**File.** `studies/mbp-03-bird-reactor-coupling/study.yaml:448-452`
**Status.** adopted (same fix as §3).

**Action.** Replace with the post-reframe mbp-01 test set:

```yaml
- study: mbp-01-time-varying-environment
  tests:
    - external_glucose_updates_propagate_within_one_step
    - cumulative-mass-balance-closes
    - zero-substrate-blocks-uptake
    - saturating-substrate-respects-vmax
    - plateau-across-saturating-range
    - static-env-baseline-unchanged
```

---

## 6. mbp-03 failure modes assume Option-A (rejected) BiRD config

**File.** `studies/mbp-03-bird-reactor-coupling/study.yaml:409, 414`
**Status.** adopted.

**Concern.** Under the Option-B decision (`BiRDTransportProcess` upstream fork;
no internal biomass ODE by construction), these signatures aren't reachable
on the primary path:

- L409 candidate cause: `"BiRD's internal biomass ODE not actually disabled
  — competing with v2ecoli's biomass"`
- L414 check: `"Confirm BiRDReactorProcess config.max_growth_rate_per_h == 0
  at boot"`

They apply only to the Option-A fallback Eran documented in
`model_change.notes`.

**Action.** Either tag both lines `(Option-A fallback only)` inline, or
remove them. Chris's preference reads as "tag if you want to keep them."
Adopt the tagging path — preserves the fallback's diagnostic coverage if
the fork PR slips.

Concretely: append ` (Option-A fallback only — under Option-B BiRDTransportProcess
has no internal biomass ODE by construction)` to the L409 string and
` (Option-A fallback only)` to the L414 string.

---

## 7. mbp-04 `regression_tests.inherits_from` references removed mbp-01 test

**File.** `studies/mbp-04-multigeneration-runs/study.yaml:263-267`
**Status.** adopted (same fix as §3 and §5).

**Action.** Replace with the post-reframe mbp-01 test set (see §3 for the
full block).

---

## 8. mbp-01 failure_modes signature points at a removed test

**File.** `studies/mbp-01-time-varying-environment/study.yaml:359`
**Status.** adopted.

**Concern.** `uptake-flux-tracks-external-glucose` is the removed test —
`limitations` (line 395) explicitly notes the removal. The failure mode block
at L358-366 is for a non-existent test.

**Action.** Repurpose to the closest current test, per Chris's suggestion:
`external_glucose_updates_propagate_within_one_step` propagation lag. New
signature:

```yaml
- signature: "external_glucose_updates_propagate_within_one_step fails (write not visible to media_update)"
  candidate_causes:
    - "EnvironmentDriver Step writes external_concentrations BEFORE media_update reads (ordering bug)"
    - "media_update caches the medium definition past one timestep"
    - "exchange_data constraints recomputed from a stale media object"
  checks:
    - "Sample external_glucose and metabolism flux at the same timepoint; both should track monotonic decline"
    - "Debug listener on metabolism.exchange_data_from_media's output set; verify it updates each step"
```

The check bullets stay the same (they were always about propagation
plumbing, not about the linear-Pearson assumption that killed the original
test).

---

## 9. mbp-01 `unblocks` references removed `do-clamp-low` sim

**File.** `studies/mbp-01-time-varying-environment/study.yaml:414`
**Status.** adopted.

**Concern.** Current: `unblocks: [linear-glucose-decline, do-clamp-low, all
primary tests]`. The `do-clamp-low` sim was removed per the §3 reframe and
replaced with `zero-substrate-clamp` + the two `saturating-glucose-*` sims
(line 120, 137, 152).

**Action.** Replace L414 with:

```yaml
unblocks: [linear-glucose-decline, zero-substrate-clamp, saturating-glucose-low, saturating-glucose-high, all primary tests]
```

---

## 10. mbp-06 `req-1-synthesis-harness.unblocks` references removed test

**File.** `studies/mbp-06-gap-analysis/study.yaml:552`
**Status.** adopted.

**Concern.** `at-least-eight-gaps-total` was removed per the §11 tactical
change (explicit removal comment at L266-269). Chris also suggested adding
the new schema-enforcement tests like `every-gap-has-resolution-type-and-venue`
(which exists at L292).

**Action.** Update L552 from:

```yaml
unblocks: [all-five-axes-covered, at-least-eight-gaps-total]
```

to:

```yaml
unblocks: [all-five-axes-covered, every-gap-has-resolution-type-and-venue]
```

---

## 11. Cross-cutting root cause + framework note for pbg-superpowers

The nine cleanup items share one root cause: **a study reshape (renaming or
removing tests / sims) doesn't currently walk its own downstream linkages**.
Three reframes (§3 mbp-01 plumbing-only, §10 mbp-05 evaluation, §11 mbp-06
tactical) all left stale FKs behind because there's no lint rule connecting
a `behavior_tests[].name` change to the `backs[]`, `inherits_from`, and
`unblocks` blocks that reference it.

**Status.** flagged as a pbg-superpowers framework item — belongs alongside
the 2026-05-26 Comment B backlog (axis-E entries in mbp-06's
`framework_axis_e_gaps`). Concretely, this is a new candidate item for that
block:

> **`study-yaml-rename-walks-internal-fks`** — lint rule that, on any
> `behavior_tests[].name` rename or removal within `studies/*/study.yaml`,
> reports every still-stale `backs[]` / `inherits_from[].tests[]` /
> `unblocks[]` / `open_questions[].blocks[]` reference within and across the
> investigation. Without it, every reframe leaves drift that has to be
> caught by reviewer pattern-matching, as happened across mbp-01/03/04/05/06.

Add to `studies/mbp-06-gap-analysis/study.yaml` under `framework_axis_e_gaps`
with `resolution_type: upstream_PR`, `resolution_venue: pbg-superpowers`,
`decision_authority: pbg-superpowers maintainer`.

---

## Address-by execution plan

One commit per logical chunk on this branch, to keep the review surface
focused:

1. **`fix(mbp-05): walk §10 reframe through claim backs[] + open_question blocks[]`**
   — addresses §2 (4 line edits on mbp-05).
2. **`fix(mbp-{03,04,05}): refresh inherits_from to post-§3 mbp-01 test names`**
   — addresses §3, §5, §7 (one shared edit pattern across three files).
3. **`fix(mbp-05): align od_to_gdw default with mbp-02 (0.34, Beulig 2025)`**
   — addresses §4 (one line).
4. **`fix(mbp-03): tag Option-A-only failure-mode bullets`** — addresses §6
   (two string edits in mbp-03).
5. **`fix(mbp-01): repurpose stale failure_mode + refresh unblocks`** —
   addresses §8, §9 (two adjacent edits in mbp-01).
6. **`fix(mbp-06): replace stale unblocks ref with new schema-enforcement test`**
   — addresses §10 (one line).
7. **`feat(mbp-06): seed study-yaml-rename-walks-internal-fks axis-E gap`** —
   addresses §11 (one new entry under `framework_axis_e_gaps`).

After (7), regenerate the workspace report (`/pbg-report`) so the
`executive.verdict` block reflects Chris's three validation observations
from §1.a, then mark PR #69 ready for review (still draft today).

No item requires source code (.py) edits — every change is to `study.yaml`
front-matter or to mbp-06's framework backlog. Total estimated diff: ~30
lines across 5 study YAMLs + one new ~10-line gap entry.

[chris_feedback_2026_05_26.md]: chris_feedback_2026_05_26.md
