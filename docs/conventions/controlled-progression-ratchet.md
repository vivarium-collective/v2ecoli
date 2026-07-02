# Controlled progression: the mechanism-add / recalibrate ratchet

> Turns a staged investigation from an ad-hoc, backward-patching sequence into a
> forward **ratchet**: the system is always brought to *satisfactory* before the
> next mechanism is added, so parameters never silently drift out of band and the
> narrative reads as a clean study-after-study progression.

## The problem

A pbg investigation is built study by study, each adding one mechanism on top of
the previous baseline. But **every mechanism perturbs the system**: adding
DnaA-box binding depletes the free pool; adding RIDA drains DnaA-ATP; adding
asynchronous initiation removes the accidental synchrony that was limiting
re-initiation. So a parameter that was in band before a mechanism is added drifts
out of band after — and has to be **recalibrated**.

If that recalibration is done ad hoc and *backward* (fix an upstream study after a
downstream one has already moved the reference config), two things rot:

1. **Stale configs.** A study keeps running on a config a later study has
   superseded, so it false-fails or shows stale plots (see
   [staged-investigation-config-propagation](staged-investigation-config-propagation.md)).
2. **Tangled narrative.** The story accumulates corrections, retractions, and
   "actually the earlier claim was wrong because…" — exactly the churn this
   convention exists to prevent.

## The rule — the ratchet

Progress through the studies so that **the system is satisfactory before every
step**. Concretely, three rules:

1. **One canonical reference config.** The investigation carries a single current
   operating point — the full set of knobs (expression V, hydrolysis k_h,
   cooperativity n/K, trigger threshold, eclipse, RIDA/DDAH/DARS multipliers, …)
   at their *satisfactory* values. Each study **inherits it explicitly** as its
   baseline; no study invents its own quietly-different config.

2. **A defined recalibration step.** Every study, after adding its mechanism, has
   an explicit recalibration step: re-tune **only** the parameters that mechanism
   perturbs, to restore **all prior** acceptance criteria. The recalibration is
   part of the study, not a later cleanup.

3. **Accumulating gates + forward write.** A study passes only if its **own new
   criterion AND every prior criterion** hold on the current config. When it
   passes, it writes the updated knobs back as the new canonical reference config,
   which the next study inherits. The gates only ever grow.

This is a **forward ratchet**: because each study re-establishes satisfactory
before proceeding, there is no need to go back and re-evaluate upstream studies.
The backward-propagation rule (the linked convention) becomes the *fallback* for
when the ratchet is broken, not the normal mode.

## The accumulating report card

The accumulating gates (rule 3) live **in each study's `tests:` block / report card**,
not just in prose. Every study:

- adds its **new criterion** as a test tagged `stage: new`, and
- carries **every prior criterion** forward as a test tagged `stage: maintained`,

so the report card literally *shows* that previous behavior is maintained. A study
passes only when its `new` test **and all `maintained` tests** pass on the current
config. The set of tests only grows as you descend the graph.

```yaml
tests:
  - name: over-replication-fixed        # the study's own contribution
    stage: new
    ...
  - name: dnaa-pool-in-band             # inherited, must still hold
    stage: maintained                   # C1 (dnaa-1)
    ...
  - name: dnaa-atp-fraction-in-band
    stage: maintained                   # C2 (dnaa-2 / dnaa-9)
    ...
```

This is what makes recalibration safe: after a mechanism perturbs a parameter, the
`maintained` tests are exactly the check that the recalibration restored every prior
behavior before the study is allowed to pass the config forward.

## Per-study template

Each study reads as the same five beats:

1. **Inherit** — start from the canonical reference config (state it explicitly).
2. **Add** — introduce exactly one mechanism.
3. **Break** — name what the mechanism perturbs (which prior criterion it stresses).
4. **Recalibrate** — re-tune the affected parameters to restore all prior criteria.
5. **Pass forward** — record the new satisfactory config as the reference; the new
   criterion joins the accumulating gate set.

A study that adds no mechanism (a pure calibration/validation sweep, e.g. an
in-sim parameter map) skips beats 2–4: it only *characterizes* the operating
basin of the current reference and hands the same config forward.

## Worked example — the DnaA / replication-initiation arc

| # | Inherit ← | Add mechanism | What it breaks | Recalibrate | New accumulating criterion |
|---|---|---|---|---|---|
| 0 | (cold cache) | succinate baseline | — | — | oriC 1↔2 periodicity |
| 1 | dnaa-0 | DnaA expression | pool level | tune V | DnaA pool ∈ [300,800] |
| 2 | dnaa-1 | ATP/ADP hydrolysis | ATP fraction | tune k_h | DnaA-ATP fraction ∈ [0.2,0.5] |
| 3 | dnaa-2 | oriC/box binding | free pool (over-binds) | resolve over-binding + box-doubling | oriC occupancy correct |
| 4 | dnaa-3 | autoregulation | needs the V override | drop V override | self-stabilizes in band |
| 5 | dnaa-4 | cooperativity (Hill) | occupancy shape | tune n/K | sharp oriC-low switch, bands hold |
| 6 | dnaa-5 | mechanistic trigger | initiation timing | threshold/pool choice | once-per-cycle on the DnaA-ATP switch |
| 8 | dnaa-6 | *(none — n×K sweep)* | — | *maps the operating basin* | n=4/K=30 confirmed in-sim |
| 9 | dnaa-8 | full-occupancy (8/8) + async | once-per-cycle (over-replicates) | re-tune k_h → 0.025 | fires at full occupancy; fraction in band |
| 10 | dnaa-9 | SeqA re-init block | — (fixes 9) | eclipse window | controlled once-per-cycle |
| 11 | dnaa-10 | RIDA | ATP fraction (RIDA drains it) | re-tune k_h | full biological re-init control |

Read top to bottom, this is the whole investigation as one controlled
progression: each row inherits the row above, adds one mechanism, recalibrates,
and passes a satisfactory config forward — with the acceptance criteria
accumulating down the column.

## Why this is more robust

- **No stale configs** — there is one reference, explicitly inherited.
- **No backward churn** — satisfactory-before-proceeding means upstream studies
  never need re-evaluation for a downstream change.
- **A legible narrative** — every study is the same five beats, so the report
  reads as a logical graph traversal instead of a sequence of corrections.
- **Recalibration is expected, not a surprise** — it is a named step, so
  "the mechanism moved a parameter out of band" is the *normal* outcome that the
  study then resolves, not a failure.
