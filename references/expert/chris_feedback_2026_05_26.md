# Chris Long — feedback on `multiscale-bioprocess` (2026-05-26)

**Source.** Inline-feedback report `rpt-20260523011545`, exported from the
v2ecoli inline-feedback widget on the rendered investigation page
(`investigation-multiscale-bioprocess-2026-05-23-2.html`) at
2026-05-26T18:15:06Z. Raw YAML preserved at
`references/expert/raw/chris_feedback_2026_05_26.yaml`.

**Reviewer.** Chris Long (v2ecoli-workspace co-maintainer; pbg-bioreactordesign
upstream owner; SMS/DARPA project).

**Audience.** This document is loaded as a methodology / scoping constraint
on the `multiscale-bioprocess` investigation. Future agents working on
mbp-* studies MUST read this before drafting spec edits, advancing a study
to Build phase, or designing simulations.

---

## How this feedback was integrated

Each item below is tagged with its integration status:

- **adopted** — translated into edits to the relevant study.yaml / investigation.yaml on this branch
- **adopted-with-decision** — the user made an architectural call between Chris's options; see notes
- **deferred-upstream** — belongs to pbg-superpowers / pbg-bioreactordesign; tracked as a candidate-future-study or upstream PR
- **flagged** — captured as an `open_question` in the relevant study, no edit yet

---

## 1. Investigation-level: acceptance_criteria need to track the mbp-05 reframe

**Concern.** Two related issues:

1. Several investigation-level acceptance_criteria still encode the
   tolerance-test framing of mbp-05 (`end-of-run-od-matches-published`,
   `acetate-profile-matches-published-shape`). If mbp-05 reframes to
   "comparison-rendered" tests (process-level, not biology-tolerance),
   the investigation success definition must shift too — to "comparison
   harness built and applied; structured gap inventory produced," not
   "Palsson benchmark achieved."
2. The criteria list doesn't differentiate "investigation-terminal" gates
   from "per-study gating." Currently 13 entries weighted equally; mixes
   process-style (e.g. `one-generation-completes-without-divergence`) with
   evaluative checks. Two cleaner shapes:
   - mark which criteria are terminal vs per-study gating;
   - or collapse the investigation-level list to mbp-06's terminal checks
     only (which subsume the upstream studies by construction).

**Status.** adopted — investigation.yaml acceptance_criteria split into
`investigation_terminal:` (mbp-06 axes + every-gap-has-resolution) and
`per_study_gating:` (the rest, marked as gating the owning study's impl
PR but not investigation-success conditions). mbp-05 entries renamed to
`-comparison-rendered` variants.

---

## 2. Framework-meta: items that belong upstream in pbg-superpowers

Items raised across the per-study comments that aren't per-investigation
work — they're upstream schema cleanups that would compound over future
investigations. Captured here once so they cluster rather than scatter:

| Item | Origin study | Status |
|---|---|---|
| `study_kind: evaluation` as first-class schema field (linter-aware; dashboard renders evaluation phases distinctly) | mbp-06 | deferred-upstream (tracked in mbp-06 candidate_future_studies → pbg-superpowers PR) |
| `comparison_overlays` — generic trajectory-pair comparison config for the comparison harness; reusable across any benchmark study | mbp-05 | adopted (added to mbp-05 schema); deferred-upstream for hoisting into pbg-superpowers |
| `functional_form` metadata on claims (saturable / step / linear / threshold) — tests pick measure shape; lint-time validation flags mismatches | mbp-01 | deferred-upstream (captured in mbp-01 open_question + mbp-06 framework-axis-E gap) |
| `provenance` on gap entries + `mandatory_findings_to_consider` block (carries known-prior gaps forward; prevents discovery-bias) | mbp-06 | adopted (added to mbp-06 schema) |
| `resolution_type` / `resolution_venue` / `decision_authority` on gap entries — distinguishes model-improvement / scope-decision / upstream-PR | mbp-06 | adopted (added to mbp-06 schema) |
| `scope_decisions` register as workspace-level artifact distinct from findings | mbp-06 + methodology pages | deferred-upstream (tracked in mbp-06 framework-axis-E gap) |
| Unit-typed stores (catches unit-mismatch bugs at compose time) | mbp-01 failure_modes | deferred-upstream (process-bigraph / pbg-superpowers extension) |
| Claim-kind typing (`biological_fact` | `design_constraint` | `regime_assumption` | …) — claims linker refuses to "back" tests with constraint-claims | mbp-03 | deferred-upstream |

These do not block the current investigation. They're a pbg-superpowers
backlog. mbp-06's findings will reify them as upstream PR candidates.

---

## 3. mbp-01 — plumbing vs biology conflation

**Concern.** The current behavior tests bundle two structurally different
kinds of check:

- Plumbing verification (does the store→constraint propagation actually
  work).
- Biological-response verification (does v2ecoli's metabolism respond with
  the correct *shape* to environment changes).

mbp-01's stated purpose is the former. The latter is a stronger claim
that depends on what kinetic / FBA-bound model `metabolism_redux`
implements internally — a question mbp-01 doesn't enhance.

**Specific issues:**

- `uptake-flux-tracks-external-glucose` (Pearson ≥ 0.7): assumes linear
  response between concentration and uptake; real biology is saturable
  (PTS Km ~3–10 µM is ~3 orders below the 5→0 g/L sweep). The current
  test fails in the realistic implementations (MM-saturable, constant-bound)
  and passes only in the unphysical linear-scaling case.
- `low-do-clamp-reduces-o2-uptake` (DO clamp at 5 µM): 5 µM is still
  25–200× above terminal-oxidase Km (cytochrome bd ~24 nM, bo3 ~200 nM).
  Biologically OUR shouldn't reduce at this clamp. The cited Vemuri 2006
  supports acetate flux RISE under low DO, not OUR drop. Test and claim
  are mismatched.

**Recommended scoping:** mbp-01 narrows to plumbing checks + qualitative
extreme-case checks. Both are robust to v2ecoli's transport-kinetics
functional form. Gradient-shape and quantitative biological-fidelity tests
defer to a downstream study explicit about its transport-kinetics
dependency.

**Recommended test set:**

Plumbing (mechanism, no biology invoked):

- `external_glucose_updates_propagate_within_one_step` — store update
  reflected in metabolism's GLC[p] exchange constraint within one timestep.
- `cumulative-mass-balance-closes` — keep as-is.
- `static-env-baseline-unchanged` — keep as-is.

Qualitative extreme-case (biology floor + ceiling, robust to functional form):

- `zero-substrate-blocks-uptake` — external_glucose held at 0 → glucose
  exchange flux ≈ 0. Same for DO.
- `saturating-substrate-respects-vmax` — external_glucose at 50 g/L
  (~30,000× PTS Km) → glucose exchange flux ≤ a published Vmax for E. coli
  glucose-PTS (~10–12 mmol/gDW/h; citation needed in claims.yaml).
- `plateau-across-saturating-range` — two runs at 5 g/L vs 50 g/L; GUR
  within ratio 0.9–1.1. Direct discriminator: saturable + constant-bound
  models predict matching GUR; linear scaling predicts ~10× difference.

Deferred to a `transport-kinetics-fidelity` study (a candidate-future-study
under mbp-06's axis A):

- Gradient-shape test (hockey-stick / step / linear — depends on impl).
- Quantitative μ response to glucose decline near Km.
- Quantitative acetate flux shift at low DO.

**Methodology-layer note (recurs everywhere).** The plumbing-vs-biology
conflation will recur on Henry's-law equilibrium tests, kLa response tests,
and any test of the form "biological observable X responds to driver Y."
Worth a study-scope-discipline rule: **mechanism tests and biological-
fidelity tests live in separate studies** because they have different
failure semantics and different remediation paths (fix the wire vs.
enhance the model).

Within biological-fidelity testing: qualitative-direction tests are
appropriate when the response shape is robust across plausible model
variants (extremes); quantitative-range tests are appropriate only when
the functional form is known and citable. The current `acceptance_form`
enum (`qualitative_direction` | `quantitative_range`) is exactly the right
vocabulary — but the rule for picking between them should be regime-aware,
not default-to-quantitative.

**Status.** adopted — mbp-01 test set replaced per Chris's recommendation.
`uptake-flux-tracks-external-glucose` and `low-do-clamp-reduces-o2-uptake`
REMOVED. New plumbing + extreme tests added. `mu-drops-when-glucose-depletes`
removed (kinetics-dependent). `transport-kinetics-fidelity` reserved as
candidate-future-study slug.

---

## 4. mbp-02 — PopulationAggregator's "literal sum" is a load-bearing architectural commitment

**Concern.** The aggregator's mechanism is `population.total_biomass_gDW
= sum(agents.*.cell_mass) × 1e-15`. This reads as a natural implementation
choice but is actually a substantive architectural commitment that
propagates through every downstream coupled study.

**Implicit consequence.** At single-cell starting inoculation in a 1 L
reactor: biomass concentration = simulated_mass / reactor_volume gives
~2.8e-13 g/L initial biomass — many orders below any density where coupled
reactor dynamics become measurable. After 4 generations (16 cells), still
~5e-12 g/L.

The PR acknowledges high-density intractability (50–80 gDW/L Beulig regime →
deferred to surrogate at mbp-06). But the same intractability applies at
modest densities (0.1–10 gDW/L, where batch dynamics get interesting),
and that's not addressed. Consequences:

- mbp-03's `cells-drop-do-below-saturation` asks whether DO drops when
  cells consume O2 — but 1–16 simulated cells in 1 L don't measurably
  consume O2.
- mbp-04's regime-transition test asks whether the population's glucose
  consumption depletes the pool — population can't deplete a 4 g/L pool
  in 240 min (or 240 days) at single-cell inoculation.
- mbp-05's batch-prefix comparison against Beulig assumes simulated
  dynamics resemble Beulig's batch phase — but Beulig's batch phase starts
  at densities the simulation cannot reach without external help.

**Alternative worth considering: representative-sampling × scale factor.**
Under the 0D well-mixed assumption (already a key_assumption), every cell
sees the same environment. The simulated lineage can legitimately stand
in for a sampled subset of an N-cell population:

```
population.total_biomass_gDW =
    sum(agents.*.cell_mass) × cells_per_agent × 1e-15
```

where `cells_per_agent` is set such that
`n_simulated_agents × cells_per_agent ≈ target_population_density ×
reactor_volume`. This isn't a hack — under the well-mixed assumption,
simulating N representative cells and scaling is mathematically equivalent
to simulating M = N × scale cells. You've downsampled for compute
tractability. Cost: stochastic variance washes out at the bulk mean as N
grows (the simulation doesn't see the variance reduction), but at high
density this is correct biology — population averages dominate.

**Status.** adopted-with-decision — Eran chose the cells_per_agent scaling
path on 2026-05-26.

- `cells_per_agent` added to `model_change.new_parameters` (default: 1.0
  preserves literal-sum; production runs set it to the target population
  size / n_simulated_agents).
- Aggregator mechanism updated in mbp-02 purpose.mechanism.
- Pure consistency checks added alongside doubling-time tests
  (`population_count_equals_len_agents`, `total_biomass_equals_sum_cell_mass_times_scale`).
- `cell_side_interface_contract.md` will be updated (followup TODO) to
  reference the scaling discipline as part of the cell-side interface —
  engines either operate at literal scale or declare an explicit
  `cells_per_agent` factor.

**OD600 framing.** Make `population.biomass_concentration_gL` the canonical
biomass observable; `population.OD600` derived from it strictly for plotting
and for cross-referencing against published OD-only data (e.g. Beulig OD
traces where dry-weight may not be reported). Static `od_to_gdw = 0.34`
(per the Beulig paper's value, not the textbook 0.33) is acceptable under
this scoping. Regime-dependence concerns already flagged in limitations
apply to *interpretation* of OD600 outputs, not to the model's internal
biomass accounting.

**Minor (adopted).** `cell-count-doubles-per-generation` cites
`od-to-gdw-conversion` "via OD600 derivation" — but the test measures
cell_count directly, not OD600. Claim FK linkage was miscategorized;
removed.

---

## 5. mbp-03 — BiRD coupling: Option B (fork to BiRDTransportProcess) wins on design grounds

**Concern.** Option B (fork to `BiRDTransportProcess`) is cleaner than the
disable-internal-biomass flag for several reasons:

- **Substitutability symmetry.** The cell-side interface contract makes
  cell-side engines variable while the reactor side stays fixed. Forking
  gives two reactor variants — one with internal biomass (use without a
  coupled cell-side) and one without (use under the contract). The
  substitutability story becomes symmetric across both sides.
- **The flag name itself.** `bird_disable_internal_biomass` signals
  "shouldn't be there in the first place" rather than "let users turn it
  off." Naming a config option "disable X" is typically evidence that the
  cleaner shape is "don't have X."
- **Tested-vs-dead-code paths.** With `disable=true`, the biomass-ODE code
  path still exists but isn't exercised; untested in this regime, no
  signal if it later quietly breaks. A fork eliminates the unused path.

**Caveat.** B done well is NOT "copy `BiRDReactorProcess` and delete the
ODE." It's an extract-shared-transport-module path — kLa / Henry /
Wilke-Chang in a shared module, with `BiRDReactorProcess` and
`BiRDTransportProcess` as two consumers — which preserves DRY while
delivering the design benefit.

**Status.** adopted-with-decision — Eran chose Option B on 2026-05-26.

- mbp-03 `req-1-bird-dep` retargeted at `BiRDTransportProcess` (does not
  yet exist).
- `bird_disable_internal_biomass` parameter removed from
  `model_change.new_parameters`.
- New candidate_future_study seeded in mbp-06:
  `pbg-bioreactor-transport-fork` — upstream PR to
  `pbg-bioreactordesign` extracting the shared transport module +
  `BiRDTransportProcess`. mbp-03 entering Build phase is **gated** on
  this upstream landing.
- Cross-investigation dependency made explicit in mbp-03
  `pipeline_gate.prerequisites` and noted in `gate_status_summary`.

---

## 6. mbp-03 — BiRD's 1-hour interval drives test-design issues

**Concern.** Is BiRD's 1-hour update interval a structural constraint or
just a default? If tunable, dropping it to a few minutes (matching the
kLa-driven timescale) resolves several issues at once:

- 60-min sim duration would contain many BiRD update points; "steady-state
  DO" tests over `window: second_half` become meaningful. Currently, the
  second half of a 60-min sim is pre-steady-state by construction — BiRD
  has at most 1–2 discrete update points across the entire run, so
  `cells-drop-do-below-saturation` and `higher-kla-raises-steady-state-do`
  can't actually resolve the behaviors they claim to check.
- The 1-hour coupler lag named in `key_assumptions` shrinks proportionally.
- The 5% tolerance on `reactor-biomass-tracks-population` (`pass_if:
  in_range, 0.95, 1.05`) can tighten to floating-point precision.
  Description says "within numerical tolerance" but the band as set admits
  5% drift, suggesting either over-tolerance or an expected drift source
  not named in failure_modes.

If the interval IS structurally fixed at 1 h, the test set needs to
accommodate it explicitly: sim durations extended to 4–6 h to span multiple
update points, "steady-state" framing replaced with "post-final-tick," and
tolerances chosen with the lag-induced drift named explicitly as the
failure-mode source.

**Status.** flagged — added as an open_question against
`pbg-bioreactor-transport-fork` (the upstream PR can either expose the
interval as tunable, or document it as structural). mbp-03 test set updated
to use 4-hour sim durations as the default until the question resolves;
tolerance band on `reactor-biomass-tracks-population` tightened to ±1%
(numerical-tolerance only) with a note that any wider band reveals a
coupling bug.

---

## 7. mbp-03 — CO2 dynamics + O2 mass balance test gaps

**Concern.** Two test-coverage gaps independent of the timestep question:

- **CO2 dynamics are invisible at the test layer.** O2 is thoroughly tested
  (DO, kLa response, Henry saturation, presence-vs-absence-of-cells), but
  symmetric tests for CO2 don't exist. The coupler writes
  `reactor.dissolved_co2 → environment.external_concentrations.CARBON-
  DIOXIDE[p]` per the mechanism, and v2ecoli's metabolism emits CO2
  exchange flux. If the CO2 leg has a wiring bug, mbp-03 passes and the
  defect surfaces downstream.

  Symmetric tests worth adding:
  - `cells-raise-dissolved-co2-above-saturation`
  - `higher-kla-lowers-steady-state-dissolved-co2`
  - `no-cells-co2-converges-to-henry-saturation`

- **O2 mass balance not tested end-to-end.** mbp-01 had
  `cumulative-mass-balance-closes` for glucose. The mbp-03 analog:
  cumulative O2 consumed by v2ecoli + dissolved-O2 change in the reactor
  ≈ kLa-driven gas-liquid transfer integrated. The four-way-coupling
  closure check. Currently no test does this.

**Minor consistency issue (adopted).** `pipeline_gate.proceed_condition`
reads "...mass-balance tests close." But there's no mass-balance test in
the current set — the gate references a test that doesn't exist. Adding
the O2 mass-balance check aligns the gate with the test set.

**Status.** adopted — three CO2-symmetric tests + one O2 mass-balance
test added to mbp-03.

---

## 8. mbp-03 — bubble-column vs stirred-tank for Beulig comparison

**Concern.** Bubble-column is the mbp-03 default; limitations note "stirred
tank / airlift not validated this phase." If mbp-05's Beulig comparison
requires stirred-tank (Beulig 2025 used stirred-tank), a geometry switch
is queued between mbp-03 and mbp-05 with its own validation work.

**Status.** flagged — added as an open_question against
`pbg-bioreactor-transport-fork`. Default geometry decision deferred until
the upstream fork stabilizes; spec to flip to stirred-tank if Beulig
methods confirm.

---

## 9. mbp-04 — MassBalanceAuditor needs explicit C/N requirements

**Concern.** Three test-design issues with the mass-balance audit:

- **CO2-carbon tracking must be an explicit auditor requirement.** The 2%
  C-balance tolerance is tractable only if CO2 is tracked. CO2-C is
  roughly half of glucose-C consumed in aerobic growth — missing it isn't
  a small error mode, it's a guaranteed test failure. The current
  `failure_modes` section names "CO2 evolved by metabolism not bookkept"
  as a candidate cause, but `implementation_requirements.req-1-mass-
  balance-auditor` doesn't list CO2-C tracking as a required behavior.
- **Acetate-carbon (and other byproducts) the same.** Vemuri 2006 is about
  aerobic acetate overflow at high growth rates; v2ecoli's metabolism
  produces acetate; the C-balance needs to track it. Same fix: explicit
  auditor requirement, not candidate failure mode.
- **Nitrogen-balance pool definition needs spec.** What's in the N-balance?
  Just NH4+, or all N-containing species (amino acids, peptides, etc.)?
  The 2% N-balance tolerance is tractable if scope is limited; less so if
  every N-containing species needs explicit accounting.

**Status.** adopted — req-1-mass-balance-auditor description rewritten to
make CO2-C and acetate-C explicit required-tracked species; N-pool defined
as `NH4+ exchange flux + biomass-N` (excluding intermediate amino acids /
peptides, which sum to << 2% under steady-state turnover).

**Other items in Chris's mbp-04 comment:**

- Timestep mismatch inherited from mbp-03 → fixed for free once mbp-03's
  4-hour default lands.
- Single-seed → already a known limitation; non-blocking.

---

## 10. mbp-05 — REFRAME as evaluation phase, not construction phase

**Concern.** The current behavior tests pair an evaluation goal (compare
against published data) with tolerance-based pass criteria (±15% on OD,
±15% on time-to-target, ±30% on acetate shape). Structural problem:

When a tolerance test fails, the AI agent (or human) has an open-ended
menu of things to adjust to make it pass — `failure_modes` literally
enumerates the menu (calibration gap, OD conversion factor, initial
conditions, regime mismatch). Even with "no silent tuning" discipline,
the impulse to reach into v2ecoli mechanics is structurally encoded in
the test design. And given everything stacked against mbp-05 (population
scaling unresolved per the mbp-02 comment; fed-batch deferred; v2ecoli
calibrated at slow-growth single-cell M9 versus Beulig's high-density
fed-batch regime), claiming benchmark-level agreement was always going
to be over-reaching.

**Cleaner shape: mbp-05 as an evaluation phase, not a construction phase.**

- No new model capability — `model_change` has no new processes or state
  variables.
- Deliverable is a structured comparison report: sim-vs-published overlays
  + categorized divergences + open questions, feeding mbp-06's gap
  analysis as input.
- Pass criteria are about whether the comparison was *executed and
  reported coherently* — not whether biology matched within tolerance.
- All biological divergences go to mbp-06 unchanged for triage.
- The `divergences-categorized-in-report-card` test (already present)
  becomes the load-bearing one; the other tests become its evidence-
  producing dependencies.

This separates construction (mbp-01..04 added capability) from evaluation
(mbp-05 applies capability + reports findings), bounds the AI-agent scope
to "execute and report" rather than "tune until match," and resolves the
mbp-02 population-sparsity issue elegantly — sparsity becomes one of the
divergences mbp-06 categorizes, not a blocker for mbp-05 itself.

**What changes in the test set:**

- `end-of-run-od-matches-published` → `end-of-run-od-comparison-rendered`
- `time-to-target-od-matches-published` → `time-to-target-od-comparison-rendered`
- `acetate-profile-matches-published-shape` → `acetate-profile-comparison-rendered`
- `divergences-categorized-in-report-card` — stays as-is; now load-bearing.

The current four-category divergence taxonomy in the YAML (v2ecoli
metabolic deficiency / coupling-adapter limitation / experimental
uncertainty / placeholder-induced) is already aligned with feeding mbp-06.

**Expand the measurables.** The supplementary directory at
`references/papers/palsson-2025-supp/` has trajectories for OD, OTR, CTR,
glucose, glucose uptake, glucose consumption, acetate, ethanol, formate,
lactate, pyruvate, succinate, melatonin, tryptophan, base addition, feed
rate, stirrer speed. Most are exchange fluxes already emitted by v2ecoli's
metabolism Process — wire them all up for completeness.

Practical mapping (WT scope):

| Beulig trajectory | v2ecoli source | Wiring |
|---|---|---|
| OD600 | `population.OD600` | already wired |
| glucose | `environment.external_concentrations.GLC[p]` | already wired |
| acetate | `environment.external_concentrations.ACET[p]` | already wired |
| dissolved O2 | `reactor.dissolved_o2` | already wired |
| glucose uptake rate | `metabolism.external_exchange_fluxes.GLC[p]` × biomass | derived |
| lactate | `environment.external_concentrations.LAC[p]` (verify ID) | new readout |
| formate | `environment.external_concentrations.FOR[p]` | new readout |
| ethanol | `environment.external_concentrations.ETOH[p]` | new readout |
| pyruvate | `environment.external_concentrations.PYR[p]` | new readout |
| succinate | `environment.external_concentrations.SUCC[p]` | new readout |
| OTR | derived: BiRD's kLa × (saturation − dissolved O2) | derived from BiRD outputs |
| CTR | derived: BiRD's kLa × (dissolved CO2 − saturation) | derived from BiRD outputs |
| base addition | derived: cumulative integration of net acid production | small bookkeeping addition |

~5 new exchange-flux readouts + ~3 derived quantities = ~8 new comparison
surfaces. Each tests a specific biological assumption that the current
4-test set doesn't probe: GUR shape ↔ transport kinetics (the mbp-01
backlog concern), mixed-acid byproducts ↔ fermentation pathway wiring +
regulation, OTR/CTR ↔ respiratory model + RQ, base addition ↔ high-density
acid load integration. Each comparison either matches Beulig (positive
evidence) or diverges (seeds a specific mbp-06 gap finding). Under the
report-comparison framing, more surfaces = more information for low
engineering cost — opposite cost/benefit calculus from the tolerance-test
framing where more tests = more knobs to turn.

Out of WT scope (legitimately ignored for mbp-05):

- Melatonin (MEL strain only; heterologous, not in WT model)
- Tryptophan (TRP/TRPp strains over-produce; WT has only native low-level)
- Feed rate, stirrer speed (reactor inputs, not v2ecoli observables)

**Framework-layer (adopted in mbp-05; deferred-upstream for hoisting).**
The comparison harness (`req-2-comparison-harness`) should be generic over
a configurable trajectory inventory, not hardcoded to specific overlays.
Shape:

```yaml
comparison_overlays:
  - sim_path: population.OD600
    published_csv: references/papers/palsson-2025-supp/reactor_OD_data.csv
    published_column: OD600
    overlay_label: "Biomass (OD600)"
    units: OD600
  - sim_path: agents.0.metabolism.external_exchange_fluxes.GLC[p]
    published_csv: references/papers/palsson-2025-supp/reactor_gluc_upt_data.csv
    published_column: GUR
    overlay_label: "Glucose uptake rate"
    units: mmol/(gDW·h)
    aggregation: per_biomass
  # ... one entry per trajectory in the inventory
```

Harness iterates the list, produces one overlay panel + derived metrics
per entry. Future investigations reuse the harness with their own list.
Generalizes cleanly.

**Status.** adopted — mbp-05 reframed in full per Chris's recommendation.
`study_kind: evaluation` declared. `model_change` zeroed out (no new
processes / variables / parameters). Three tolerance tests renamed to
`-comparison-rendered`; pass criteria changed to "overlay artifact rendered
+ divergences logged." `divergences-categorized-in-report-card` promoted
to load-bearing. ~8 new comparison surfaces added to `readouts` +
`comparison_overlays`. `req-2-comparison-harness` rewritten as generic
overlay-iterator. Investigation-level `acceptance_criteria` updated to
match (see §1).

---

## 11. mbp-06 — synthesis design needs carried-forward priors + resolution-type vocabulary

**Concern 1 — carried-forward priors aren't part of the synthesis design.**
mbp-06's synthesis walks mbp-01..05 artifacts → enumerates gaps —
discovery-from-this-investigation by construction. The pre-seeded
`candidate_future_studies` block acknowledges some gaps are obvious before
mbp-01..05 even run, but it's narrow (three items, all things this
investigation touches). Known-prior gaps that weren't necessarily exercised
by mbp-01..05 are at risk of going undocumented.

Concrete examples that should appear regardless of what mbp-05's report
turns up:

- **v2ecoli transport kinetics** (the mbp-01 backlog concern) — wouldn't
  necessarily show up as an mbp-05 divergence if the simulation never
  reaches a glucose regime where it bites; invisible to the investigation
  but still load-bearing for the next iteration.
- **v2ecoli mass-balance at low-GUR** (Eran-identified, deferred from
  prior work) — not exercised by mbp-01..05 (no test specifically probes
  low-GUR mass balance).
- **pbg-bioreactor fed-batch support** — touched by mbp-05's batch-prefix
  scoping, but the gap itself exists in the upstream package; mbp-06's
  discovery framing might position it as a Beulig-scope deferral rather
  than the upstream-contribution-needed it actually is.

**Recommendation.** Add a `mandatory_findings_to_consider` block at the
top of mbp-06, pre-seeding known-prior gaps the synthesis MUST address.
Each entry: gap signature, source (where it came from outside this
investigation), why it's still load-bearing, what would resolve it. The
synthesis either incorporates each prior with new evidence from this
investigation, or explicitly justifies dropping it.

Plus a `provenance` field on each gap entry: `surfaced_by:
this_investigation | carried_prior | external_input | expert_elicitation`.
Priority assessed independently of provenance — a carried prior can be
priority 1 even if mbp-05 didn't surface it fresh.

This pattern matters beyond this investigation: every gap analysis carries
forward known priors, doesn't only discover from what happened.

**Concern 2 — divergence-resolution categorization needs more vocabulary.**
The current `candidate_resolution` field is a single string. mbp-05
divergences split into at least three categories with different recipients,
decision processes, and durable artifacts:

- **Model improvement** — "v2ecoli's Pta-AckA kinetics don't trigger
  overflow at the right DO threshold." Development task; v2ecoli PR;
  calibration_log entry; assigned to metabolism Process maintainer.
- **Scope decision** — "Resistance/persistence bistable expression requires
  regulatory dynamics not in v2ecoli's current model; we choose not to
  model this." No development task; documented as an explicit
  non-modeling decision; decided by program-level authority.
- **Upstream contribution** — "pbg-bioreactor needs fed-batch operations."
  Resolution lives in another repo; decided by whoever owns that
  investigation.

Schema addition:

```yaml
resolution_type: model_improvement | scope_decision | upstream_PR | expert_elicitation | further_investigation
resolution_venue: <where the resolution work or decision lives>
decision_authority: <who decides — for scope decisions especially>
```

This subsumes the in-repo / upstream-contribution / cross-investigation
distinction the current 5-axis taxonomy doesn't carry: `resolution_venue`
makes explicit whether a gap's resolution lives inside v2ecoli, upstream
in pbg-bioreactor, in a sibling investigation, or as a program-level
decision.

Possibly a separate `scope_decisions` register at investigation or
workspace level — distinct from findings. Each entry: what wasn't modeled,
why (decision rationale), what comparison data showed divergence at this
point, who decided. Durable artifact for the program's overall scope
discipline.

**Methodology-layer note.** Upstream methodology pages don't have
vocabulary for intentional non-modeling as a first-class artifact. Neither
a placeholder (implies later resolution), nor an open question (implies
unresolved), nor a deferred candidate-future-study (implies future work).
Worth a concept-page addition over in the upstream methodology — "scope
decisions" or "intentional non-modeling" as a recognized artifact type.

**Concern 3 — pre-seeded `candidate_future_studies` should include two
more items:**

- **BiRD biomass-as-input refactor → upstream PR to pbg-bioreactordesign.**
  Cross-investigation coordination raised in §5 above. Concrete,
  prioritizable (priority 1), resolution venue is the parallel
  pbg-bioreactordesign investigation. Pre-seeding makes the dependency
  explicit. *(Now also a hard prerequisite of mbp-03 entering Build —
  Eran's Option-B decision per §5.)*
- **Population scaling architectural decision** (the mbp-02 question).
  Foundational — mbp-08 (surrogate engine) presumes some resolution.
  Pre-seeding makes the dependency explicit and surfaces that mbp-08's
  design depends on mbp-02's resolution. *(Now resolved per Eran's
  cells_per_agent decision in §4 — the gap shifts to "validate the scaling
  factor against a multi-density real-comparison.")*

**Concern 4 — `study_kind: evaluation` is a framework-layer schema
extension.** Currently introduced inline in this one study. The
pbg-superpowers schema doesn't have this field yet; linter would need to
know about it; downstream tooling (dashboard rendering, status aggregation)
could use it to differentiate construction from evaluation phases. Worth
acknowledging as a schema-update candidate for a pbg-superpowers PR.

**Concern 5 — mbp-06 timing depends on upstream design resolutions.**
mbp-02's architectural question (population scaling) and mbp-05's reframe
both materially change what mbp-01..05 impl PRs will produce. mbp-06's
prerequisites correctly list mbp-01..05, but the gate timing depends on
those architectural questions being resolved first — otherwise mbp-06
synthesizes against a baseline that's not what was intended.

**Smaller tactical items:**

- `findings:` schema should be specified before impl. Behavior tests
  reference `findings.entries` and `findings.summary.axis_X_count` but
  the schema isn't given. Structural-check predicates need shape to
  validate against.
- 8-gap lower bound is arbitrary. Better as a per-axis floor (e.g., axes
  A and D expected ≥ 3 each based on prior known limitations) rather than
  a global count.
- `every-gap-has-candidate-resolution` placeholder detection should
  specify the banned-string list explicitly ("TBD", "figure out later",
  "to be determined", etc.) so the predicate is auditable.

**Status.** adopted — mbp-06 schema extended with
`mandatory_findings_to_consider`, gap `provenance`, `resolution_type` /
`resolution_venue` / `decision_authority` fields. Two new
candidate_future_studies pre-seeded (`pbg-bioreactor-transport-fork`,
`scaling-factor-multi-density-validation`). Per-axis floors set for axes
A and D. `every-gap-has-candidate-resolution` predicate now lists banned
strings explicitly. `findings:` schema sketched in
implementation_requirements. Timing-dependency on mbp-02/05 noted in
`gate_status_summary` (now historically resolved per §4 and §10 above).
`study_kind: evaluation` framework hoisting tracked as upstream PR (§2).

---

## 12. Recurring methodology rule that fell out of this review

> **Mechanism (plumbing) tests and biological-fidelity tests live in
> separate studies.** They have different failure semantics and different
> remediation paths (fix the wire vs. enhance the model). Bundling them
> conflates the AI agent's response when something fails.

This rule applies recursively to every future study in this investigation
arc and to the cell-side interface contract. Captured as a methodology
constraint in `investigation.yaml.discipline.scope_discipline` (followup
edit — see `mbp-06` axis-E `methodology` gap entry).
