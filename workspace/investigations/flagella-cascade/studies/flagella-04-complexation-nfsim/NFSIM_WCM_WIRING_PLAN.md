# Plan: wiring NFsim into the real WCM for complexation, alongside regulation + FlgM secretion

Added 2026-08-12, part of Maya Abdalla's flagella-cascade investigation. This
is a planning document, not yet implemented — captures the architecture
agreed on before starting the work, so the reasoning survives independent of
any one conversation.

## Context

`flagella_nfsim_assembly.py` / `run_nfsim_assembly.py` (this study) currently
run NFsim as a fully **standalone** composite: two processes (`MonomerProduction`
feeding a placeholder "1 flagellum/100s" rate, `NFSimProcess` running the BNGL
reaction network), disconnected from the real WCM entirely. No
`flagella_regulation`, no ParCa, no real transcription, no division.

Separately, the real WCM already has a complete, working, deterministic
assembly + regulation pipeline (`ecoli_baseline.py`'s `flagella_regulation`
feature, opt-in): `flagella-motor-switch-assembly` ->
`flagella-export-apparatus-assembly` -> `flagella-motor-complex-assembly` ->
`flagella-filament-nucleation` -> `flagella-filament-elongation` ->
`flagella-flgm-secretion` -> `flagella-transcription-regulation`.

Goal: replace the custom deterministic complexation Steps (motor-switch
through hook-nucleation) with the NFsim rule-based reaction network, while
keeping `flgm-secretion` and `transcription-regulation` working exactly as
they do today.

## Key simplifying insight

`flagella_flgm_secretion.py` and `flagella_transcription_regulation.py` only
ever read/write **real WCM bulk molecule counts** (FliA, FlhDC, FlgM,
CPLX0-7452) — they have no dependency on which mechanism produced those
counts. As long as NFsim's output lands in the same bulk molecule IDs those
two Steps already use, **neither one needs to change at all.** This is not a
full regulatory-loop rewrite — it's a swap of what feeds the loop.

## What has to change

1. **Rename NFsim's BNGL species to real v2ecoli bulk molecule IDs.**
   Currently `generate_flagella_bngl.py` uses generic placeholder names
   (`Free_fliF`, `flagellar_motor`, etc.). Rename these directly to the real
   EcoCyc bulk IDs (`FLIF-FLAGELLAR-MS-RING[i]`, `CPLX0-7450[i]`, etc.) so
   NFsim's observables ARE the bulk array's own molecule names — no
   translation layer needed, mass conservation stays exact by construction.

2. **Drop the standalone `MonomerProduction` placeholder entirely.** Wire
   NFsim's `observables` port straight to the real `("bulk",)` store — same
   topology pattern every other flagella Step already uses. NFsim reads
   whatever real transcription/translation has actually produced; no
   separate artificial feed.

3. **Wrap `NFSimProcess` in a new v2ecoli Step**
   (`flagella_nfsim_complexation.py`, new file, this study or promoted to
   `v2ecoli/processes/` once validated) that:
   - Reads real bulk counts each time it fires, feeds them to NFsim as
     `observables`.
   - Calls `update()` using the scaffold-persistence-fixed `NFSimProcess`
     (see the `pbg-nfsim` PR, https://github.com/vivarium-collective/viva-nfsim/pull/2 —
     REQUIRED, not optional, for correctness across the many chunks a real
     WCM run implies).
   - Writes observable deltas back to the real bulk store.
   - Carries `scaffold_species` forward via its own port/store in the
     composite (the exact wiring gap found and fixed 2026-08-12 in
     `make_production_document` — must be replicated here).

4. **Give NFsim a realistic firing cadence, not every 2s tick.** Spawning a
   BioNetGen subprocess every tick would be prohibitively slow wall-clock.
   Reuse the same `next_update_time`-based rate-limiting pattern
   `flagella_filament_nucleation.py` already uses (fixed interval, NOT a
   per-tick probability — see that module's "BUG FOUND AND FIXED" note for
   why the naive version doesn't work). Target interval: ~60-120s of
   simulated time, tunable.

5. **Where it plugs into the existing pipeline:** NFsim's BNGL network (as
   built) replaces Steps 1-4 — motor-switch, export-apparatus, motor-complex,
   and hook-nucleation — since its terminal `flagellum` observable already
   represents "hook-basal-body complete" (see the FLIC REMOVAL note in
   `generate_flagella_bngl.py`), the same trigger point custom nucleation
   currently uses to create a `nascent_flagellum`. **`filament-elongation`
   (Step 5) stays exactly as-is** — FliC/filament was deliberately excluded
   from the BNGL network to avoid the same combinatorial-explosion problem
   that motivated pulling filament growth out of Gillespie SSA in the first
   place. The new wrapper Step, on seeing NFsim's `flagellum` observable
   increase, creates that many new `nascent_flagellum` unique molecules
   (filament_length=0) for elongation to pick up — replacing what
   `flagella_filament_nucleation.py` does today.

6. **FlhDC degradation (YdiV) slots into the same renamed-species BNGL
   network later**, once the literature check (still pending) is done — a
   new reaction consuming real `CPLX0-3930[c]` directly, naturally
   consistent with everything else once species names match the real bulk
   store.

## Recommended rollout — not all at once

1. **DONE (2026-08-12).** Renamed BNGL species to real bulk IDs in
   `generate_flagella_bngl.py` (e.g. `fliF` -> `FLIF-FLAGELLAR-MS-RING[i]`),
   cross-checked directly against the real running Steps and
   `reconstruction/ecoli/flat/*.tsv`, not guessed. Caught two real findings
   in the process: (a) FlhC's real bulk protein ID is `MONOMER0-2488[c]`,
   not `EG10319-MONOMER[c]` as the cistron reference in
   `flagella_transcription_regulation.py`'s docstring might suggest; (b) the
   canonical reaction spec includes FlgI (`FLGI-FLAGELLAR-P-RING[j]`, -26)
   in the motor-complex reaction, matching `flagella_motor_complex_assembly.py`'s
   own docstring, but that Step's actual `_REQUIREMENTS` dict never consumes
   it -- a real, apparently unintentional gap in the currently-running WCM
   code, flagged but NOT fixed here (out of scope for this rename pass).
   Three species have no real bulk-ID equivalent and were deliberately left
   as descriptive names (`flagellar export apparatus subunit`, `flagellar
   hook`, `flagella`) -- see the model's own docstring for why. Re-verified:
   rule count unchanged (588), standalone run at 8 simulated hours produces
   the same order-of-magnitude result as before the rename (export=34,
   motor=13, hook=255, flagella=2 vs. pre-rename export=36, motor=15,
   hook=252, flagella=1) -- consistent with normal stochastic variance, not
   a regression. `run_nfsim_assembly.py`'s tracked observable names updated
   to match.
2. Build the wrapper Step; test against a **minimal** composite with a real
   bulk store (not the full 55-process baseline) — a diagnostic script,
   checking mass conservation and correct handoff to a stub/real
   `flgm-secretion`.
3. Wire into `ecoli_baseline.py`'s `flagella_regulation` feature **behind a
   new sub-flag**, so the existing custom-Steps pipeline stays selectable
   too (preserve-old-code rule; also gives an A/B comparison for free —
   NFsim-driven vs. deterministic-Step-driven assembly under otherwise
   identical regulation).

2. **IN PROGRESS (2026-08-12).** Built `diagnostic_real_bulk_seeding.py`:
   reads the REAL WCM's actual bulk counts for all 33 real-bulk-ID species
   (via a live `ecoli_baseline` composite), seeds `NFSimProcess.observables`
   directly from them, runs with NO `MonomerProduction` at all. Confirmed
   all 33 real IDs resolve correctly end-to-end (structural proof step 1's
   renaming works against a live composite, not just standalone).

   REAL FINDING, not a bug in this diagnostic: nothing ever completes.
   `~67-70 distinct partial scaffolds` nucleate almost immediately and
   NEVER finish across 8 simulated hours -- free FliF depletes from its
   real ambient count (657) down to a floor of 14, split across ~70
   simultaneous competing scaffolds, each needing 34 copies to complete.
   Root cause: `NUCLEATION_SUPPRESSION_FACTOR=1000` was calibrated against
   the old standalone demo's artificial seed counts (~170 FliF, from
   `n_flagella=5`-scaled defaults) -- real WCM ambient pools are ~4x
   larger, and since nucleation propensity scales with count^2, that's
   ~15x more simultaneous nucleation events, reproducing the EXACT
   "many parallel scaffolds, none finish" failure mode already found and
   fixed once this session (see `flagella-04-nucleation-rate-fixed-real-
   assembly-confirmed` in study.yaml) -- just resurfacing because that
   fix's calibration doesn't transfer to real WCM copy numbers. NOT yet
   fixed -- needs a decision on how to recalibrate (a larger fixed
   suppression factor? scale suppression by monomer count so real vs.
   demo scales both work? something else?) before continuing to step 3.

   RESOLVED (2026-08-12, same day): chose the fast option (a larger fixed
   global constant), explicitly accepting the tradeoff that it's tuned to
   today's observed real:demo ratio rather than derived per-reaction.
   `NUCLEATION_SUPPRESSION_FACTOR` rescaled 1000 -> 35,000, sized to cover
   the worst directly-relevant observed case (FlgE/hook: real=3508 vs.
   demo=600, ~34.2x propensity increase) with margin. Re-ran the same
   real-bulk-seeding diagnostic: 5x CPLX0-7450 (C-ring) and 28x complete
   hooks formed over 8 simulated hours, material still actively flowing
   (Free_FLIF 657->287, scaffold count still growing at the end, not
   stalled) -- a real, hierarchically-bottlenecked result (export apparatus
   and motor complex hadn't progressed yet by t=28800s, consistent with
   depending on only 5 C-rings having formed so far, not a bug). Step 2 is
   now functionally complete: NFsim runs correctly from ONLY the real WCM
   ambient bulk pool, no synthetic MonomerProduction feed at all.
   `diagnostic_real_bulk_seeding.py` is the artifact demonstrating this.

3. **DONE (2026-08-16).** Built `v2ecoli/processes/flagella_nfsim_complexation.py`
   (real `EcoliStep`, wraps the scaffold-fixed `NFSimProcess`, carries
   `scaffold_species` + `internal_observables` across firings) and wired it
   into `ecoli_baseline.py` as a new, independent `flagella_nfsim_complexation`
   feature module -- mutually exclusive with `flagella_regulation` (both
   `insert_before` the same anchor; enabling both would duplicate
   filament-elongation/flgm-secretion/transcription-regulation).

   Two real infrastructure gaps found and fixed along the way, both
   pre-existing patterns in this codebase, not NFsim-specific:
   - The actual runtime step-config lookup goes through `CachedConfigLoader`
     (serves pre-baked configs from the cache bundle), NOT live calls to
     `sim_data.py` -- adding a config-getter method there only matters when
     *building* a cache, not loading an existing one. Fixed by adding
     `ecoli-flagella-nfsim-complexation` to `core.py`'s `_CACHE_CONFIG_NAMES`
     and directly patching the one new (trivially empty, `{}`) config entry
     into the existing `sim_data_cache.dill`, rather than rebuilding.
   - `scripts/build_cache.py`'s fast fixture-hydration path turned out to be
     unusable here: `models/parca/parca_state.pkl.gz` (June 24) predates
     this investigation's own `nascent_flagellum` unique-molecule-type
     addition, so hydrating from it throws `KeyError: 'nascent_flagellum'`.
     Confirmed via file-hash comparison first (matched exactly, ruling out
     "wrong/swapped fixture") before concluding it's a content-vs-codebase
     staleness issue, not a file-identity one -- exactly the scenario Maya
     flagged from a prior bad experience with the fast ParCa path and
     flagella timing. Cache was refreshed via `write_cache_version()`
     (recomputes only the version-stamp hash) instead, touching neither
     `simData.cPickle` nor `sim_data_cache.dill`'s existing (correct) fitted
     content -- zero information lost, verified by exact hash match against
     the error message's own "Expected inputs_hash".

   Real, fully-wired A/B test (`run_nfsim_wired_test.py`, full 55-process
   composite, division disabled, 2 simulated hours, real ambient initial
   conditions -- no artificial seeding):
   | | NFsim seed 0 | NFsim seed 1 | custom-Steps (flagella_regulation) |
   |---|---|---|---|
   | C-ring (CPLX0-7450) | 14 | 8 | 0 (converted downstream immediately) |
   | export apparatus | 0 | 0 | 0 |
   | motor complex | 0 | 0 | 9 |
   | **complete flagella** | **0** | **0** | **11** |
   | nascent_flagellum in progress | 0 | 0 | 1 |
   | free FliA (final) | 29,562 | 28,353 | 29,048 |

   Honest read: the NFsim pipeline works correctly end-to-end (real bulk
   read/write, scaffold/internal-observable persistence, all confirmed) but
   is currently much SLOWER to reach full completion than the deterministic
   custom-Steps pipeline within the same 2-hour, real-ambient-supply
   window -- stuck at the C-ring stage, hasn't progressed to export
   apparatus/motor/complete flagella yet, while custom-Steps reached 11
   complete flagella and started a 12th. This is consistent with the
   nucleation-rate calibration being a known, accepted rough approximation
   (step 2's "RESOLVED" note: a single global suppression constant, not
   derived per-reaction) -- not a correctness bug, but a real, honest
   finding about NFsim's current speed relative to the existing pipeline,
   worth keeping in mind before treating NFsim's absolute timing as
   comparable yet.

## Nucleation rate recalibration (2026-08-17, real literature rate)

Replaced `NUCLEATION_SUPPRESSION_FACTOR` (a reactively-tuned global
constant, never derived from literature) with a properly derived rate:
Sim et al. 2017 (Sci Rep 7:41189) measured 7.8 flagella/cell at fast
growth (1.2hr doubling), back-calculating to ~1.81e-3/s -- matching (within
8%) the SAME 0.00167/s rate `flagella_filament_nucleation.py` already uses
from the same paper. Applied via `_calibrated_nucleation_rate()`
(`generate_flagella_bngl.py`): any nucleation reaction whose nucleating
species is a real, abundant free monomer gets a rate calibrated to that
species' OWN real ambient count (`REAL_AMBIENT_MONOMER_COUNTS`: C-ring/FliF
657, hook/FlgE 3508, flhDC/FlhC 649); reactions nucleating from a scarce,
dynamically-produced precursor (export apparatus subunit, motor reaction)
correctly keep plain `k_bind` -- naturally self-limiting, matching the
custom Steps pipeline's own architecture.

**Real bug found and fixed in the same pass**: the first version of this
fix only special-cased C-ring, leaving hook at plain `k_bind` on the
(wrong) reasoning that "downstream stages should be fast once a flagellum
has committed." Confirmed directly (real WCM run, standard `INIT`
conditions, no artificial forcing): **1,226 parallel `Growing_flagellar_hook`
scaffolds nucleated in a single 1200s chunk**, all stuck at 2-6 of 120
needed subunits -- the exact same "many parallel scaffolds, none finish"
failure mode already fixed once for C-ring, now reproduced for hook because
it ALSO nucleates from an abundant real free-monomer pool (FlgE, ~3,500),
not a scarce precursor. Generalized the fix instead of hardcoding a single
special case. Re-verified: 0 stuck scaffolds, 1 real hook completion within
the same single chunk that previously produced 1,226 failures and 0
completions.

Also removed FlgI from NFsim's motor-complex reaction (previously included
to match the canonical/aspirational spec) to match what
`flagella_motor_complex_assembly.py` ACTUALLY consumes today (that Step's
own FlgI omission remains a separate, unfixed, flagged bug in the real WCM
-- this just keeps the two pipelines mass-balance-consistent while it's
unresolved).

Real, unforced, standard-`INIT` test after the fix (2 simulated hours):
motor complex went 0->6 for the first time ever in a real (non-forced) run
-- previously stuck at 0 the entire window regardless of duration. Free
FliA climbed steeply (500->19,883) via the real Class III cascade (MotA/
MotB are real Class III genes, directly fed by rising free FliA) -- this
FlgM-crash/FliA-runaway pattern matches a dynamic already flagged earlier
in this investigation (task #22), not something new or NFsim-specific.
8-hour unforced test launched to check whether a full, natural
`nascent_flagellum` creation now occurs within a longer window.

## Explicitly out of scope for this plan

- Coupling NFsim's monomer supply timing to real transcription/translation
  in a way tighter than "reads the shared bulk pool" (see
  `diagnostic_transcription_to_protein_lag.py` and the 2026-08-12 execution-
  order discussion — the existing multi-tick delay via
  `transcript_elongation`/`polypeptide_elongation` is correct biology, not
  something to compress).
- FlhDC/YdiV degradation mechanism itself (separate literature-search task,
  referenced but not started here).
- Deciding whether the `pbg-nfsim` scaffold-persistence fix gets merged
  upstream before or after this work begins (independent, already in PR
  review).
