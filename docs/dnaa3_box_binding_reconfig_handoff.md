# Study 3 (dnaa-3-box-binding) — reconfigure to use the dnaa-4 baseline

## For the dashboard agent

**Action requested**: edit `studies/dnaa-3-box-binding/study.yaml` (or whichever
file defines Study 3's sim configuration) to use the dnaa-4 reference
configuration listed under "Reconfigure path (preferred)" below, then re-run
Study 3.

Specifically:
1. Update the study's code commit reference to `eb4ea39` on
   `feat/aim2-dnaa-oric-box-binding`.
2. Build the cache `cache_dnaa2_v1.5e-3_kd3nm_apoATP_kinetic` by applying
   two patches to the dnaa-2 cache (see the **Cache patches** table below):
   the V perturbation and the F-05 `integrate_dt_mask[24] = True`.
3. Re-run Study 3 with that cache, seed = 1, generations = 12, resuming from
   `out/steady_state_inputs/succinate_default_gen3_start_dnaa3.dill`.

Use the `/pbg-*` skills for any study.yaml edits rather than hand-editing.

4. **Run the Study 3 tests** after the re-run completes (POST `/api/study-tests-run`,
   not `/api/run-tests`) and update the study's `runs[].outcomes` and verdict
   pills accordingly. With the dnaa-4 reference configuration, all six tests
   are expected to pass (the V-tension drift test that was failing with V=1e-3
   is exactly what autoregulation was scoped to resolve).

After re-running, the binding readouts should match those already produced
inside Study 4 (search for *"dnaa-3 — DnaA-box binding readouts on the dnaa-4
reference config"* in the latest investigation HTML).

## Context

In the current `investigation-dnaa-replication-2026-06-17-2.html`:

| study | name | status |
|---|---|---|
| Study 3 | dnaa-3-box-binding | ⛔ Blocked / ❌ Failing (5/6 tests pass) |
| Study 4 | dnaa-4-autoregulation | ✅ Passed (Preliminary, 20 runs) |

Study 3's failure is **not** in the binding mechanism — it's a configuration
issue. Study 3 is currently pinned to:

- Cache: `cache_dnaa2` (the dnaa-2 / V=1e-3 baseline)
- V = 1e-3 (Mechanism A perturbation)
- No F-05 (apo+ATP charging is fast-equilibrium)
- k_h = 0.046 / min (Sekimizu 1987 default)

That configuration has a known V-tension: the DnaA pool drifts above the
[300, 800] band in later generations (gens 6-8 reach 931-1077). This is the
exact failure that dnaa-4 autoregulation was scoped to resolve — and it does,
on the autoregulation-enabled config.

Notably, the dashboard agent **already** ran the box-binding readouts on the
dnaa-4 reference config and reported them inside Study 4 (search for the
section labelled *"dnaa-3 — DnaA-box binding readouts on the dnaa-4 reference
config (linear autoregulation s=0.6 + F-05, V=1.5e-3, k_h=0.025/min)"*). So
the box-binding work was reproduced; it just lives in the wrong study panel.

## What we want

Get Study 3 to **re-run on the dnaa-4 reference config** so its panel reports
the binding result on the autoregulation-stabilized lineage (which doesn't
drift). This unblocks Study 3 cleanly and lets Study 4 focus on the
autoregulation-specific outcomes.

## Two equivalent paths

**Either** edit Study 3's configuration to point at the dnaa-4 cache + commit
(preferred — keeps the biology-correct ordering of binding → autoregulation),

**OR** swap Studies 3 and 4 in the investigation order (autoregulation first,
then box-binding observes the autoreg-stabilized baseline). This is more
mechanical but disrupts the existing roadmap.

## Reconfigure path (preferred)

Update Study 3's `study.yaml` (or whichever file defines the study's sim
configuration) to use:

| field | new value |
|---|---|
| code commit | `eb4ea39` on `feat/aim2-dnaa-oric-box-binding` (already pushed) |
| cache | `cache_dnaa2_v1.5e-3_kd3nm_apoATP_kinetic` (built from V + F-05 patches; see below) |
| `KD_HIGH_M` | 3 nM (chromosomal_high, oriC_high, promoter_high — already in `eb4ea39`) |
| `HYDROLYSIS_RATE_PER_MIN` | 0.025 / min (in `dnaa_box_binding.py` — already in `eb4ea39`) |
| `_HYDROLYSIS_RATE_PER_SEC` | 0.025 / 60.0 (in `equilibrium.py` — already in `eb4ea39`) |
| `AUTOREG_STRENGTH` (s) | 0.6 (linear, in `transcript_initiation.py` — already in `eb4ea39`) |
| V (Mechanism A) | 1.5e-3 (cache patch — `configs["ecoli-transcript-initiation"]["perturbations"]["TU00259[c]"] = 1.5e-3`) |
| F-05 (apo+ATP kinetic) | `integrate_dt_mask[24] = True` in `configs["ecoli-equilibrium"]["fluxesAndMoleculesToSS"]["_data"]` (cache patch) |
| Starting dill | `out/steady_state_inputs/succinate_default_gen3_start_dnaa3.dill` |
| Seed | 1 (the seed used for the autoreg result) |
| Generations | 12 on succinate |

The two patches above (V + F-05) are sim_data patches written into the cache
pickle at build time. They are **not** in the source commit — applying them
is part of the cache-build step. See the dnaa-4 autoregulation handoff at
`docs/dnaa4_autoregulation_handoff.md` for the same recipe in detail.

## Why Study 3 should keep its position (binding → autoreg, not the other way)

Biologically, **binding is prerequisite to autoregulation**: autoregulation
*reads* DnaA-promoter occupancy. The promoter occupancy doesn't exist without
the box-binding mechanism. Putting binding first matches the causal arrow
in the model.

The current Study 3 failure is a configuration mismatch, not a violation of
that ordering. Reconfiguring Study 3 to the dnaa-4 baseline lets the binding
mechanism be observed on a stable cell cycle, then Study 4 layers the
autoregulation feedback on top of that (a vertical extension of the same
configuration, not a replacement).

## Swap path (workaround if reconfiguring is harder)

If the study YAML edit is more work than reordering, swapping the two studies
in the investigation roadmap gets the same end state:

- Study 3 becomes "Test Autoregulation" (formerly Study 4) — passes
- Study 4 becomes "Evaluate Chromosomal Binding" (formerly Study 3) — now
  passes because it inherits the autoreg-stabilized baseline

The science is preserved either way. The reconfigure path is cleaner; the
swap is more mechanical.

## What's pushed and where

- Commit: `eb4ea39` (`feat(dnaa-4): dynamic dnaA autoregulation + K_d/k_h refinements`)
- Branch: `feat/aim2-dnaa-oric-box-binding` (on remote `origin`)
- Pushed 2026-06-15
- Status: clean fast-forward on top of `c648d51`
- The cooperativity work (dnaa-5) is still local-only and not relevant to
  this Study 3 reconfiguration.
