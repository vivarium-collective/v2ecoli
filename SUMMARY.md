# Hardening Plan C — parameter-uq: deep-param injection adapter + param-uq-04

Branch `harden/paramuq-deep-params`. Answers the investigation's one open decision
— *"Expand to the DEEP sim_data physiological parameters (ParCa-level injection)?"* —
with **YES: built it, ran it, it passes.**

## What was delivered

1. **Deep sim_data injection adapter** — `pbg_v2ecoli/uq_sim_data_injection.py`.
   Bridges pbg-uq's sampling (`CallableAdapter` + `ForwardUQ`) to the in-memory
   `sim_data` a v2ecoli run consumes. Each deep param declares one of two routes,
   decided empirically and documented in the module docstring:
   - **POST_PARCA** — the param surfaces as a runtime `configs` key → inject via
     `config_overrides` on the shared bundle. **No rebuild.**
   - **REBUILD** — the param shapes `initial_state`/config projection (a config
     override is inert) → deep-copy the fitted `sim_data`, mutate the attribute,
     regenerate the bundle with `save_sim_input`.

2. **param-uq-04-deep-params run** — `workspace/studies/param-uq-04-deep-params/`
   `artifacts/` (sobol_results.json, per-seed sample npz, run + figure scripts) and
   `charts/deep_param_sobol.png` (real data, not the old mockup).

3. **Decisions filled** — investigation `decisions_needed` → `decisions_resolved`;
   study.yaml verdict/gate/statuses updated to PASSED; param-uq-03's `(pending)`
   fields filled honestly (param-uq-04 does **not** unblock it).

## Headline results (order-2 PCE, 26+8 samples × 2 seeds, 150-step single cell)

Total-order Sobol, mean [min,max] across seeds:

| observable | cell_dry_mass_fraction (rebuild) | rnap_elongation_rate (post-ParCa) | kinetic_objective_weight (post-ParCa) | PCE test err |
|---|---|---|---|---|
| dry_mass  | **1.00 [1.00,1.00]** | 0.00 | 0.00 | 0.0% |
| cell_mass | **1.00 [1.00,1.00]** | 0.00 | 0.00 | 0.1% |
| growth rate | 0.52 [0.21,0.82] | 0.50 [0.25,0.75] | 0.11 | 1.6% |

- **Gate PASSES**: a single deep sim_data param (dry-mass fraction) dominates MASS
  variance (S_T ≈ 1.0, stable). Max deep-param S_T = 1.0 (primary test `in_range[0.5,1.0]`).
- **Growth rate** is a co-dominance: dry-mass fraction and RNAP elongation rate ~0.5 each.
- FBA kinetic-objective weight is **inert** — a genuine physiological null control.
- **100% crash-free** (68/68 samples).

## Two premises of the original PIVOT were wrong (and it's good news)

- The "rebuild" is a **~2 s `save_sim_input` bundle regeneration** (reuses the fitted
  sim_data, no ParCa refit) — **not** the ~2.5-min full ParCa the study budgeted for.
  68 samples ran in 19.5 min. The deep physiological surface is now cheaply sweepable.
- **Not all deep params need a rebuild**: RNAP elongation rate and the FBA weights are
  reachable POST_PARCA via `config_overrides`. Only params read at bundle-build time
  (dry-mass fraction) need the rebuild path.
- One deep param is **MASKED**, not reachable post-ParCa: `rnap_active_fraction`
  (`fracActiveRnapDict`) is recomputed each tick from ppGpp with `ppgpp_regulation`
  on, so a static-dict override is inert. Substituted RNAP *elongation rate*
  (live) and documented the masking.

## Caveats
- Proof-of-capability budget: 3 deep params, 150-step single-generation window. Mass
  observables move mainly through the initial mass partition here; multi-generation
  runs would let dynamic (transcription/translation) params accumulate.
- The rebuild path captures a param's **direct** initial-state/config effect, not
  ParCa refit feedback — the standard forward-UQ-on-sim_data assumption (uqEcoli's
  `sim_data_setattr`).
- uqEcoli bounds adopted as-is (nominal values match); not re-calibrated to v2ecoli's
  current ParCa state — an open follow-up, along with the full deep-param sweep.

## Environment note (for the driving session)
This worktree's v2ecoli (origin/main) needs a newer **bigraph-schema** (the
`bigraph_schema.contract` module, pinned at commit `4b208e13` in `uv.lock`) than the
1.4.2 installed in `~/code/v2ecoli/.venv`. Resolved **non-invasively** — a pinned-commit
git worktree at `/tmp/bgs-pinned` prepended to `PYTHONPATH`; no venv package was changed:

```
PYTHONPATH=/tmp/bgs-pinned:~/code/v2e-hparamuq ~/code/v2ecoli/.venv/bin/python ...
```

The venv would need `bigraph-schema` synced to the locked commit for a clean run
without the PYTHONPATH shim. `pbg_uq` was present and editable-installed as expected;
`out/cache` symlink to the canonical ParCa cache was used read-only (all rebuilds
wrote to fresh temp dirs).

## Reproduce
```
cd ~/code/v2e-hparamuq
PYTHONPATH=/tmp/bgs-pinned:~/code/v2e-hparamuq ~/code/v2ecoli/.venv/bin/python \
  workspace/studies/param-uq-04-deep-params/artifacts/run_param_uq_04.py
PYTHONPATH=/tmp/bgs-pinned:~/code/v2e-hparamuq ~/code/v2ecoli/.venv/bin/python \
  workspace/studies/param-uq-04-deep-params/artifacts/make_figure.py
```
