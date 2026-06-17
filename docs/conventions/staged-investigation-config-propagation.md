# Config propagation in staged investigations

> Kills the *"why is the earlier study failing / showing stale plots when the
> mechanism is fine?"* confusion. When a later study changes the reference
> configuration, the earlier studies that share its lineage must be
> re-evaluated on that new config — or they read as broken for the wrong reason.

## The trap

A pbg investigation is built study-by-study, each adding a mechanism on top of
the previous baseline (e.g. dnaa-0 baseline → dnaa-1 expression → dnaa-2
hydrolysis → dnaa-3 box-binding → dnaa-4 autoregulation). The **biological
ordering** is fixed (binding is prerequisite to autoregulation — autoregulation
*reads* promoter occupancy). But the **configuration dependency often runs the
other way**: a *downstream* study can establish the corrected reference config
that the *upstream* study should actually be observed on.

dnaa-4 is the worked example. dnaa-3 (box-binding) was originally pinned to the
fixed-V baseline (V=0.8e-3/1e-3, no F-05, k_h=0.046). That config has a known
**V-tension**: the DnaA pool drifts above the [300,800] band in later
generations — which is *exactly* the failure dnaa-4 autoregulation was scoped to
resolve. So dnaa-3 "failed 5/6 tests" not because its mechanism was wrong, but
because it was evaluated on a config that predates the fix. The result was a
study that looked **blocked/failing** while the science was correct.

Two failure surfaces it produced:

1. **False test failures.** The V-tension drift test failed on the stale config;
   it passes on the dnaa-4 reference (linear s=0.6 + F-05, V=1.5e-3, k_h=0.025).
2. **Provenance/chart drift.** Across repeated reconfigurations, *some* fields
   were updated (charts, verdict, gate) while others were left stale
   (`enforced_params` still said V=0.8e-3; three runs were flagged
   `canonical: true`; charts and tests cited different runs). A reader can't tell
   which config the study is really on.

## The rule

**When a study establishes a new reference configuration, propagate it to every
upstream study that shares its lineage, and re-evaluate them on it.** Don't leave
an earlier study pinned to a config a later study has already corrected.

Concretely, when you adopt or change a reference config, for each affected study:

- [ ] **`enforced_params`** declares the *current* config (commit, V, K_d, k_h,
      autoreg form/strength, cache patches). No stale knob values. If the real
      config lives in a cache patch (not the source commit), say so — see
      [`run-provenance.md`](run-provenance.md).
- [ ] **One `canonical: true` run** per study, and it is on the current config.
      Demote historical/superseded runs to `canonical: false` with a one-line
      reason. (Multiple canonical runs = ambiguous provenance.)
- [ ] **Tests are re-run on the current config** (`/api/study-tests-run`) and
      `runs[].outcomes` recorded from that run — not carried over from the old
      config.
- [ ] **Charts render from the current-config run.** If you reproduce an
      upstream study's readout on the new config, the figure belongs in the
      **upstream study's** panel, not the downstream one that happened to
      generate it first.
- [ ] **The verdict/prose says which config the study is on** and frames the old
      config as history (the "journey"), not as the current state.

## Biology order vs config baseline — keep them separate

Reconfiguring an upstream study to a downstream reference does **not** reorder
the investigation. dnaa-3 stays before dnaa-4 (binding → autoregulation, the
causal arrow). dnaa-3 is simply *observed on* the autoregulation-stabilized
baseline; dnaa-4 then layers the feedback-specific outcomes on top of the same
config. Reordering the studies to dodge a config mismatch is a workaround, not a
fix — it scrambles the causal narrative to paper over a provenance gap.

## Smell tests (catch it early)

- An earlier study is "failing" a test that a later study was explicitly scoped
  to fix → it's probably on the pre-fix config. Re-evaluate, don't debug the
  mechanism.
- A study's `enforced_params` names a different V / K_d / k_h than its verdict or
  its canonical run → stale config declaration.
- More than one `canonical: true` in `runs:` → reconcile to one.
- The same readout figure appears under two studies → it belongs to one; move it
  to the upstream study it actually characterizes.
