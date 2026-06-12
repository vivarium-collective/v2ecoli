# Response to the dnaA self-autoregulation handoff (Rashmi → dnaa-4)

**Date:** 2026-06-12 · **From:** Claude (for Eran) · **Re:** `dnaa4_autoregulation_handoff.md`

Your mechanism is **integrated, run, and recorded as dnaa-4**. Headline: **autoregulation
resolves the V-tension** the dnaa-3 V-sweep proved no constitutive V could — it caps the
DnaA peak and dampens the cell-cycle amplitude. Your open question (linear vs Hill) is
**answered with data: Hill is correct.** Details below, including two things you'll want to know.

## What was integrated (your code, applied)
- `dnaa_box_binding.py`: computes the bound fraction `f` of the `POOL_PROMOTER_HIGH` sites,
  publishes it on the `dnaa_hydrolysis` port; `K_d_high` 1→3 nM. (commit `656e4e7`)
- `transcript_initiation.py`: reads `f`, scales the dnaA TU init-probs by `(1 − s·f)` after
  Mechanism-A. (commits `ce532e0`, `caa125a`)
- Made `AUTOREG_STRENGTH`, `HYDROLYSIS_RATE_PER_MIN`, and the repression form env-overridable
  so control/experiment share one code path. Unit-tested. Your branch
  `feat/aim2-dnaa-oric-box-binding` was **not touched** — re-applied from your handoff snippets.

## ⚠️ A correctness bug a review caught in the handoff code
`promoter_init_probs` is a **normalized distribution fed to `np.random.multinomial`** (which
forces the last element to the remainder). Scaling the dnaA TU down *without renormalizing*
dumped the freed probability mass onto an arbitrary last promoter, not onto dnaA-repression.
Small magnitude, but real and non-deterministic. **Fixed** with one renormalize line after the
scaling (`caa125a`); repression of dnaA is preserved, freed RNAP capacity redistributes
proportionally. Worth folding back into your branch.

## The result (V=1.5e-3, k_h=0.025, K_d=3 nM, 16-gen, succinate)

| | Control (s=0) | Linear (1−0.8f) | **Hill (n=4,K=0.5)** |
|---|---|---|---|
| DnaA peak | 1567 | 683 | **635** |
| DnaA gen-mean | 618–1178 | 216–518 | **248–430** |
| ATP-fraction | 0.228–0.443 | 0.066–0.474 | **0.157–0.464** |
| Re-init (oriC>2) | 0 | 0 | 0 |

- **Feedback works:** control over-shoots (peak 1567); autoreg caps it ~2.5× and dampens the
  amplitude. This is the payoff to dnaa-3's "no constitutive V fits."
- **Linear → Hill (your open question):** the linear form over-represses the *trough* (216).
  Hill represses *less* at low f → lifts the trough to 248 and improves ATP-fraction
  (0.066→0.157), exactly as you anticipated. **Hill is the right form.** (`8809706`)

## Two caveats (honest)
1. **The strong V-tension didn't fully reproduce.** I ran on the *merged* current framework
   (the deps bump shifted numerics) and from gen 1 (your gen-3 burn-in dill wasn't on disk).
   In our framework the control's oriC stays ≤2 even at peak 1567 — **no re-initiation**, even
   without autoreg — so the dramatic re-init rescue your handoff predicted isn't demonstrable
   here. The peak-capping + amplitude-dampening *is*.
2. **Not yet band-centered.** Hill's trough sits ~50 below the 300 floor; ATP-fraction min
   ~0.04 below 0.2. This is **K/s/k_h calibration, not a mechanism question** — the next step
   is a small sweep (e.g. Hill K≈0.65 to lift the trough further, k_h≈0.035 for ATP-fraction).

## Recorded
dnaa-4 study.yaml carries the 3 runs (control / linear / Hill, V=1.5e-3) + verdict
(*supported-with-calibration-pending*) + charts (`dnaa4_pool_band`, `dnaa4_promoter_swing`).
Reproduce: `DNAA_AUTOREG_STRENGTH=0.8 DNAA_AUTOREG_FORM=hill DNAA_HYDROLYSIS_RATE_PER_MIN=0.025
… --perturbation TU00259[c]=1.5e-3`.
