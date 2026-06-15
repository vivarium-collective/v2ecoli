# Response to the dnaA self-autoregulation handoff (Rashmi → dnaa-4)

**Date:** 2026-06-12 · **From:** Claude (for Eran) · **Re:** `dnaa4_autoregulation_handoff.md`

Your mechanism is **integrated, run, and recorded as dnaa-4**. Headline: **autoregulation
resolves the V-tension** the dnaa-3 V-sweep proved no constitutive V could — it caps the
DnaA peak and dampens the cell-cycle amplitude. Your open question (linear vs Hill) is
**answered with data: Hill is correct.** Details below, including two things you'll want to know.

---

## Update — 2026-06-14: parameters, tests, and box-binding under the autoreg expression

You asked three things. All three are addressed; honest status below.

**(1) "Which parameters were used — V, DnaA-ATP/ADP binding K_d, hydrolysis rate, box-binding K_d?"**
Now documented in the dnaa-4 study (`model_settings → canonical-run-parameters`) so the report carries them. The canonical s=0.7 runs (`dnaa4_s07_seed{0,1,2}`, succinate, 16 gen):
- **dnaA expression V = 1.5e-3** (`TU00259[c]` synth-prob override). Note: the runs use this *fixed base V* and the autoregulation scales it DOWN — i.e. "base V × Hill feedback", not the pure no-V the design first aimed at.
- **DnaA-box K_d:** high-affinity boxes (chromosomal 302, oriC-high 3, dnaA-promoter 2) = **3 nM** for *both* DnaA-ATP and DnaA-ADP; oriC-low (8 sites) = **100 nM**, DnaA-ATP only.
- **DnaA-ATP intrinsic hydrolysis rate = 0.025 /min.**
- **Autoregulation:** Hill, **s = 0.7, K = 0.5, n = 4**.

**(2) "Run the tests."**
Done (evaluator on the s=0.7 runs): **cell-cycle-preserved PASS** (robust — all 3 seeds, oriC ≤ 2, 0 re-inits); **DnaA-pool-in-band PARTIAL** (2/3 seeds hold the band across every full generation g3–16; the strict per-gen check fails only on the truncated final gen + seed0's noise). Honest read: **mechanism validated, but the band is PARTIAL — not a clean pass.**

**(3) "If it passes, redo the box-binding dynamics study on the achieved expression."**
Because the band is PARTIAL (not a clean pass) and dnaa-4 **already runs box-binding *under* the autoregulated expression**, we answered this by analyzing the box dynamics in the existing s=0.7 runs rather than launching a fresh dedicated study (Eran's call). Result is a positive one: under autoregulation the **oriC-low over-binding is further resolved** — low-affinity (100 nM) oriC occupancy is **0.10–0.28** (was ~0.8 read-only, ~0.46 adopted), because feedback lowers DnaA and depletes the free DnaA-ATP that drove over-occupancy; the high-affinity boxes stay filled (0.67–0.95). Recorded as finding **F-04** + chart **`dnaa4_box_dynamics`**. If you'd prefer a *fresh, dedicated* box-binding study on this expression (rather than the existing-run analysis), say so and we'll run it — and if you want the band tightened to a clean pass first, that's a small s/seed sweep.

---

## Update — 2026-06-13: your s=0.7 round (V=1.5, K=0.5, n=4, multi-gen)

You asked to **run V=1.5 + Hill K=0.5 + s=0.7 + n=4 for multiple generations**, then continue
a lineage from a steady-state `.dill` to confirm stability. Done — and **s=0.7 is the answer
to band-centering**. I ran it across **3 seeds** for robustness (your "advance this" ask):

| seed | DnaA gen-mean (g3–16) | peak (<800) | oriC / re-inits | in-band? |
|---|---|---|---|---|
| 0 | wanders 187–489 | 692 | 2 / 0 | most gens; stochastic dip to ~187 at g11–12 |
| **1** | **333–487** | **655** | 2 / 0 | **every full gen g3–16** |
| **2** | **355–594** | **759** | 2 / 0 | **every full gen g3–16** |
| control | 618–1178 | 1567 | 2 / 0 | no (runaway) |

- **s=0.7 centers the band.** Seeds 1 & 2 hold DnaA inside [300,800] across *every full
  generation* — better than s=0.6 (trough-dips) and s=0.8 (low setpoint). The only sub-300
  points are the *partial* final gen g17 (truncated) and seed0's noise.
- **Peak-cap (<800) and cell-cycle (oriC≤2, 0 re-inits) are ROBUST across all 3 seeds.**
- **seed1 was the cleanest** — and it's the one that needed the fix below.

### A baseline bug your run surfaced — now fixed (#209)
seed1 first **crashed**: `PROTON[c] NegativeCountsError` ~gen 3 — an allocator over-draft
after an FBA `GLP_NOFEAS` infeasibility tick. It's **platform-sensitive** (crashed on the
mini, ran clean on the laptop) and is a **baseline-model robustness defect, not an
autoregulation issue.** Fixed in **v2ecoli #209** (`resolve_overdraft` clamps the over-draft
and warns instead of crashing the lineage; behavior-neutral, byte-identical parity). With it,
seed1 completed all 16 gens (6 graceful clamps, 0 crashes).

### The resume-from-dill stability check — blocked by a tooling bug
I could **not** complete the "continue a lineage from a steady-state gen-10 dill" step: the
`--resume-dill` path produces a daughter that **never re-initiates replication** (oriC pinned
at 2, grows to ~2× mass, never divides) — reproduced across **two seeds**, while the
*in-process* continuous divide of the identical state divides fine. So it's a **dill-roundtrip
tooling bug** (the serialized `cell_state['unique']` loses replication-init state), **not
instability**. Write-up + a no-sim diagnostic: `docs/resume_dill_replication_init_bug.md`.
**The multi-generation stability you wanted is already evidenced by the continuous 16-gen runs**
(clean, every gen divides); the dill-resume confirmation is deferred until that tooling is fixed.

Reproduce s=0.7: `DNAA_AUTOREG_STRENGTH=0.7 DNAA_AUTOREG_FORM=hill DNAA_HYDROLYSIS_RATE_PER_MIN=0.025
… --perturbation "TU00259[c]=1.5e-3" --seed {0,1,2} --generations 16`.

---

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
