# Scoping — DnaA negative autoregulation (next study)

Drafted 2026-06-10, off the back of: dnaa-1/dnaa-2 V tension + per-gen drift,
the adopted box-binding (PR #162), and the verified autoregulation gap.

## The main question

**Can DnaA negative autoregulation hold the DnaA pool self-stabilized in the
[300,800] band across generations WITHOUT the manual V override (Mechanism A) —
by wiring DnaA-ATP repression of its own promoter into the transcription
recruitment — and does this dissolve the V tension (dnaa-1 wanted 0.8e-3,
dnaa-2 needed ~1e-3) and the per-gen drift at their root?**

## Why this is the key question

DnaA represses its own transcription (binds the dnaA promoter → down-regulates
dnaA). This homeostatic loop is what sets the DnaA level + initiation timing.
Across dnaa-1/2 we've been hand-setting the level with a fixed V override
(TU00259[c]=V, ~100x the natural basal) precisely BECAUSE this feedback is
missing — so the pool drifts (no restoring force) and no single constitutive V
closes both the in-band pool AND the ~1/min init rate. Working autoregulation
would make the cell find its own level → plausibly closing both at once.

## What we verified (the precise gap)

- The recruitment IS dynamic at runtime (transcript_initiation.py:21):
  `p_i = max(0, basal_prob_i + Σ_j delta_prob[i,j]·bound_TF_j)`.
- BUT the dnaA promoter has NO DnaA-ATP recruitment edge: `target_tf[TU00259[c]]
  = None` and `delta_prob[TU00259[c], MONOMER0-160] = 0`. So there is no path for
  the feedback to flow, and the self-edge fit to zero.
- The genetic_perturbations V override additionally REPLACES the dnaA synth prob,
  bypassing the recruitment entirely.
- (A quick stopgap — patch basal+delta_prob post-hoc — FAILED: DnaA collapsed to
  ~17 because the patch didn't propagate to the cache's basal_prob param AND
  there's no bound_TF edge at the dnaA promoter. So the real work is wiring the
  edge, not a one-liner.)

## Suggested mechanism (design options)

1. **Wire the recruitment edge** (required): register DnaA-ATP (MONOMER0-160) as a
   TF on the dnaA promoter (TU00259[c]) with a NEGATIVE delta_prob, so
   `bound_TF[DnaA-ATP at dnaA promoter]` modulates dnaA synthesis. Two sources for
   that bound_TF fraction:
   - **(a) existing TF machinery** — use the standard p_promoter_bound for
     DnaA-ATP (the recruitment's own equilibrium). Lightest; reuses ParCa's TF path.
   - **(b) PRINCIPLED — couple to the adopted box-binding** — drive the bound_TF
     fraction from Rashmi's now-adopted **dnaA-promoter box occupancy** (the 2
     promoter boxes). This unifies the box-binding adoption with the regulation:
     DnaA bound to the dnaA-promoter boxes → represses dnaA. Strongly preferred.
2. **Calibrate the magnitudes**: set the basal (max unrepressed) synth + the
   repression strength (anchored on the −2.31 log2 FC ≈ 5x) so the feedback
   equilibrium lands DnaA in [300,800]. A small 2-D sweep (basal × strength).
3. **Drop the V override** for this study (the whole point — let feedback set the
   level). Keep V available as a fallback/comparison.

## Acceptance criteria (proposed)

- DnaA pool self-stabilizes WITHIN [300,800] across ≥6 generations with **no V
  override** (the thing a constitutive V couldn't do steadily).
- **Per-gen drift reduced** vs the fixed-V runs (which ramped 73→880).
- Clean cell cycle preserved (oriC 1↔2, one init/gen, periodic mass).
- STRETCH: closes the dnaa-1 tension — in-band pool AND ~1/min dnaA init rate
  simultaneously (no constitutive V achieved both).

## First concrete steps

1. **Check the binding substrate**: confirm whether DnaA-ATP binding to the dnaA
   promoter is observable (her dnaA-promoter box occupancy from the adopted run) —
   this is the signal that would drive the feedback. (We already emit
   `promoter_high_bound_*` in her listener.)
2. **Wire option (a) first** (cheapest test): add the DnaA-ATP→TU00259 negative
   recruitment edge in ParCa (so it survives the fit + propagates to the cache),
   calibrate basal, drop V, run → does DnaA self-stabilize in band?
3. If (a) works → then **(b)** couple it to the box occupancy for the principled
   version. If (a) doesn't engage (bound_TF stays ~0) → go straight to (b).

## Where it lives

This is the regulation/initiation mechanism — fits **dnaa-04
(mechanistic-initiation)** reframed as the autoregulation study, or a dedicated
dnaa-09-autoregulation. It builds directly on the adopted box-binding (dnaa-3).
