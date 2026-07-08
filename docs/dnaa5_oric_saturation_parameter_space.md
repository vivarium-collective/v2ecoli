# DnaA-oriC saturation — parameter space for tuning cooperativity

This is a parameter-space map for fine-tuning whichever cooperativity form
(Hill on DnaA-ATP, K_d-reduction by occupancy, or something else) is being used
to drive chromosome initiation. The point is the *categories of knobs* that
need attention together — implementation specifics will differ across branches.

**Goal:** at steady-state, the oriC cluster fills to **full saturation exactly
once per cell cycle**, fires initiation, and resets. No second filling, no
daughter-cluster re-saturation, no skipped cycles.

Outputs you want held in band over a long lineage:

- **roughly constant total DnaA level across steady-state generations** (no monotonic drift)
- total DnaA in [300, 800], ATPfr in [0.2, 0.5]
- cycle τ < cell-doubling time (no overlapping cycles)
- bulk DnaA-ATP stable from gen to gen

**What this document is for:** finding the parameter combination across the
axes below that simultaneously delivers all of those outputs. No single knob
solves it on its own — the cooperativity functional form, the DnaA supply,
the autoregulation strength, and the hydrolysis rate are all coupled, so
fine-tuning means moving them together. The sections below lay out each axis
with what it does, its starting values, and how to think about moving it.

---

## 1. DnaA supply

| Knob | Effect |
|---|---|
| dnaA promoter strength (perturbation V) | sets baseline dnaA mRNA → DnaA synthesis. Single most impactful lever on bulk DnaA + total DnaA. Applied as a multiplier on transcription initiation probability at the dnaA TU. |
| dnaA translation efficiency | multiplies protein output per dnaA mRNA. Orthogonal to V — same total DnaA can come from many mRNAs × low TE or few mRNAs × high TE. |

V is typically tuned in the 1e-3 to 2.5e-3 range to land totA in [300, 800].
Too low → starvation; too high → bulk runaway. The right V is coupled to
autoregulation strength (§2) — the two have to be tuned together.

---

## 2. Autoregulation

DnaA represses its own promoter via DnaA-bound boxes in the dnaA promoter
region. Two pieces: the functional form (linear vs. Hill) and the strength
parameter that scales the repression.

| Form | Equation | Notes |
|---|---|---|
| **Linear** | scaled_prob = base × (1 − s · P_DnaA) | one parameter (s); simple; what this branch uses |
| **Hill** | scaled_prob = base × (1 − s · P_DnaA^n / (K^n + P_DnaA^n)) | three parameters (s, n, K); sigmoidal in DnaA promoter occupancy |

where `P_DnaA` is the fraction of dnaA promoter sites bound by DnaA (in 0..1).

**Autoregulation strength is the primary lever for keeping total DnaA constant
across gens.** Too weak → totA drifts up across generations; too strong →
DnaA pool gets clamped too aggressively and the cluster can't refill between
cycles. Strength of ~0.6 (linear) is a reasonable starting point.

---

## 3. Cooperativity functional form at oriC

This is where the bulk of the fine-tuning attention should go. Whatever
form is chosen, it has to deliver "fills once, resets once" within a single
cell cycle.

**Two valid framings are being tried in parallel:**

- **Hill function on free DnaA-ATP concentration** — classical formulation
  (dashboard / Eran's implementation). Cluster saturation is sigmoidal in bulk
  DnaA-ATP.
- **K_d reduction by occupancy** — neighbor-influence on each site's
  dissociation constant (this branch). K_d at each site drops as local
  occupancy on the same chromosome rises.

Per Haochen, both are legitimate; the underlying parameter-space structure is
the same either way.

**Suggested escalation path (Haochen) for whichever form is in use:**

| Shape | Description | When to try |
|---|---|---|
| **Linear** | linear from "no-help" endpoint to "max-help" endpoint across the occupancy / concentration axis | start here |
| **Quadratic** | quadratic in occupancy / concentration | if linear is too weak / cluster never fills |
| **Hill** | sigmoidal | if quadratic still doesn't give clean once-per-cycle saturation |

Two static parameters span the envelope regardless of form:

- low-affinity endpoint (no help): currently **100 nM** at oriC_low. Sets nucleation difficulty.
- high-affinity endpoint (max help): currently **1 nM** at oriC_low. Sets the floor on tightening.

**Within a chosen form, scan its shape parameters too** — don't fix them at a
single guess. For example, for a Hill function, scan **both** the Hill
coefficient `h` (transition sharpness) **and** `K` (transition midpoint).
Different (h, K) combinations give very different cluster dynamics even at the
same low/high endpoints.

---

## 4. Optional dynamic kick

Static K_d (or Hill) sometimes isn't enough on its own — the cluster can stall
mid-fill if local DnaA-ATP supply is borderline. A "stuck-time" mechanism that
gives the K_d a small temporary boost when occupancy plateaus can help.

If used, the things to think about are:

- **stuck-time threshold** — how long without progress before the kick fires
- **kick magnitude** — how much the K_d is allowed to drop
- **nucleation guard** — require ≥1 site already bound before the kick fires (else there's no cooperativity to amplify)
- **bulk-DnaA-ATP gate** — only kick when bulk DnaA-ATP is meaningful (prevents post-init daughter clusters from firing on crashed bulk)
- **persistence** — does the kick commit once fired, or reset on any upward progress?

All of this is optional — the static functional form (§3) should be tried alone
first.

---

## 5. Equilibrium solver

At every tick the model has to solve for the DnaA distribution across (free,
ATP-bound, ADP-bound, apo-bound) at every K_d-governed site. This step runs
between the upstream events (synthesis, hydrolysis, fork release) and the
downstream readouts (cluster occupancy, initiation decision).

| Component | Role |
|---|---|
| K_d at each site | sets the equilibrium ratio of bound:free at that site for each form |
| total DnaA conservation | sum across pools = total DnaA pool at that tick |
| ATP / ADP / apo partitioning | per-form K_d (typically apo ≈ ATP ≈ ADP at oriC_high, ATP-only at oriC_low) |
| solver method | analytic, fixed-point iteration, or root-finding |

What matters for tuning: the solver has to reach equilibrium each tick and
deliver a sensible DnaA distribution. Fixed-point methods can stall or
oscillate; root-finding (e.g. `scipy.root`) is more robust. A damped
fixed-point iteration is known to give wrong steady states under fast
DnaA-pool changes — verify whichever solver is in use against a hand-computed
equilibrium for a few static configurations.

---

## 6. Initial state

Cold-start runs have a multi-generation transient before DnaA + cell mass
settle. Burned-in initial states from a prior steady-state lineage are
preferable for testing cooperativity dynamics. Validating the same config from
multiple starting points helps rule out initial-state dependence.

---

## 7. Stochastic seed

Massive variance across seeds. Same configuration can run cleanly under one
seed and crash early under another. Validate across ≥3 seeds before claiming a
config "works."

---

## Failure modes to watch for

1. **DnaA starvation** — bulk DnaA-ATP crashes to <1 nM, cluster never fills, cycle τ blows up, cell stops dividing. (V too low, or autoreg too strong.)
2. **Bulk runaway** — bulk DnaA-ATP creeps up generation-over-generation; daughters fire cluster within their first 10 min; multiple inits per cycle. (V too high, or autoreg too weak.)
3. **totA drift** — total DnaA grows or shrinks monotonically across gens instead of holding steady. Symptom of mismatched V vs. autoreg.
4. **Metabolic crash** — `NegativeCountsError` on `PROTON[c]` from the allocator. Typically downstream of one of the above (cycles stretched too long).
5. **Cluster lock-in** — once cooperative help fires, the cluster stays at full saturation the rest of the cycle and constantly leaks DnaA-ATP into bulk. Carry-forward persistence is the relevant knob.
6. **Daughter firing** — after parent oriC fires, daughter clusters refill and trigger a second cluster event in the same gen. Bulk-DnaA-ATP gating and nucleation guard prevent this.

---

## Knobs ranked by effect size (rough)

1. **DnaA supply (V)** — single biggest dial; sets everything downstream
2. **Autoregulation strength** — primary lever for constant totA across gens; long-term drift control
3. **Cooperativity functional form (§3)** — linear vs. quadratic vs. Hill changes how aggressively cluster saturates as it fills
4. **Bulk-DnaA-ATP gate on dynamic kick** (if used) — biggest single lever against daughter firing
5. **Carry-forward persistence on dynamic kick** (if used) — whether cluster locks at full once fired
6. **Stuck-time threshold on dynamic kick** (if used) — kick frequency
7. **Cooperative envelope (low-affinity / high-affinity endpoints)** — minor unless extreme
8. **Translation efficiency** — orthogonal to V, similar effect
