# Root cause (branch `harden/143-o2-rootcause`)

**TL;DR — not an instrumentation bug. The low O₂ is real wcEcoli-FBA behavior; the −40 % O₂
(v2 vs v1) is an averaging-window fragility of a front-loaded flux, not a localized respiratory
defect. Closable as "real, understood."**

### What I checked (fresh seed-0 baseline, `build_composite("ecoli_baseline")`)

1. **Instrumentation ruled out.** O₂ and CO₂ are *unconstrained* in the media (freely available,
   not O₂-limited), and the `listeners.fba_results.external_exchange_fluxes` O₂ equals the net O₂
   consumed by the internal reactions read from `reaction_fluxes` (mass-balanced). The listener
   does **not** under-report O₂. The report-card extraction (`card_vectors.extract_vectors`) is
   also symmetric v1↔v2 — same units (mmol/gDCW/h), same sign (uptake −), same burn-in. So it is
   neither a listener bug nor a units/sign/window comparison bug.

2. **The exchange fluxes are bimodal within a generation.** A ~2500-tick run shows:

   | phase | ticks | O₂ | CO₂ | RQ |
   |---|---|---|---|---|
   | early | 0–440 | **−1.31** | 0.00 | 0 (respiratory burst) |
   | late  | 460–2520 | **−0.005** | **+1.94** | ~400 (decarboxylation, no O₂) |

   The cell grows and consumes glucose throughout, but for ~90 % of the generation it meets ATP
   demand glycolytically and balances redox reductively — so it barely respires while still
   emitting biosynthetic CO₂. **O₂ is front-loaded; CO₂ is back-loaded.** The generation
   time-average (≈ −0.28 O₂ / +1.6 CO₂ for one seed) matches the report-card v2 numbers.

3. **Why −40 % O₂ ≫ −20 % CO₂.** Because O₂ is front-loaded, its per-cell time-average is highly
   sensitive to generation length (truncating at tick 1600 vs 2520 changes mean O₂ by **−36 %**
   but mean CO₂ by **<1 %**). v2's ensemble has longer / non-dividing generations (this is #142 —
   24/256 cells hit the 3600 s cap), which dilute its mean O₂ ~40 % below v1's. CO₂
   (length-insensitive) diverges less. Exactly the observed pattern.

### Mechanism (the specific model term)
The metabolism FBA objective is **homeostatic (+ mixed kinetic)** with a **secretion penalty**
(`metabolism.py:1075`, "limit secretion (e.g. of CO₂)") and **NGAM/GAM ATP maintenance**
(`metabolism.py:959-960,1078`). It respires only as much as the per-tick ATP/redox deficit
requires — far below the fully-oxidative KETCHUP 13C-MFA cores (O₂ ≈ −135 per 100 glucose). This
is shared with vEcoli (v1 O₂:glucose ≈ 0.14 vs v2 ≈ 0.09). Corroborated by the Millard FBA-bridge,
which pins the respiratory branch and lifts O₂ ~3× (to −26) with the cell staying viable — i.e.
O₂/CO₂ sit on a weakly-determined branch of the LP.

### Recommendation
- **Close #143 as real/understood.** The listener and report card are correct.
- **Optional hardening:** grade O₂/CO₂ over a *matched averaging window* (fixed early-generation
  window, or exclude duration-capped cells) so the front-loaded flux isn't confounded by v2/v1
  generation-length differences — this collapses most of the −40 % O₂ delta.
- **To get a physiological O₂:glucose (~1.5):** add a kinetic O₂ / terminal-oxidase target or pin
  the respiratory branch. That's a modeling choice, not a bug fix.

Full trail: `workspace/studies/showcase-6-equivalence-large/143-rootcause-findings.md`.
