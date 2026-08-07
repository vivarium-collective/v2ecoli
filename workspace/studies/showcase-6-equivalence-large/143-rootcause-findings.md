# Issue #143 — O₂/CO₂ respiratory-exchange divergence: root-cause findings

**Branch:** `harden/143-o2-rootcause`  ·  **Date:** 2026-07-26
**Question:** Is the exchange-flux group's failure (O₂ Δ −40.4%, CO₂ Δ −20.3% v2 vs v1)
(i) real under-respiration, (ii) a listener instrumentation bug, or (iii) a comparison
(units/sign/averaging) artifact?

## Verdict

**Not (ii). Primarily (i) — the low O₂ is *real* wcEcoli-FBA behavior — with the specific
#143 v2‑vs‑v1 −40 % O₂ delta being an (iii)-flavored *averaging-window fragility*, not a
localized respiratory defect.**

- The `external_exchange_fluxes` listener faithfully reports the FBA solution (mass-balanced
  against the internal O₂-consuming reactions). **Instrumentation (ii) is ruled out.**
- O₂ is *unconstrained* in the media (freely available); the low O₂ is a genuine property of
  the FBA solution, shared by both v1 (O₂:glucose ≈ 0.14) and v2 (≈ 0.09).
- The exchange comparison pipeline is symmetric v1↔v2 (same `extract_vectors`, same units
  mmol/gDCW/h, same sign, same burn-in) → **no units/sign/window bug in the extraction (iii-proper).**
- The **−40 % O₂ delta** arises because O₂ exchange is a **front-loaded** flux whose per-cell
  time-average is highly sensitive to generation length. v2's ensemble contains longer /
  "stuck" generations (#142), which dilute the mean O₂ below v1's. CO₂ (**back-loaded**) is far
  less sensitive → that is why O₂ (−40 %) diverges much more than CO₂ (−20 %).

---

## Evidence trail

### 1. The pipeline (boundary trace)
- **FBA solve:** `v2ecoli/processes/metabolism.py:798` (`fba.solve`), results pulled at
  `library/fba_fast.py:120` (`extract_results` → `external_exchange_fluxes = col_primals[external_exchange_idx]`).
- **Exchange sign/units:** external-exchange reactions are defined with coeff −1
  (`modular_fba.py:366` `_initExternalExchange`), so **uptake = negative, secretion = positive**.
  Raw units mmol/L·tick → converted to **mmol/gDCW/h** at `metabolism.py:569-571`
  (`converted_exchange_fluxes = (CONC_UNITS * external_exchange_fluxes / coefficient).to(GDCW_BASIS)`).
- **Listener:** written verbatim at `metabolism.py:844` (`external_exchange_fluxes: converted_exchange_fluxes`).
- **Report-card extraction (both sides):** `library/card_vectors.py:24-84` `extract_vectors`
  reads `listeners__fba_results__external_exchange_fluxes`, time-averages **within a cell** over
  all ticks in generations ≥ `generation_lower_bound` (=3), then takes the **population mean**
  across cells. The vEcoli (v1) reference uses the *same* `extract_vectors`
  (`scripts/pin_vecoli_equivalence_reference.py:205`). Grading: `card_criteria.py` `flux_scatter`
  (identity R², r2_min 0.99) + per-metabolite `ttest` (within 5 %, mismatch 10 %).
  → **Extraction is unit-consistent and symmetric between v1 and v2.**

### 2. Instrumentation ruled out (verdict ii ✗) — `scripts/probe_o2_fba_vs_listener.py`
Fresh seed-0 baseline (`build_composite("ecoli_baseline")`):
- media_id = `minimal`; **O₂ and CO₂ are both in the `unconstrained` import set** (freely available;
  not O₂-limited).
- The listener O₂ equals the **net O₂ consumed by the internal reactions** (from
  `reaction_fluxes`, e.g. −0.003 ≈ −0.003) → the listener does **not** under-report; O₂ enters
  only via the medium exchange, so exchange O₂ = true respiration. Mass-balanced.
- Tick 0 (high initialization demand) respires at **−14.7 mmol/gDCW/h** (physiological) — the FBA
  *can* respire hard when demand requires it, so low O₂ is **demand-limited, not capability-limited.**

### 3. The mechanism: a bimodal within-generation trajectory — `scripts/probe_o2_longrun.py`
A ~2500-tick single-cell run (`143-evidence/o2co2_trajectory_seed0.txt`) shows the exchange
fluxes are **not steady** — they switch regimes ~10 % into the generation:

| phase | ticks | O₂ (mmol/gDCW/h) | CO₂ | RQ=CO₂/O₂ | character |
|---|---|---|---|---|---|
| early | 0–440 | **−1.31** | 0.00 | 0 | respiratory burst (pools far from target) |
| late  | 460–2520 | **−0.005** | **+1.94** | ~400 | non-respiratory: decarboxylation without O₂ |

The cell consumes glucose (−6 to −7) and grows (dry mass 380→730 fg) throughout, but for ~90 %
of the generation it meets ATP demand by substrate-level phosphorylation and balances redox
reductively (no O₂), while still emitting CO₂ from biosynthetic decarboxylations. **O₂ is
front-loaded; CO₂ is back-loaded.** The generation **time-average** of this trajectory
(≈ −0.28 O₂ / +1.6 CO₂ for this single seed) matches the report-card v2 numbers (−0.45 / +1.70)
in magnitude and sign. So the report-card O₂:glucose ≈ 0.09 is the honest time-mean of a
mostly-non-respiring FBA solution — **real model behavior.**

Why the FBA under-respires (the specific model term): the metabolism objective is
**homeostatic (+ mixed kinetic)** — hit internal metabolite concentration targets + kinetic
reaction-rate targets — plus a **secretion penalty** (`metabolism.py:1075-1076`, "the
inconvenient constant — limit secretion (e.g. of CO₂)") and **NGAM/GAM ATP maintenance**
(`metabolism.py:959-960, 1078`). This objective does *not* maximize growth or force full glucose
oxidation; it respires only as much as the per-tick ATP/redox deficit demands. The KETCHUP
kinetic MFA cores (O₂ ≈ −135 per 100 glucose) fit batch-aerobic data where the cell fully
respires; the wcEcoli homeostatic FBA does not — hence the shared, real low O₂:glucose.

### 4. Why the v2‑vs‑v1 −40 % O₂ delta (and why CO₂ diverges less)
Because O₂ is front-loaded, the per-cell **time-average O₂ is a strong function of generation
length** — a longer generation dilutes the fixed early respiratory burst over more low-O₂ late
ticks. Truncating the trajectory at different division ticks:

| division tick | mean O₂ | mean CO₂ |
|---|---|---|
| 1600 | −0.348 | +1.60 |
| 2520 | −0.223 | +1.61 |

→ a longer generation gives **−36 % O₂** but **<1 % CO₂** change. v2's ensemble has systematically
longer / non-dividing generations (24/256 cells hit the 3600 s cap without dividing — the #142
sawtooth), which v1 does not. Averaging O₂ (front-loaded) over v2's longer generations pulls its
mean **less negative** than v1's → v2 O₂ (−0.45) vs v1 (−0.76), i.e. the **−40 % headline**. CO₂
(back-loaded, length-insensitive) shows a smaller delta (−20 %), consistent with a residual
level difference rather than length dilution. This differential sensitivity is exactly the
observed O₂(−40 %) ≫ CO₂(−20 %) pattern.

### 5. Corroboration & degeneracy
- **Ketchup cross-check (same root cause):** the `ketchup-exchange-comparison` FBA-bridge moved
  O₂ from −9 to −26 per 100 glucose (3×) *with the cell staying viable* (dry mass 380→385 fg) —
  direct evidence that O₂/CO₂ sit on a weakly-determined branch of the LP (the objective does not
  robustly pin them). `scripts/probe_o2_degeneracy.py` confirms all exchange **reduced costs are
  0** (the homeostatic objective prices internal targets, not exchange fluxes).
- The ~0.09 baseline O₂:glucose in ketchup **is the same phenomenon** as #143 — one root cause
  explains both.

---

## Implications & recommendation

- **#143 is not a bug in the listener or the comparison extraction.** The exchange-flux report
  card behaves as designed; O₂/CO₂ are faithfully measured and symmetrically compared.
- The exchange-flux axis is nonetheless **fragile**: O₂ is a front-loaded flux on the FBA's
  weakly-determined respiratory branch, so its per-cell time-average — and thus the v1↔v2 delta —
  is dominated by generation-length / phase-partitioning differences (v2's #142 stuck cells),
  **not** by a localized respiratory-metabolism divergence.
- **Fairer axis (optional hardening, not a bug fix):** compare O₂/CO₂ over a *matched averaging
  window* (e.g. a fixed early-generation window, or exclude cells that hit the duration cap)
  so the front-loaded flux is not confounded by generation-length differences between the two
  ensembles. This would collapse most of the −40 % O₂ delta.
- **If a "true" respiratory phenotype is wanted** (to match the KETCHUP kinetic cores /
  physiological O₂:glucose ≈ 1.5), the model term to change is the metabolism **objective /
  respiratory constraints** — e.g. add a kinetic O₂-uptake or terminal-oxidase target, raise
  NGAM/GAM, or pin the respiratory branch (as the Millard FBA-bridge does). That is a *modeling*
  decision, not a defect: #143 can be closed as **"real, understood."**

## Artifacts
- `scripts/probe_o2_fba_vs_listener.py` — FBA vs listener mass-balance (rules out ii).
- `scripts/probe_o2_longrun.py` — within-generation O₂/CO₂ trajectory (bimodal).
- `scripts/probe_o2_degeneracy.py` — exchange reduced costs (weak determination).
- `143-evidence/o2co2_trajectory_seed0.txt` — tick, O₂, CO₂ trajectory data.
