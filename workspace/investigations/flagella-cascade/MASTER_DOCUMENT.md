# Flagella-Cascade Investigation — Master Document

*Living document. Started 2026-08-28. Current State, Parameters, History
Appendix, and References sections are all up to date as of this date —
this consolidates and supersedes the older per-date (`CHANGES_*.md`) and
per-study (`study.yaml`) notes. Those files, along with
`NFSIM_WCM_WIRING_PLAN.md` and the `feedback/` notes, have been removed
from the working tree (2026-08-28) now that their content lives here —
all still fully recoverable from git history if ever needed.
`investigation.yaml` was trimmed the same day to a short pointer back to
this file; it is no longer the working record.*

---

## 1. Current State

### 1.1 Regulatory cascade (Class I → II → III)

- **Class I**: FlhDC (master regulator, transcription factor complex).
  Drives Class II gene expression. Not a structural component — never
  consumed by any assembly reaction.
- **Class II**: structural genes (C-ring, export apparatus, rod, rings,
  hook, and their associated ATPase/chaperone machinery). Expression is
  FlhDC-dependent.
- **Class III**: late genes (`fliC`, `fliD`, `flgK`/`flgL`, `motAB`,
  `cheAW`, `fliS`, `flgM`). Expression requires free FliA (σ28), which is
  normally sequestered by FlgM until the hook-basal body (HBB) is complete.

**FlgM:FliA sequestration** — modeled as an equilibrium reaction
(`FLGM-FLIA-CPLX_RXN`), still solved by the shared, general-purpose
equilibrium Step (numerical ODE, not yet moved to a dedicated exact-solve
Step). Current model Kd = 2×10⁻⁷ M — a **known, deliberate relaxation**
from the real, measured value (see §2). Not yet corrected — two attempts to
tighten it toward the real Kd both failed in testing (see History
Appendix, once written).

**FlgM secretion** — its own dedicated Step
(`flagella_flgm_secretion.py`). As of 2026-08-27, the trigger is
`count(nascent_flagellum)` alone (in-progress HBBs), **not**
`count(CPLX0-7452)` (fully complete flagella including filament). This
matches the real, direct literature finding that the substrate-specificity
switch enabling FlgM export happens at HBB completion, before the filament
is built (see §2 citations). The `secretion_rate` constant (0.1
molecules/HBB/s) is flagged **UNVERIFIED** — its original justifying
comment cited no real source; the real half-life data that should ground
it has been found but not yet converted into a corrected rate.

### 1.2 Assembly cascade (NFsim / BNGL rule-based network)

Fixed 2026-08-27/28 to the real, literature-confirmed dependency order.
Each stage requires the previous stage's product — no stage can be skipped
or built out of order:

**C-ring → export apparatus (2 steps) → rod → P-ring → L-ring → hook →
complete HBB (→ `nascent_flagellum`)**

- C-ring (FliF/FliG/FliM/FliN) nucleates from free monomer, deliberately
  slow (real, literature-derived rate, not a placeholder).
- Export apparatus assembles onto the C-ring (2 sub-reactions, generator
  bookkeeping only, not 2 real biological stages).
- Rod (FliE, FlgB, FlgC, FlgF, FlgG) requires the export apparatus.
- P-ring (FlgI, 26 copies) requires the rod. **Newly added 2026-08-28** —
  previously skipped entirely.
- L-ring (FlgH, 26 copies) requires the rod+P-ring. This is the real,
  confirmed trigger for the rod-to-hook transition (see §2).
- Hook (FlgE, 120 copies) now requires the finished rod+P-ring+L-ring base
  (`FLAGELLAR-MOTOR-COMPLEX[j]`). **This dependency did not exist before
  2026-08-28** — hook used to form independently from free FlgE alone.
- **Stator (MotA/MotB) removed from this entire chain.** Real biology
  confirms structural completion does not require the stator — only
  rotation does. Free MotA/MotB are currently untouched by this pathway.
- Completion produces the `flagella` species, which becomes a real
  `nascent_flagellum` unique molecule.

**Filament elongation** is a separate, non-NFsim incremental Step
(combinatorial explosion in the rule-based engine at real subunit counts
made a single-shot reaction impossible — see History Appendix). Growth
follows an injection-diffusion law, `dL/dt = a/(b+L)`. Target length is
5,000 subunits (short end of the real 20,000–40,000 range — a deliberate
modeling simplification for practical simulation windows, not a claimed
biological value).

**FliS:FliC chaperone binding** — as of 2026-08-28, this reaction has its
**own dedicated Step** (`flagella_flis_flic_equilibrium.py`), solved by an
exact closed-form quadratic (not the shared numerical ODE solver). This
was a direct fix for a repeated division-triggered crash (see §1.3, §2).
Confirmed by direct testing: free FliS/FliC never overshoot negative,
by construction, at any scale.

### 1.3 Division handling

- Division splits all bulk species, including complete flagella and
  in-progress `nascent_flagellum` entries, binomially between daughters —
  consistent with direct Salmonella live-imaging evidence (see §2) that
  division "forcefully splits the number of existing filaments into
  half."
- Division does **not** pause for flagella assembly, and flagella assembly
  does not pause division. This was checked directly against the
  literature: the real mechanism connecting motility and division in
  E. coli is resource competition (shared ribosome/RNAP/ATP pools, already
  correctly modeled), not a hard on/off checkpoint. Direct checkpoint
  mechanisms of that kind exist in the literature (Caulobacter, FlhA-based
  sensor; Campylobacter, FlhG-based pole exclusion) but not in E. coli.
- **Known, still-live risk**: the equilibrium solver is most likely to
  crash right at division, because binomial splitting can suddenly cut a
  reaction's already-small absolute molecule counts in half. Confirmed
  directly (diagnostic printout) that FliS/FliC was the specific reaction
  crashing, twice, at increasing severity — this drove the exact-solve
  Step in §1.2. **FlgM:FliA is on the same shared, general solver, with a
  Kd relaxed for the same kind of reason, and has not yet been checked
  for the same vulnerability.**

### 1.4 Known open problems (unresolved, as of 2026-08-28)

1. FliS:FliC exact-equilibrium fix is implemented and passed one clean
   test. **Not yet validated** across a multi-seed batch (the standard
   validation protocol used throughout this investigation).
2. FlgM:FliA remains on the shared, general ODE solver with a relaxed Kd.
   Same class of fix (dedicated exact-solve Step) has not yet been applied.
3. `secretion_rate` (FlgM secretion) is an unverified placeholder pending
   proper recalibration from real half-life data.
4. `complexation_reactions_modified.tsv` (a separate, ParCa-facing static
   file) may still record the old motor-complex stoichiometry, stator
   included — not yet checked against the new rod/P-ring/L-ring order.
5. NFsim's own internal random seed is not controlled by this codebase's
   outer `--seed` — confirmed, real, unfixed. Lives in a separate sibling
   package (`pbg-nfsim`), not this repository.
6. The generated `.bngl` model file is not automatically rebuilt when its
   generator script changes — a real, already-encountered footgun. No
   permanent safeguard yet.
7. Temporary diagnostic print statements are currently live in
   `equilibrium.py` and `function_registry.py`, explicitly marked for
   removal once the division-crash investigation is fully closed out.
8. FliD double-consumption fix (2026-08-21, see §3.1) is applied but has
   not yet been re-run through a live NFsim test to confirm the corrected
   behavior before trusting it in a longer population run.

### 1.5 Execution-layer order (flagella-related Steps, real order)

1. `ecoli-flagella-nfsim-complexation` — C-ring → export apparatus → rod →
   P-ring → L-ring → hook → complete HBB
2. `ecoli-flagella-flis-flic-equilibrium` — exact closed-form solve
3. `ecoli-flagella-filament-elongation` — incremental FliC growth, draws
   protected pool first
4. `ecoli-flagella-flgm-secretion` — triggered by `nascent_flagellum` count
5. `ecoli-flagella-transcription-regulation` — Class II / III promoter
   activity

**Why this order does not need to match the conceptual regulation → transcription →
translation → assembly loop**: the real biological lag between a regulatory
change and its effect on the free-monomer pool is hundreds of ticks
(transcription, then translation). So assembly in any one tick always draws
from monomers made by *past* regulatory decisions, never the current tick's.
Confirmed directly: `diagnostic_transcription_to_protein_lag.py` showed free
FliC jump 0 → 10,807 → 17,576 molecules within the first ~600s of a run,
entirely from pre-existing mRNA and ribosomes, before that run's own
regulatory changes could have had any effect. Step ordering within one tick
is therefore a bookkeeping choice, not a biology-accuracy question.

---

## 2. Parameters and Sources

| Parameter | Value | Source |
|---|---|---|
| Kd, FlgM:FliA (real) | ~1.8–2.0×10⁻¹⁰ M | Chadsey, Karlinsey & Hughes 1998, *Genes Dev* 12:3123 (*Salmonella*, SPR: ka=8.9×10⁵ M⁻¹s⁻¹, kd=1.6×10⁻⁴ s⁻¹) |
| Kd, FlgM:FliA (current model) | 2.0×10⁻⁷ M | Deliberately relaxed from the real value above, for solver stability. Not yet corrected. |
| Kd, FliS:FliC | 5.26×10⁻⁸ M | Muskotal et al. 2006, *FEBS Lett* 580:3916 (ITC, Ka=1.9×10⁷ M⁻¹, 1:1 stoichiometry) — real value, unchanged by the 2026-08-28 Step move |
| k_bind (generic proxy rate) | 8.5×10⁴ M⁻¹s⁻¹ (→ ~1.412×10⁻⁴ /molecule/s) | McMurry et al. 2015 (real, measured FlhA:FlhB rate — used as a borrowed proxy for ~30 other reactions lacking their own measured kinetics) |
| Nucleation rate (C-ring, FlhDC) | 1.67×10⁻³ /s | Sim et al. 2017, *Sci Rep* 7:41189 (*E. coli* RP437, chemostat growth-rate study) |
| Filament elongation rate_a | 26,450 subunit²/s | Renault et al. 2017, *eLife* 6:e23136 (*Salmonella*, injection-diffusion model). Literature does not cleanly resolve to one value; kept deliberately (Maya's call, 2026-08-21). |
| Filament elongation rate_b | 575 subunits | Same source as rate_a |
| Filament target_length | 5,000 subunits | Short end of real 20,000–40,000 range, PMC7696725 — a modeling simplification, not a claimed real value |
| P-ring (FlgI) copy number | 26 | Real, confirmed structural literature |
| L-ring (FlgH) copy number | 26 | Real, confirmed structural literature |
| Hook (FlgE) copy number | 120 | Real, structural (~55 nm hook, 11 protofilaments) |
| FliC degradation half-life (unprotected) | 7.4 min (k=0.00156/s) | Nagar et al. 2022, pulsed-SILAC |
| FlgM half-life, HBB-complete (Fla+) | 7.3 min | Karlinsey et al. 1998, *J Bacteriol* 180:5384 (*Salmonella*, strain TH2592) |
| FlgM half-life, HBB-incomplete | no detectable turnover | Same source, ring-mutant strains (ΔflgHI, flgB) |
| FlgM `secretion_rate` (current model) | 0.1 /HBB/s | **Unverified placeholder** — not yet derived from the Karlinsey et al. 1998 numbers above |
| Hook-basal-body dependency on L-ring | rod-to-hook transition requires L-ring | Cohen & Hughes 2014, *J Bacteriol* 196:2387 (*Salmonella*) |
| Stator not required for structural completion | confirmed | General literature on *motA*/*motB* mutants — paralyzed but structurally complete flagella |
| Flagella split binomially at division | confirmed, direct imaging | Aizawa & Kubori 1998, *Genes to Cells* 3:625 (*Salmonella*, live dark-field imaging through real division events) |

*Note: several of these citations are from Salmonella, not E. coli directly
— used because the mechanism is treated as conserved across these closely
related species, and no direct E. coli data was found (checked directly,
not assumed) for several of them. This is flagged per-row, not hidden.*

---

## 3. History Appendix

Chronological-by-topic record of what was tried across this investigation
(2026-06-20 through 2026-08-28), drawn from `investigation.yaml`, the
`CHANGES_2026-08-0{6,7,8,10}.md` series, each study's `study.yaml`, the two
`archive/*/README.md` files, the `feedback/` review-response docs, and (for
everything from ~2026-08-11 onward not yet captured in any file) direct
session history.

### 3.1 Tried and kept

- **Kalir & Alon SUM-gate transcriptional cascade** — promoter activity
  `p_i = (β·X + β'·Y)/(β+β')`, X=FlhDC activity, Y=free-FliA activity,
  driving Class II vs Class III `init_prob_override`. Ported from Maya's
  vEcoli `biofilm` branch (feature name `flagella_regulation` itself taken
  from her vEcoli config script name). Confirmed byte-neutral when off
  (`init_prob_override`==0 everywhere, cell grows/divides identically to
  pre-port main) — lets every downstream change be attributed to the
  cascade, not the port. Class II > Class III ordering holds stably across
  generations and the division boundary (Kalir & Alon 2004).
- **FlgM secretion gate** — `exported = min(FlgM, round(n_flagella·0.1·dt))`,
  the anti-sigma-factor negative-feedback loop that releases free FliA as
  FlgM is exported — flgM is itself a Class III gene, so rising free FliA
  drives more FlgM, which re-sequesters FliA, the cap that prevents runaway
  free sigma28 (Stefan et al. 2015). Confirmed to keep free FliA bounded
  (not runaway) within a single generation and across a division boundary.
  `secretion_rate`
  (0.1/HBB/s) remains an unverified placeholder (§1.1, §1.4) — real
  Karlinsey et al. 1998 half-life data has been found but not yet converted
  into a corrected rate.
- **Real flagellar assembly stoichiometry, cryo-EM/structural-literature
  grounded** — corrected from mostly-unsourced -1 placeholders across many
  sessions: FliC×20,000 (later reduced for practical reasons, see 3.2),
  FlgK/FlgL×11, FliF×34, FliG/FliM/FliN (PMC10128058), FlgH/FlgI×26,
  MotA×55/MotB×22, FliP:FliQ:FliR 5:4:1 (Kuhlen et al. 2018, resolving a
  1:1:1 placeholder both of Maya's spreadsheets and the vendored default had
  wrong), FlhA×9 (homononameric export-gate ring), FliH₁₂FliI₆FliJ₁. Checked
  independently against Maya's two spreadsheets AND primary structural
  literature (not spreadsheet-only) after finding the spreadsheets
  themselves had at least one unsourced error. FliO left at -1 (not
  changed) — see 3.3, its consumed-vs-scaffold role is a real open question.
- **Real assembly hierarchy, deterministic-Step era (2026-08-11)** —
  MS-ring (FliF) → C-ring (FliG/FliM/FliN) merged into one stage;
  C-ring → export apparatus (previously modeled as two independent
  branches only merging at the very last step — a real ordering bug, not
  matching Minamino & Namba 2008 / Chevance & Hughes 2008); export
  apparatus + rod/L-ring/stator → motor complex → hook completion →
  nucleation. Carried forward unchanged into the later NFsim rule-based
  network's own dependency order (3.1 below) and into the current
  C-ring→export→rod→P-ring→L-ring→hook chain (§1.2).
- **Injection-diffusion filament elongation law** — `dL/dt = a/(b+L)`
  (Renault et al. 2017), pulled out of Gillespie SSA into its own
  incremental Step specifically because FliC's real ×20,000 stoichiometry
  caused a combinatorial propensity blowup in the generic complexation
  framework (root-caused via macOS `sample`+`strings` on the compiled
  solver, confirmed empirically: a coefficient of 60 was safe, one of
  20,000 was not). Still the live elongation mechanism (§1.2); NFsim's own
  rule-based network deliberately excludes filament growth for the same
  combinatorial reason (fliC=-5000 alone ballooned 237 rules to 5,588) and
  defers to this same Step once WCM-coupled.
- **`RUNTIME_EXCLUDED_REACTIONS`** (`complexation.py`) — the general
  mechanism for pulling specific numerically-dangerous reactions out of
  Gillespie SSA into dedicated deterministic/exact Steps instead. Used for
  every flagella structural-assembly reaction since 2026-08-06, most
  recently the export-apparatus reaction (2026-08-11, see 3.1 SSA race fix)
  and conceptually reused by the FliS:FliC exact-solve Step (2026-08-28,
  see 3.1 below) via zeroing the shared solver's rates for that reaction
  rather than deleting the row.
- **Export-apparatus/C-ring same-tick SSA-vs-Step race fix (2026-08-11)** —
  after the hierarchy fix made export-apparatus assembly depend on a C-ring
  molecule that only existed transiently inside another Step's own tick,
  moved `CPLX0-7451_RXN` out of ordinary Gillespie SSA into its own
  deterministic Step, alongside the other three flagella assembly
  reactions. Fixed a cross-mechanism timing gap, not the real (FlhA-driven)
  scarcity underneath it — motor-complex pool went from monotonically
  draining to a stable 5-6 oscillation.
- **Nucleation-rate literature grounding** — Sim et al. 2017 (*Sci Rep*
  7:41189, *E. coli* RP437 chemostat data), ≈1.67×10⁻³–1.81×10⁻³/s
  depending on derivation, used consistently for both the deterministic
  nucleation Step and (from 2026-08-17) the NFsim BNGL rule set's own
  nucleation calibration — two independently-derived numbers from the same
  paper agreeing within 8%.
- **Fixed-interval rate-limiting pattern** (`next_update_time`-based, not a
  per-tick probability) — first built for
  `flagella_filament_nucleation.py` after discovering `round(nucleation_
  rate*timestep)` silently rounds to 0 forever at small per-tick
  probabilities; reused directly for NFsim's own firing cadence
  (2026-08-12 wiring plan) and every other rate-limited flagella Step since.
- **Nucleation first-tick "free event" bug (2026-08-11)** — a second,
  separate bug from the rounds-to-0 issue above, found the same session.
  `next_update_time` defaults to 0.0, and `update_condition` fires whenever
  `next_update_time <= global_time` — so at t=0, `0.0 <= 0.0` is true, and
  the deliberately-rare nucleation Step fired immediately on its very first
  call, skipping the intended ~600s wait entirely (every single-generation
  run got one unintended "free" nucleation event at simulation start).
  Fixed by special-casing the first call (detected via the same lazy
  bulk-index-resolution check every flagella Step already does): schedule
  `next_update_time` one full interval ahead and return, no flagellum
  created. Verified directly: pre-fix, first nascent flagellum appeared at
  t≈120s already 2,016 subunits grown; post-fix, `n_nascent` correctly
  stayed 0 until t=600s. Rationale for rate-limiting nucleation at all:
  existing structures preferentially absorb material over nucleating new
  ones (Chang, Sung & Hong 2025).
- **Bottleneck correction: FlhA → FliN (2026-08-11)** — overturned an
  earlier same-session finding. Before the C-ring→export-apparatus
  hierarchy fix (3.1 above), FlhA looked genuinely scarce (1–8 copies,
  drawn down aggressively by ungated SSA firing). After the fix, FlhA held
  steady at 105–190 copies (ample) because `CPLX0-7451_RXN` now only fires
  when gated by `CPLX0-7450`, itself gated by FliN. The real bottleneck
  moved to FliN: FliN and FliM are co-transcribed on the same operon
  (`fliLMNOPQR`, TU0-1441, produced in lockstep), but FliN costs 111 copies
  per C-ring event versus FliM's 34, so each event drains FliN's pool a
  proportionally much bigger bite (observed range 10–110 vs. FliM's
  430–660). Recommendation at the time: leave FliN alone — the
  co-transcription fact rules out a synthesis-rate error, the stoichiometry
  is already literature-cited (111±13, PMC10128058), and the motor-complex
  pool it ultimately feeds stayed healthy (5–7) throughout — no evidence
  this constrained flagella completions. Relevant context for the still-open
  §3.3 question of what, if anything, actually limits flagella count
  layer-by-layer.
- **Binomial division splitting** (`divide_nascent_flagellum`,
  `v2ecoli/library/division.py`) — in-progress and complete flagella split
  binomially between daughters at division, consistent with direct
  Salmonella live-imaging evidence (Aizawa & Kubori 1998) that division
  "forcefully splits the number of existing filaments into half." Verified
  2026-08-06 (fixed a real bug where `nascent_flagellum` was duplicated,
  not split), re-verified more rigorously 2026-08-11 against `unique_index`
  identity (not just `filament_length` values that could coincidentally
  collide) — exact partition confirmed, zero loss, zero duplication.
  Extended to 3 real generations 2026-08-12 — all 3 divisions passed.
- **Scale-aware mass-balance tolerance** (`atol=1e-8 + rtol=1e-13·
  max_term_magnitude`, `complexation.py`) — a general, non-flagella-specific
  framework fix for floating-point mass-balance false-positives, found
  during flagella diagnostics.
- **Opt-in `features=` threading through daughter rebuild** — a general
  division-boundary bug where opt-in feature lists (e.g.
  `flagella_regulation`) were silently dropped when a daughter cell's
  composite was rebuilt post-division, raising unrelated-looking
  `ValueError`s. Fixed by threading `features=` through the rebuild path.
- **FliC synthesis-rate 10x override** — measured real-time FliC synthesis
  (0.71→6.0 molecules/s) and applied a 10x transcription-rate override via
  ParCa's `adjust_rna_expression` mechanism, deliberately NOT closing the
  full ~65x gap to Renault-implied demand (≈46/s) since that would require
  ~13.5% of the total mRNA budget going to one gene; chose ~2.3% instead.
- **NFsim rule-based assembly, ported into v2ecoli and stoichiometry-
  corrected (2026-08-12)** — the BNGL model (`generate_flagella_bngl.py`,
  `flagella_complexation.bngl`) moved from `pbg-nfsim`'s own generic
  bundled example (unsourced -1 placeholders, sourced from an unrelated
  older codebase) into this study's own `models/` directory so it can
  evolve alongside v2ecoli's real reaction network, per Maya's explicit
  direction ("I want it to be coupled to the whole cell model"). Restoichi-
  ometried to exactly match v2ecoli's own (v11) reaction network, same
  citations throughout. `pbg-nfsim` itself reverted to supplying only the
  runtime engine.
- **NFsim nucleation-rate suppression, tied to `k_bind` not raw
  propensity** — fixed a first-real-run failure (zero completions despite
  thousands of real reaction events firing, because new scaffolds
  nucleated faster than any existing one could finish). A fixed ratio to
  `k_bind` (later replaced by the ambient-count-scaled, literature-derived
  version, 2026-08-17) let existing scaffolds dominate — first-ever real
  completions confirmed at realistic multi-hour timescales (8hr: first
  export-apparatus completion; 48hr: 930 motor complexes, real assembly at
  scale).
- **NFsim scaffold-species persistence across chunks** — root-caused why
  `flagellar_hook` (120 sequential binds) never completed across any
  chunked run while shorter-chain stages did: `pbg-nfsim`'s `NFSimProcess`
  discards all in-progress `Growing_X` scaffold state between chunk
  invocations by design, so any structure needing more sequential binds
  than fit in one chunk could never accumulate progress. Patched
  `NFSimProcess` to persist scaffold state via BNG's own `.species` file
  and a new `scaffold_species` port; verified end-to-end (isolated hook
  test: 0→4 scaffolds persisting, completed at chunk 32; full model:
  hook=255, first-ever `flagella`-stage completion=3). PR upstream to
  `pbg-nfsim` (viva-nfsim#2).
- **NFsim real-bulk-ID species renaming (2026-08-12)** — BNGL species
  renamed directly to real v2ecoli EcoCyc bulk IDs so NFsim's observables
  ARE the bulk array's own molecule names, no translation layer, mass
  conservation exact by construction. Caught two real, previously-unflagged
  discrepancies in passing: FlhC's real bulk ID is `MONOMER0-2488[c]`, not
  `EG10319-MONOMER[c]`; the canonical motor-complex spec includes FlgI but
  the currently-running deterministic Step never actually consumes it (a
  real, unfixed gap, flagged not fixed).
- **`flagella_nfsim_complexation.py` Step, wired into `ecoli_baseline.py`
  (2026-08-16)** — real `EcoliStep` wrapping the scaffold-fixed
  `NFSimProcess`, feeding off real WCM bulk counts (no synthetic
  `MonomerProduction`), as a new feature module mutually exclusive with the
  older deterministic-Steps `flagella_regulation` pipeline (A/B comparable
  by construction). Confirmed correct end-to-end (real read/write,
  scaffold persistence) but markedly slower to reach full completions than
  the deterministic pipeline within the same 2hr real-ambient-supply
  window — an honest, current speed/calibration gap, not a correctness bug.
- **NFsim nucleation-rate recalibration to real ambient copy numbers
  (2026-08-17)** — replaced a reactively-tuned global suppression constant
  with a rate derived from Sim et al. 2017 and scaled per-reaction by each
  nucleating species' own real WCM ambient count (C-ring/FliF 657,
  hook/FlgE 3508, flhDC/FlhC 649). Caught and fixed a real bug in the same
  pass: the first version only special-cased C-ring, leaving hook
  vulnerable to the identical "many parallel scaffolds, none finish"
  failure (confirmed: 1,226 stuck `Growing_flagellar_hook` scaffolds in one
  chunk) until the fix was generalized to every abundant-monomer-nucleated
  reaction.
- **Real hook/rod/P-ring/L-ring assembly-order fix (2026-08-27/28)** — the
  full C-ring→export apparatus→rod→P-ring→L-ring→hook chain, replacing an
  earlier version where hook formed independently from free FlgE alone and
  P-ring was skipped entirely. L-ring is the confirmed real trigger for the
  rod-to-hook transition (Cohen & Hughes 2014). Stator (MotA/MotB) removed
  from the structural-completion chain entirely — real biology confirms
  rotation, not structural completion, requires the stator; motAB mutants
  assemble structurally complete, paralyzed flagella.
- **FliS:FliC exact closed-form equilibrium Step
  (`flagella_flis_flic_equilibrium.py`, 2026-08-28)** — direct fix for a
  division-triggered equilibrium-solver crash that hit twice, confirmed via
  diagnostic printout to be this exact reaction both times. Root cause: the
  shared numerical ODE solver's default `atol` (1e-6 M) is >600x the
  concentration of a single real molecule in this cell's real volume at the
  post-division scale involved — slowing the reaction's rate constants
  (tried twice, 1000x then another 100x, Kd unchanged both times) never
  touched this, since it was the wrong knob. A simple 1:1:1 binding
  reaction has an exact quadratic solution; this Step computes it directly,
  every firing, with no tolerance to get wrong and no possible overshoot.
  Confirmed by direct testing: free FliS/FliC never go negative, by
  construction. Not yet validated across a multi-seed batch (§1.4).
- **FliD double-consumption fix (2026-08-21)** — found once NFsim was
  enabled: two separate consumption points fired for the same real
  flagellum, NFsim's own flagellum reaction (5× FliD, at hook-basal-body
  completion) and `flagella_filament_elongation.py`'s completion event (5×
  FliD, at filament completion) — 10 FliD consumed per completed flagellum
  instead of the real 5. Real biology: FliD caps the growing tip once,
  before elongation begins, and cannot accept another monomer once capped
  (Song et al. 2017; corroborated by Postel et al. 2020 cryo-EM). Fix kept
  the consumption on `flagella_filament_elongation.py`'s side (the old
  deterministic pipeline's only FliD accounting point, still the default
  feature) and removed it from NFsim's own flagellum reaction
  (`generate_flagella_bngl.py`, old line kept as a comment;
  `flagella_complexation.bngl` regenerated). Verified directly that FliD no
  longer appears in the model's real-bulk-ID mapping (31 tracked IDs, down
  from 32). **Not yet re-run** through a live NFsim test to confirm the
  corrected behavior (§1.4).

### 3.2 Tried and dropped

- **Artificial nucleation cap** (a hard ceiling on flagella count, applied
  as a stopgap once the FlgM/FliA positive-feedback runaway was found) —
  implemented, confirmed working (held count at exactly 10 for the
  remainder of a run), then **fully removed** per Maya's explicit
  instruction ("i dont want the artificial cap at all so remove that and
  any plots, scripts associations with it") — an artificial patch, not a
  real mechanism, rejected on exactly those grounds.
- **`v2ecoli/processes/flagella_nucleation_cap.py`** — the NFsim-side
  equivalent stopgap (Option A, a ceiling Step scoped to
  `flagella_regulation`), confirmed working (held `max_flagella=10` for 36
  of 42 minutes) but explicitly logged as the less "textbook correct" of
  two options considered (vs. Option B, a real consumed nucleation-site
  species) and superseded by the FliT:FlhDC checkpoint attempt below, then
  by the NFsim migration.
- **FliT:FlhDC checkpoint** (`flagella_flit_flhdc_checkpoint.py` +
  `flagella_flhdc_degradation.py`, 2026-08-05/10) — FliT-dimer (released
  once FliD is exported) binds FlhD4C2 and enhances its degradation via a
  fast-equilibrium reduction (Utsey & Keener 2020). Real, biochemically
  confirmed mechanism **in Salmonella** (Yamamoto & Kutsukake 2006, pull-
  down/far-Western). Basal ClpXP turnover (Tomoyasu et al. 2003; Claret &
  Hughes 2000 rate) was bundled in since FlhD4C2 otherwise has literally no
  degradation pathway (`protein_degradation.py` only handles monomers,
  never assembled complexes). Confirmed necessary-but-not-sufficient before
  removal: FlhDC degradation alone brought FlhDC down from an unbounded
  625 to a stable ~173-174, but flagella count kept climbing 4→19
  regardless, because the SUM-gate's `X=FlhDC/(K+FlhDC)` term stays deep in
  its saturated regime even at the lower level — the real runaway engine is
  the separate FlgM-secretion feedback loop. Removed 2026-08-10 because
  Albanna et al. 2018 directly tested a Δ*fliT* mutant in **E. coli MG1655
  — this WCM's exact K-12 reference strain** — and found no significant
  phenotype there, versus a clear effect in *Salmonella*. `FLIT-DIMER_RXN`
  (real, checkpoint-independent FliT homodimer biology) kept as a building
  block for the planned NFsim replacement. Archived at
  `archive/flit-flhdc-regulation-2026-08/`. Known, accepted tradeoff:
  `flagella_regulation` now has no FlhD4C2 shutdown mechanism at all until
  an NFsim rule replaces it — reopens the original unbounded-runaway risk.
- **Four deterministic assembly Steps, superseded by NFsim
  (removed 2026-08-21)** — `flagella_motor_switch_assembly.py`,
  `flagella_export_apparatus_assembly.py`, `flagella_motor_complex_
  assembly.py`, `flagella_filament_nucleation.py`. Real, carefully
  cross-checked structural biology, not wrong — removed because they were
  always meant as a bridge to the NFsim rule-based network (which enforces
  assembly order per-instance via pattern matching rather than by careful
  same-tick Step ordering, and had already had its own real bugs found and
  fixed), and Maya committed to the NFsim path as the one way forward,
  making two parallel, mutually-exclusive assembly pipelines pure
  maintenance surface. `flagella_filament_elongation.py`, `flagella_flgm_
  secretion.py`, and `flagella_transcription_regulation.py` were NOT
  archived — shared infrastructure reused as-is by the NFsim pipeline.
  Archived at `archive/deterministic-flagella-assembly-2026-08/`.
- **FlgM:FliA rate-constant slowdown, two attempts** (2026-08 division-
  crash investigation) — tried scaling FLGM-FLIA-CPLX's rate constants down
  1000x, then another 100x on top (Kd preserved both times), to try to
  stop the same kind of division-triggered equilibrium-solver crash later
  root-caused and fixed for FliS:FliC. Both attempts failed (6/6 seeds
  still crashing at one Kd value tested, ~1.8×10⁻¹⁰ M; failed again at
  ~2×10⁻⁸ M) — the real problem was the shared solver's absolute tolerance
  being far larger than one real molecule's concentration at this scale,
  which rate-constant scaling never touches. This diagnosis directly
  motivated the FliS:FliC exact-solve Step (3.1); the same fix has not yet
  been applied to FlgM:FliA, which remains on the shared solver with a
  deliberately relaxed Kd (§1.1, §1.4).
- **FliC target filament length, three successive reductions** —
  20,000 (real, PMC7696725-cited) → 10,000 (2026-08-10) → 5,000
  (2026-08-11, current). Each real literature-range value, not an
  arbitrary diagnostic override — chosen because minimum completion time
  scales ~L² under the real `dL/dt=a/(b+L)` growth law: 20,000 needs ~133
  min minimum (structurally impossible within one ~42 min generation);
  10,000 needs ~35 min; 5,000 was chosen after the 10,000 target still
  nearly fully drained free FliC (51,967→14) within one generation once
  the export-apparatus/C-ring hierarchy fix increased completion
  throughput. Maya's explicit call (2026-08-12): hold at 5,000 for now
  ("10,000 is a lot right now"; 20,000 "is too much"), planned to move
  back toward 10,000 once the reaction network is fully validated
  post-NFsim migration — not a permanent value.
- **Manual multi-generation state-splicing methodology** — an early
  approach to testing flagella completion across many generations by
  hand-splicing simulation state between runs, replaced after it produced
  its own real, separate bug (a dry-mass drift artifact of the splicing
  method itself, 706.7→262.0 fg over 7 generations) that was initially
  mistaken for a model bug. Replaced by driving real generations through
  the actual `Division` machinery instead, which is what surfaced the
  real (not diagnostic-artifact) division/mass-homeostasis gap (3.3).
  A separate stdout-log-parsing feature (captured `Division`'s own
  internal timing via a Tee+regex) was also removed after it reported an
  implausible ~5-8 min generation time — traced to a stale buffer bug in
  the parser itself, not the simulation; the simulation's own
  independently-recorded timestamps were correct throughout.

### 3.3 Still undecided / parked

- **FliO's structural role in the export apparatus** — real literature
  describes FliO as a transient assembly scaffold, not part of the final
  mature complex (a Δ*fliO* mutant is rescued to wild-type motility by
  FliP overexpression alone), but it is currently modeled as an ordinary
  consumed reactant (-1) in the reaction forming the final complex.
  Maya's explicit call: leave as-is, since a copy-number fix doesn't
  address a role question; defer to being represented properly once
  NFsim rules can express a non-consumed scaffold relationship.
- **Motor-complex upstream supply chain, never examined the way FliC's
  was** — across every diagnostic run this investigation, something else
  (FliC supply, then motor-complex supply, then division/mass-homeostasis
  corruption) always capped flagella count before the FliT/FlhDC
  checkpoint mechanism was ever directly observed to be binding. Whether
  the checkpoint (or its NFsim successor) would ever actually become the
  real limiting factor, or whether the system is bottlenecked
  layer-by-layer before that regardless, is unresolved. Parked at Maya's
  explicit call (2026-08-08).
- **YdiV as the E. coli-native FlhD4C2 anti-regulator candidate** — since
  the FliT:FlhDC checkpoint was found to be Salmonella-specific (not
  significant in E. coli MG1655), Maya's revised direction is to check
  YdiV (an EAL-domain pseudo-phosphodiesterase that binds FlhD4C2 and
  targets it for ClpXP degradation, real and E. coli-native rather than
  borrowed from Salmonella) against the literature before building any
  NFsim rule around it. Not yet checked or implemented.
- **Division/mass-homeostasis corruption** — a general (non-flagella-
  specific) WCM limitation: dry mass drifts down generation-over-generation
  past ~3-4 generations, with growing `GLP_NOFEAS` infeasibility flooding.
  Most likely cause (not yet confirmed as fixable): `division.py`'s
  D-period timer fires on a fixed duration regardless of mass threshold,
  producing underweight daughters that FBA can't feed, and
  `allocator.py`'s negative-pool handling doesn't heal by design, so the
  error compounds across generations. Documented as a known limitation;
  Maya's explicit call was to document and not attempt a fix in-session,
  since it is outside flagella-specific code. Blocks any multi-generation
  flagella-completion test past ~3 clean generations (~2hr) — the original
  "does an inherited, in-progress flagellum complete across real
  generations" question this whole investigation thread has been chasing
  is therefore still not fully answered either way.
- **NFsim as its own investigation** — once NFsim assembly is fully
  WCM-coupled (rather than the current standalone/partially-coupled
  state), should it split out into its own investigation with its own
  acceptance criteria (rule-based vs. Gillespie complexation as a distinct
  methods thread) rather than staying inside flagella-cascade as Aim 2B?
  Maintainers'/Maya's call, not yet made.
- **`secretion_rate` (FlgM export) recalibration** — currently an
  unverified 0.1/HBB/s placeholder with no original citation. The real
  half-life data needed to correct it (Karlinsey et al. 1998: 7.3 min
  HBB-complete vs. no detectable turnover HBB-incomplete) has been found
  but not yet converted into a corrected rate (§1.1, §1.4).
- **`complexation_reactions_modified.tsv` motor-complex stoichiometry
  audit** — this ParCa-facing static file may still record the pre-2026-
  08-27/28 motor-complex stoichiometry (stator included); not yet checked
  against the new rod/P-ring/L-ring order (§1.4).
- **NFsim's own internal random seed** — not controlled by this
  codebase's outer `--seed`; lives in a separate sibling package
  (`pbg-nfsim`), confirmed real and unfixed (§1.4).
- **Report-study visual differentiation** — Maya's request for more
  detail / bigger font to differentiate studies in the generated report is
  a vivarium-workbench report-template change, out of scope for this
  investigation's own code; logged, not actioned.
- **Chart 04 (bursts-are-division-transients)** — the transient bursts at
  division boundaries in the multigen mean-override trace are hypothesized
  to be a re-initialization transient (pools halve, gate re-settles over a
  few ticks), but this has not been directly confirmed against logged
  division times — status remains `untested`.

---

## 4. References

- Aizawa SI & Kubori T (1998). Bacterial flagellation and cell division.
  *Genes to Cells* 3:625-634. (Direct Salmonella live dark-field imaging of
  flagellar filament number through real division events — binomial
  splitting.)
- Albanna A, Sim M, Hoskisson PA, Gillespie C, Rao CV, Aldridge PD (2018).
  Driving the expression of the *Salmonella enterica* sv Typhimurium
  flagellum using *flhDC* from *Escherichia coli* results in key regulatory
  and cellular differences. *Sci Rep* 8:16705.
  https://doi.org/10.1038/s41598-018-35005-2
- Chang YR, Sung YS & Hong DF (2025). Intrinsic clustering of flagellar
  basal body proteins in *Escherichia coli*. *Biochem Biophys Reports*
  42:102051. (Rationale for rate-limited nucleation: existing structures
  preferentially absorb material over nucleating new ones.)
- Chadsey MS, Karlinsey JE & Hughes KT (1998). The flagellar anti-sigma
  factor FlgM actively dissociates *Salmonella typhimurium* sigma28 RNA
  polymerase holoenzyme. *Genes Dev* 12:3123-3136.
- Chevance FFV & Hughes KT (2008). Coordinating assembly of a bacterial
  macromolecular machine. *Nat Rev Microbiol* 6:455-465.
- Claret L & Hughes KT (2000). Flagellar hook length control. *J Bacteriol*
  182:833. (FlhD/FlhC half-life estimate, *Proteus mirabilis*, used for the
  basal FlhD4C2 degradation rate.)
- Cohen EJ & Hughes KT (2014). Rod-to-hook transition for extracellular
  flagellum assembly is catalyzed by the L-ring-dependent rod scaffold
  removal. *J Bacteriol* 196:2387-2395.
- Fukumura T et al. (2017). Assembly and stoichiometry of the core
  structure of the bacterial flagellar type III export gate complex.
  *PLOS Biol* 15:e2002281.
- Kalir S & Alon U (2004). Using a quantitative blueprint to reprogram the
  dynamics of the flagella gene network. *Cell* 117:713-720. (Class I/II/III
  regulatory hierarchy and the SUM-gate promoter model.)
- Karlinsey JE, Tanaka S, Bettenworth V, Yamaguchi S, Boos W, Aizawa SI &
  Hughes KT (1998). Completion of the hook-basal body complex of the
  Salmonella typhimurium flagellum is coupled to FlgM secretion and fliC
  transcription. *J Bacteriol* 180:5384-5397. (Strain TH2592; FlgM
  half-life 7.3 min when HBB-complete, no detectable turnover when
  HBB-incomplete.)
- Kuhlen L et al. (2018). Structure of the core of the type III secretion
  system export apparatus. *Nat Struct Mol Biol* 25:583-590. (Cryo-EM,
  building on Fukumura et al. 2017; FliP:FliQ:FliR 5:4:1.)
- McMurry JL, Van Arnam JS, Kihara M & Macnab RM (2015 reprint of original
  characterization). Analysis of the cytoplasmic domains of Salmonella
  FlhA and interactions with components of the flagellar export machinery.
  *PLOS One* 10.1371/journal.pone.0134884. (FlhA:FlhB association rate
  constant, biosensor data, used both directly and as the literature-
  grounded proxy default for other flagellar binding reactions lacking
  their own measured kinetics.)
- Minamino T & Namba K (2008). Distinct roles of the FliI ATPase and
  proton motive force in bacterial flagellar protein export. *Nature*
  451:485-488. (Real assembly order: MS-ring/C-ring before export
  apparatus before rod/hook.)
- Muskotal A, Kiraly R, Sebestyen A, Gugolya Z, Vegh BM & Vonderviszt F
  (2006). Interaction of FliS chaperone with a flagellin-specific export
  signal. *FEBS Lett* 580:3916-3920. (Isothermal titration calorimetry,
  Ka=1.9×10⁷ M⁻¹, 1:1 FliS:FliC stoichiometry.)
- Nagar N et al. (2022). Pulsed-SILAC proteomics reveals flagellin
  turnover kinetics. (FliC degradation half-life, unprotected: 7.4 min.)
- Renault TT, Abraham AO, Bergmiller T, Paradis G, Rainville S, Charpentier
  E, Guet CC, Tu Y, Namba K, Keener JP, Minamino T & Erhardt M (2017).
  Bacterial flagella grow through an injection-diffusion mechanism. *eLife*
  6:e23136. (`dL/dt = a/(b+L)` filament growth law, a≈26,450, b≈575.)
- Postel S, Deng Z, Xu C, Sun S & Zhou J (2020). Cryo-EM structure of the
  bacterial flagellar filament tip and its molecular role in dual filament
  formation. *Nat Commun* 11:1965. (Cryo-EM mechanism for the FliD
  pentameric cap; corroborates single-cap-event behavior.)
- Sim M et al. (2017). Growth rate control of flagellar assembly in
  *Escherichia coli* strain RP437. *Sci Rep* 7:41189. (Chemostat data;
  nucleation rate ≈1.67-1.81×10⁻³/s, and 7.8 flagella/cell at 1.2 hr
  doubling used as the direct back-calculation source for the same rate.)
- Song WS, Jung HW & Yoon SI (2017). Structural insight into flagellar
  cap-mediated flagellin secretion and filament formation. *J Mol Biol*
  429:847-861. (FliD pentameric cap forms once, before elongation, and
  cannot accept a second binding event once complete.)
- Stefan D et al. (2015). FlgM/FliA feedback loop and bounded free sigma28.
  *PLoS Comput Biol* 11:e1004028.
- Tomoyasu T, Ohkishi T, Ukyo Y, Tokumitsu A, Takaya A, Suzuki M, Sekiya K,
  Matsui H, Kutsukake K & Yamamoto T (2003). The ClpXP ATP-dependent
  protease regulates flagellum synthesis in *Salmonella enterica* serovar
  typhimurium. *Mol Microbiol* 48:443-452. (ClpXP degrades assembled
  FlhD4C2, not free subunits.)
- Utsey B & Keener JP (2020). A mathematical model for regulation of the
  MS ring switch complex assembly in bacterial flagellum biogenesis. *PLOS
  Comput Biol* 16:e1007689. (Fast-equilibrium reduction used for the
  FliT:FlhDC checkpoint's math.)
- Yamamoto S & Kutsukake K (2006). FliT acts as an anti-FlhD2C2 factor in
  the transcriptional control of the flagellar regulon in *Salmonella
  enterica* serovar Typhimurium. *J Bacteriol* 188:5124-5131.
  https://pubmed.ncbi.nlm.nih.gov/16952964/
- PMC7696725 — cryo-EM structural source for FliC subunit copy number
  (~20,000) and the 20,000-40,000 subunit / 5-20 μm real filament-length
  range.
- PMC10128058 — cryo-EM structural source for FliG/FliM/FliN switch-complex
  copy numbers.
