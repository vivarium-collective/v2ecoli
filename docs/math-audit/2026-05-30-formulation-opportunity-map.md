# v2ecoli — Mathematical-Formulation Opportunity Map

**Date:** 2026-05-30
**Branch:** `docs/math-formulation-audit`
**Method:** [design doc](2026-05-30-formulation-opportunity-map-design.md) — hybrid audit
(13 parallel per-process auditors + single-context architecture & cross-cutting passes).
**Scope:** every process in `v2ecoli/processes/` + math-bearing steps + the
partitioning/allocation architecture. Read-only; no code changed.

Parity legend: 🟢 behavior-preserving · 🟡 behavior-changing (needs justification).
Effort: S/M/L. Impact: H/M/L.

---

## 1. Executive summary

The model's *kernels* are, by and large, mathematically sound and faithful to
upstream vEcoli. The opportunities cluster not in the equations themselves but in
**four recurring structural problems**, plus a handful of high-value individual
items.

The four cross-cutting themes (detailed in §5) account for most of the findings:

- **A — "Mathematical Model" docstrings describe different math than the code runs.**
  Pervasive and sometimes severe: `metabolism` documents a `max cᵀv` biomass FBA but
  solves a homeostatic *quadratic minimization*; `polypeptide_elongation` documents a
  ppGpp ODE and a "min f_charged" rule the code doesn't implement; `chromosome_structure`
  documents a supercoiling subsystem that is never executed; `complexation` opens by
  describing a legacy run-to-completion algorithm contradicting its Gillespie code.
  These are 🟢 doc-only fixes with high clarity payoff.
- **B — Shadowed duplicate `inputs()`/`outputs()` definitions** in 6 processes — a
  mechanical v1→v2 migration artifact. The first definition is dead; some diverge
  (drop a real port). One mechanical sweep. 🟢.
- **C — Three divergent copies of the proportional-allocation math**
  (`calculatePartition`, `reconcile_requests`, naive departitioned), with correctness
  that has *already drifted between the copies* — the allocator can crash on an edge
  case the reconciled copy guards against. This is the architecture-layer simplification
  the request specifically asked for.
- **D — Magic hardcoded array lengths** in listener defaults (`1088`, `49`, `99`, `3`)
  that silently desync from the config-derived true sizes. 🟢.

The highest-value **individual** items:

1. **Two latent crash bugs in `transcript_elongation`** — `update["bulk"]` is appended to
   before it's created, and `attenuated_rna_indices_lookup` is used but never defined.
   Both are masked by default-off config, so they'll fire the moment recycling or
   attenuation is enabled. 🟢 S-effort fixes. *(M-ELONG-1)*
2. **`metabolism_simple` documents three constraints it never enforces** — nutrient
   exchange bounds, maintenance energy, and reaction irreversibility are all wired into
   config and then ignored in `update`, so the reduced LP is unbounded and
   media-insensitive. 🟡 — but it ships with no behavior test and no composite uses it. *(M-METSIMPLE-1)*
3. **`chromosome_structure` supercoiling subsystem is dead** — documented, half-coded
   (~290 lines of unreachable helpers), and its companion empty-fork RNAP-removal logic
   is silently dropped with it. Decide: restore or delete. *(M-CHROMSTRUCT-1)*
4. **`equilibrium` & `two_component_system` dropped upstream's correctness guards** when
   the ODE solves were inlined — the negativity/steady-state-residual checks and the
   propensity-weighted reaction-reversal loop are gone, so both can emit negative counts.
   🟡 rigor regression vs vEcoli. *(M-EQUIL-1, M-TCS-1)*
5. **`chromosome_initiation.py` is non-functional stubs** advertising a DnaA model that
   doesn't exist (the real criterion lives in `chromosome_replication`). *(M-CHROMINIT-1)*
6. **`division` carries a dead `divide` flag** — two decoupled division criteria, only
   the mass+chromosome one fires. *(M-DIV-1)*

A note on the architecture goal: the partition→evolve→merge cycle is, mathematically, a
first-order operator split with a proportional-rationing conflict resolver. It is nowhere
written down as such, and it is implemented three times. §4 proposes unifying it behind a
single documented `proportional_allocate()` kernel and writing the splitting scheme down —
the biggest single lever for "clarify the mathematics of the architecture."

---

## 2. Ranked opportunity table

Ranked by (impact × confidence), favouring 🟢 low-effort wins. 🟡 items grouped as
"rigor / parity candidates" at the bottom — real, but each needs a justification + fixture
regeneration before pursuing.

### Tier 1 — High-value, behavior-preserving, do-first

| ID | Opportunity | Axis | Imp | Eff | Parity | Conf |
|---|---|---|---|---|---|---|
| M-ELONG-1 | Fix two latent crashes in transcript_elongation (`update["bulk"]` init; undefined `attenuated_rna_indices_lookup`) | rigor | H | S | 🟢 | high |
| M-DOC-1 | Rewrite drifted "Mathematical Model" docstrings to match code (theme A: metabolism, polypeptide_elongation, polypeptide_initiation, transcript_initiation, complexation, rna_degradation) | clarity | H | S–M | 🟢 | high |
| M-PORTS-1 | Delete shadowed duplicate `inputs()`/`outputs()` in 6 processes (theme B) | clarity | M | S | 🟢 | high |
| M-CHROMSTRUCT-2 | Declare the collision/ribosome listener outputs `chromosome_structure` actually writes (port-contract violation) | rigor | M | S | 🟢 | high |
| M-MAGIC-1 | Derive listener default lengths from config, not literals `1088`/`49`/`99`/`3` (theme D) | rigor | M | S | 🟢 | high |
| M-ALLOC-2 | Add the `replace=False` size guard to `calculatePartition` that `reconcile_requests` already has (latent crash) | rigor | M | S | 🟢 | high |

### Tier 2 — Architecture & structural (medium effort, high clarity)

| ID | Opportunity | Axis | Imp | Eff | Parity | Conf |
|---|---|---|---|---|---|---|
| M-ARCH-1 | Unify the 3 allocation implementations behind one documented `proportional_allocate()` kernel | architecture | H | M | 🟢 | high |
| M-ARCH-2 | Write down partition→evolve→merge as an explicit operator-splitting scheme (order, conservation, contention semantics) | clarity | H | M | 🟢 | high |
| M-CHROMSTRUCT-1 | Resolve the dead supercoiling/linking-number subsystem: restore (parity) or delete (~290 lines) — and decide the coupled empty-fork RNAP removal | architecture | H | M | 🟡 | high |
| M-CHROMINIT-1 | Resolve chromosome_initiation.py stubs (delete, or implement the real DnaA-box model) | architecture | H | M | 🟡 | high |
| M-DIV-1 | Resolve the dead `divide` flag — pick one authoritative division criterion | architecture | H | M | 🟡 | high |
| M-REPL-1 | Delete dead `_prepare` request machinery + duplicate `buildSequences` in chromosome_replication (Process→Step leftover) | simplification | M | S | 🟢 | high |
| M-ARCH-3 | Fix `reconciled`'s false "same fairness guarantees as Allocator" claim — it's single-tier; reconcile priorities or document the divergence | rigor | M | S | 🟡 | high |
| M-ARCH-4 | Guard `departitioned` against negative counts under contention (it has no mediation; only safe when no two processes contend) | rigor | M | M | 🟡 | med |

### Tier 3 — Math-driven performance (mostly behavior-preserving)

| ID | Opportunity | Axis | Imp | Eff | Parity | Conf |
|---|---|---|---|---|---|---|
| M-MET-PERF-1 | Warm-start the dFBA solve across ticks (successive LPs are near-identical — dFBA's whole premise) | performance | M | M | 🟢 | med |
| M-METSIMPLE-PERF-1 | Stop densifying `S`/`M` and rebuilding `A_eq` every tick in metabolism_simple — assemble sparse once, vary only `b_eq` | performance | M | S | 🟢 | high |
| M-WATERFILL-1 | Replace the iterative overcrowding `while` loops in transcript_initiation **and** polypeptide_initiation with the closed-form capped-simplex (water-filling) projection (theme E) | performance/rigor | M | M | 🟢 | med |
| M-PPELONG-PERF-1 | Hoist the loop-invariant branch + reuse a preallocated buffer in the charging-ODE RHS (187k calls/sim); consider analytic Jacobian | performance | M | M | 🟢 | med |
| M-VEC-1 | Vectorize per-element Python hot loops (trna_attenuation stop-prob; chromosome_structure recycling; rna_degradation class masks) | performance | L–M | M | 🟢 | high |
| M-EQUIL-PERF-1 | Remove the redundant double ODE solve in the TCS shortfall path | performance | M | S | 🟢 | high |

### Tier 4 — Rigor / parity candidates (real, but each needs justification + fixture regen)

| ID | Opportunity | Axis | Imp | Eff | Parity | Conf |
|---|---|---|---|---|---|---|
| M-METSIMPLE-1 | Enforce the documented-but-ignored constraints (exchange bounds, maintenance, irreversibility) + resolve `S@v=0` vs `delta=S@v`; add a behavior test | rigor | H | M | 🟡 | high |
| M-EQUIL-1 | Restore equilibrium's dropped negativity + steady-state-residual guards | rigor | H | S | 🟡 | high |
| M-TCS-1 | Restore TCS's dropped propensity-weighted reaction-reversal correction loop | rigor | H | M | 🟡 | high |
| M-EQUIL-2 | Replace `solve_ivp` to `t=1e20` with a per-reaction algebraic/Newton equilibrium solve | rigor/perf | M | L | 🟡 | med |
| M-MET-2 | Make the FBA retry loop meaningful (relax kinetic weight to the always-feasible homeostatic core) instead of re-solving the identical problem | rigor | M | S | 🟡 | high |
| M-MET-3 | Use stochastic rounding (not truncation) for exchange-flux→count deltas to remove a systematic boundary bias | rigor | M | S | 🟡 | high |
| M-RNADEG-1 | Resolve the tRNA uniform-vs-MM contradiction + the three inconsistent water-accounting statements in rna_degradation | rigor | M | S | 🟡 | high |
| M-TFBIND-1 | Restore the upstream MarA TODO caveat (documented kludge presented as settled biology) + derive `34` from the regulon | rigor | M | S | 🟡 | med |
| M-ATTEN-1 | Clamp/assert attenuation `P_stop ∈ [0,1]` (negative if any fold-change > 1) | rigor | M | S | 🟢 | med |
| M-DIV-2 | Align division RNG seeding with upstream + decouple it from aggregate molecule count | rigor | L | S | 🟡 | med |

---

## 3. Per-opportunity detail (selected)

Only the items where the table row needs expansion are detailed here; the rest are
self-contained above. File:line citations are from the audit and should be re-confirmed
at implementation time.

### M-ELONG-1 — Two latent crashes in transcript_elongation 🟢
`transcript_elongation.py`. (a) `update` is initialized without a `"bulk"` key (line ~406);
the stalled-recycle branch (~599-603) does `update["bulk"].append(...)` *before* the
`setdefault("bulk", [])` at ~633 → `KeyError` whenever `recycle_stalled_elongation=True`
and a stall occurs. (b) Line ~584 uses `self.attenuated_rna_indices_lookup`, which is never
assigned in `initialize` (only `attenuated_rna_indices` is) → `AttributeError` the first
time attenuation fires. Fix: initialize `update["bulk"] = []` up front; add
`self.attenuated_rna_indices_lookup = {idx: i for i, idx in enumerate(self.attenuated_rna_indices)}`
in `initialize` (mirrors upstream). Both are 🟢 — outputs identical, they only un-break
currently-broken paths. Add a behavior test that exercises each flag.

### M-DOC-1 — Docstring↔code drift program 🟢
Worst offenders, in priority order:
- **metabolism** — docstring claims `max cᵀv` biomass FBA; code solves
  `homeostatic_kinetics_mixed` *quadratic minimization* (`min (1−λ)Σ(fᵢ−1)² + λ·s·Σ relaxⱼ`).
  Also a phantom `molecular_weight` term in the documented count formula, and ppGpp
  documented as "reduces the growth objective" when it's actually an added homeostatic
  target. The docstring is v2ecoli-authored (no upstream equivalent) → free to correct.
- **polypeptide_elongation** — names nonexistent flags (`include_ppgpp`,
  `steady_state_trna_charging`), gives a ppGpp ODE and a "min f_charged across species"
  rule the code doesn't implement (RelA uses A-site competitive inhibition; `v_rib` uses a
  sum-over-species denominator).
- **chromosome_structure** — see M-CHROMSTRUCT-1 (documents an unexecuted subsystem).
- **polypeptide_initiation** — documents a footprint geometric-spacing overcrowding test;
  code implements a probability cap + rescale. Documented `n_to_activate` formula is a
  different (simpler) model than the termination-balance code.
- **transcript_initiation** — documented `n_to_activate = round(f·n)−n_active` vs the
  real rate-balance activation; ppGpp "fitted functions" that are actually read pre-computed.
- **complexation** — opening paragraph describes a legacy "one reaction, run to
  completion within the timestep" algorithm; code runs a `dt`-bounded Gillespie.
- **rna_degradation** — three mutually inconsistent statements of the water stoichiometry
  (Lᵢ vs Lᵢ−1 vs +1 for the 5′ diphosphate).

These are independent, parallelizable, 🟢, and individually S. Treat as one "doc-truth"
PR series — and add a lightweight CI check (see Deferred) so they can't silently re-drift.

### M-ARCH-1 / M-ARCH-2 — Unify and document the allocation math
See §4. The single concrete deliverable: one `proportional_allocate(requests, available,
priorities, rng)` in (e.g.) `v2ecoli/library/allocation.py`, used by `Allocator`,
`ReconciledStep`, and the standalone path, with the priority-tiered largest-remainder
apportionment + stochastic remainder rounding written down once and unit-tested for
integer conservation and non-negativity.

### M-CHROMSTRUCT-1 — Dead supercoiling subsystem
`chromosome_structure.py`. `update()` never reads `chromosomal_segments`, never calls
`_compute_new_segment_attributes` (~721-1009) or `get_last_known_replisome_data`
(~1012-1087); it sets `update["chromosomal_segments"] = {}` and leaves it. The docstring
(lines 29-39) devotes a section to the `Lk = Tw + Wr` math. **Coupling caveat:** upstream's
empty-fork RNAP removal lives inside the same gated block — deleting the subsystem also
drops that removal, a behavior change beyond supercoiling. Decide deliberately; don't leave
it half-present.

### M-METSIMPLE-1 — Unenforced constraints in the reduced metabolism 🟡
`metabolism_simple.py` declares and builds `exchange_rxn_indices`, `ngam`/`dark_atp`/
`ngam_rxn_idx`, and reads `gtp_to_hydrolyze`, but `update` applies none of them — every
bound is `(None, None)` except enzyme-zeroing. So the LP has no uptake ceiling, no forced
maintenance, and all reactions are reversible. Plus the objective is un-normalized L1
(upstream is fractional/quadratic), and `delta_conc = S@v` contradicts the enforced
`S@v=0`. No composite references this module and it has no behavior test. Recommendation:
either complete it (enforce the three constraint families, normalize the objective, fix the
delta extraction, add a test) or mark it explicitly experimental. 🟡 throughout.

---

## 4. Architecture: partitioning, allocation & operator splitting

**What the architecture actually is, mathematically.** Each timestep the cell state is
advanced by an operator split: every partitioned process first *predicts* its bulk
consumption against the shared state (`calculate_request`), a mediator resolves contention
by **proportional rationing**, then each process *evolves* against its private allocation
and the deltas are merged back. This is a first-order (Lie–Trotter-style) split with a
conflict-resolution projection between the predict and evolve half-steps. Nowhere in the
codebase is this written down as the model's integration scheme, its order of accuracy, or
its conservation guarantees. **M-ARCH-2** is to write it down — this is the core of
"clarify the mathematics of the architecture."

**The allocation kernel.** `calculatePartition` (allocator.py:204) is priority-tiered
largest-remainder apportionment: within each priority level, molecules in excess demand are
shared as `requestᵢ · available / total_requested`, and the fractional remainders are
distributed by a stochastic draw weighted by remainder. This is a clean, conservative,
integer-exact method — but it is **implemented three times with divergent correctness**:

1. `Allocator.calculatePartition` — priority-tiered, the baseline path.
2. `reconcile_requests` (reconciled.py:35) — *single-tier* proportional reconcile + a
   post-hoc `_clamp_bulk_deltas` negativity fix.
3. `DepartitionedStep` — no mediation at all; each process takes exactly what it requests.

Three concrete problems fall out of having three copies:

- **M-ALLOC-2 (latent crash):** the allocator's remainder draw
  `random_state.choice(options, size=count, p=remainder/total_remainder, replace=False)`
  can raise "fewer non-zero entries in p than size" when `count` exceeds the number of
  nonzero-probability options. `reconcile_requests` guards this with
  `count = min(count, len(options))` (reconciled.py:104); the allocator does **not**. The
  correctness already drifted between the copies.
- **M-ARCH-3 (false equivalence claim):** `reconciled`'s docstring claims "the same
  fairness guarantees as the Allocator," but it reconciles at a *single* priority tier.
  When `custom_priorities` differ across processes, the two architectures are **not**
  equivalent — and the architecture-comparison report would silently show a divergence
  attributable to this. Either implement priority tiers in the reconcile, or document the
  limitation honestly.
- **M-ARCH-4 (unsound under contention):** `departitioned` applies each process's full
  request with no mediation, so two processes consuming the same scarce molecule can drive
  its count negative. It is only mathematically safe when no contention occurs. This
  precondition is undocumented and unenforced.

**Layer cleanup that clarifies the math:** the `update_condition` time-gating block is
duplicated verbatim across `Requester`, `Evolver`, and `DepartitionedStep` (and a
map-valued variant in `ReconciledStep`) — a shared mixin would make the clock contract one
thing. And `chromosome_replication` still carries a full `_prepare` request pass whose
result is explicitly discarded (`_ = requests  # no longer used in Step form`), a
Process→Step migration leftover that also duplicates `buildSequences` (M-REPL-1).

**Recommended sequence:** M-ALLOC-2 (the crash guard, trivial) → M-ARCH-1 (extract the one
kernel, which makes M-ALLOC-2/M-ARCH-3 structurally impossible to diverge again) →
M-ARCH-2 (document the splitting scheme) → M-ARCH-3/4 (decide the reconciled/departitioned
guarantees explicitly).

---

## 5. Cross-cutting patterns

- **A — Docstring↔code drift (M-DOC-1).** 8 of 13 audited units flagged it; 3 are severe
  (wrong objective / unimplemented subsystem / contradictory algorithm). Root cause: the
  v2ecoli value-add was expanding upstream's terse docstrings into "Mathematical Model"
  sections, which then drifted as the code was refactored. Highest-ROI theme: all 🟢,
  parallelizable.
- **B — Shadowed `inputs()`/`outputs()` (M-PORTS-1).** 6 files
  (`chromosome_initiation`, `polypeptide_elongation`, `polypeptide_initiation`,
  `rna_degradation`, `transcript_elongation`, `transcript_initiation`) define each method
  twice; Python keeps only the second. Some first copies *omit a real port* (e.g.
  `ppgpp_state`, `attenuation_config`), so the dead copy is also wrong — and a future
  editor may "fix" the wrong one. One mechanical sweep.
- **C — Divergent allocation copies & dropped inline-port guards.** Covered in §4
  (allocation) and Tier-4 (the `equilibrium`/`two_component_system` ODE inlines that
  dropped upstream's negativity/convergence/reversal guards). Same underlying pattern:
  a piece of math got reimplemented and the reimplementation lost guarantees the original
  had.
- **D — Magic hardcoded shapes (M-MAGIC-1).** `complexation` (`1088` reactions),
  `rna_maturation` (`49`/`99`/`3`) hardcode listener-default lengths that are really
  `len(config_list)`. Directly relevant to the *expanded-variant-scope* goal: a variant
  that adds/removes a reaction silently desyncs these.
- **E — Iterative "capped renormalization" loops (M-WATERFILL-1).** The overcrowding caps
  in both `transcript_initiation` and `polypeptide_initiation` are the *same* mathematical
  object — projection onto a capped simplex (water-filling) — implemented as unbounded
  `while` loops. There is a one-pass closed form. Unifying them clarifies the math and
  removes two unbounded loops from hot paths. (The analogous capped-multinomial loops in
  `rna_degradation`/`polypeptide_elongation` are *stochastic* and parity-bound — leave
  those, see Deferred.)
- **F — Per-element Python hot loops (M-VEC-1).** Several updates build NumPy arrays and
  then loop in Python over molecules/transcripts. The behavior-preserving subset (dense
  scatter/gather, masked bincount) is worth doing; the subset that would change the RNG
  draw order is parity-bound — keep separate.

---

## 6. Deferred / rejected (so the next pass doesn't re-litigate)

- **Rewriting the stochastic sampling loops** in `rna_degradation` (`_get_rnas_to_degrade`)
  and `polypeptide_elongation` (`distribution_from_aa`) for speed — **rejected for now.**
  These are correct, their termination is bounded, and any algorithmic change reorders the
  RNG stream → breaks bit-level parity with vEcoli. Document the termination invariant
  instead.
- **Replacing `stochasticRound`-sum with `np.random.binomial`** in `tf_binding` — **deferred.**
  Faster and distributionally near-identical, but a different RNG consumer → parity-breaking.
  Only if exact-trace parity is dropped.
- **Normalizing the FBA mixed objective / forking `modular_fba.py`** (M-MET / metabolism_simple
  objective) — **deferred to the PDMP investigation.** This lives in shared `wholecell` code,
  needs a ParCa re-fit, and the active `investigations/v2ecoli-pdmp/` track is the right place
  for a metabolism reformulation. Flagged so the two efforts don't collide.
- **A docstring↔code CI lint** — **proposed, not scoped here.** The doc-drift theme will
  recur without a guard. A lightweight check (e.g. assert documented flag names exist as
  config keys; assert each written listener key is declared in `outputs()`) would catch the
  M-DOC-1 and M-CHROMSTRUCT-2 classes mechanically. Worth its own small spec.
- **`emit_unique` default mismatch** in `chromosome_structure` (config default `False` vs
  `initialize` default `True`, and never used) — folded into M-CHROMSTRUCT cleanup, not a
  standalone item.

---

## Appendix — coverage

13 auditors, full model: transcript_initiation · transcript_elongation ·
polypeptide_initiation · polypeptide_elongation · rna_degradation · rna_maturation ·
protein_degradation · complexation · equilibrium · two_component_system · tf_binding ·
tf_unbinding · chromosome_replication · chromosome_initiation · chromosome_structure ·
metabolism · metabolism_simple · fba_bridge · trna_attenuation · ppgpp_initiation ·
division. Architecture pass (single context): allocator · partition (Requester/Evolver/
PartitionedProcess) · departitioned · reconciled · global_clock.
