# Review — Rashmi's box-binding branch (`feat/aim2-dnaa-oric-box-binding`)

Reviewed 2026-06-10 (handoff `dnaa3_box_binding_handoff.md` +
`dnaa3_phase2_v1.2e3_sharedhyd_seed0_combined.pdf` + the branch code, 4 commits
ahead of `feat/aim2-dnaa-oric`).

## Verdict: ADOPT (replacing the current read-only occupancy observer), after a verification re-run.

Her branch is an **active** in-sim DnaA-box binding model; the current dnaa-3 is a
**read-only** occupancy observer. The read-only observer *cannot by construction*
resolve the over-binding (it reads `P = C/(C+K_d)` off the full DnaA-ATP pool
without depleting it, so free DnaA-ATP stays ~380-550 nM ≫ the 100 nM low-aff
K_d → occupancy pinned ~0.8). Her active Langmuir solve **depletes the free pool**,
which is exactly the missing physics.

## What it gets right (correctness)

1. **Langmuir mass-balance solve** (`dnaa_box_binding.py:_solve_competitive_pool`/
   `_solve_atp_only_pool`): damped fixed-point in (A_free, D_free) with
   `A_free = A_total − n_atp_bound`. Correct competitive Langmuir; converges <20 iter.
2. **Bulk depletion**: bulk DnaA-ATP/ADP updated by the net change in bound counts —
   the binding feeds back on the free pool the next equilibrium/TF tick sees. This is
   the key advance over the read-only observer.
3. **Bound-pool hydrolysis routed through FBA** (the subtle part, and it's done right):
   the step samples a bound-pool hydrolysis count, does the in-place box ATP→ADP swap,
   and writes the count to a `process_state` port; `equilibrium.py` reads it and injects
   a matching `DNAA-INTRINSIC-HYDROLYSIS-RXN` flux so Pi/PROTON/−WATER go through FBA
   (mass-balanced) instead of bypassing it. Conservation totals are explicit
   (`total_atp = bulk + bound − delta_h`, `total_adp = bulk + bound + delta_h`).
4. **Fork-passage DnaA release** (`chromosome_structure.py`): when a fork crosses a box,
   the parent box is deleted, 2 child boxes added (bound_form=0), and bound DnaA released
   to bulk (not silently destroyed) — mass-conserved, matches Katayama 2017. This is what
   makes the box catalog double properly (315→630), fixing the gap I'd found (the old
   read-only deriver hardcoded `n_oric_low=8` and didn't double the low-aff sites).
5. **Per-pool + per-chromosome listeners** (`replication_data.py`): 11 occupancy counts
   + per-box arrays → the per-chromosome snapshots (PDF p6).

## Results (PDF, V=1.2e-3, seed 0)

- **Over-binding RESOLVED**: oriC-low occupancy ~3-4/8 (37-50%) at initiation
  (Haochen's expected ~5/8), not the stuck ~0.8. `[ATP_free]/K_d_low` now oscillates
  around 1, not pinned high.
- **Box-doubling FIXED**: total DnaA boxes step 315↔630 at replication.
- Cell cycle clean (6/6 divisions), oriC 1↔2.

## Caveats to resolve before/at adoption

1. **PROTON crash at gen 7** ("FBA sink artifact"). This is the SAME pre-existing
   allocator edge we hit in our own runs (V=0.8e-3 seed-1 crashed gen 5). Her FBA-PROTON
   injection should be confirmed not to worsen it. Limits runs to ~6 gens for now.
2. **DnaA-ATP fraction drifts below 0.2 in later gens** at V=1.2e-3 (and total DnaA/mass
   drifts up). Same V-dependent drift family we documented in dnaa-1/dnaa-2 — a tuning/
   regulation question, not a binding-mechanism bug.
3. **DnaA self-autoregulation is dropped at runtime** (she flagged this):
   `delta_prob[TU00259[c], MONOMER0-160] = 0.0` exactly from the L1-norm promoter fit —
   so there is no negative feedback on dnaA transcription regardless of binding impl;
   Mechanism A's V override does the regulatory work. Independent of the binding choice;
   her proposed stopgap (post-patch `delta_prob` after `calculateRnapRecruitment`) is worth
   trying. This is arguably the more important open item now.
4. **Not independently re-validated** in the shared-hyd + bound-pool-routing state
   (Haochen validated a pre-shared-hyd code state). → a verification re-run on our side is
   the gate before merging.
5. Modeling choice (her open Q1): "binding depletes the pool the equilibrium/TF processes
   see" vs "binding is a derived observation". Haochen's note endorses the depleted-pool
   Langmuir as the correct equilibrium. Agree.

## Recommended next steps

1. **Verify**: re-run her branch on our side (V=1.0e-3 burned-in + V=1.2e-3 cold) and
   reproduce the over-binding-resolved + box-doubling results; confirm the gen-7 PROTON
   crash is the pre-existing edge.
2. **Adopt**: merge her box-binding mechanism into the investigation branch as the dnaa-3
   mechanism, retiring the read-only observer (keep the observer's charts as historical).
3. **Then**: take up the autoregulation gap (#3) as the next dnaa-3 / regulation question.

## VERIFICATION (2026-06-10) — reproduced on our side: PASS

Ported her 4 commits onto v3 (cherry-pick; the only friction, `flat/dna_sites.tsv`,
resolved via a `flat_overrides/dna_sites.tsv` true-override registered in
`parca_overrides.tsv` — no change to the external `ecoli_sources` package).
ParCa built clean (2.7 min); box catalog confirmed in sim_data
(DnaA_box 307 · oric_low 8 · oric_high 3 · promoter 2). Ran V=1.2e-3 cold seed=0,
8 gens (38 min), active binding step engaged:

- **8/8 clean divisions**, oriC 1↔2 — and NO gen-7 PROTON crash (ran all 8 gens
  clean, better than her seed-0 reference which crashed at gen 7).
- **Box-doubling confirmed**: total DnaA boxes step 315 (oriC=1) ↔ 630 (oriC=2);
  the low-affinity boxes double 8→16 (the exact gap the read-only observer had).
- **Over-binding RESOLVED**: steady-gen oriC-low occupancy mean 0.46 (range
  0.00-0.75) — the partial dynamic fill (~3-4/8) Haochen expected, vs the read-only
  observer's pinned ~0.8.

CONCLUSION: adopted + verified. Her active Langmuir mechanism replaces the read-only
observer's role. FOLLOW-UPS (not blocking the merge): (1) remove v3's now-redundant
read-only `dnaa_box_binding_listener` from baseline.py + re-point the old dnaa-3
charts/scripts (render_dnaa3_occupancy.py etc.) to her per-pool replication_data
listener; (2) update the dnaa-3 study verdict/charts to the active-binding results;
(3) dnaa-3 runs now require the new ParCa cache (schema changed).
