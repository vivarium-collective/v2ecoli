# Response to Haochen's feedback (2026-06-27)

Point-by-point reply to the three comments on the oriC-saturation parameter-space
plan, plus the report question about lowering K. Evidence is from the dnaa-5/dnaa-6
runs against the succinate reference cache (`cache_dnaa4_s06_F05`).

---

## H1 — Scan shape parameters (n *and* K); show the K-lowering result

**Comment:** for the Hill form, scan both the coefficient n (sharpness) and K
(midpoint), not just n. Report question: it mentioned lowering K but didn't *show*
the result — does lowering K increase the chance of oriC saturation?

**n — already scanned (dnaa-5).** n = 1/2/4/6/8 (analytic `dnaa5_switch_curve` +
in-sim): n=4 is the operating point (sharp switch, homeostasis preserved); n=6
over-depletes the pool; n=1 recovers the gradual Langmuir. So sharpness is covered.

**K — in-sim scan (this round).** K = 15/20/25/30/40 nM at n=4, 8 generations, under
the **mass trigger** (the regime where dnaa-5 showed late-generation under-saturation:
per-gen saturation episodes `[11,93,95,0,0,0,0,0,0]`). Metric: per-generation
oriC-low saturation events (does the cluster reach the fire threshold each gen?).

**Result: lowering K increases oriC saturation chance — monotonically.** Per-generation
oriC-low saturation (1 = the cluster reached the fire threshold that gen, debounced):

| K (nM) | per-gen saturation (g1→) | gens that saturate | steady ATP fraction |
|---|---|---|---|
| 15 | 1,0,0,1,1,1,1,1,(0) | 6/8 | 0.082 |
| 20 | 1,0,1,1,1,2 *(lineage ended g6)* | 5/6 | 0.12 |
| 25 | 0,0,1,1,1,0,1,1,(0) | 5/8 | 0.087 |
| **30 (ref)** | 0,0,0,0,1,0,1,1,(0) | **3/8** | 0.126 |
| 40 | 0,0,0,1,0,0,0,1,(0) | **2/8** | 0.079 |

**Conclusion:** at the reference K=30 most generations FAIL to saturate (the cluster
can't reach threshold as free DnaA-ATP drifts below the midpoint — the dnaa-5
late-gen `[...,0,0,0,0]` problem); raising K to 40 is worse (2/8); **lowering K to
15–25 trips the switch left so the cluster saturates in ~5–6 of 8 gens.** So yes —
lower K raises the chance of oriC saturation, monotonically, confirming your
intuition. (Caveat: it does NOT close the ATP-fraction band — all K stay ~0.08–0.13,
below [0.2,0.5]; that's the recharge-rate k_r / code item, not K. Also brief gen-1
oriC→4 transients at K=20/30, and the K=20 lineage stopped at gen 6 — a mild
instability worth a multi-seed check before settling on an operating K.)

**Important framing:** under the **mechanistic** trigger (H3), this question is largely
moot — firing on oriC saturation already produces reliable once-per-cycle saturation
regardless of K. The mass-trigger K-scan above isolates K's effect on saturation
*chance* in the regime where it was failing.

---

## H2 — Hydrolysis at oriC is NOT the cluster-reset mechanism ✅

**Comment:** reset should happen at initiation (bound DnaA falls back to the
cytoplasm); since the oriC-saturation window << DnaA-ATP half-life, oriC-bound
hydrolysis is negligible and shouldn't be the reset. Double-check the code.

**Confirmed — you're right.**
- `dnaa_box_binding.py` is explicitly *"pure bookkeeping — does NOT gate replication
  initiation."* The oriC box pools are set in `initial_conditions.py` and
  **propagated by `chromosome_structure.py` on fork passage** — i.e. the cluster is
  reset at initiation when the fork passes, not by hydrolysis.
- k_h = 0.025/min runs uniformly across all DnaA-ATP pools (bound + free) in the
  background; it is not what cleans out the cluster after initiation.
- **Empirical corroboration:** the stage-3 k_h triage (k_h = 0.010/0.015/0.020 vs
  0.025) did **not** change the once-per-cycle reset, and the observed reset
  (`oriC_low_bound_atp` → 0) coincides exactly with oriC 1→2 (initiation), not a
  gradual hydrolytic decay.

So k_h is not a cluster-reset lever; the "§5 hydrolysis as a reset tuning knob"
framing was incorrect and is dropped from the plan.

---

## H3 — Try real initiation triggering by oriC saturation (not mass) ✅

**Comment:** do real triggering by oriC saturation. Initiation is a feedback on oriC
state; without it you may always see over/under-saturation (as Eran & I saw), which
could be a mass-triggering artifact. Try it.

**Already done (dnaa-6, finding F-04) — and it confirms your hypothesis.**
Replacing the mass clock with the oriC-saturation gate
(`DNAA_INITIATION_TRIGGER=mechanistic`, `DNAA_INIT_TRIGGER_POOL=low`, fire when
`oriC_low_bound_atp ≥ 6`) gives:
- **clean once-per-cell-cycle initiation** — exactly one fill→fire→reset per gen;
- **oriC strictly 1↔2** (never 3/4), 6–8 divisions, no runaway cycle length;
- **seed-robust** across seeds 0/1/2 over 8 generations;
- with the cooperative switch alone — SeqA eclipse / RIDA not required.

This directly supports your reasoning: the over/under-saturation under the mass
trigger (the dnaa-5 `[11,93,95,0,...]` signature) **was a mass-triggering artifact**;
the mechanistic feedback clamps it. The earlier "over-initiates to oriC→4" was a
property of the older `59e108fb` build and does not reproduce on current code.

**Open (both code-level, not parameter sweeps):** the DnaA-ATP *fraction* stays
seed-variable / mostly below [0.2,0.5] (lever = the code-baked recharge rate k_r), and
firing is a brief fill-spike rather than a sustained full-saturation plateau (would
need a dynamic-kick with persistence).
