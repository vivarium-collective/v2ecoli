# Parameter-map doc — revisions per Haochen's feedback

This companion note summarises the changes made to
`dnaa5_oric_saturation_parameter_space.md` in response to Haochen's three
comments on the first version, and flags what is still open.

---

## Haochen's comment 1 — scan shape parameters within a chosen form

> Part 3. Cooperativity functional form at oriC: when exploring different
> function forms, also explore parameters of those forms. For example, for
> Hill function, not only explore Hill coefficient n (how sharp the
> transition is), but also scan different K (where the transition happens).

**Change made:** §3 now ends with an explicit note that within a chosen form,
the shape parameters must be scanned too — for Hill specifically, scan **both**
the Hill coefficient `h` (sharpness) **and** `K` (midpoint).

---

## Haochen's comment 2 — hydrolysis at oriC should not be the cluster-reset mechanism

> Part 5. DnaA-ATP hydrolysis at oriC (cluster reset): I'm confused by this
> part, because cluster reset should always happen right after initiation
> because all bound DnaA at oriC should fall back to the cytoplasm at that
> moment. Because the time window of oriC saturation is typically much shorter
> than DnaA-ATP half-life, the hydrolysis of DnaA-ATP bound at oriC should be
> negligible and should not be the main reason for the cluster reset. Please
> double-check if this is true in the current implementation.

**Code check:** the comment in `dnaa_box_binding.py` explicitly says *"fork
release is what actually destroys the state."* So fork release at initiation
IS the dominant reset mechanism in the current implementation. Hydrolysis at
oriC runs in the background at `k_h = 0.025/min` (uniform across all
DnaA-ATP pools, bound and free), but it is not what cleans out the cluster
after initiation.

**Change made:** §5 (the hydrolysis / cluster-reset section) has been
removed entirely. It was framing hydrolysis as a tuning lever for cluster
reset, which is incorrect. Sections downstream have been renumbered;
the hydrolysis entry has also been dropped from the ranked-knobs list.

---

## Haochen's comment 3 — try real initiation triggering by oriC saturation

> One thing I realize is that maybe we should do real initiation triggering at
> this stage — not artificially by mass, but by oriC saturation events. My
> concern is that initiation itself is a feedback to oriC state — if oriC
> saturation is too fast, initiation happens early, which will generate more
> DnaA boxes during replications and titrate redundant DnaA-ATP; if oriC
> saturation is too slow, initiation will not happen, allowing DnaA-ATP to
> keep accumulating, until oriC saturates. Without this feedback, you might
> always see over- or under-saturation at oriC, as you and Eran recently
> obtained, but it can be an artifact by mass-triggering initiations. The
> caveat is that this could mess up the regular cell cycles (being very long
> or short) and affect other processes in the background. I would still give
> it a try and see what would happen.

**Status:** open. Not yet incorporated as a documented parameter-space axis.

**Next step proposed:** replace the mass-clock gate in
`chromosome_replication.py` (currently `criticalMassPerOriC ≥ 1.0`) with an
oriC-saturation gate (init fires when at least one chromosome's oriC_low
cluster is fully bound). This is a code change rather than a parameter sweep;
to be tried as a focused experiment with the current best lineage config
(V=1.5 + Hill K_d) to see whether the feedback Haochen describes does in fact
clamp the late-gen drift.

