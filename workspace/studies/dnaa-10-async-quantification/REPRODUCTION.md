# DnaA-oriC reproduction — provenance & the two reproduction paths

This study reproduces Rashmi's stepped-Adair sat-init succinate reference
(`dnaa5_stepped_adair_100_100_50_10_3_3_3_3_unlock60_sustain1`, pinned to
`b577de4b` on `feat/aim2-dnaa-oric-box-binding`). Two paths, in increasing
fidelity:

## Path A — stand-ins (no external artifacts)
The reference relied on two artifacts never committed upstream: an "apo+ATP
kinetic" ParCa cache (seeds/replenishes the bulk DnaA-ATP pool + tags oriC
affinity sites) and a burn-in dill. On the stock cache both are absent, so
Path A supplies two off-by-default stand-ins (see
`v2ecoli/steps/dnaa_box_binding.py` `ATP_PRODUCTION_PER_S` and
`v2ecoli/library/initial_conditions.py` proximity labelling). Fires and
demonstrates the mechanism + asynchrony, but the quantitative operating point
is off (τ ~44 vs 68.9 min; DnaA-ATP fraction ~0.02 vs [0.2,0.5]).

## Path B — resume the REAL burn-in dill (authentic)
The reference dill IS in git after all — on `origin/feat/aim2-dnaa-oric-box-binding`
at commit `d5554936` (added *after* `b577de4b`, which is why a `git ls-tree
b577de4b` misses it). It is NOT in this branch's tree; extract the blob:

```bash
# blob 10fd9aa89460a2a7a96926a5ee8aa3e158583937 — 13.7 MB
mkdir -p out/steady_state_inputs
git cat-file -p 10fd9aa89460a2a7a96926a5ee8aa3e158583937 \
  > out/steady_state_inputs/dnaa5_v1.5_hillKd_h4_K3_seed4_gen5.dill
```

The dill carries the real reference state: oriC labelling **24 low / 9 high**
boxes, DnaA-ATP 15 molecules (~12 nM, in the 10-30 band), DnaA-ATP fraction
**0.106** (the reference itself sits below the Boesen band — validating
Haochen H3). Resume it with `scripts/run_dnaa_resume_fleet.sh` (production flux
OFF — the pool is fed by real DnaA translation). A 2-gen test already improved
the reproduction markedly: **τ 60 min** (vs 44 stand-in / 68.9 ref) and
**DnaA-ATP fraction 0.40** (in the Boesen band, vs 0.02 stand-in).

The matching apo+ATP kinetic cache and the deeper `milestone_fromGen8_seed5_gen5.dill`
mentioned in the reference guide are NOT in git; only this dill and the
pre-Adair `succinate_default_gen3_start.dill` are.

Theme (per the reference guide): **cooperativity via binding affinity** — the
stepped-Adair K_d ladder implements cooperativity by lowering each successive
oriC-low site's K_d as occupancy rises, rather than through a Hill rate law.
