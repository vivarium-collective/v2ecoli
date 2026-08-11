# dnaa-3 box-binding — alternative implementation for comparison

Branch: `feat/aim2-dnaa-oric-box-binding` (4 commits ahead of `feat/aim2-dnaa-oric`).

Two new commits on top of the existing Phase 1 schema work:

- `4fe5cde` feat(dnaa-3 Phase 2): DnaA-box equilibrium binding + bound-pool hydrolysis
- `c648d51` feat(plots): dnaa-3 Phase 2 plotting scripts

The two earlier commits already on the branch
(`93f1129` Phase 1 schema additions, `5998a83` known-box catalog) supply the
chromosomal box coordinates and unique-molecule slots that the binding step
consumes.

## What this branch is

An alternative in-sim DnaA-box binding implementation offered as a possible
approach, not a claim that it's the correct one. Posting here for Eran and
the dashboard agent to compare against the existing read-only occupancy
listener and decide which direction to take.

The mechanism: a fast-equilibrium Langmuir solver (`scipy.optimize.root` on
the 2-eq mass-balance system in `A_free`, `D_free`) actively binds DnaA-ATP
and DnaA-ADP to the 307 chromosomal + 11 oriC + 4 dnaA-promoter boxes,
updating the bulk pool each tick. Bound-pool hydrolysis is sampled in the
binding step and the byproducts routed through FBA via a shared port to
`equilibrium.py`.

Spec parameters used (matching `studies/dnaa-03-box-binding/study.yaml`):
- 307 chromosomal high-affinity (K_d = 1 nM, binds ATP or ADP)
- 3 oriC high-affinity (K_d = 1 nM, binds ATP or ADP)
- 8 oriC low-affinity (K_d = 100 nM, ATP only)
- 2 + 2 promoter high-affinity (K_d = 1 nM)
- Intrinsic hydrolysis k = 0.046/min on free + bound DnaA-ATP

## How this differs from the read-only listener on `feat/aim2-dnaa-oric`

The existing implementation computes occupancy as `P = C / (C + K_d)` with
`C = total DnaA-ATP`. The dashboard report
(`investigation-dnaa-replication-2026-06-10`) flagged this as an open
question:

> "Free DnaA-ATP ~380–550 nM (≈4× the 100 nM oriC-low K_d) for 100% of every
> cycle across 7 gens, oriC-low occupancy ~0.8 throughout — NOT resolved by
> the correct DnaA-ATP fraction (0.23, in band). An honest open question
> within the provided mechanisms for Rashmi/Haochen."

The difference between the two approaches is whether DnaA-ATP binding
actively depletes the bulk pool or not. With active depletion the free
pool the next tick of equilibrium / TF binding sees is smaller; the
listener-only approach leaves the bulk pool intact and uses the total for
the `P = C/(C+K_d)` calculation.

| | Read-only listener (current) | This branch |
|---|---|---|
| Pool used in P calc | C = total DnaA-ATP | A_free from mass-balance solve |
| Bulk depleted by binding? | no | yes |
| oriC_low occupancy across cycle | ~0.8 throughout | partial fill, rises pre-init (see V=1.0 PDF) |

Whether the depleted-pool dynamics are the right physics is the question
for Eran / Haochen.

## Validation runs available

Two combined PDFs in `out/figures/` showing this branch's behaviour at the
V values that Eran's `dnaa-1` study explored:

- **`dnaa3_phase2_v1e3_steadystart_combined.pdf`** — V = 1.0e-3, steady-state
  start (resume from gen 5 of a prior burned-in lineage, `--seed 7 --start-gen 1`).
  8/8 clean divisions, DnaA mean 574–712 (mid-band), peaks 704–1045,
  ATPfr 0.13–0.20.
- **`dnaa3_phase2_v1.2e3_sharedhyd_seed0_combined.pdf`** — V = 1.2e-3, cold
  seed=0. 6 clean gens with ATPfr 0.20–0.26 in spec band, DnaA peaks
  624–954.

Each PDF has 6 pages: schematic, 6-panel trajectory, 4-panel box partition,
nM concentrations with K_d reference lines, raw counts with [300, 800] target
band, gen-4 per-chromosome region snapshots.

The Haochen 2026-06-07 validation covered the equilibrium physics for this
branch at V = 1.2e-3:

> "Our scipy.root + fork-release + Langmuir-with-free physics IS the correct
> equilibrium. Our sim correctly shows oriC_low at ~5/8 (62%) at initiation."

(Validation was on a pre-shared-hyd code state. Current shared-hyd + bound-
pool routing has not been independently re-validated.)

## Phase 2 additions on top of bf8b82e

1. **bound-pool hydrolysis routing** (`v2ecoli/processes/equilibrium.py`):
   `dnaa_box_binding` samples a hydrolysis count once for the bound pool,
   writes to a `process_state` port; equilibrium reads it and injects a
   matching `DNAA-INTRINSIC-HYDROLYSIS-RXN` flux so the byproducts (Pi,
   PROTON, −WATER) go through FBA. The bulk DnaA-ATP / DnaA-ADP delta for
   the bound portion is then reverted so the form swap stays in-place on
   the unique DnaA_box rows.

2. **fork-passage DnaA release** (`v2ecoli/processes/chromosome_structure.py`):
   when a replication fork crosses a DnaA box, bound DnaA-ATP / DnaA-ADP is
   released back to bulk instead of being silently destroyed when the parent
   box is deleted.

3. **per-pool listener** (`v2ecoli/steps/derivers/replication_data.py`):
   emits 11 occupancy counts + per-box arrays.

## Open issues we'd want Eran / Haochen to weigh in on

1. **Does the depleted-pool framing match what Haochen and the dnaa-3 spec
   intend?** Both approaches are defensible Langmuir treatments of the same
   physics — the choice between "binding depletes the bulk pool the
   equilibrium / TF binding processes see" vs "binding is a derived
   observation that doesn't feed back" is a modeling decision.

2. **DnaA self-autoregulation is dropped at runtime.**
   `fold_changes_nca.tsv` has `dnaA → dnaA = −2.31 log2 FC`, but the runtime
   `delta_prob[TU00259[c], MONOMER0-160] = 0.0` exactly (other 10 DnaA
   targets keep meaningful coefficients). The self-edge specifically lands
   at zero from the L1-norm promoter fit. This means there's no active
   negative feedback on dnaA transcription from box binding regardless of
   which binding implementation is chosen — Mechanism A's V override is
   doing the regulatory work. A possible stopgap is to post-patch
   `delta_prob` after `calculateRnapRecruitment`.

## Reproducing

```bash
# V=1.0 steady-state start
.venv/bin/python scripts/run_condition_multigen_parquet.py \
  --cache-dir out/cache_dnaa2_v1e3 \
  --out-dir out/dnaa3_phase2_v1e3_steadystart_parquet \
  --experiment-id dnaa3_phase2_v1e3_steadystart \
  --generations 8 --max-min 180 --seed 7 \
  --resume-dill out/dnaa3_phase2_v1e3_burnedin_seed2/gen_dills/gen5.dill \
  --start-gen 1

# V=1.2 spec-band ATPfr
.venv/bin/python scripts/run_condition_multigen_parquet.py \
  --cache-dir out/cache_dnaa2_v1.2e-3 \
  --out-dir out/dnaa3_phase2_v1.2e3_sharedhyd_seed0_parquet \
  --experiment-id dnaa3_phase2_v1.2e3_sharedhyd_seed0 \
  --generations 8 --max-min 180 --seed 0
```
