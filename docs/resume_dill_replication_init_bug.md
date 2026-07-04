# Tooling bug: dill-resume of a saved generation breaks replication initiation — reproduction + fix handoff

**Status:** open · **Scope:** multigen runner / `divide_cell` pickle-fidelity (tooling, not biology) · **Target:** its own PR.

## Summary

`scripts/run_condition_multigen_parquet.py` can checkpoint a generation's final state to a `.dill` and later resume a fresh lineage from it (`--resume-dill gen{N}.dill`). **The resumed daughter never re-initiates replication** — `number_of_oriC` stays pinned at 2, the cell grows to ~2× normal mass without dividing, hits the per-gen time cap, and the lineage terminates. The *same* cell state divided fine when continued **in-process** (no dill round-trip), so the defect is in the **dill serialize→deserialize of the cell state**, which corrupts or drops the replication-initiation / unique-molecule state that `divide_cell` depends on. This blocks the "resume from a steady-state generation to confirm stability" protocol.

## The discriminating evidence

`gen{N}.dill` is exactly the `last_state` that the *continuous* run divides in-memory to make gen N+1. So resume-from-dill should be identical to the continuous path. It is not:

| Path | seed | gen N+1 outcome |
|---|---|---|
| **Continuous** (in-process `divide_cell(last_state)`) | 11 | divides normally, mean DnaA 246, oriC cycles 2→4→2 |
| **Resume** `--resume-dill gen10.dill --start-gen 11` | 11 | **no division**: 7638 steps, mass 522→1020 fg, oriC stuck at 2 |
| **Resume** `--resume-dill gen10.dill --start-gen 1` | 1 | **no division**: 7089 steps, mass 522→1020 fg, oriC stuck at 2 |

Two different seeds → byte-identical pathology (mass 522→1020, oriC=2, ~7k steps, never divides). The only variable that predicts the failure is "was the state round-tripped through dill," not the RNG seed and not the autoregulation (the dnaa-4 s=0.7 Hill feedback) — the continuous 16-gen autoreg runs divide cleanly every generation.

## Reproduce

```bash
# On investigation/dnaa-replication-v3 with out/cache_dnaa4_autoreg + a completed
# multigen run that saved per-gen dills (out/<exp>/gen_dills/gen{N}.dill).
DNAA_AUTOREG_STRENGTH=0.7 DNAA_AUTOREG_FORM=hill DNAA_HYDROLYSIS_RATE_PER_MIN=0.025 \
PYTHONPATH=. .venv/bin/python scripts/run_condition_multigen_parquet.py \
  --cache-dir out/cache_dnaa4_autoreg --out-dir /tmp/resume_repro --experiment-id resume_repro \
  --generations 6 --max-min 180 --seed 0 --perturbation "TU00259[c]=1.5e-3" \
  --resume-dill out/dnaa4_s07_seed0_16gen/gen_dills/gen10.dill --start-gen 11
# -> first (and only) generation runs ~7600 steps, dry mass ~520->1020 fg, divided=False,
#    "did not divide — stopping lineage". Inspect number_of_oric in the parquet: pinned at 2.
```

## Where the state flows (file references)

- **Checkpoint write:** `scripts/run_condition_multigen_parquet.py:457-461` — `prev_cell_data = last_state; dill.dump(last_state, f)`.
- **Resume read:** same file `:393-395` — `prev_cell_data = dill.load(f)`.
- **Daughter construction:** same file `:428` — `d1_state, _ = divide_cell(prev_cell_data)`, fed as `initial_state` into a fresh `baseline_doc`/`Composite`.
- **The divider:** `v2ecoli/library/division.py:254 divide_cell()` — rebuilds daughters from `cell_state['unique']` (full_chromosome, chromosome_domain, active_RNAP, replisomes, oriC, …) via per-type `UNIQUE_DIVIDERS`. These are numpy (often **structured**) arrays. If dill alters their dtype, field layout, index/domain linkage, or an "unknown molecule type → just copy to both" array (`division.py:286-289`) that should have been domain-divided, the daughter's replication machinery is inconsistent and never fires a new round.

## Likely root + suggested diagnostic

The continuous path keeps `last_state`'s arrays live in memory; the resume path reconstructs them from dill. So the dill round-trip is dropping/altering something in `cell_state['unique']` (or a process-internal field needed by initiation). **Isolate it without running a sim:**

```python
import dill, numpy as np
from v2ecoli.library.division import divide_cell
orig = dill.load(open("out/dnaa4_s07_seed0_16gen/gen_dills/gen10.dill","rb"))
roundtrip = dill.loads(dill.dumps(orig))
# 1) structural diff of the unique stores
for k, a in orig["unique"].items():
    b = roundtrip["unique"][k]
    if getattr(a, "dtype", None) != getattr(b, "dtype", None) or np.shape(a) != np.shape(b):
        print("DIFF unique", k, getattr(a,"dtype",None), "->", getattr(b,"dtype",None), np.shape(a), np.shape(b))
# 2) diff the daughters the two states produce
do1,_ = divide_cell(orig); dr1,_ = divide_cell(roundtrip)
# compare do1["unique"] vs dr1["unique"] field-by-field; the first mismatch is the culprit
```

Whatever key first diverges (most likely a chromosome-domain / replisome / oriC structured array, or one falling through the `UNIQUE_DIVIDERS.get(name) is None` copy-to-both branch) is what to fix — either make that store dill-stable, or have `divide_cell`/the resume path reconstruct it deterministically rather than trusting the pickled arrays.

## Scope / acceptance

- Tooling fix in the resume/checkpoint path (or `divide_cell` pickle-fidelity), independent of any investigation's biology. Own PR.
- **Acceptance:** resuming from `gen{N}.dill` produces a daughter that re-initiates replication (oriC cycles past 2) and divides on a normal timescale — i.e. the resumed lineage is statistically indistinguishable from the continuous run's gens N+1… A direct check: `divide_cell(dill_roundtrip(last_state))` yields daughter unique-stores field-identical to `divide_cell(last_state)`.
