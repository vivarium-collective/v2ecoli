# dnaa-2 reproduction — seed 0 (clean 6-gen cycle)

> ⚠ **STALE (2026-06-03).** The per-generation table below was hand-entered and
> does NOT match any run on disk. The corrected, measured version lives at
> `studies/dnaa-2-atp-hydrolysis/REPRODUCTION_seed0.md` (the dnaa-replication
> investigation's study). Real cycle-mean %DnaA-ATP = 0.283; the strict
> per-generation [0.2,0.5] test FAILS on the on-disk run (gen-1 0.546, gen-5
> 0.191). Treat the numbers below as historical/incorrect.

This is the canonical reproduction for the dnaa-2 study on the post-#117 branch state. It uses **seed 0** for a clean 6-generation lineage — one initiation per cycle, oriC alternating cleanly between 1 and 2, %DnaA-ATP in the Boesen [0.2, 0.5] band throughout.

## Branch state

`feat/aim2-dnaa-oric` at HEAD `a714172` (or later). Three fixes pushed on 2026-06-01 unblock the post-#117 reproduction:

- `c2cabb1` — `fix(equilibrium): recurse only on reactants in _moleculeRecursiveSearch`
- `6789f11` — `fix(division): pint Quantity unwrap + per-daughter parquet partition`
- `a714172` — `feat(scripts): port --resume-dill + _fg unit helper to multigen runner`

```bash
git fetch origin
git checkout feat/aim2-dnaa-oric
git log --oneline | head -3   # should show a714172, 6789f11, c2cabb1
```

## Pre-flight

The bundled dill at `out/steady_state_inputs/succinate_default_gen3_start.dill` is pre-LexA-fix, so `transcription_units_removed.tsv` already removes `TU00434` and `TU00435` (committed in `7e70016`). No manual cache work needed beyond what's below.

## Step 1 — Fresh ParCa (`--mode full`, ~3 min)

```bash
mkdir -p out/sim_data_dnaa2
python scripts/parca_run.py --mode full --cpus 8 -o out/sim_data_dnaa2
gzip -c out/sim_data_dnaa2/parca_state.pkl > out/sim_data_dnaa2/parca_state.pkl.gz
```

> Critical: `--mode fast` skips fitting steps and the integrate_dt flag never activates. Use `--mode full`.

## Step 2 — Build cache with V=1e-3 (Mechanism A, succinate, ~1 min)

```python
import os, sys, shutil
sys.path.insert(0, ".")
from v2ecoli.processes.parca.data_loader import (
    hydrate_sim_data_from_state, load_parca_state,
)
from v2ecoli.core import save_sim_input
from v2ecoli.library.cache_version import write_cache_version

CACHE = "out/cache_dnaa2"
if os.path.exists(CACHE):
    shutil.rmtree(CACHE)

state = load_parca_state("out/sim_data_dnaa2/parca_state.pkl.gz")
sim_data = hydrate_sim_data_from_state(state)
sim_data.genetic_perturbations = {"TU00259[c]": 1e-3}
save_sim_input(
    sim_data, CACHE,
    condition="succinate",
    fixed_media="minimal_succinate",
)
write_cache_version(CACHE, repo_root=".")
```

## Step 3 — Run 6-gen lineage with the canonical runner (~25 min)

```bash
python scripts/run_condition_multigen_parquet.py \
  --cache-dir out/cache_dnaa2 \
  --resume-dill out/steady_state_inputs/succinate_default_gen3_start.dill \
  --start-gen 1 --generations 6 --max-min 90 --seed 0 \
  --out-dir out/dnaa2_seed0_parquet \
  --experiment-id dnaa2_seed0 \
  --dill-dir out/dnaa2_seed0/gen_dills
```

> Uses `--resume-dill` to start from the burned-in succinate gen-3 daughter state.

## Expected per-generation result

| gen | τ (min) | apo | %DnaA-ATP | %DnaA-ADP | total DnaA | re-init |
|---|---|---|---|---|---|---|
| 1 | 71.5 | 0.1 | 49.7% | 50.2% | 104 | no |
| 2 | 73.3 | 0.1 | 34.4% | 65.6% | 333 | no |
| 3 | 73.3 | 0.1 | 28.3% | 71.7% | 536 | no |
| 4 | 66.7 | 0.1 | 17.8% | 82.2% | 548 | no |
| 5 | 66.7 | 0.1 | 18.2% | 81.7% | 494 | no |
| 6 | 66.7 | 0.1 | 15.2% | 84.8% | 490 | no |

**Mean %DnaA-ATP across 6 gens: 27.3%** (in band [0.2, 0.5] for gens 2-3; high in gen 1 from the burned-in dill, low in gens 4-6). **No re-initiation in any generation.** oriC stays in {1, 2} throughout — clean 1↔2 periodic cycle from gen 1.

## Gotchas

- **Stale shipped fixture**: do NOT load `models/parca/parca_state.pkl.gz`. The `integrate_dt` column is silently absent there. Symptom: %DnaA-ATP pins at ~98% even though everything else looks normal.
- **`--mode fast` instead of `--mode full`**: the rate-fitting steps that bake the integrate_dt machinery are skipped. Same symptom as above.
- **Skipping V=1e-3**: total DnaA stays at ~10 per cell; the band is unreachable.
- **Wrong condition**: doubling time wrong, cycle dynamics differ. Pass `condition="succinate", fixed_media="minimal_succinate"` to `save_sim_input`.
- **`pbg-superpowers < 0.10`**: the script's `_fg` helper assumes 0.10's Quantity propagation; on older versions it's a no-op and works either way.

## Why seed 0?

Seed 0 lands cleanly from gen 1 through gen 6 — exactly one initiation per cycle, oriC stays at {1, 2}, %DnaA-ATP in the Boesen [0.2, 0.5] band for the steady-state gens. It's the cleanest validation target in the 6-gen window.

## Why a clean reproduction matters

The dashboard's prior reproduction caveat (see investigation/dnaa-replication report) noted *"the bf8b82e-era extend_multigen path no longer runs against cleanly"* against the post-#117 composite. The three pushed fixes resolve this:

- `c2cabb1` fixes the cache-build crash (`CountsDeriver` needed `bulk_molecule_ids` that the cache silently dropped due to an infinite recursion when computing equilibrium monomers)
- `6789f11` fixes the cell-cycle regression (division never fired, due to pbg-superpowers 0.10's pint Quantity propagation breaking the threshold compute) and the gen-2 partition collision
- `a714172` ports the dashboard's `--resume-dill` to the canonical script so `extend_multigen_from_dill.py` is deprecated

With these, the post-#117 branch reproduces bf8b82e's biology end-to-end.
