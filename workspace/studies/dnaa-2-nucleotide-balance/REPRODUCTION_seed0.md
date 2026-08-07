# dnaa-2 reproduction — seed 0 (V=1e-3, 6-gen burned-in lineage)

Reproduction protocol for the dnaa-2 study on the post-#117 branch state
(`feat/aim2-dnaa-oric`). Uses **seed 0**, V=1e-3, burned-in resume from the
succinate gen-3 steady-state dill, 6 generations.

> **Status (2026-06-03): QUALIFIED, not a clean reproduction.** On the run this
> doc produces (`out/dnaa2_seed0_parquet`), the DnaA-ATP fraction lands in the
> Boesen [0.2, 0.5] band on a **cycle-mean basis (0.283)** but **not strictly
> every generation**: gen-1 = 0.546 (burned-in-resume transient, above band) and
> gen-5 = 0.191 (just below). The MECHANISM (Haochen's bf8b82e integrate_dt) is
> validated — the fraction is driven off the ~0.997 fast-equilibrium pin into the
> band region. A clean per-generation in-band 7-gen reproduction (Rashmi's) is
> still needed for a strict PASS. The earlier hand-entered table in this file was
> stale (did not match any run on disk) and has been replaced with the measured
> values below.

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

## Measured per-generation result (from `out/dnaa2_seed0_parquet`, 2026-06-03)

Values extracted directly from the bulk species on the followed all-zeros lineage
(<5-min daughter-stub generations dropped):

| gen | τ (min) | %DnaA-ATP | total DnaA | oriC | in [0.2,0.5]? |
|---|---|---|---|---|---|
| 1 | 71.3 | 54.6% | 131 | 2 | **no** — above (burned-in transient) |
| 2 | 66.6 | 28.4% | 275 | 2 | yes |
| 3 | 60.0 | 25.1% | 418 | 2 | yes |
| 4 | 66.6 | 20.9% | 491 | 2 | yes |
| 5 | 73.3 | 19.1% | 472 | 2 | **no** — just below 0.2 |
| 6 | 73.3 | 21.4% | 625 | 2 | yes |

**Cycle-mean %DnaA-ATP across the 6 gens: 28.3%** (in band). apo ≈ 0.000 every
gen; background **[ATP]:[ADP] ≈ 20.3**; oriC stays clean at 2 throughout (the
burned-in resume starts at steady state — no overshoot to 4).

**Strict per-generation [0.2, 0.5] test: FAILS** (gen-1 0.546 above, gen-5 0.191
below). The run is a cycle-mean pass, **not** a clean per-generation reproduction.

## Extraction recipe (how the table above was measured)

DnaA forms are read straight from the bulk store (no bespoke listener — the
bf8b82e equilibrium machinery interconverts the three molecules):

- apo-DnaA = `PD03831[c]` · DnaA-ATP = `MONOMER0-160[c]` · DnaA-ADP = `MONOMER0-4565[c]`
- background pools = `ATP[c]`, `ADP[c]`
- follow only the canonical all-zeros lineage: `agent_id` matching `^0+$`
- fraction = `MONOMER0-160[c] / (PD03831[c] + MONOMER0-160[c] + MONOMER0-4565[c])`

See `scripts/render_dnaa2_sixpanel.py` for the full per-panel logic.

## Gotchas

- **Stale shipped fixture**: do NOT load `models/parca/parca_state.pkl.gz`. The `integrate_dt` column is silently absent there. Symptom: %DnaA-ATP pins at ~98% even though everything else looks normal.
- **`--mode fast` instead of `--mode full`**: the rate-fitting steps that bake the integrate_dt machinery are skipped. Same symptom as above.
- **Skipping V=1e-3**: total DnaA stays at ~10 per cell; the band is unreachable.
- **Wrong condition**: doubling time wrong, cycle dynamics differ. Pass `condition="succinate", fixed_media="minimal_succinate"` to `save_sim_input`.
- **`pbg-superpowers < 0.10`**: the script's `_fg` helper assumes 0.10's Quantity propagation; on older versions it's a no-op and works either way.

## Why this run is not yet a clean PASS

Gen-1's high fraction (0.546) is the burned-in-resume transient — the dill starts
mid-cycle so the first generation is not yet relaxed. Gen-5 dips marginally below
the band (0.191). A clean strict-PASS reproduction (Rashmi's authoritative 7-gen
seed-0 cycle, per-gen 13.8–34.9% all in band) is the target; its raw parquet is
not currently on this branch (only the committed chart
`charts/dnaa2_rashmi_reproduction_seed0.png` survived). Re-running that 7-gen
cycle on the current code, with raw data retained, is dnaa-2's primary next action.
