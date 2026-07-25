# Plasmid Replication Experiment

How to run the ColE1/pBR322 plasmid replication experiment from a fresh
clone of v2ecoli (`plasmids` branch) and regenerate
`reports/plasmid_replication_report.html`.

The plasmid-aware ParCa fixture
(`models/parca/parca_state.pkl.gz`) and the vendored ParCa code
(`v2ecoli/processes/parca/`) both ship in this branch — there is no
external `v2parca` checkout or install required.

## Prerequisites

- v2ecoli at the `plasmids` branch
- Python 3.12.9, `uv` installed
- GLPK (`brew install glpk` on macOS, `apt install libglpk-dev` on Linux)
- A known-good baseline `simData.cPickle` from a vEcoli ParCa run
  (see step 2). Required because the v2ecoli fast-mode ParCa fixture has
  initial-state issues that crash the online sim on the first step.

## 1. Install + build vendored Cython extensions

```bash
git clone https://github.com/vivarium-collective/v2ecoli.git
cd v2ecoli
git checkout plasmids
uv venv
uv sync                              # installs pyproject.toml deps incl. vEcoli
bash scripts/parca_cython_build.sh   # compiles the 3 .pyx → .so files
```

The Cython step is mandatory — the `.so` files are gitignored, so every
fresh clone needs this. Skipping it produces:

```
ModuleNotFoundError: No module named
'v2ecoli.processes.parca.wholecell.utils.mc_complexation'
```

## 2. Get a baseline `simData.cPickle`

The plasmid-aware fixture in `models/parca/` is fast-mode parca and has
NaN/initial-state issues (cell hits division threshold at t=0,
chromosomes already at 4) that crash the online sim. The workaround is
to graft the plasmid fields from the fast-mode fixture onto a known-good
baseline `simData.cPickle` produced by vEcoli's monolithic ParCa.

Run vEcoli's parca **once** (~30-70 minutes) and copy the result:

```bash
cd $VECOLI_REPO          # vEcoli is installed editable as a v2ecoli dep
uv run python -m runscripts.parca \
    --config configs/templates/parca_standalone.json \
    --outdir out/parca
mkdir -p ../v2ecoli/out/workflow
cp out/parca/kb/simData.cPickle ../v2ecoli/out/workflow/simData.cPickle
cd ../v2ecoli
```

`scripts/build_plasmid_cache.py` looks for the baseline at:

  1. `out/workflow/simData.cPickle`
  2. `out/kb/baseline_simData.cPickle`
  3. `$VECOLI_REPO/out/test_installation/parca/kb/simData.cPickle`
     (`VECOLI_REPO` defaults to `../vEcoli`)

so any of those locations will be picked up automatically.

## 3. Build the plasmid-enabled cache

```bash
uv run python scripts/build_plasmid_cache.py
```

This script:
  - hydrates the plasmid-aware fast-mode fixture
    (`models/parca/parca_state.pkl.gz`) into a `SimulationDataEcoli`,
  - patches its plasmid fields onto the baseline (molecule_ids,
    process.replication, getter, internal_state.unique_molecule),
  - writes `out/kb/simData_plasmid_patched.cPickle`,
  - calls `save_cache(..., has_plasmid=True)` → `out/cache_plasmid/`.

`has_plasmid=True` tells `LoadSimData` to seed one `full_plasmid`, one
`oriV`, and one `plasmid_domain` at start.

Override paths via `--baseline`, `--parca-fixture`, `--cache-dir`, or
`--seed` if needed.

## 4. Run the simulation

```bash
PLASMID_DURATION=600 PLASMID_INTERVAL=10 \
    uv run python scripts/run_plasmid_experiment.py
```

Writes `out/plasmid/timeseries.json` with per-snapshot counts of plasmid
unique molecules, replisome subunit pools, and the RNA I/II/hybrid state
from the `plasmid_rna_control` port.

Defaults are 60 s duration / 2 s emission interval. For visible
replication events you generally need ≥ 600 s (one RNA II initiation
interval is 360 s).

## 5. Generate the HTML report

```bash
uv run python scripts/plasmid_report.py
open reports/plasmid_replication_report.html   # macOS
```

Six plots:
  1. Plasmid unique molecule counts
  2. Active replisomes (plasmid vs chromosome)
  3. RNA I / II / hybrid dynamics (Ataai-Shuler 1986 ODE)
  4. Initiation accumulator + RNA II interval countdown
  5. Replisome subunit pool
  6. Cell + DNA mass

## Known issues

- v2ecoli fast-mode ParCa output (`models/parca/parca_state.pkl.gz`)
  has NaN/initial-state issues when used directly: cell mass already at
  division threshold, 4 chromosomes at t=0, immediate division on first
  step which then crashes in bigraph-schema's resolver. The patching in
  step 3 is the working workaround. Fixing this at the v2ecoli ParCa
  level is follow-up work.
- Short runs (< 360 s) won't show replication events — RNA II control
  only fires once per RNA II initiation interval.
- `save_cache` reports one expected non-fatal omission:
  `ecoli-polypeptide-elongation` (`AttributeError: 'Metabolism' object
  has no attribute 'aa_enzymes'`). Doesn't affect the plasmid
  experiment — those processes get omitted from the cache and the run
  proceeds without them.
- Multi-generation, multi-seed, and copy-number-distribution
  visualizations from the upstream
  [vEcoli_Plasmid#2](https://github.com/vivarium-collective/vEcoli_Plasmid/pull/2)
  require division + multiple lineages, out of scope for this
  single-cell run.
