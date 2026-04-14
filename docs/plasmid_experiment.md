# Plasmid Replication Experiment

How to run the ColE1/pBR322 plasmid replication experiment added in [PR #10](https://github.com/vivarium-collective/v2ecoli/pull/10) and regenerate `reports/plasmid_replication_report.html`.

## Prerequisites

- v2ecoli at the `plasmids` branch of this repo
- v2parca at the `plasmid` branch: https://github.com/vivarium-collective/v2parca/pull/1 — installed editable as a dep (see below)
- GLPK (`brew install glpk`) and the usual dev stack (`ecos`, `swiglpk`, `autograd`, `ipython`, `pint`, `stochastic-arrow`, `vivarium-core`) installed into v2ecoli's `.venv`

## 1. Generate the sim_data

The v2parca `plasmid` branch produces `data/parca_state.pkl.gz` containing a `SimulationDataEcoli` with plasmid fields wired in (plasmid sequence, oriV coordinate, unique molecule definitions, full_plasmid mass).

```bash
cd ../v2parca   # or wherever your v2parca checkout lives
git checkout plasmid
# If you want to regenerate rather than use the committed pickle:
uv run python scripts/parca_bigraph.py --mode fast --cpus 4 -o out/sim_data
# Resume step 6 if step 6 fails with ECOS:
uv run python scripts/parca_bigraph.py --mode fast --cpus 4 \
    -o out/sim_data --resume-from-step 6 --resume-pickle out/sim_data/checkpoint_step_5.pkl
```

## 2. Install v2parca as a dep inside v2ecoli's venv

v2ecoli's `SimulationDataEcoli` unpickler needs the v2parca module to be importable:

```bash
cd ../v2ecoli
uv pip install --python .venv/bin/python -e ../v2parca
```

## 3. Extract / patch the sim_data

The fast-mode parca output currently has initial-state NaN issues that break the equilibrium ODE solver on its own. The short-term workaround is to patch plasmid fields into a known-good baseline sim_data (e.g. `out/workflow/simData.cPickle` from a prior workflow run). A helper script does this:

```python
# scripts/patch_plasmid_into_workflow_simdata.py  (one-off, run once)
import pickle

with open('out/workflow/simData.cPickle', 'rb') as f:
    sd_good = pickle.load(f)
with open('out/kb/simData.cPickle', 'rb') as f:     # extracted from parca_state.pkl.gz
    sd_p = pickle.load(f)

sd_good.molecule_ids.full_plasmid = sd_p.molecule_ids.full_plasmid
sd_good.molecule_ids.plasmid_ori = sd_p.molecule_ids.plasmid_ori
for attr in ('plasmid_sequence', 'plasmid_sequence_rc', 'plasmid_length',
             'plasmid_A_count', 'plasmid_T_count', 'plasmid_G_count', 'plasmid_C_count',
             'plasmid_ori_coordinate', 'plasmid_forward_sequence',
             'plasmid_forward_complement_sequence', 'plasmid_replichore_lengths',
             'plasmid_replication_sequences'):
    setattr(sd_good.process.replication, attr, getattr(sd_p.process.replication, attr))
for k in ('full_plasmid', 'plasmid_domain', 'oriV', 'plasmid_active_replisome'):
    sd_good.internal_state.unique_molecule.unique_molecule_definitions[k] = \
        sd_p.internal_state.unique_molecule.unique_molecule_definitions[k]
    if k not in sd_good.molecule_groups.unique_molecules_domain_index_division:
        sd_good.molecule_groups.unique_molecules_domain_index_division.append(k)
sd_good.getter._all_plasmid_coordinates = sd_p.getter._all_plasmid_coordinates
sd_good.getter._all_submass_arrays['full_plasmid'] = sd_p.getter._all_submass_arrays['full_plasmid']
sd_good.getter._all_total_masses['full_plasmid'] = sd_p.getter._all_total_masses['full_plasmid']
sd_good.getter._all_compartments['full_plasmid'] = \
    sd_good.getter._all_compartments.get('full_chromosome', ['c'])

with open('out/kb/simData_plasmid_patched.cPickle', 'wb') as f:
    pickle.dump(sd_good, f)
```

Once v2parca fast-mode is producing a directly usable pickle, this patching step goes away and you just use `out/kb/simData.cPickle` directly.

## 4. Build the plasmid-enabled cache

```bash
uv run python -c "
from v2ecoli.composite import save_cache
save_cache('out/kb/simData_plasmid_patched.cPickle',
           cache_dir='out/cache_plasmid', seed=0, has_plasmid=True)
"
```

`has_plasmid=True` tells `LoadSimData` to initialize one `full_plasmid`, one `oriV`, and one `plasmid_domain` at start.

## 5. Run the simulation

```bash
PLASMID_DURATION=200 PLASMID_INTERVAL=10 \
    uv run python scripts/run_plasmid_experiment.py
```

Writes `out/plasmid/timeseries.json` with per-snapshot counts of plasmid unique molecules, replisome subunit pools, and the RNA I/II/hybrid state from the `plasmid_rna_control` port.

Defaults are 60 s duration and 2 s emission interval. For visible replication events you generally need ≥ 600 s (one RNA II initiation interval is 360 s).

## 6. Generate the HTML report

```bash
uv run python scripts/plasmid_report.py
```

Writes `reports/plasmid_replication_report.html` with six plots:
1. Plasmid unique molecule counts
2. Active replisomes (plasmid vs chromosome)
3. RNA I / II / hybrid dynamics (Ataai-Shuler 1986 ODE)
4. Initiation accumulator + RNA II interval countdown
5. Replisome subunit pool
6. Cell + DNA mass

Open it with `open reports/plasmid_replication_report.html` on macOS.

## Known issues

- v2parca fast-mode sim_data has equilibrium ODE NaN issues when used directly. The patching step above is a workaround; fixing this at the v2parca level is follow-up work.
- Short runs (< 360 s) won't show replication events — RNA II control only fires once per interval.
- Multi-generation, multi-seed, and copy-number-distribution visualizations from the upstream [vEcoli_Plasmid#2](https://github.com/vivarium-collective/vEcoli_Plasmid/pull/2) require division + multiple lineages, which is out of scope for this single-cell run.
