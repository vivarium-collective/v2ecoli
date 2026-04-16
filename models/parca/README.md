# ParCa state — pre-computed

`parca_state.pkl.gz` is the final composite state of the full ParCa
pipeline (`v2ecoli/processes/parca/`) in `--mode fast` (debug=True,
reduced TF condition set).  Produced by `scripts/parca_run.py` and
gzipped for the repo.  Consumers can use this as a drop-in
`sim_data`-shaped dict without re-running the 70-minute step 5.

**Format note.**  This is currently a gzipped Python pickle.  The
planned migration to a bigraph-schema JSON bundle (a true `.pbg`
artifact) is tracked separately — see
`v2ecoli/processes/parca/schema.py` for the subsystem types that will
underpin the serialize/deserialize dispatches.

## Contents

- 24 top-level stores: the nine `process/*` subsystem objects
  (`transcription`, `translation`, `metabolism`, `rna_decay`,
  `complexation`, `equilibrium`, `two_component_system`,
  `transcription_regulation`, `replication`) plus top-level dataclasses
  (`mass`, `constants`, `growth_rate_parameters`, `adjustments`,
  `molecule_groups`, `molecule_ids`, `relation`, `getter`,
  `external_state`), top-level data dicts (`conditions`,
  `condition_to_doubling_time`, `condition_active_tfs`,
  `condition_inactive_tfs`, `expected_dry_mass_increase_dict`,
  `pPromoterBound`), plus `internal_state/bulk_molecules` and
  `cell_specs`.
- 7 conditions in `cell_specs`: `basal`, `with_aa`, `acetate`,
  `succinate`, `no_oxygen`, `CPLX-125__active`, `CPLX-125__inactive`.
- `runtimes.json`: per-step wall-clock timings from the run that
  produced the pickle.

## Loading

```python
from v2ecoli.processes.parca.data_loader import load_parca_state
state = load_parca_state()

# Access any store:
transcription = state['process']['transcription']
cell_specs    = state['cell_specs']
mass          = state['mass']
```

The loader transparently aliases the legacy `v2parca.*` and `vparca.*`
module paths that the pickle was originally written against, so old
artifacts deserialize cleanly under the merged package name.

## Provenance

- Pipeline: `scripts/parca_run.py --mode fast --cpus 2`
- Duration: 71.6 min end-to-end (step 5: 70 min; steps 1-4 + 6-9: ~1.5 min)
- `debug=True`, `operons_on=True`, `remove_rrna_operons=False`, `stable_rrna=False`

To regenerate from scratch (~70 minutes), first build the Cython
extensions:

```bash
bash scripts/parca_cython_build.sh
python scripts/parca_run.py --mode fast --cpus 2
gzip -k out/sim_data/parca_state.pkl
mv out/sim_data/parca_state.pkl.gz models/parca/
cp out/sim_data/runtimes.json models/parca/
```

To rerun only steps 6-9 using a cached step-5 checkpoint (~15 s):

```bash
bash scripts/parca_rerun_from_step5.sh
```
