# Run-config provenance

> Kills the *"was V=1.7e-3 actually applied to this run?"* confusion: any
> run or figure traces back to the exact config that produced it.

## The model

```
experiment_id  ==  run_id  ==  top parquet partition key
       │
       │  runner records run_config ──► runs_meta.params_json   (per-study runs.db)
       │
       └─ figure .meta.json carries run_id ──► pbg_superpowers.provenance ──► run_config
```

Four moving parts:

1. **`experiment_id` is the run id.** It is the top parquet partition key
   (`out/<experiment_id>/<experiment_id>/configuration/experiment_id=<id>/…`).
   One run, one id, everywhere.

2. **The runner records — and explicitly applies — the config.**
   `scripts/run_condition_multigen_parquet.py`:
   - `--perturbation RNA_ID=SYNTH_PROB` (repeatable, e.g.
     `--perturbation 'TU00259[c]=1.7e-3'`) sets the fixed RNA synthesis
     probability **at run time** into the cache config TranscriptInitiation
     reads, *overriding* whatever the cache baked in. So the perturbation is a
     recorded run-time decision, not an opaque cache name.
   - It assembles a `run_config` dict — the decision-relevant knobs:
     `perturbations` (applied) + `perturbations_detail` (which overrode a
     cache-baked value), `cache_dir` + a cheap `cache_fingerprint` (a short
     hash derived from sim_data_cache.dill's size + mtime — a plain string,
     `None` if the cache is missing; the full diagnostic dict, including
     size/mtime, is under `cache_fingerprint_detail`), `seed`, `generations`,
     `max_min`, `resume_dill`, and **`dnaA_synth_prob_from_cache`** — the
     ParCa-derived basal probability read back from the cache for each
     perturbed RNA (the ground truth the perturbation overrides; this is the
     number that answers the V=1.7e-3 question).
   - It writes `run_config` into the run summary JSON **and registers it**
     into the per-study `runs.db` via
     `pbg_superpowers.run_registry.register_run(run_id=experiment_id, …)`
     (`status=running` up front, `status=complete` at the end — the recorded
     params survive both calls).
   - `--dry-run` prints + registers the `run_config` **without** running any
     simulation. Use it to verify wiring and `--perturbation` parsing.

3. **Figures carry the run id.** Each `render_dnaa*.py` writes the canonical
   `run_id` into every chart's `.meta.json` (alongside the legacy free-text
   `source_run_id`). It is derived from the run dir's
   `experiment_id=<id>` partition segment (authoritative) via
   `scripts/_run_provenance.run_id_from_run_dir`.

4. **Lookup CLI.**
   ```
   python -m pbg_superpowers.provenance <figure.png | .meta.json | run_id> \
       [--runs-db PATH] [--workspace DIR] [--json]
   ```
   Resolves figure → `run_id` → `run_registry.get_run_params`, falling back to
   the on-disk parquet `configuration` partition (which still yields
   experiment_id / seed / generations) for runs that predate this system.

## Examples

```bash
# Record-and-run (real sim): the perturbation is applied AND recorded.
python scripts/run_condition_multigen_parquet.py \
    --cache-dir out/cache_dnaa1 --out-dir out/dnaa1 \
    --experiment-id dnaa1_v17e3 --generations 8 --max-min 200 --seed 1 \
    --perturbation 'TU00259[c]=1.7e-3' \
    --study-dir studies/dnaa-1-expression --spec-id dnaa-1-expression

# Verify config assembly without simulating:
python scripts/run_condition_multigen_parquet.py … --perturbation 'TU00259[c]=1.7e-3' --dry-run

# Trace any figure back to its config:
python -m pbg_superpowers.provenance studies/dnaa-1-expression/charts/dnaa1_decision.png
```

For a registered run, `provenance` prints the applied perturbation, the seed,
generations, cache fingerprint, and the ParCa synth prob it overrode — so the
*"was V=1.7e-3 applied?"* question is answerable straight from the figure.

## `run_identity.json` — the code-identity sidecar (v2ecoli#472/#473)

Everything above answers *"what config produced this run?"*. It doesn't answer
*"what commit ran it?"* — no runner recorded that anywhere until now.

Every `run_*` entrypoint that produces sweep output writes
`<out_dir>/run_identity.json` via
`v2ecoli.library.run_provenance.write_run_identity`, combining:

- **code** — `code_provenance()`'s `{commit, dirty, diff_sha256, untracked}`
- **cache_version** — the sweep's `cache_version.json` content fingerprint
  (`inputs_hash`), read fresh at write time and copied in (never a pointer —
  `cache_version.json` is mutable and gets silently regenerated later)
- **design** — whatever grid/config metadata (`experiment_id`, seed,
  generations, perturbations, ...) is already available at the call site;
  the write-side half of #473's seed×generation design record

`v2ecoli.library.sim_vector_cache._run_commit` reads this file (local or
`s3://`) to answer `run_commit` in a cached vector's provenance block, with a
legacy flat-key fallback for sweeps produced before this convention existed.
A dedicated file rather than a key added to each runner's own
`summary.json`/`run_config.json` — see `RUN_IDENTITY_FILENAME`'s docstring in
`run_provenance.py` for why (`v2ecoli/workflow/run.py`'s `summary.json` shape
in particular can't take a sibling key without becoming ambiguous to its
existing readers). Those files still carry their own `run_identity` copy for
anyone reading them directly.
