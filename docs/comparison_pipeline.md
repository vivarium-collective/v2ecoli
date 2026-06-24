# v2ecoli ↔ vEcoli comparison pipeline

A deterministic, **AI-agent-free, one-command** pipeline that compares a v2ecoli
build against **any vEcoli fork + its config**, both run as process-bigraph
composites on Ray (GovCloud), emitting the same v2ecoli-format zarr so one
standardized report card reads both sides.

The whole run is pinned by a committed manifest (`comparison_spec.json`). Nothing
in the run path calls an LLM — re-run the spec to reproduce, edit it to vary.

## One-command contract

```bash
bash scripts/comparison_harness.sh all --spec comparison_spec.json
```

`all` = `register → launch → wait → report`:

1. **register** — register **ONE** v2ecoli image with sms-api. The image bundles
   the spec's vEcoli fork (cloned at build time, see below), so it serves BOTH
   engines. No separate vEcoli registration in the default route.
2. **launch** — per condition, submit **two** Ray jobs on that single
   `simulator_id`: `composite=v2ecoli` and `composite=vecoli`. Both emit
   v2ecoli-format zarr to S3 (`v2ecoli_seed*.zarr` / `vecoli_seed*.zarr`).
3. **wait** — poll S3 until each condition has its expected per-seed zarr stores.
4. **report** — `comparison_report_card.py --pbg-vs-pbg` reads BOTH engines from
   S3 via the zarr reader and renders the standardized multi-section report.

Preview the exact requests without sending anything (no creds needed):

```bash
bash scripts/comparison_harness.sh launch --dry-run --v2-sim <SIM> --tag demo
```

## Spec schema (`comparison_spec.json`)

```jsonc
{
  "v2ecoli": { "repo": "...", "commit": "...", "branch": "..." }, // engine to register
  "vecoli":  { "repo": "https://github.com/CovertLab/vEcoli",     // the fork to WRAP
               "commit": "", "branch": "master" },
  "vecoli_engine": "upstream-wrapper",        // | "nextflow" (legacy). default upstream-wrapper
  "from_vecoli_config": "configs/default.json", // optional; drives BOTH engines (see below)
  "defaults": { "seeds": 4, "gens": 2 },
  "conditions": [ { "name": "basal", "config": "cond_basal.json", "seeds": 4, "gens": 2 }, ... ]
}
```

Field resolution for seeds/gens: **per-condition > defaults > CLI** (`--seeds/--gens`).

`scripts/_read_spec.py` (stdlib-only, the spec→bash bridge) exposes:
`engine {v2ecoli|vecoli}`, `vecoli-fork` (→ `repo<TAB>ref`, commit > branch >
`master`), `vecoli-engine`, `from-vecoli-config`, `conditions`.

## How the fork gets into the image

`docker/build-and-push-ecr.sh` reads `vecoli.{repo,commit}` from the spec
(`_read_spec.py vecoli-fork`) and passes them as Docker build-args:

```
--build-arg VECOLI_UPSTREAM_REPO=<repo> --build-arg VECOLI_UPSTREAM_REF=<commit-or-branch>
```

The `Dockerfile` clones `$VECOLI_UPSTREAM_REPO` and checks out
`$VECOLI_UPSTREAM_REF` into `/app/vEcoli` (`V2E_VECOLI_DIR`). It is wrapped, with
**zero edits**, by `v2ecoli.library.vecoli_pbg_upstream` +
`upstream_division` (`composite=vecoli`). If the spec omits the fork, the
Dockerfile defaults clone `CovertLab/vEcoli@master`.

## How one config drives BOTH engines

When `from_vecoli_config` is set, `run_comparison_ensemble.py` resolves it from
the fork (`V2E_VECOLI_DIR`) with v2ecoli's own inheritance loader (no fork venv
needed) and then:

- `--composite v2ecoli` gets the **translated** config
  (`config_adapter.translate_vecoli_config` → baseline overrides);
- `--composite vecoli` runs the **original** config — its native run knobs
  (`condition`, `time_step`, `exclude_processes` via
  `config_adapter.vecoli_native_kwargs`) are threaded into the upstream wrapper.

The driver resolves the config from, in order: `--from-vecoli-config` flag →
`$V2E_FROM_VECOLI_CONFIG` → the `from_vecoli_config` field in the
`comparison_spec.json` **baked into the image** (`COPY . .`). So the default Ray
route is fully spec-driven — no per-job API parameter, **no sms-api change**.

## To compare a NEW vEcoli fork

1. Edit `comparison_spec.json`: set `vecoli.repo`, `vecoli.commit` (pin for
   determinism), and `from_vecoli_config`.
2. Rebuild + push the image: `bash docker/build-and-push-ecr.sh`.
3. Run: `bash scripts/comparison_harness.sh all --spec comparison_spec.json`.

No code change. No agent.

## Legacy Nextflow route

Set `vecoli_engine: "nextflow"` to register vEcoli separately (its own
`simulator_id`, Nextflow, parquet output); the report then reads vEcoli parquet
instead of zarr. Kept for back-compat; the upstream-wrapper route is the default.

## Rendering the report for an existing pbg-vs-pbg run

```bash
.venv/bin/python scripts/comparison_report_card.py --pbg-vs-pbg \
    --out out/4x4x5_compare --only all
```

reads the condition→[v2_dir, ve_dir] pairs from `<out>/experiments.json` and
loads `v2ecoli_seed*.zarr` / `vecoli_seed*.zarr` from S3 for both engines.

## Known sms-api dependency (flagged, not changed)

The default route needs **no** sms-api change: it relies only on the existing
`composite=vecoli` Ray path and the image-baked spec. The only thing that would
require an sms-api change is passing `from_vecoli_config` as a *per-job request
parameter* (instead of baking it into the image) — not done here by design.
