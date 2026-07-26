# `redux_cards` fixture — matched 1-generation basal MetabolismRedux pair

## What this is

`v2ecoli_seed00.zarr` and `vecoli_seed00.zarr` are a matched pair of
**1-generation, basal-condition, MetabolismRedux** runs — one from the
v2ecoli engine, one from the vEcoli-pbg engine — both emitted in the SAME
v2ecoli compact-zarr format (see
`scripts/compare_matched_trajectories.read_v2ecoli_trajectory`). They are
small (~150-165 KB each) and committed to the repo so report-card tests run
offline, in seconds, with no S3/AWS credentials.

Both stores hold at least `cell_mass` and `dry_mass` under
`lineage_seed=00/generation=0` (verified directly — see Task B0 report).

## `make_card_state()` (in `tests/conftest.py`)

```python
from conftest import make_card_state
state = make_card_state()
```

Import it bare (`from conftest import ...`), NOT `from tests.conftest
import ...`: a site-packages `nose`-provided `tests` package (in
`v2ecoli/.venv/lib/python3.12/site-packages/tests/`) shadows this repo's
`tests/conftest` for dotted imports, so `tests.conftest` resolves to the
wrong module. Copy this bare form in B1-B4 card tests.

Returns a `state` dict matching the `CARD_INPUTS` contract in
`scripts/_compare/report_cards/__init__.py`, with `v2_dir` **and** `ve_dir`
BOTH set to the absolute path of this directory
(`tests/fixtures/redux_cards`). `name="metabolism_redux_basal"`,
`condition="basal"`, `seeds=1`, `generations=1`, `variant=0`.

## New cards read the zarr directly — `observables`/`plot_trajs` are empty here

The four NEW report cards (trajectory, distribution, metabolism,
composition — built in later tasks) read the fixture zarr stores directly
off `state["v2_dir"]`/`state["ve_dir"]`, via:

- `scripts.compare_matched_trajectories.read_pbg_local(zarr_path, observables)`
  — the local-path reader; works for BOTH engines since both emit the
  v2ecoli-format zarr. This is the reader proven against this fixture (see
  `tests/test_card_fixture.py`).
- `read_v2ecoli_trajectory` / `read_vecoli_pbg_trajectory` — these also
  parse the v2ecoli-format zarr, but as currently written they only build
  **S3** store URIs when called with a bare `experiment_dir` (no
  `store_uri` kwarg) — see `scripts/compare_matched_trajectories.py` lines
  ~74-163. To read this LOCAL fixture through them, pass the resolved
  zarr path via `store_uri=` explicitly (that's exactly what
  `read_pbg_local` does internally). A future B-task adding local-dir
  support to `read_v2ecoli_trajectory`/`read_vecoli_pbg_trajectory`
  directly would let new cards call
  `read_v2ecoli_trajectory(state["v2_dir"], seed, obs)` /
  `read_vecoli_pbg_trajectory(state["ve_dir"], seed, obs)` unmodified; until
  then, cards operating on local fixture dirs should go through
  `read_pbg_local(os.path.join(v2_dir, f"v2ecoli_seed{seed:02d}.zarr"), obs)`
  and `read_pbg_local(os.path.join(ve_dir, f"vecoli_seed{seed:02d}.zarr"), obs)`.

Because the new cards ignore `observables`/`plot_trajs`/`v2_bounds`/`config`,
`make_card_state()` fills them with minimally-valid empty values
(`{}`/`[]`). Do NOT expect those keys to be populated for this fixture.

## For reference — the REAL shape of `observables`/`plot_trajs`

The OLDER cards (standard/parca/statistical/config_diff) DO consume
`observables`/`plot_trajs`. Their real shape, as built by
`scripts/comparison_report_card.py`:

### `state["observables"]` — `per_obs`, built in `build()` (~lines 154-210)

```python
per_obs: dict[str, list[dict]]
# {
#   "cell_mass": [
#     {  # one dict per seed that produced a matched gen-1 window
#        "init_t": <float>,      # first matched-grid time (s)
#        "init_v2": <float>,     # v2ecoli value at init_t
#        "init_ve": <float>,     # vEcoli value at init_t (reference)
#        "init_rel": <float>,    # (v2-ve)/ve at init_t
#        "v2_mean": <float>,     # mean over matched grid
#        "ve_mean": <float>,
#        "median_rel": <float>,  # median |relative Δ| over matched grid — THE grading stat
#        "max_rel": <float>,
#        "n": <int>,             # matched-grid point count
#        "seed": <int>,          # appended after _matched() returns (build(), line 199)
#     },
#     ...  # one entry per seed with data (see _matched(), ~lines 115-140)
#   ],
#   "dry_mass": [...], "protein_mass": [...], "rna_mass": [...],
#   "instantaneous_growth_rate": [...], "active_RNAP": [...],
#   "active_ribosome": [...],
# }
```

Consumers: `eval_section()` (~lines 810-838) takes `np.median`/`np.max`
of `median_rel`/`max_rel` across the per-seed dicts and grades via
`_grade()` (5%/10% bands); `overview_section`/report-card grading do the
same via `report_card_section.build_report_card`.

### `state["plot_trajs"]` — built alongside `per_obs` in the same loop

```python
plot_trajs: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]]
# {
#   "cell_mass": {
#     "v2": [(times_seed0, values_seed0), (times_seed1, values_seed1), ...],
#     "ve": [(times_seed0, values_seed0), (times_seed1, values_seed1), ...],
#   },
#   "dry_mass": {...}, ...
# }
```

One `(times, values)` tuple appended per seed per engine — the FULL
trajectory (all generations), not windowed to gen-1. Consumer:
`runs_section()` (~lines 786-808) feeds `pt["v2"]`/`pt["ve"]` straight into
`overlay_svg_multi()` for the multi-seed overlay SVG plots.

### `state["v2_bounds"]` and `state["config"]`

- `v2_bounds: list[float]` — generation-boundary times on the v2ecoli axis,
  from `_gen_bounds()` (~line 146), used as vertical dashed lines in the
  overlay plots.
- `config: dict` — for `assemble_from_studies()` (~lines 840-884) this is
  `{"condition": ..., "seeds": ..., "generations": ..., "cards": [...]}`
  (the study's run spec, rendered by the `config` card).

`per_obs`/`plot_trajs` are read straight off `cond_data[cond]` (the tuple
returned by `build()`) inside `assemble_from_studies()` (~lines 861-875,
where the actual `state = {...}` dict is constructed one condition/study at
a time and handed to each assigned card's `step.update(state)`).
