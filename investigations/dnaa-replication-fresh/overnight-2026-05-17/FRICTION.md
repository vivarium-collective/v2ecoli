# Investigation-Design Friction Log — 2026-05-17

> Each friction point is something that slowed me down or made me less
> effective driving the dnaA investigation forward. Each entry includes
> a concrete improvement suggestion so the studies-infrastructure can
> evolve to support deeper, faster work.

## F-01: Spawned follow-up study.yaml is mostly empty templates

**Where**: `studies/dnaa-01f-recalibrate-eg10235-translation-efficiency-in-parca/study.yaml`

**What I hit**: The spawned study has `purpose.question` (good, copied
from parent), but `model_change.notes: "Populate during Build phase"`,
`pipeline_gate.proceed_condition: "TBD — define before Simulate"`, and
`behavior_tests: []`. Every section asks me to invent context that
already exists in the parent's finding evidence.

**Concrete suggestion**: When the dashboard's "Seed new study" action
fires, in addition to copying `purpose`, it should:
1. Copy the parent finding's `evidence` block into `seeded_from.evidence`.
2. Pre-populate `pipeline_gate.proceed_condition` from the parent's
   `next_action` (or at minimum quote it as a starting point).
3. Pre-populate `behavior_tests` with the parent's tests that
   originally FAILED (these are exactly what the follow-up must make
   pass). For dnaa-01f-recalibrate, that's `dnaA-count-in-range` +
   `autorepression-correlation`.
4. Pre-populate `model_change.modified_processes` with a "TBD"
   placeholder per the finding's `mechanism` hint, but at least
   point at the right file.

**Why it matters**: Driving from a fresh `planned`-status study takes
30+ minutes of re-reading the parent before any real model work
starts. With the suggested seed, that drops to 5 minutes.

## F-02: Test gate logic is implicit in conclusion_logic.if_primary_tests_pass

**Where**: `study.yaml.conclusion_logic`

**What I hit**: To decide whether dnaa-01 can pass its gate, I had to
read the conclusion_logic block AND mentally match it against the
findings list. There's no programmatic way to ask "is this study's
gate passed yet?" — the dashboard infers it from running pytest, but
the decision rules live in YAML prose.

**Concrete suggestion**: Add a `gate_evaluator` field to study.yaml,
e.g.:

```yaml
gate_evaluator:
  expr: "outcomes['dnaA-count-in-range'].result == 'PASS' AND outcomes['autorepression-correlation'].result == 'PASS'"
  result: blocked  # auto-set by evaluator after last run
  blocked_by: ["dnaA-count-in-range", "autorepression-correlation"]
```

Then the dashboard can show a binary "gate open/closed" badge on
each study card, and the next-study auto-status update can flip from
`blocked` → `planned` when its prerequisite's gate opens.

## F-03: Joint parameter sweeps require manual scripting

**Where**: F-10 explicitly recommends a "TE × autorepression-fold-change
3×3 grid". To implement this I'd have to write a new variant runner
that takes TWO override params, then a new evaluator that aggregates
by both axes.

**Concrete suggestion**: Promote the variant-sweep pattern in
study.yaml to a first-class declarative spec:

```yaml
simulation_set:
  - kind: grid_sweep
    name: te_x_foldchange_grid
    base: baseline
    axes:
      - param: dnaa_te_multiplier
        values: [10, 20, 30, 40, 50]
      - param: dnaa_autorep_multiplier
        values: [0.5, 1.0, 2.0, 5.0]
    seeds: [0, 1, 2]
    duration_min: 10
```

Then a generic runner expands this into N×M×S simulations + a generic
evaluator that produces a per-axis heatmap visualization
automatically. This pattern recurs in dnaa-02 (ATP/ADP/apo split
hydrolysis-rate sensitivity) and dnaa-03 (DnaA-box K_d sensitivity).

## F-04: Listeners hide multi-state pools behind a single index

**Where**: `listeners.monomer_counts[3861]` — apparently the dnaA "count"
— is the observable that the dnaA-count-in-range test reads.

**What I hit**: At the bulk level, DnaA actually lives across THREE bulk
species (PD03831 apo, MONOMER0-160 ATP-bound, MONOMER0-4565 ADP-bound),
maintained by the `ecoli-equilibrium` process. The
`listeners.monomer_counts[3861]` reading aggregates these (probably) so
that `count_in_range = 115` describes total DnaA, not just apo. **But you
can't tell from the listener API alone** — there's no metadata that says
"this is a sum over an equilibrium pool". I had to write a probe that
reads the live `bulk` vector directly to verify.

**Concrete suggestion**: Add an `expansion` field to listener-config
schemas:

```python
listener_schema(monomer_counts, expansion={
    "PD03831[c]": ["MONOMER0-160[c]", "MONOMER0-4565[c]"],
    # ... more equilibrium-coupled species
})
```

…and the dashboard's Observables tab can render the breakdown so a
biologist exploring "DnaA count" sees apo + ATP-bound + ADP-bound on
a stacked-area chart automatically.

## F-05: Equilibrium and metabolic-reaction databases are de-coupled from kinetic constraints

**Where**: `metabolic_reactions.tsv` declares RXN0-7444 (DnaA-ATP intrinsic
hydrolysis) catalyzed by CPLX0-10342, BUT `metabolism_kinetics.tsv` has
no entry for it. The FBA solver therefore never selects this reaction.

**What I hit**: I went looking for "why is DnaA-ADP always 0?" and traced
through three databases (`metabolic_reactions.tsv` → `metabolism_kinetics.tsv`
→ `disabled_kinetic_reactions.tsv`) before realizing the reaction is silently
inactive because no rate constraint exists for it.

**Concrete suggestion**: Add a validation step that runs after ParCa cache
build, checking: for each reaction in `metabolic_reactions.tsv`, is there
EITHER a kinetic constraint, OR an entry in `disabled_kinetic_reactions.tsv`
explicitly excluding it? Unannotated reactions get silently ignored by
the FBA solver, which is the worst kind of failure mode — they look
"available" but never fire. A startup-time warning ("4 reactions declared
but neither constrained nor disabled; behavior depends on FBA objective")
would have saved me an hour.

## F-06: study.yaml.runs[] is divorced from runs.db simulations

**Where**: A study's `runs[]` array (with `simulation`, `seed`, `started_at`,
`outcomes`) does NOT carry a run_id field. Meanwhile `runs.db` tracks
simulations via UUID `simulation_id`. The dashboard's Runs tab shows the
yaml entries; Simulations DB shows the DB rows — they're disjoint
datasets, so a user can't trivially click "see this run's data".

**Concrete suggestion**: When the evaluator writes outcomes back to
study.yaml.runs[], it should also record the `run_id` from the most
recent matching `simulations.name` row, so the two views can join. The
Dashboard's Runs-tab "View" button would then have a real URL to navigate
to (instead of the currently-404 /composite-explorer?run_id=...).

## F-07: New variant runners proliferate; no canonical pattern

**Where**: To run the joint (TE × fc) sweep I had to write a NEW runner
(`run_baseline_with_fc.py`) instead of extending the existing
`run_baseline.py`. The existing runner has `--dnaa_te_multiplier` but
not `--dnaa_autorep_multiplier`. There's no clean way to add a new
"pre-build patch" without copying the entire runner.

**Concrete suggestion**: Refactor the runner around a `BundlePatcher`
plugin pattern:

```python
class BundlePatcher:
    def apply(self, bundle, args): ...

REGISTRY = {
    'dnaa_te_multiplier': lambda v: lambda b: scale_te(b, 3861, v),
    'dnaa_autorep_multiplier': lambda v: lambda b: scale_deltaV_col(b, 12, v),
    # ... add more
}

# In runner:
for name, value in args.__dict__.items():
    patcher = REGISTRY.get(name)
    if patcher and value != 1.0:
        patcher(value)(bundle)
```

Then any new variant adds one row to REGISTRY, no new runner needed.

## F-08: Path-resolution bug pattern in seed scripts

**Where**: Wrote `run_baseline_with_fc.py` using `Path(__file__).resolve().parents[2]`
to locate V2ECOLI_DIR. Wrong — the file is at
`investigations/dnaa-replication/overnight-2026-05-17/run_baseline_with_fc.py`,
so parents[2] = `investigations/` (off by one). Got
`FileNotFoundError: ...investigations/out/cache/initial_state.json`.

**Concrete suggestion**: Provide a one-import helper:

```python
# In v2ecoli/library/paths.py
def workspace_root() -> Path:
    """Walk up from caller's file until you find workspace.yaml or .git/."""
    p = Path(sys._getframe(1).f_code.co_filename).resolve().parent
    while p != p.parent:
        if (p / 'workspace.yaml').exists() or (p / '.git').exists():
            return p
        p = p.parent
    raise RuntimeError('no workspace root found')
```

Then every script does `V2ECOLI_DIR = workspace_root()` and is location-independent.

## F-09: deltaV [DnaA col] regulation graph is opaque

**Where**: `delta_prob.deltaV[deltaJ==12]` has 10 entries — these are
the 10 TUs DnaA regulates. The TU IDs are `TU0-1193[c]`, `TU0-14305[c]`,
etc. — opaque locus IDs. Cross-referencing to gene names requires reading
`rnas.tsv` and following `cistron_id` → `gene_symbol`.

**What I hit**: Trying to verify whether dnaA self-autorepresses (a key
biological assumption), I had to grep `rnas.tsv` for `EG10235_RNA`
to find its cistron ID, then trace cistron→TU via cistron_to_rna_indexes
machinery in transcription dataclass. Took 10 minutes that should have
been 30 seconds.

**Concrete suggestion**: `delta_prob` should carry a `regulation_map`
field: `{TF_id: {regulated_TU_id: {target_gene: ..., effect: ...}}}`.
At least at debug-time, exposing readable names with deltaV would make
autorepression triage much faster.

