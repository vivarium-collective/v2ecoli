# pbg-superpowers test-run notes — DnaA studies scaffolding (2026-05-16)

> **Audience.** Another Claude Code session is reading these notes live and
> using them as input for pbg-superpowers improvements. Treat each
> section as a self-contained brief: what I did, what was awkward, where
> the gap is in pbg-superpowers / pbg-template, and a concrete proposal
> (skill name, CLI signature, file path) the receiving session can act on
> without re-reading my full transcript.
>
> Repo paths used in this run:
> - `~/code/v2ecoli/` — the v2ecoli source repo I scaffolded as a workspace.
> - `~/code/pbg-superpowers/` — the source of installed skills.
> - `~/code/pbg-template/` — the workspace template repo.
> - `~/code/viva-munk/` — another pbg workspace, used as a v3 `study.yaml` example.
> - `~/code/v2ecoli-workspace/` — a sibling pbg-template-cloned v2ecoli workspace.

## Cold-start orientation

The task: a user dropped two PDFs (a multi-phase replication-initiation **plan**
and an **expert-knowledge** brief) and asked to (a) register the expert doc
under "Inputs > Expert curation" and (b) scaffold one study per phase, viewable
in the dashboard. The user wanted to stay in `~/code/v2ecoli` and treat it as
a workspace (not a sibling repo).

`origin/main` of v2ecoli had just merged PR #50 (`feat(template): add workspace
scaffolding kit`) which contributes generic `references/`, `experiments/`,
`scripts/`, `.pbg/schemas/`, `pbg_v2ecoli/`, `NEXT_STEPS.md` to v2ecoli, but
**intentionally** omits the workspace-specific files (`workspace.yaml`,
`.gitmodules`, `.claude/settings.json`). So I:

1. Branched `dnaa-replication-studies` off `b927dc3` (one behind main).
2. Hand-rolled `workspace.yaml`, populated `references/expert/` with both PDFs,
   wrote `references/papers.bib` + `claims.yaml`, and authored
   `studies/dnaa-01..06/study.yaml` v3-shape drafts.
3. Rebased onto `origin/main` (PR #50); resolved add/add conflicts on
   `references/{README.md, papers.bib, claims.yaml}` by keeping mine.
4. `python scripts/lint-workspace.py` → `workspace lint: OK`.

Resulting tree on the branch (new files only):

```
workspace.yaml
references/expert/replication_initiation_molecular_info.pdf
references/expert/chromosome_replication_plan.pdf
references/papers.bib                 # 17 bib entries (was 1-line skeleton)
references/claims.yaml                # 16 claim mappings (was {})
studies/dnaa-01-expression-dynamics/study.yaml
studies/dnaa-02-atp-hydrolysis/study.yaml
studies/dnaa-03-box-binding/study.yaml
studies/dnaa-04-initiation-mechanism/study.yaml
studies/dnaa-05-rida-ddah-dars/study.yaml
studies/dnaa-06-seqa-sequestration/study.yaml
docs/superpowers/notes/2026-05-16-dnaa-studies-test-run.md   # this file
```

The receiving session can `git checkout dnaa-replication-studies` on the
v2ecoli repo to see the exact scaffolding.

---

## Findings, in priority order

Each finding has: **What I did** · **Friction** · **Gap in pbg-superpowers** ·
**Proposal** (concrete enough to start coding).

### 1. "Is this cwd a pbg workspace?" took ~6 tool calls

**What I did.** Tried to figure out where to put the scaffolding. Searched
v2ecoli for `workspace.yaml`, `studies/`, `investigations/`, `study.yaml`,
`spec.yaml`. None present in v2ecoli; found one in viva-munk, partial in
v2ecoli-workspace.

**Friction.** The cwd had a stale `.pbg/server/server-info` pointing at port
61341 (server dead), plus `docs/workspace-template/investigations/` template
files. Both were red herrings. The actual signal — "no `workspace.yaml` =>
not a workspace" — wasn't surfaced by any skill.

**Gap.** No `/pbg-status` or `/pbg-workspace status` subcommand. `pbg-workspace`
is bootstrap-only.

**Proposal.** New skill `/pbg-status` in `pbg-superpowers/skills/pbg-status/`:

```
$ /pbg-status
cwd: /Users/eranagmon/code/v2ecoli
workspace.yaml: NOT FOUND
  -> this directory is not a pbg workspace.
nearest workspace.yaml above cwd: (none)
catalog (~/.pbg/workspaces.json): 3 registered workspaces:
  - v2ecoli-workspace  /Users/eranagmon/code/v2ecoli-workspace  (last opened: 2026-05-15)
  - viva-munk          /Users/eranagmon/code/viva-munk          (last opened: 2026-05-15)
  - pbg-biomodels      /Users/eranagmon/code/pbg-biomodels      (last opened: 2026-04-30)
dashboard servers: 1 alive (viva-munk on http://127.0.0.1:61349)
stale .pbg/server/server-info in cwd: yes (PID 60307 not running; safe to delete)

next:
  /pbg-workspace ... --upstream <repo>       # spawn a sibling workspace
  /pbg-workspace bootstrap-in-place          # turn this checkout into a workspace
  cd /Users/eranagmon/code/viva-munk         # open an existing one
```

Argument: none, or optional `--path <dir>`. Tools: `Bash` only.

### 2. `/pbg-workspace --upstream` clones into a new dir; there's no in-place mode  ✅ LANDED 2026-05-16

> **Update:** the receiving Claude session shipped this. The `pbg-workspace`
> skill description now reads "Three modes: (1) upstream-branch ... (2)
> standalone ... (3) in-place — promote an existing git checkout into a
> workspace branch without cloning." Verified by inspecting the skill list
> mid-session. Leaving the original finding below for context.


**What I did.** The user explicitly asked to stay in `~/code/v2ecoli`. I
couldn't use upstream-branch mode (it `gh repo clone`s into a fresh target).
Hand-rolled `workspace.yaml` + populated the scaffolding instead.

**Friction.** Two-thirds of `workspace.yaml` is boilerplate that should not be
authored from scratch. I duplicated pbg-template's content from memory.

**Gap.** `pbg-workspace` SKILL.md has two modes (upstream-branch, standalone).
Neither handles "I already have the upstream cloned at cwd — promote this
checkout into a workspace branch."

**Proposal.** Add a third mode to `pbg-workspace`:

```
/pbg-workspace bootstrap-in-place [--branch <name>] [--package <pkg>]
```

Pre-flight:
- refuse if `workspace.yaml` already exists at cwd-root
- require cwd-root is a git repo with a clean working tree
- require currently on a branch with an upstream (so we can branch off it
  consistently)

Steps:
1. `git checkout -b <branch>` (default: kebab-case of repo name + `-workspace`)
2. Apply pbg-template's workspace-specific files: `workspace.yaml`,
   `.claude/settings.json`, `.gitignore` additions, `NEXT_STEPS.md` if missing.
   (Generic scaffolding from PR #50 may already be present; merge, don't
   overwrite.)
3. `python -m pbg_superpowers.workspace_catalog add --path . --name <name> --package <pkg>`
4. `git add -A && git commit -m "feat(workspace): bootstrap in-place"`

This is exactly what I did manually; it should be one command.

### 3. Scaffolding 6 studies from a plan PDF should be one skill invocation

**What I did.** Read `chromosome-replication-plan-2.pdf`, identified six
phases/studies, wrote `studies/dnaa-{01..06}/study.yaml` by hand. Each
study.yaml has: objective, description, baseline composite ref, variants,
observables, visualizations, implementation_tasks, expected_behavior,
expert_questions, references. ~600 lines total YAML.

**Friction.** This is mechanical transcription with light judgment calls.
The plan PDF already segments by `## Study N:` headers; each section has
predictable subheaders (Objective / Required knowledge / Implementation
Tasks / Read outs / Expected Behavior / References / Expert questions). I
spent ~20 min on plumbing that should be ~2 min of review.

**Gap.** `/pbg-study new` (in source pbg-superpowers/skills/pbg-study/) creates
ONE empty study. There is no "decompose a plan into N drafts" path.

**Proposal.** Sub-command:

```
/pbg-study scaffold-from-plan <plan.pdf> \
    [--expert <expert.pdf> ...] \
    [--composite <dotted-ref>] \
    [--slug-prefix <prefix>] \
    [--bib-key-prefix <prefix>]
```

Behavior:
1. Read the PDF; ask Claude (the host model — this is a skill, so this happens
   in-band) for a structured list of studies/phases with the canonical fields.
2. For each, write `studies/<slug-prefix>-NN-<slugified-name>/study.yaml` with
   `status: draft`, baseline pointing at `--composite` (or prompt if absent).
3. For each PDF given as `--expert`, copy under `references/expert/` (sha256
   dedup) and register in `workspace.yaml.expert_docs`.
4. Surface authored references; offer to append BibTeX skeletons to
   `references/papers.bib` and stub `references/claims.yaml` entries.
5. Print a summary tree + recommended next step
   (`/pbg-study open dnaa-01-...`).

The plan I just consumed was unusually well-structured, so a v0 that requires
exactly that structure is fine; document the expected section layout.

### 4. v3 `study.yaml` shape is implicit and inconsistent

**What I did.** Picked the field shape by reading three contradictory sources.

**Friction.**

- `~/code/viva-munk/studies/bending-pressure-260515/study.yaml` uses
  `baseline:` as a **dict** (`{composite, params}`).
- v2ecoli `reports/{workflow,multigeneration,colony,compare}_report.py`
  `_load_study()` expects `baseline:` as a **list of dicts**
  (`[{name, composite, params}, ...]`).
- The viva-munk study has `runs:` populated; my drafts have `runs: []` and
  there's no schema saying "runs may be absent" or "if status is draft,
  runs must be empty."

**Gap.** No `.pbg/schemas/study.schema.json`. `scripts/lint-workspace.py`
doesn't validate `studies/**/study.yaml`. `pbg-study` has 14 subcommands all
operating on this shape with no schema-of-record.

**Proposal.**

1. Add `.pbg/schemas/study.schema.json` to pbg-template/template/.pbg/schemas/.
   Required: `schema_version` (const 3), `name`, `created`, `status`,
   `objective`, `baseline`, `variants`, `runs`, `visualizations`.
   - `baseline`: anyOf [dict, list-of-dicts]. Document the single-arch /
     multi-arch convention OR pick one (recommend always-list; matches v2ecoli
     reports).
2. Extend `scripts/lint-workspace.py` to glob `studies/**/study.yaml` and
   validate each. Print one line per study (see finding #8).
3. Migrate viva-munk if the list shape wins.

### 5. expert_docs registration is manual paperwork

**What I did.**
```bash
cp ~/Downloads/Molecular\ information\ ...pdf references/expert/replication_initiation_molecular_info.pdf
shasum -a 256 references/expert/replication_initiation_molecular_info.pdf
# -> 4c4b85ff4d88b46b816bacc84b494fcdf073ef1d901e77b56329effa92a0dab2
# then hand-edited workspace.yaml.expert_docs with name + path + sha256 + description + contributor + claims_supported
```

**Friction.** Path slug, sha256, description, and `claims_supported:` cross-
references all hand-typed. Easy to mistype the sha or forget to register.

**Gap.** `scripts/add-reference.sh` exists for BibTeX. No analogue for
expert PDFs. The dashboard description claims PDFs auto-extract metadata, but
no shell/skill entry-point.

**Proposal.** Either a shell script `scripts/add-expert.sh <pdf>` mirroring
`add-reference.sh`, OR (preferred) a skill `/pbg-data add-expert <pdf>`:

```
/pbg-data add-expert ~/Downloads/foo.pdf \
    --name foo_brief \
    [--contributor "Alice / Bob lab"] \
    [--claims claim.id.1,claim.id.2]
```

Behavior:
1. Copy under `references/expert/<name>.pdf` (refuse if sha256 collides).
2. Compute sha256.
3. Extract first-page text; ask Claude for a 2-line description.
4. Append a complete entry to `workspace.yaml.expert_docs`.
5. Lint workspace.

### 6. Pre-create `studies/<name>/{composites,viz}/` and `runs.db`?

**What I did.** Created only `study.yaml`, not the sibling subdirs that
viva-munk and v2ecoli-workspace have (`composites/`, `viz/`, `runs.db`).

**Friction.** When the user runs the first baseline, where does the run land?
Is `runs.db` auto-created or must `/pbg-study run-baseline` create it?

**Gap.** Undocumented. Need to read `pbg-study run-baseline` source to find out.

**Proposal.** Document in `pbg-study`'s SKILL.md: "`run-baseline` lazily
creates `runs.db` + `viz/`; `study.yaml` is the only required file for
`status: draft`." Or — opposite — have `/pbg-study new` pre-create the
subdirs with `.keep` files so the layout is immediately greppable.

### 7. Dashboard model needs to be linked from `pbg-workspace`

**What I did.** Searched for the dashboard's API endpoints (`/api/study-*`,
`/api/expert-doc`, etc.) to know what to populate. Found them by reading
`pbg-study` subcommand listings.

**Gap.** `pbg-workspace`'s "Next steps" mentions tabs but not the dashboard
data model document.

**Proposal.** Add a single line at the end of `pbg-workspace/SKILL.md`:
"See `docs/concepts/vivarium-dashboard-model.md` for the dashboard's data
model and `/api/*` endpoints."

### 8. `scripts/lint-workspace.py` is silent on what it found

**Current output (success):** `workspace lint: OK`

**What I'd want.** After running on my scaffolded v2ecoli:

```
workspace lint: OK
  workspace: v2ecoli  (package: v2ecoli)
  expert_docs: 2  (replication_initiation_molecular_info, chromosome_replication_plan)
  bib: 17 keys
  claims: 16 (all resolve to bib keys)
  studies: 6  (all status: draft)
    - dnaa-01-expression-dynamics       baseline: v2ecoli.composites.baseline.baseline
    - dnaa-02-atp-hydrolysis            ... parent_studies: [dnaa-01-expression-dynamics]
    - dnaa-03-box-binding               ...
    ...
  runs: 0 active, 0 completed
  composites discovered in registry: 4  (baseline, departitioned, reconciled, colony)
```

**Gap.** Current `scripts/lint-workspace.py` introspects `build_core()` but
doesn't print findings on success — exits silently. The verbose summary
would be the agent's primary "did the scaffolding register?" feedback.

**Proposal.** Add `--summary` flag (or make verbose the default; quiet on `--quiet`):

```python
# at the end of main() in scripts/lint-workspace.py
if not args.quiet:
    print(f"workspace lint: OK")
    print(f"  workspace: {ws_data['name']}  (package: {ws_data.get('package_path')})")
    print(f"  expert_docs: {len(ws_data.get('expert_docs', []))} ...")
    # etc.
```

### 9. `parent_studies:` exists; nothing renders the DAG

**What I did.** Used `parent_studies:` to encode dnaa-02 ← dnaa-01,
dnaa-04 ← dnaa-03, dnaa-05 ← {dnaa-02, dnaa-04}, dnaa-06 ← dnaa-04.

**Gap.** Nothing reads `parent_studies:`. No tooling.

**Proposal.** `/pbg-study dag` prints:

```
dnaa-01-expression-dynamics
└── dnaa-02-atp-hydrolysis
    └── dnaa-05-rida-ddah-dars (also depends on dnaa-04)
dnaa-03-box-binding
└── dnaa-04-initiation-mechanism
    ├── dnaa-05-rida-ddah-dars
    └── dnaa-06-seqa-sequestration
```

And/or a `--mermaid` flag emits a mermaid graph that the dashboard's Studies
tab can render natively.

---

## Concrete deliverable shortlist for the other Claude session

If you're improving pbg-superpowers and want a sequence to attack:

1. **`scripts/lint-workspace.py --summary`** (finding #8) — smallest change,
   biggest UX win for agents authoring scaffolding.
2. **`.pbg/schemas/study.schema.json`** (finding #4) — eliminates the
   "dict vs list" ambiguity I just lived through.
3. **`/pbg-workspace bootstrap-in-place`** (finding #2) — pairs with PR #50 to
   cover the "I'm already cloned" path.
4. **`/pbg-status`** (finding #1) — single most useful agent-onboarding skill.
5. **`/pbg-study scaffold-from-plan <plan.pdf>`** (finding #3) — biggest
   end-user win but largest implementation.
6. **`/pbg-data add-expert <pdf>`** (finding #5) — pairs with #5 nicely.

---

## Live log

Appended as I work the rest of this session with the user iterating on the
studies and viewing them in the dashboard. New entries go below; do not
edit older ones.

### 2026-05-16 (post-scaffolding)

- Branch `dnaa-replication-studies` lives at HEAD `2891f01` on v2ecoli.
  Two commits: scaffolding + this notes file.
- `scripts/lint-workspace.py` passes.
- Next step in this session: have user `/pbg-server start` from the workspace
  and confirm the 6 studies show up in the dashboard's Studies tab.
- Open question: does the running dashboard auto-discover
  `studies/**/study.yaml`, or does each study need to be registered via
  `/api/study-new`? My drafts don't have associated runs, so they may not
  appear if discovery is run-driven.

### 2026-05-16 (server started)

- `vivarium-dashboard serve --workspace /Users/eranagmon/code/v2ecoli`
  bound on port 49428. `GET /api/state` returns 200, workspace.yaml +
  both expert_docs visible.
- Cleaned a **stale `.pbg/server/server-info` left in cwd by a long-gone
  server**. The pre-existing pid file was already gone. Re-iterates
  finding #1: agents need `/pbg-status` to surface this without manual
  `kill -0 $PID`.
- **Open question answered:** the dashboard DOES auto-discover studies.
  `GET /api/studies` returns 6 entries with metadata derived directly
  from each `studies/<name>/study.yaml`. No `/api/study-create` round-trip
  needed.
- **New friction (now finding #10 below):** the dashboard auto-migrates
  any `schema_version: 3` study to v4 in memory and runs the v4 validator.
  My custom `references:` (dict) + `implementation_tasks:` (list) collided
  with the v4 reserved field shapes (list-of-{file} + string respectively),
  so all 6 studies came back as `status: invalid` until I renamed them to
  `bibliography:` + `tasks:`. Fix is commit `f181977`.
- All 6 are now `status: draft, n_baseline: 1, baseline_source:
  v2ecoli:baseline.baseline`, n_variants matches the plan (4/3/3/5/5/3).
- **Friction next to surface:** `composite: ""` and `composites: []` are
  empty in the dashboard's per-study response, even though
  `baseline_source` resolved. Hints that the dashboard expects a separate
  "register the baseline composite" step. To investigate.

### 2026-05-16 (finding #2 landed)

- The receiving Claude session updated `pbg-workspace` SKILL.md so the
  description now lists three modes including in-place. So the cycle is
  working as designed: notes → improvement → installed skill. Worth
  amplifying: putting **a single proposal block per finding** (skill name,
  CLI signature, behavior steps) made the receiving end's job easy.
  Vague "we should have X" doesn't.

---

## Finding #10 (added live): v3→v4 auto-migration trips user-defined fields

**What I did.** Wrote `schema_version: 3` study.yaml files with custom keys
`references:` (a structured dict) and `implementation_tasks:` (a list of
strings). Loaded the dashboard. Got back six
`{"status": "invalid", "error": "references must be a list"}` from
`/api/studies`.

**Friction.** The error message ("references must be a list") doesn't tell
you *why* the validator wants a list. The actual reason is in
`vivarium_dashboard/lib/spec_migration.py:170-191` — `migrate_v3_to_v4`
is called unconditionally on read, promotes my v3 spec to v4, then the
v4 validator (`vivarium_dashboard/lib/investigations.py:170-190`) trips
because v4 reserves `references:` for `[{file: "path"}, ...]` and
`implementation_tasks:` for a markdown string. None of this is documented
on the v3 user-facing contract.

**Gap.**
- v4 reserved field names aren't called out in v3 docs.
- The error message doesn't say "v4 reserved field"; it says "must be a
  list", which suggests v3 had a `references:` field that wanted a list,
  which isn't true.

**Proposal.**
1. **Error message:** when the v4 validator trips on a field that didn't
   exist in v3, prefix the error: "v3→v4 auto-migration: field
   `references` collides with a v4 reserved name; rename it or migrate
   your spec to v4 explicitly".
2. **Documentation:** add a "Reserved field names in v4" subsection to
   the v3 study.yaml contract (when it lands per finding #4).
3. **Tooling:** `lint-workspace.py` should call the same migration +
   validation path that the dashboard uses, so the agent gets the error
   without having to start the server. Right now lint passes but the
   dashboard rejects — that mismatch ate 6 tool calls of debugging.

### 2026-05-16 (refining dnaa-01)

- Concrete iteration: mapped the dnaa-01 study to the real v2ecoli
  identifiers (EG10235, EG10235_RNA, MONOMER0-160[c]) and existing
  listener paths (rnap_data.rna_init_event, rna_synth_prob.n_actual_bound).
- Surfaced an important user-experience insight: the study.yaml's
  `observables` block needs a per-entry `status:` flag (available /
  derived-needed / aspirational), because in a plan-driven workflow many
  observables are *intentions* not measurements. Without that flag the
  user can't tell at a glance which plots will populate today.
- Also added per-observable `index_by: {type, value}` to express the
  numpy-indexed-by-id pattern used by v2ecoli's `bulk_array` and TF /
  TU-indexed listeners. The current observable schema (name/store_path/
  units/description) has no shape for this, but the dashboard tolerated
  the extension field. Worth formalizing.
- Added `gaps:` block at study top level with id/title/why/approach for
  each piece of new code needed. Effectively a study-local TODO list. The
  dashboard tolerates this; it would be more useful if it were rendered.

## Finding #11 (added live): YAML traps in flow-style values

Within an hour of authoring I hit two YAML parse errors that the
dashboard surfaced (because v3→v4 migration walks the whole spec):

- `{translation_efficiency_override={EG10235: 0.0}}` in a flow mapping —
  the bare `:` makes YAML think we're declaring a sub-mapping. Need to
  quote the whole string.
- `index_by: {type: bulk_id, value: MONOMER0-160[c]}` — the bare `[c]`
  is parsed as a flow sequence inside a flow mapping. Need to quote.

**Proposal.** `scripts/lint-workspace.py` should validate every
`study.yaml` by parsing it through PyYAML (already happens implicitly)
*and* through `vivarium_dashboard.lib.spec_migration.migrate_v3_to_v4 +
_validate_study_v3_or_v4`. The current lint silently passes specs that
the dashboard rejects, forcing the agent to start the server just to
catch trivial syntax problems.

This is a sharper version of finding #4's "ship a schema" — the schema
isn't enough; the same migration code needs to be reachable from the
CLI.

## Finding #12 (added live): per-observable `status:` and `index_by:` are missing from the v3 schema but are essential for plan-driven authoring

**What I did.** Each `observables[i]` entry in my refined dnaa-01 has:

```yaml
- name: dnaA_protein_count
  status: available
  description: |
    Bulk DnaA monomer count ...
  store_path: agents.0.bulk
  index_by: {type: bulk_id, value: "MONOMER0-160[c]"}
```

**Why this matters.**
- `status: available | derived-needed | aspirational` tells the user
  which observables will plot today vs which require new derived
  listeners or are deferred to a later study. In plan-driven work that
  decomposes intent before code lands, most observables in a draft
  study are aspirational; the user needs to see which are which.
- `index_by` covers the very common case where the underlying store is
  a numpy structured array (e.g. v2ecoli's `bulk_array`) addressed by
  a string ID. Without it, the `store_path` either points at the whole
  array (no scalar to plot) or has to bake the index into a synthetic
  path the dashboard can't resolve.

**Proposal.** Extend the v3/v4 observable schema with both fields,
documented, and have visualizations honor them:

```jsonschema
{
  "status": {"enum": ["available", "derived-needed", "aspirational"]},
  "index_by": {
    "type": "object",
    "required": ["type", "value"],
    "properties": {
      "type":  {"enum": ["bulk_id", "rna_id", "tf_id", "tu_id", "literal_index"]},
      "value": {"type": ["string", "integer"]}
    }
  }
}
```

The dashboard's observable resolver can then dispatch on `index_by.type`:
look up the index in sim_data, then read `store_path[index]`. Without
this, every dnaa-* study has to reinvent the convention.

### 2026-05-16 (Q+H, status vocab, behavioral_tests for dnaA-01)

- Added `question:` and `hypothesis:` to all 6 dnaa-* studies. Each Q
  states the measurable prediction; each H states quantitative thresholds
  drawn from the plan + expert doc (e.g. dnaa-02 hypothesises ATP-fraction
  0.2-0.5 from Boesen 2024; dnaa-05 says inter-initiation CV narrows by
  ≥30% vs intrinsic-only).
- Migrated all 6 from `status: draft` (informal) to `status: planned`
  (the canonical `_VALID_STATUSES` member in
  `vivarium_dashboard/lib/investigations.py:30`). The dashboard now
  reports `status=planned` for all 6 instead of the freeform `draft`
  string. **User flagged:** the existing `draft` marker is visible in
  the dashboard pop-out today — that's because reads tolerate any string
  but writes go through `update_spec_status` which enforces the six.
  Worth surfacing in `pbg-status` (finding #1) which value is canonical.
- Implemented BT-01..BT-04 as real pytest assertions in
  `studies/dnaa-01-expression-dynamics/tests/test_expression_dynamics.py`.
  Each ties back 1:1 to a `behavioral_tests:` entry in study.yaml via
  the `pytest_node:` field. BT-05 (post-initiation gene-dosage) is an
  xfail stub until dnaa-04 lands.
- `conftest.py` reads this study's `runs.db` directly (SQLite,
  runs_meta + history(state JSON), schema documented at
  `vivarium_dashboard/lib/composite_runs.py`). Tests `pytest.skip`
  cleanly when no completed run exists, so the suite is meaningful
  documentation even pre-run.

## Finding #13 (added live): Registry tab leaks processes from other workspaces

**Reported by the user mid-session.** The dashboard's Registry tab shows
discovered processes from `multi_cell` (viva-munk) and other workspaces
while reporting that this workspace has **no installed modules**. The
two views contradict each other.

**Diagnosis (probable).** `vivarium_dashboard/server.py` builds the
"discovered processes" list by calling `build_core()`, which walks
*every* installed entry point — including processes pip-installed into
the same Python environment as the dashboard server, not filtered by
the active workspace. The "Installed modules" panel reads
`workspace.yaml.imports` which is empty here. Two distinct discovery
paths, no reconciliation.

**Proposal.**
1. Filter the discovered-processes table by the active workspace's
   imports + its own package_path. Anything outside that set is hidden
   (or shown under a separate "Available in environment but not
   imported" section).
2. If a process is discoverable but not declared in `workspace.yaml`,
   surface it as an actionable item: "Add `<pkg>` to imports?" with a
   one-click registry-import button.
3. The "Installed modules" panel should at minimum list `package_path`
   from workspace.yaml (here: `v2ecoli`) as installed-by-default, since
   that's what `build_core()` was given to import.

This is finding-priority HIGH; it's the kind of bug that erodes trust in
the dashboard for a new user.

## Finding #14 (added live): Pop-out should render `status`, `question`, `hypothesis`

**Reported by the user mid-session.** When they clicked into a study and
the pop-out / details panel opened, the new `question:` and
`hypothesis:` fields and the canonical `status:` (planned vs draft)
were not visible there. The list view shows them but the detail
pop-out doesn't.

**Proposal.** In the study detail / pop-out panel, render:
- A status chip (planned / running / ran / complete / failed / invalid)
  with the canonical color coding.
- The `question:` field as italicised text under the title.
- The `hypothesis:` field as a callout box ("Predicted outcome").
- The `behavioral_tests:` entries as a checklist with status icons
  (implemented = ✓, stub = ◯, gated-by-gap = ⏳) and the English
  description as hover text.

This is essential for the plan-driven authoring loop: the user iterates
on the spec, then opens the pop-out to read back what the study
*claims* to test, before running anything.

## Finding #15 (added live): new skill `/pbg-study fill-overview <slug>`

**What the user asked for, verbatim.** "Can you also fill out the
Question and Hypothesis of these different studies for me? and this
should be a pbg-superpower to fill in these fields."

**Proposal.** New subcommand on `pbg-study`:

```
/pbg-study fill-overview <slug> \
    [--from-plan references/expert/<plan>.pdf] \
    [--from-expert references/expert/<doc>.pdf ...] \
    [--fields question,hypothesis,objective,description]
```

Behavior:
1. Read the study's current `study.yaml`.
2. Read the linked plan / expert docs from
   `workspace.yaml.expert_docs`.
3. Ask the host Claude to draft each requested field. Each field is
   bounded:
   - `question:` — one paragraph, scientifically framed, ends with `?`.
   - `hypothesis:` — one paragraph with quantitative thresholds where
     the source documents give them.
   - `objective:` — imperative present tense, one paragraph.
   - `description:` — multi-paragraph, citing the source section.
4. Preview-and-confirm flow: print the diff, ask the user
   yes/no/edit-prompt, then write.
5. Update the study via `/api/study-set-overview` (which already
   accepts `{question, hypothesis, topic, status}` per
   server.py:6136) so the dashboard sees it immediately without
   needing a re-read of disk.

This is the most-leveraged proposal in the notes today: I filled 12
fields by hand (6 questions + 6 hypotheses); a tool that does this from
the plan PDF in one call replaces ~30 minutes of mechanical work.

## Finding #16 (SUPERSEDED → see #16-rev): structured `expected_behavior:` DSL

> The first pass of this finding proposed a parallel `behavioral_tests:`
> block alongside `expected_behavior:` (list of strings). After one
> more iteration with the user — who asked for a precise + reproducible
> grammar matching simple English statements like *"If you stop
> synthesized DnaA, look for decrease in concentration"* — the two
> blocks have been unified into one structured `expected_behavior:`
> list. See finding #16-rev below.

## Finding #16-rev (built live in dnaa-01): expected_behavior DSL

**What I built today.** A small grammar that pairs each English
prediction with a precise machine-readable triple so one YAML entry
auto-generates exactly one pytest assertion. Lives at:

- `studies/dnaa-01-expression-dynamics/study.yaml` — the structured
  `expected_behavior:` list (6 entries today).
- `studies/dnaa-01-expression-dynamics/tests/_behaviors.py` — the
  evaluator (~250 LOC, stdlib + statistics).
- `studies/dnaa-01-expression-dynamics/tests/test_behaviors.py` — one
  parametrized test that walks the list.
- `studies/dnaa-01-expression-dynamics/tests/test_with_synthetic_history.py`
  — demonstrates 5 green tests against an in-memory plausible history
  before any real simulation has been executed.

**Grammar (final shape).**

```yaml
expected_behavior:
- name: <stable-slug>
  en: "<one-sentence English description; goes in Overview tab>"
  given:
    run: baseline | variant
    variant: <variant-name>          # if run == variant
    window: full | second_half | post_initiation_10min
  measure:
    kind: bulk_count | listener_path | listener_sum | xy_correlation
    # ...kind-specific args
    reduce: median | mean | series | first_and_last | pre_post_event_ratio
  expect:
    op: in_range | rolling_cv_below | ratio_at_most | ratio_at_least |
        monotonic_decreasing | pearson_below | pearson_above
    # ...op-specific args (low, high, threshold, ratio, window_steps, ...)
  status: implemented | stub | gated
  requires:
    - gap: <gap-id>
    - listener: <listener-id>
    - variant_hook: <hook-id>
```

**Why this matters.** The user's exact phrasing: *"is there a way to do
this very precisely and reproducibly?"* The DSL hits both axes:

- **Precisely:** every English sentence has a unique `(given, measure,
  expect)` triple. No ambiguity about what "look for decrease" means;
  it's `op: ratio_at_most, ratio: 0.7` over `window: full`.
- **Reproducibly:** any future Claude session reading the YAML can
  re-derive the same assertion. The evaluator is deterministic.

**Proposal for pbg-superpowers.**

1. **Adopt the DSL upstream.** Add `expected_behavior:` (structured) to
   the v3/v4 study schema. Document the grammar in
   `docs/concepts/expected-behavior-grammar.md`.
2. **Lift the evaluator.** Move `_behaviors.py` into the dashboard's
   `vivarium_dashboard/lib/expected_behavior.py` so every workspace
   uses the same dispatch logic instead of copying it.
3. **Lift the conftest helpers.** Provide
   `vivarium_dashboard/testing/study_fixtures.py` with
   `baseline_history`, `variant_history(name)`, `bulk_count(state, id)`,
   `listener_value(state, path)`. Workspaces import from there; per-study
   conftest.py shrinks to ~10 lines.
4. **Render in the dashboard.** On the Overview tab (per finding #14),
   show each `expected_behavior:` entry as: ⬛ status icon · `en:` text ·
   "Show assertion ▶" disclosure that prints the structured form. On the
   Tests tab, group pytest nodes by their `expected_behavior[i].name` so
   pass/fail aligns with the English sentence.
5. **Extension primitives needed.** A few common measures are still
   missing from my v0 evaluator:
   - `event_count(predicate)` — count timesteps satisfying a predicate
     (e.g., initiations).
   - `pre_post_event(event, before_min, after_min)` — slice around an
     event time, paired with `reduce: pre_post_event_ratio`. This unblocks
     BT-05 / dnaa-04's gene-dosage test.
   - `concentration(molecule, volume_path)` — derived from bulk_count +
     volume; closes finding gap-1 without a custom listener Step.

## Finding #17 (added live): Observables tab with bigraph-tree picker

**Reported by the user.** *"The Study needs also a tab for 'Observables'
which are the experimental measurements. This should select them from
the bigraph structure, with paths."*

**Current state.** Observables are author-only YAML in
`study.yaml.observables[]` with `name + store_path + (custom) index_by`.
There's no UI to discover what's available in the composite's bigraph
state tree. I had to spelunk into the source — `v2ecoli/processes/
transcript_initiation.py:249`, `transcript_elongation.py:184`,
`tf_binding.py:122`, etc — to find legal `store_path` values. Per
finding #1, this is a slow path for an agent and an impossible path
for a non-developer user.

**Proposal: new Observables tab in the per-study UI.**

1. **Backend.** `GET /api/study/<name>/bigraph-paths` returns the JSON
   tree of stores for this study's baseline composite. Build it by
   importing the composite (via `v2ecoli.build_composite` or the workspace
   equivalent) and walking the resulting `Composite.state` schema. Cache
   per (workspace, composite_ref, sim_data_fingerprint).

2. **Tree shape.**

   ```jsonc
   {
     "path": ["agents", "0", "listeners", "rnap_data"],
     "kind": "store",
     "type_hint": "map",
     "children": [
       {
         "path": [..., "rna_init_event"],
         "kind": "store",
         "type_hint": "overwrite[array[integer]]",
         "leaf_meta": {
           "indexed_by": {"type": "rna_id", "lookup": "sim_data.process.transcription.rna_data.id"},
           "units": null,
           "first_seen_in_process": "transcript_initiation"
         }
       },
       ...
     ]
   }
   ```

3. **UI.**
   - Left pane: collapsible tree of paths under `agents.<id>.*`. Type hints
     and indexed-by metadata visible inline.
   - Right pane: form to add an observable — name, store_path (auto-filled
     from the tree click), optional index_by (auto-suggested when the
     leaf carries `indexed_by`), units, description.
   - Save → appends to `study.yaml.observables[]` and refreshes the
     Visualizations tab's dropdown.

4. **Cross-link to expected_behavior.** When the user references an
   observable name in an `expected_behavior:` entry's `measure:` block,
   the dashboard validates the name resolves and the path/index_by line up.

**Why this matters.**

- For the user: replaces "ask the agent to find the right path" with a
  click. Closes the loop between the bigraph state model and the study
  schema.
- For agents: an HTTP endpoint that returns the legal-paths tree means
  no more grepping the codebase. Same data, two interfaces.

This pairs naturally with finding #1 (`/pbg-status`) and #14 (pop-out
rendering) — together they give the workspace a coherent
"introspect-and-author" surface.

---

## Concrete deliverable shortlist for the other Claude session (updated 2)

| # | Improvement | Effort | Priority |
|---|---|---|---|
| 8     | `lint-workspace.py` enumerates findings + runs the dashboard's own validator (#11) | XS | P0 |
| 13    | Filter Registry tab discovered-processes by workspace imports | M | P0 |
| 14    | Render expected_behavior on Overview tab (status pill + Q + H already render today) | M | P0 — PARTIAL ✅ |
| 17    | Observables tab with bigraph-tree picker | L | P0 (user asked) — VIEW ONLY ✅ |
| 16r   | Lift expected_behavior DSL + evaluator + fixtures to the dashboard package | M | P1 |
| 4     | `.pbg/schemas/study.schema.json` (now needs to include the DSL) | M | P1 |
| 15    | `/pbg-study fill-overview <slug>` | M | P1 |
| 1     | `/pbg-status` | S | P2 |
| 3     | `/pbg-study scaffold-from-plan <plan.pdf>` | L | P2 |
| 5     | `/pbg-data add-expert <pdf>` | S | P2 |
| 6     | Document study-subdir layout | XS | P3 |
| 9     | `/pbg-study dag` | S | P3 |

### 2026-05-16 (dashboard UI patches — user wanted these visible NOW)

User said: *"i dont see expected behavior in my dashboard Study. can
we get this up? and add Observables tab"*. The data was already in the
API (`/api/study/<name>.spec.expected_behavior` is a list of 6 entries);
the UI just didn't render it.

Shipped two patches to `vivarium-dashboard-tests-investigations` branch
`feat/studies-with-tests-and-investigations` (commit `21d6a07`):

1. **Overview tab → Expected Behavior section.** Renders each entry's
   `en:` sentence in a card with a status-coded left border
   (implemented=green, stub=amber, gated=slate) and a collapsible
   "Assertion" disclosure that prints the `given/measure/expect/requires`
   structured form. Section omits cleanly when the list is empty.
2. **New Observables tab.** Read-only table of
   `{name, status, store_path, indexed_by, description}` with status
   icons (available / derived-needed / aspirational). Placeholder
   disclosure documents the planned `GET /api/study/<name>/bigraph-paths`
   picker (the actual bigraph-tree picker is still finding #17).
3. **Status dropdown** extended to include `planned`, `running`, `ran`,
   `complete` (the canonical `_VALID_STATUSES`) — v2ecoli's
   `status: planned` previously wasn't in the dropdown options.

Caveats / follow-ups for the other Claude:
- The "+ Add observable" form is a static placeholder; the
  bigraph-tree picker still needs the backend endpoint + JS interaction.
  Same pattern as the existing variant-add form.
- No JS changes were needed — `_setStudyTab()` already switches on
  `data-kind` which matches the new tab.
- Status pill CSS classes (`.status-planned`, `.status-running`, etc.)
  may not exist yet; the inline status icons in the Expected Behavior
  list don't depend on them but the title-bar pill might fall through
  to default styling. Worth a follow-up CSS pass.

## Finding #18 (added live): dashboard's status dropdown vocab doesn't match `_VALID_STATUSES`

While shipping the patches above, I found that the old
`templates/study-detail.html` dropdown offered
`{draft, in-progress, completed, archived}` while the validator at
`vivarium_dashboard/lib/investigations.py:30` declares
`_VALID_STATUSES = {planned, running, ran, complete, failed, invalid}`.
**Two different vocabularies in the same package.** The dropdown's
values aren't canonical; status writes via `/api/investigation-set-overview`
would reject `in-progress`, `completed`, `archived` on `_VALID_STATUSES`
membership unless that endpoint bypasses validation.

**Proposal.** Pick one vocabulary and unify:
- The validator vocab is run-centric. The dropdown vocab is design-
  process-centric. The user wanted finer wording for the latter
  ("planning, implementing, done") which neither covers.
- Suggest the union: `{planned, implementing, runnable, running, ran,
  analyzing, complete, archived, failed, invalid}` with a documented
  state-transition map.
- Make `_VALID_STATUSES` the source of truth and have the template
  render `<option>` entries by iterating that set, so the two stay
  in sync automatically.

### 2026-05-16 (width fix + bigraph-tree picker shipped)

User: *"the embedded Study within the window is underutilizing the
size of the browser window… Fix that and the bigraph-tree picker"*.
Both shipped on the dashboard branch `feat/studies-with-tests-and-investigations`
(commit `2d693f0`).

**Width fix (one CSS rule, three overrides).**
- `main { max-width: 1800px }` (was 1200) — bumps the global container.
- `#page-studies { max-width: none }` — Studies tab fills the window.
- `main.study-page { max-width: none; padding: 16px 20px }` — embedded
  study-detail content fills its iframe.
- `#study-detail-frame { height: calc(100vh - 160px); min-height: 720px }`
  — iframe also gets more vertical room.

**Bigraph-tree picker (new GET endpoint + new tree walker + two-pane UI).**
- `GET /api/study-bigraph-paths?study=<slug>[&baseline=<name>][&max_depth=N]`
  resolves the study's first baseline composite, looks for a serialized
  state file under `<workspace>/models/<basename>.{pbg,json}` (with a
  v2ecoli legacy fallback to `partitioned.pbg`), and returns
  `{composite, source_file, max_depth, node_count, nodes:[{path,kind,...}]}`.
  Cached by `(path, mtime, max_depth)`. Returns 404 with `looked_in` +
  `hint` when no snapshot is on disk.
- New `walk_state_snapshot()` in `composite_recipes.py`: companion to
  `walk_state_tree()`; handles state snapshots without `_type` metadata;
  special-cases the `__structured_array__` marker (extracts the column
  names from `dtype`); bounded depth.
- Observables tab in `study-detail.html` now renders a two-pane picker
  on the left side. Right side carries a "Selected path" panel +
  auto-generated observable YAML snippet + Copy button. For
  `structured_array` leaves the snippet pre-fills an `index_by` stub.

**Verified.** Against v2ecoli baseline:

```
GET /api/study-bigraph-paths?study=dnaa-01-expression-dynamics&max_depth=4
-> 238 leaves from models/partitioned.pbg
   including agents.0.bulk (structured_array) and
   agents.0.unique.DnaA_box (structured_array).
```

**Still open (finding #17 follow-ups).**
- "Click → Save" (no copy/paste) — needs a `POST /api/study-observable-add`
  endpoint that appends to `study.yaml.observables` and re-validates.
  Right now the snippet must be copy-pasted into study.yaml manually.
- For `bulk`-like stores indexed by molecule id, ideally the picker
  should let the user drill into the actual id list. The .pbg snapshot
  doesn't carry the id list — only the dtype — so this needs either a
  separate sim_data-introspection endpoint (expensive) or a runtime
  fetch from the most recent run's history.
- The depth limit (default 8) hides deeper paths. Acceptable today for
  the v2ecoli baseline (238 leaves at max_depth=4 is enough to find
  every interesting top-level store); revisit if any study needs
  deep-indexed paths.

## Finding #19 (added live): walk_state_tree vs walk_state_snapshot

While building the bigraph picker I added `walk_state_snapshot()`
alongside the existing `walk_state_tree()`. The two address different
needs:

- `walk_state_tree(doc)` — reads a process-bigraph **composite document**
  with `_type` / `_default` / `address` metadata. Used by the existing
  Composites tab and the loom-explore renderer.
- `walk_state_snapshot(state)` — reads a **state snapshot** (no type
  metadata, just values). Used by the new bigraph-paths endpoint.

**Proposal.** Document the distinction at the top of
`composite_recipes.py`, since the naming is too close. Or merge under
a single `walk(...)` with a `kind=` parameter. Either way, the next
agent will trip on this if it isn't called out.

---

### 2026-05-16 (Investigations as study collections)

User asked: *"I want to bring back a concept of Investigation, which
is a set of connected studies with dependencies between them."* —
Investigation is now the **higher-level container** (a named
collection of studies with an explicit DAG), and Study stays as the
per-experiment unit. The legacy "investigations as a synonym for
studies" naming is fully retired in the UI; backend keeps the legacy
aliases for back-compat.

Shipped end-to-end across both repos:

**v2ecoli** (`investigations/dnaa-replication/investigation.yaml`)
- New YAML format at `investigations/<name>/investigation.yaml` with
  `{schema_version, name, title, status, question, hypothesis,
    description, studies:[study-slug, ...], expert_docs:[...],
    acceptance_criteria:[{study, behavior}, ...]}`.
- File lives under `investigations/` but uses `investigation.yaml`
  (not `spec.yaml`) so the legacy walker correctly skips it.

**vivarium-dashboard** (`feat/studies-with-tests-and-investigations`
commits `a39c975`, `5b353a8`, `bc6c061`, `bc0c165`)
- `_iter_iset_dirs()` in `server.py` yields only dirs that contain
  `investigation.yaml` (vs `_iter_study_dirs` which yields per-study
  dirs by `study.yaml` / `spec.yaml`).
- New GET endpoints:
  - `/api/iset-list` — summaries (name, title, status, n_studies).
  - `/api/iset/<name>` — investigation + resolved studies. Each study
    carries `parent_studies:` normalized into the
    `[{study, condition}]` shape for the DAG.
- Rail rendering: studies grouped by their iset membership; group
  headers collapsible; topological depth ordering within each group;
  `[DAG]` link in each group header jumps to the investigation
  detail view. Ungrouped studies fall into a final "Ungrouped" bucket.
- New page `#page-investigations` with two views: list of investigation
  cards on entry, DAG canvas detail view on click. Width treatment
  mirrors `#page-studies` (`:has()` selector + full-bleed overrides).

## Finding #20 (added live): DAG visualization pattern

**What I built.** A simple SVG + absolute-positioned div layout for
the study DAG inside an investigation:

- BFS depth from roots: y = depth × (CARD_H + Y_GAP), x = within-depth
  slot. Rows centered horizontally.
- Each row is a horizontal band; cards have status-coded left border
  (planned/running/ran/complete/failed colors).
- SVG `<defs><marker>` arrowhead, cubic-Bezier paths between parent
  bottom-mid and child top-mid with vertical control offsets so curves
  flow downward.
- Edge labels show the parent's `condition:` (tests-passed / ran /
  complete) at the curve's midpoint.
- Shell uses `overflow-y: visible` + JS-driven `height = canvasH` so
  the **page** scrolls through the DAG instead of an inner box.
  Critical UX detail — easy to miss.

**Proposal.**
1. Lift the layout into `vivarium_dashboard/static/dag-layout.js` (or
   `lib/`) so it can be reused by future "study-of-studies" views
   (e.g., a workspace-wide DAG showing all investigations).
2. Replace the hand-rolled BFS-depth with a real layered-DAG
   algorithm (dagre-like) for cases with many cross-edges. v0 is
   fine for ≤10 nodes; the dnaA investigation has 6.

## Finding #21 (added live): In-place embed pattern for collection items

**What I built.** Clicking a node in the investigation DAG opens the
full study **in the same page**, below the DAG, in an iframe pointing
at `/studies/<name>`. Mirrors the existing study-detail embed inside
the Studies tab — no page switch, no context loss.

**Pattern (shareable across collection views):**
- Embed panel = `<div>` with title bar (name + Pop out + ×) + `<iframe>`.
- Click handler sets `iframe.src = '/<resource>/<name>'`, scrolls
  panel into view.
- Pop-out button uses `_openDetachedWindow(url, w, h)` (see #22).
- Close button clears `iframe.src` and hides the panel.

The same pattern works for "click a study in the rail → open in the
current page" (which I implemented with a back-compat shim that picks
the right embed target based on `window._currentIset`).

**Proposal.** Extract the embed panel + handlers into a reusable
component / partial (`templates/_embed_panel.html.j2` +
`static/embed.js`) so the Studies tab, the Investigations tab, and
any future tab can drop in the same affordance with one include.

## Finding #22 (added live): Pop-out → detached window, not new tab

**What the user wanted.** *"the pop out button opens up in a new
window in the same browser rather than truly popping out"* — they
want a real detached window for the popped-out study or
investigation.

**Cross-browser facts:**
- `window.open(url, '_blank')` without features → tab (modern default).
- `window.open(url, '_blank', 'popup=yes,width=N,height=N')` → popup
  window in Chromium and Safari. `popup=yes` is the modern keyword.
- Firefox respects `width=`/`height=` when present (default prefs).
- Hard-coded tab-only browsers (user pref) ignore the JS request;
  no JS escape hatch exists.

**Helper shipped:**

```js
function _openDetachedWindow(url, width, height) {
  var features = [
    'popup=yes',
    'width=' + (width || 1280),
    'height=' + (height || 900),
    'left=' + Math.max(0, (screen.availWidth - width) / 2),
    'top='  + Math.max(0, (screen.availHeight - height) / 2),
    'menubar=no', 'toolbar=no', 'location=no', 'status=no',
    'resizable=yes', 'scrollbars=yes',
    'noopener',           // discourage tab grouping with opener
  ].join(',');
  return window.open(url, '_blank', features);
}
```

All three pop-outs (`_popoutStudy`, `_popoutInvestigation`,
`_popoutInvestigationStudy`) route through it. Popup-blocker fallback
is `console.warn` + `alert`.

**Proposal.** Lift `_openDetachedWindow` into a shared client helper
in `vivarium_dashboard/static/client.js`. Document the
`browser-side prefs override JS` caveat in the dashboard's pop-out
README so users aren't surprised when their Arc/Firefox-config
returns a tab.

## Finding #23 (added live): URL-param round-trip for "open in detached window"

**The pattern.** A popped-out window needs to know which resource to
display. Two options:

1. Dedicated route per resource: `/studies/<name>` (already exists),
   `/investigations/<name>` (NEW — would need a server-side template).
2. Shared root URL + query param: `/?investigation=<name>#investigations`
   — JS reads the param on load and auto-opens the detail view.

I went with (2) for investigations to avoid adding a server-side
template for a feature that's frontend-only. Two gotchas:

- **Race**: the URL-param handler may fire before the
  page-investigations DOM is mounted. Solution: retry up to 20×
  with 100ms spacing (already in `walkthrough.js`).
- **Origin assumption**: build the URL from
  `window.location.origin + window.location.pathname`, NOT a hardcoded
  `/`. The dashboard might be deployed under a sub-path.

**Proposal.** For collection views, prefer option (1) (dedicated
route) when the resource has a stable identifier, AND option (2) (URL
param) for ephemeral views. Add a route handler for
`/investigations/<name>` that returns the dashboard HTML pre-rendered
with the right investigation pre-selected (eliminates the race).

## Finding #24 (added live): Sidebar grouping needs both a list endpoint and a member endpoint

The rail's grouped sidebar (collapsible investigations with member
studies) needs BOTH:

1. `/api/iset-list` — the investigations (group headers).
2. `/api/investigations` — the studies (group members, with their
   own status + blocked state).

The frontend `_renderRailInvestigationGroups` runs after BOTH have
loaded; if either is missing it falls back to the legacy flat
rendering. This works but it's brittle.

**Proposal.** Add a single `GET /api/iset-list?include=studies` flag
that returns the grouped representation in one round-trip:

```json
{
  "investigations": [
    {"name": "...", "title": "...", "studies": [<full study summary>, ...]},
    ...
  ],
  "ungrouped": [<study summaries>]
}
```

Cuts the number of fetches the rail makes from 2 to 1 and lets the
backend handle the membership computation that the frontend currently
re-does on every render.

---

## Concrete deliverable shortlist for the other Claude session (updated 3)

| # | Improvement | Effort | Priority |
|---|---|---|---|
| 8     | `lint-workspace.py` enumerates findings + runs the dashboard's own validator (#11) | XS | P0 |
| 13    | Filter Registry tab discovered-processes by workspace imports | M | P0 |
| 14    | Render expected_behavior on Overview tab (LANDED PARTIAL ✅) | M | P0 — PARTIAL |
| 17    | Observables tab with bigraph-tree picker (VIEW ONLY ✅) | L | P0 |
| 21    | Reusable embed-panel component | S | P0 (highest reuse) |
| 22    | Lift `_openDetachedWindow` into shared client helper | XS | P1 |
| 23    | Dedicated `/investigations/<name>` route to eliminate URL-param race | S | P1 |
| 24    | `/api/iset-list?include=studies` to reduce rail round-trips | S | P1 |
| 20    | Lift DAG-layout into a reusable module | M | P1 |
| 16r   | Lift expected_behavior DSL + evaluator + fixtures to the dashboard package | M | P1 |
| 4     | `.pbg/schemas/study.schema.json` + investigation.schema.json | M | P1 |
| 15    | `/pbg-study fill-overview <slug>` | M | P1 |
| 1     | `/pbg-status` | S | P2 |
| 3     | `/pbg-study scaffold-from-plan <plan.pdf>` | L | P2 |
| 5     | `/pbg-data add-expert <pdf>` | S | P2 |
| 6     | Document study-subdir layout | XS | P3 |
| 9     | `/pbg-study dag` (now subsumed by the in-dashboard DAG view, but still useful for CLI use) | S | P3 |

---

## Concrete deliverable shortlist for the other Claude session (updated)

Picking up where the prior list left off; #2 has already landed; #8 is
the cheapest remaining win.

| # | Improvement | Estimated effort | Priority |
|---|---|---|---|
| 8 | `lint-workspace.py` enumerates findings + runs the dashboard's own validator path (#11) | XS | P0 (cheapest UX win) |
| 13 | Filter Registry tab discovered-processes by workspace imports | M | P0 (user-visible bug) |
| 14 | Render `status` + Q + H in study pop-out | M | P0 (user just asked) |
| 4 | `.pbg/schemas/study.schema.json` + observable extensions (#12) | M | P1 |
| 15 | `/pbg-study fill-overview <slug>` | M | P1 (highest agent leverage) |
| 16 | Canonical `behavioral_tests:` block + shared test fixtures | S | P1 |
| 1  | `/pbg-status` | S | P2 |
| 3  | `/pbg-study scaffold-from-plan <plan.pdf>` | L | P2 |
| 5  | `/pbg-data add-expert <pdf>` | S | P2 |
| 6  | Document study-subdir layout (lazy vs pre-created) | XS | P3 |
| 9  | `/pbg-study dag` | S | P3 |
