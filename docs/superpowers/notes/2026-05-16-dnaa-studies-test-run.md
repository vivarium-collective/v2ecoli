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
