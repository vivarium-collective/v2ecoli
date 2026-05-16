# pbg-superpowers test-run notes — DnaA studies scaffolding (2026-05-16)

Context: scaffolded a v2ecoli pbg workspace (workspace.yaml + references/expert/
+ 6 dnaa-* study.yaml files) on a branch off main. PR #50 had just merged the
generic scaffolding kit; the new `/pbg-workspace --upstream` upstream-branch
mode is in the installed skill.

What follows are friction points and proposed improvements, in priority order.

---

## 1. Discovery: "where is the workspace?" was the slowest step

I burned ~6 tool calls answering "is v2ecoli already a pbg workspace, or do I
need a sibling workspace dir?" before landing on the answer (`workspace.yaml`
is the marker; the cwd v2ecoli didn't have one until PR #50). The signal was
scattered across:

- `.pbg/server/server-info` in v2ecoli (stale — suggested a server had run)
- `docs/workspace-template/investigations/` (templates, not a real workspace)
- `~/code/v2ecoli-workspace/` (a separate pbg-template-cloned workspace)
- `~/code/viva-munk/` (a v3 study.yaml example)
- pbg-template's `template/` (the canonical source-of-truth)

**Proposal:** `/pbg-status` or `/pbg-workspace status` that, given a cwd:

1. Walks up to find `workspace.yaml`. If found, prints name, package_path,
   imports, and a one-line "this IS a pbg workspace" verdict.
2. If not found, looks for sibling pbg workspaces in `~/.pbg/workspaces.json`
   and suggests them.
3. Prints whether a dashboard server is alive (and where).

Right now the user has to invoke `pbg-workspace` (bootstrap-mode skill), read
its prose, and infer the answer. A dedicated status skill is much faster for
both humans and agents.

---

## 2. Study scaffolding from a multi-phase plan PDF should be one command

The user handed me a multi-phase plan PDF and an expert-knowledge PDF. The
**obvious** flow is "decompose plan into studies, register expert doc". I did
it by hand — 6 study.yaml writes + workspace.yaml expert_docs entries + papers.bib
+ claims.yaml. ~600 lines of YAML.

**Proposal:** `/pbg-study scaffold-from-plan <plan.pdf> [--expert <expert.pdf>]`:

1. Read the plan PDF; ask Claude to identify each study/phase as a section.
2. For each, emit a `studies/<slug>/study.yaml` draft with status: draft,
   objective + description from the section, observables + variants + expected
   behavior + expert_questions extracted into structured fields.
3. If `--expert` given, copy under `references/expert/` and register in
   `workspace.yaml.expert_docs` with sha256 + description.
4. Pull bib keys mentioned in the plan/expert into a stub `papers.bib`.

This is exactly the test-run task I just performed. It should be a skill.

---

## 3. The v3 study.yaml shape is implicit — schema would help

`workspace.yaml` has a JSON Schema (`.pbg/schemas/workspace.schema.json`).
`study.yaml` does not. I had to infer the shape from:

- `viva-munk/studies/bending-pressure-260515/study.yaml` (full v3 example —
  `baseline` is a dict, not a list)
- v2ecoli's `reports/{workflow,multigeneration,colony,compare}_report.py`
  `_load_study` validators (`baseline` is a LIST of architectures)

The list-vs-dict mismatch will bite. I picked the list shape (matches v2ecoli
reports) but viva-munk uses dict. **Proposal:** ship
`.pbg/schemas/study.schema.json`, lint study.yaml in `lint-workspace.py`,
and document the dict/list rule (it's "one composite → dict; many → list" or
similar — pick one).

I also added "extension" fields (`implementation_tasks`, `expected_behavior`,
`expert_questions`, `references.{expert,bib_keys,claims}`) that aren't read by
any report or by `/pbg-study`. They're documentation. The schema should
either accept them or formalize them — right now they're floating in YAML
limbo.

---

## 4. /pbg-workspace upstream-branch mode arrived just in time — needs a "from existing checkout" mode

Today's exact situation: I was already in `~/code/v2ecoli` (on main, with some
work-in-progress and a stale `.pbg/server/`) when the user asked to scaffold.
The new `/pbg-workspace ... --upstream vivarium-collective/v2ecoli` clones
into a **new** directory. It doesn't help "scaffold this existing checkout
in-place".

**Proposal:** `/pbg-workspace bootstrap-in-place [--branch <name>]`:

1. Refuse if a workspace.yaml already exists.
2. Create the branch off the current HEAD (no clone).
3. Apply the same scaffolding files as upstream-branch mode (pyproject
   imports, references/, experiments/, scripts/, .pbg/schemas/, NEXT_STEPS.md,
   workspace.yaml).
4. Commit on the new branch.

Otherwise the agent has to either (a) abandon the working checkout and start
fresh, or (b) hand-roll the scaffolding (what I just did). Option (b) means
duplicating template content the agent has to maintain in sync with
pbg-template by hand. PR #50 is great, but it only addresses (a).

---

## 5. expert_docs PDF ingestion — auto-extract metadata

I had to:

- shasum each PDF
- write `path:` + `sha256:` + `description:` by hand in workspace.yaml
- duplicate the description in `expert_docs[i].description` and in the
  per-study `description:`/`references.expert:` fields

**Proposal:** `/pbg-data add-expert <pdf>` or dashboard "Workspace inputs >
Expert knowledge > Add PDF" that:

1. Copies the PDF under `references/expert/` (deduped by sha256).
2. Reads the first page (or asks Claude for a 2-line description).
3. Appends to `workspace.yaml.expert_docs`.
4. Optionally suggests `claims_supported:` IDs based on a Claude pass.

The current `scripts/add-reference.sh` handles BibTeX-only; there's no
equivalent for expert PDFs. The skill description says PDFs auto-extract
metadata — the implementation gap is what I just filled by hand.

---

## 6. studies/<name>/composites/ + runs.db + viz/ — pre-create or lazy?

viva-munk's study dirs have `composites/`, `runs.db`, `viz/` alongside
study.yaml. v2ecoli-workspace's investigations have `runs.db` + `viz/`.
My dnaa-* studies have JUST `study.yaml` (no runs yet).

**Question:** should `/pbg-study new` pre-create those subdirs (with .keep)
or are they lazy? Document this explicitly. I made the call to NOT
pre-create (status: draft, no runs yet) but a future `/pbg-study run-baseline`
will need them to exist.

---

## 7. Implicit MEMORY: dashboard schema vs workspace schema

The dashboard's "5 tabs" (Workspace inputs / Registry / Simulation Setup /
Visualizations / Build Model) — none of the per-tab API endpoints are
discoverable from CLI alone. I gathered them from `/pbg-study`'s
sub-commands.

**Proposal:** `docs/concepts/vivarium-dashboard-model.md` should be linked
from `pbg-workspace`'s SKILL.md "Next steps" section (it's mentioned in
pbg-study, but not pbg-workspace — which is the first skill a new user
hits).

---

## 8. Lint-workspace.py is silent on study.yaml

`python scripts/lint-workspace.py` says `workspace lint: OK` even with 6
study.yaml files that have unknown extension fields. Linter should at least
ENUMERATE the studies it found.

**Proposal:** lint output:

```
workspace lint: OK
  - 2 expert_docs, 17 bib keys, 16 claims
  - 6 studies: dnaa-01-expression-dynamics, dnaa-02-atp-hydrolysis, ... (all status: draft)
  - 0 active runs, 0 completed runs
```

Gives the agent (and the user) instant confidence the scaffolding registered.

---

## 9. Cross-study links: `parent_studies:` is good, no tooling around it

I used `parent_studies:` to express the dependency chain (dnaa-02 depends on
dnaa-01, dnaa-04 on dnaa-03, dnaa-05 on dnaa-02 + dnaa-04, dnaa-06 on
dnaa-04). Nothing consumes this. **Proposal:** the dashboard's Studies tab
should render the DAG. `/pbg-study dag` should print the DAG as ASCII or
mermaid.

---

## Summary of recommended new skills / sub-commands

| Skill / command | Purpose |
|---|---|
| `/pbg-status` | Detect cwd's workspace state in one shot |
| `/pbg-workspace bootstrap-in-place` | Scaffold an existing checkout (no clone) |
| `/pbg-study scaffold-from-plan <plan.pdf>` | Decompose a plan PDF into N study.yamls |
| `/pbg-data add-expert <pdf>` | Auto-register expert PDFs with metadata |
| `/pbg-study dag` | Print parent_studies DAG |
| `study.schema.json` | Formalize the v3 shape |

And one ergonomic fix: `scripts/lint-workspace.py` should enumerate what it
found instead of just printing OK.
