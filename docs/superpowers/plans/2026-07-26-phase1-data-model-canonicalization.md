# Phase 1 — Data-Model Canonicalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the v2ecoli workspace one drift-free study/investigation data model — single canonical study layout, one composite/config schema, `inputs.from` as the sole ordering source, investigations keyed by `members:` — enforced by a resolver that can no longer shadow, and an interim lint guard.

**Architecture:** Build comment-preserving migrators + a resolver fix in `pbg-superpowers` (import pkg `viva_superpowers`), tested on fixtures (Group A). Then editable-install that branch into the v2ecoli venv and run the migrators over the real workspace, delete the nested duplicates, fix the DAG breakage the migrators flag, and add an interim conformance guard (Group B). Three PRs: `pbg-superpowers` (tools + resolver), `vivarium-workbench` (mirror the vendored resolver copy), `v2ecoli` (migrated data + guard).

**Tech Stack:** Python 3, `ruamel.yaml` (comment-preserving round-trip), `argparse` CLIs, `pytest`, `uv`, `git worktree`.

## Global Constraints

- **`viva_superpowers` source repo:** `/Users/eranagmon/code/pbg-superpowers` (remote `viva-superpowers.git`). v2ecoli pins `pbg-superpowers = { git = ..., branch = "main" }`, so **all tool changes must target `main`** (NOT `rebrand/dist-name-viva`). Work on a worktree off `origin/main`.
- **The resolver is a VENDORED COPY in two repos:** `viva_superpowers/workspace_paths.py` AND `vivarium-workbench/vivarium_workbench/lib/workspace_paths.py`. A drift guard (`tests/test_workspace_paths.py`) fails if `LAYOUT_DEFAULTS` diverges. **Edit both copies identically.**
- **Comment preservation is mandatory** for any `study.yaml`/`investigation.yaml` edit. Use `ruamel.yaml` round-trip + `viva_superpowers.study_io.atomic_write`. **Never** `yaml.safe_dump`/`save_yaml_atomic` on a study/investigation file.
- **ruamel idiom** (match existing code — there is no shared load module):
  ```python
  from ruamel.yaml import YAML
  from io import StringIO
  from viva_superpowers import study_io
  def _ruamel():
      y = YAML(); y.preserve_quotes = True; y.width = 4096
      y.indent(mapping=2, sequence=4, offset=2)  # match study.yaml block-seq convention
      return y
  def _load_rt(path):
      y = _ruamel(); data = y.load(path.read_text(encoding="utf-8"))
      return y, (data if data is not None else {})
  def _dump_rt(y, path, data):
      buf = StringIO(); y.dump(data, buf); study_io.atomic_write(path, buf.getvalue())
  ```
- **Migrator convention:** a pure transform `f(spec[, known_slugs]) -> report` that mutates a ruamel `CommentedMap` in place; a file wrapper `migrate_study_file(study_dir, ..., write=False)` where `write=False` is a byte-preserving dry-run; a thin `main(argv=None) -> int` argparse CLI resolving paths via `WorkspacePaths.load(root)`. Register CLIs under BOTH `viva-<name>` and `pbg-<name>` → `viva_superpowers.<module>:main`.
- **Per-model config key stays `params`** (the Model panel reads `b.params`). Canonical model form is the **`conditions:` block** (`conditions.baseline` single + `conditions.variants[]`). `model_settings` stays under `conditions.model_settings`.
- **Worktree discipline (from the user's CLAUDE.md):** one dedicated worktree per repo; never commit in a shared canonical checkout; verify `git branch --show-current` + `git rev-parse --short HEAD` before every commit.
- **Test env for `pbg-superpowers`:** the repo has a `.venv`; run `\.venv/bin/python -m pytest tests/test_<x>.py -v`. If the worktree lacks a synced venv, `uv sync` in the worktree (it is self-contained; no sibling-repo path dependency like the workbench has).

---

## Group A — Tooling (`pbg-superpowers` off `main`, + workbench resolver copy)

### Task A0: Set up worktrees

**Files:** none (environment only)

- [ ] **Step 1: Create the pbg-superpowers worktree off main**

```bash
git -C ~/code/pbg-superpowers fetch origin main
git -C ~/code/pbg-superpowers worktree add ~/code/pbg-superpowers--repro-audit -b feat/study-canonicalization origin/main
cd ~/code/pbg-superpowers--repro-audit && git branch --show-current && git rev-parse --short HEAD
```

- [ ] **Step 2: Sync its venv**

```bash
cd ~/code/pbg-superpowers--repro-audit && uv sync 2>&1 | tail -3
.venv/bin/python -c "import viva_superpowers, ruamel.yaml; print('ok')"
```

- [ ] **Step 3: Create the workbench worktree off main (for the mirrored resolver copy)**

```bash
git -C ~/code/vivarium-workbench fetch origin main
git -C ~/code/vivarium-workbench worktree add ~/code/vivarium-workbench--repro-audit -b feat/resolver-toplevel-only origin/main
cd ~/code/vivarium-workbench--repro-audit && git branch --show-current && git rev-parse --short HEAD
```

---

### Task A1: Resolver — top-level-only `iter_study_dirs()` (both copies)

**Files:**
- Modify: `~/code/pbg-superpowers--repro-audit/viva_superpowers/workspace_paths.py` (`iter_study_dirs`, ~lines 149-168)
- Modify: `~/code/vivarium-workbench--repro-audit/vivarium_workbench/lib/workspace_paths.py` (identical method)
- Test: `~/code/pbg-superpowers--repro-audit/tests/test_workspace_paths.py`

**Interfaces:**
- Produces: `WorkspacePaths.iter_study_dirs()` now yields ONLY `studies/<slug>/` dirs containing `study.yaml`; never scans `investigations/*/studies/`.

- [ ] **Step 1: Write the failing regression test** in `tests/test_workspace_paths.py`

```python
def test_iter_study_dirs_ignores_nested(tmp_path):
    from viva_superpowers.workspace_paths import WorkspacePaths
    (tmp_path / "workspace.yaml").write_text("name: t\n")
    # top-level canonical study
    top = tmp_path / "studies" / "alpha"; top.mkdir(parents=True)
    (top / "study.yaml").write_text("name: alpha\n")
    # a study that exists ONLY nested (must NOT be yielded)
    nested_only = tmp_path / "investigations" / "inv" / "studies" / "ghost"
    nested_only.mkdir(parents=True); (nested_only / "study.yaml").write_text("name: ghost\n")
    # a nested duplicate of a top-level slug (must NOT shadow)
    nested_dup = tmp_path / "investigations" / "inv" / "studies" / "alpha"
    nested_dup.mkdir(parents=True); (nested_dup / "study.yaml").write_text("name: alpha-STALE\n")

    wp = WorkspacePaths.load(tmp_path)
    slugs = sorted(p.name for p in wp.iter_study_dirs())
    yielded = [str(p) for p in wp.iter_study_dirs()]
    assert slugs == ["alpha"]                         # ghost not yielded, alpha once
    assert all("investigations" not in p for p in yielded)   # never a nested path
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd ~/code/pbg-superpowers--repro-audit && .venv/bin/python -m pytest tests/test_workspace_paths.py::test_iter_study_dirs_ignores_nested -v`
Expected: FAIL (current impl yields `ghost` and the nested `alpha`).

- [ ] **Step 3: Replace `iter_study_dirs` in the pbg-superpowers copy**

```python
    def iter_study_dirs(self):
        """Yield every study dir from the canonical top-level ``studies/`` layout.

        Studies live ONLY at ``studies/<slug>/study.yaml``. Nested
        ``investigations/<inv>/studies/`` are intentionally NOT scanned: a single
        canonical location prevents an older nested copy from shadowing the
        registry copy (study-reproducibility contract, L0)."""
        flat = self.dir("studies")
        if not flat.is_dir():
            return
        for s in sorted(p for p in flat.iterdir() if p.is_dir()):
            if (s / "study.yaml").is_file():
                yield s
```

- [ ] **Step 4: Run the test to verify it passes, and the drift/existing tests still pass**

Run: `.venv/bin/python -m pytest tests/test_workspace_paths.py -v`
Expected: PASS (new test + existing, including the `LAYOUT_DEFAULTS` drift guard).

- [ ] **Step 5: Apply the identical replacement to the workbench copy**

Edit `~/code/vivarium-workbench--repro-audit/vivarium_workbench/lib/workspace_paths.py` — replace its `iter_study_dirs` with the exact same body as Step 3. Then run the workbench's own resolver test if present:
Run: `cd ~/code/vivarium-workbench--repro-audit && (test -d .venv && .venv/bin/python -m pytest tests/test_workspace_paths.py -v || echo "no local venv; rely on CI")`

- [ ] **Step 6: Commit each repo on its own worktree**

```bash
cd ~/code/pbg-superpowers--repro-audit && git add viva_superpowers/workspace_paths.py tests/test_workspace_paths.py
git commit -m "fix(resolver): iter_study_dirs yields top-level studies only (kill nested shadow)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
cd ~/code/vivarium-workbench--repro-audit && git add vivarium_workbench/lib/workspace_paths.py
git commit -m "fix(resolver): iter_study_dirs yields top-level studies only (mirror viva_superpowers)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task A2: `canonicalize_models` — pure model-schema transform

**Files:**
- Create: `~/code/pbg-superpowers--repro-audit/viva_superpowers/study_canonicalize.py`
- Test: `~/code/pbg-superpowers--repro-audit/tests/test_study_canonicalize.py`

**Interfaces:**
- Produces: `canonicalize_models(spec) -> dict` — mutates a ruamel `CommentedMap` (or plain dict) IN PLACE into the canonical `conditions.{baseline,variants,model_settings}` form; returns a report `{"changed": bool, "style": "canonical"|"B"|"both", "flags": list[str], "inherited_composites": list[str]}`.
  - Canonical target: `conditions.baseline` is a single mapping `{name, composite, params}`; `conditions.variants` is a list of `{name, composite, params, ...preserved}`; `conditions.model_settings` defaults to `[]`.
  - Every variant gets an explicit `composite` (inherit `conditions.baseline.composite` when absent); variant `parameter_overrides` is renamed to `params` (value kept).
  - Style B (top-level `baseline:` list, no/empty `conditions`): move the single baseline entry into `conditions.baseline`; move any top-level `variants:` into `conditions.variants`; delete the top-level `baseline`/`variants` keys.
  - "both" (top-level `baseline:` AND `conditions:`): if `conditions.baseline` already has a `composite`, keep it and just delete the redundant top-level `baseline`/`variants` (flag `both_dropped_toplevel`); else move top-level into `conditions.baseline`.
  - A top-level `baseline:` list with >1 entry is NOT auto-migrated — flag `multi_baseline_needs_human` and leave the study unchanged.

- [ ] **Step 1: Write failing tests**

```python
from viva_superpowers.study_canonicalize import canonicalize_models

def test_style_A_already_canonical_inherits_variant_composite():
    spec = {"conditions": {
        "baseline": {"composite": "c.base", "params": {"seed": 0}},
        "variants": [{"name": "ko", "params": {"knockouts": ["EG10526"]}}],
    }}
    report = canonicalize_models(spec)
    assert spec["conditions"]["variants"][0]["composite"] == "c.base"  # inherited
    assert "conditions" in spec and "baseline" not in spec  # no top-level baseline
    assert report["changed"] is True and "ko" in report["inherited_composites"]

def test_style_B_toplevel_baseline_list_moves_into_conditions():
    spec = {"baseline": [{"name": "m", "composite": "c.x", "params": {"seed": 0}}]}
    report = canonicalize_models(spec)
    assert "baseline" not in spec  # top-level removed
    assert spec["conditions"]["baseline"]["composite"] == "c.x"
    assert report["style"] == "B" and report["changed"] is True

def test_both_keeps_conditions_drops_toplevel():
    spec = {"baseline": [{"name": "d", "composite": "c.dupe", "params": {}}],
            "conditions": {"baseline": {"composite": "c.real", "params": {"seed": 1}}}}
    report = canonicalize_models(spec)
    assert "baseline" not in spec
    assert spec["conditions"]["baseline"]["composite"] == "c.real"  # conditions wins
    assert "both_dropped_toplevel" in report["flags"]

def test_multi_baseline_is_flagged_not_migrated():
    spec = {"baseline": [{"name": "a", "composite": "c.a"}, {"name": "b", "composite": "c.b"}]}
    report = canonicalize_models(spec)
    assert "multi_baseline_needs_human" in report["flags"]
    assert "baseline" in spec and report["changed"] is False  # untouched

def test_variant_parameter_overrides_renamed_to_params():
    spec = {"conditions": {"baseline": {"composite": "c.b"},
            "variants": [{"name": "v", "parameter_overrides": {"media": "rich"}}]}}
    canonicalize_models(spec)
    v = spec["conditions"]["variants"][0]
    assert v["params"] == {"media": "rich"} and "parameter_overrides" not in v

def test_idempotent():
    spec = {"conditions": {"baseline": {"composite": "c.b", "params": {}},
            "variants": [{"name": "v", "composite": "c.b", "params": {}}],
            "model_settings": []}}
    r1 = canonicalize_models(spec); r2 = canonicalize_models(spec)
    assert r2["changed"] is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_study_canonicalize.py -v`
Expected: FAIL (`ModuleNotFoundError: study_canonicalize`).

- [ ] **Step 3: Implement `study_canonicalize.canonicalize_models`**

```python
"""Canonicalize a study's model declaration into the conditions.{baseline,variants}
form. Pure, in-place, comment-preserving-safe (moves value nodes, never reserializes)."""
from __future__ import annotations


def _as_single_baseline(entry):
    """Return a {name?, composite, params} mapping from a top-level baseline list entry."""
    return entry


def canonicalize_models(spec) -> dict:
    report = {"changed": False, "style": "canonical", "flags": [], "inherited_composites": []}
    top_baseline = spec.get("baseline")
    conditions = spec.get("conditions")

    # --- classify + move top-level baseline/variants into conditions (Style B / both) ---
    if isinstance(top_baseline, list):
        if len(top_baseline) > 1:
            report["flags"].append("multi_baseline_needs_human")
            return report  # leave untouched
        report["style"] = "both" if isinstance(conditions, dict) and conditions.get("baseline") else "B"
        if conditions is None or not isinstance(conditions, dict):
            spec["conditions"] = {}
            conditions = spec["conditions"]
        if conditions.get("baseline") and conditions["baseline"].get("composite"):
            report["flags"].append("both_dropped_toplevel")   # conditions wins
        elif top_baseline:
            conditions["baseline"] = top_baseline[0]           # move node (keeps its comments)
        # move a top-level variants list in, if present and conditions lacks one
        top_variants = spec.get("variants")
        if isinstance(top_variants, list) and not conditions.get("variants"):
            conditions["variants"] = top_variants
        for k in ("baseline", "variants"):
            if k in spec:
                del spec[k]
        report["changed"] = True

    conditions = spec.get("conditions")
    if not isinstance(conditions, dict) or not conditions.get("baseline"):
        return report  # nothing canonical to normalize (e.g. parca / non-model study)

    base_composite = conditions["baseline"].get("composite")

    # --- normalize variants: inherit composite, rename parameter_overrides -> params ---
    for v in (conditions.get("variants") or []):
        if not isinstance(v, dict):
            continue
        if "parameter_overrides" in v and "params" not in v:
            v["params"] = v.pop("parameter_overrides"); report["changed"] = True
        if not v.get("composite") and base_composite:
            v["composite"] = base_composite
            report["inherited_composites"].append(v.get("name", "?")); report["changed"] = True

    # --- ensure model_settings key exists (kept separate) ---
    if "model_settings" not in conditions:
        conditions["model_settings"] = []; report["changed"] = True

    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_study_canonicalize.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/code/pbg-superpowers--repro-audit
git add viva_superpowers/study_canonicalize.py tests/test_study_canonicalize.py
git commit -m "feat(canonicalize): pure canonicalize_models transform (Style A/B/both -> conditions)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task A3: `canonicalize_ordering` — prerequisites → `inputs.from`

**Files:**
- Read first: `~/code/pbg-superpowers--repro-audit/viva_superpowers/migrate_inputs.py` (an existing inputs migrator — reuse/extend if it already converts prereqs; otherwise add the pass here).
- Modify: `~/code/pbg-superpowers--repro-audit/viva_superpowers/study_canonicalize.py`
- Test: `~/code/pbg-superpowers--repro-audit/tests/test_study_canonicalize.py`

**Interfaces:**
- Consumes: `known_slugs: set[str]` — every top-level study slug in the workspace.
- Produces: `canonicalize_ordering(spec, known_slugs) -> dict` — mutates the spec IN PLACE. Report `{"changed": bool, "added_inputs": list[str], "outputs_declared": bool, "flags": list[str]}`.
  - **Convention:** a study's default output artifact is named after its own slug. A prerequisite producer `P` (valid, in `known_slugs`) becomes a consumer edge `inputs += [{"artifact": P, "from": P}]` (dedup on `(artifact, from)`), and the consumer declares no producer output here (the producer declares `outputs: [{artifact: P}]` when *it* is processed — done workspace-wide in Group B, Task B2 Step 4, or via `--declare-outputs`).
  - Prereqs read from top-level `pipeline_gate.prerequisites` (list of slug strings OR `{study: slug}` maps) and top-level `parent_studies`. Nested `pipeline_gate` blocks deeper in the file are NOT touched.
  - A prereq slug not in `known_slugs` → flag `dangling_prereq:<slug>`, NO edge added, and the prereq is NOT deleted (leave for a human fix in Task B4).
  - After all prereqs are either converted or flag-preserved, delete top-level `pipeline_gate.prerequisites`, `pipeline_gate.enables`, and `parent_studies` **only if there are no dangling flags** (so ordering info is never silently lost). If dangling flags exist, leave `pipeline_gate` in place and flag `pipeline_gate_retained`.

- [ ] **Step 1: Write failing tests**

```python
from viva_superpowers.study_canonicalize import canonicalize_ordering

def test_valid_prereq_becomes_inputs_from():
    spec = {"name": "b", "pipeline_gate": {"prerequisites": ["a"], "enables": ["c"]}}
    report = canonicalize_ordering(spec, known_slugs={"a", "b", "c"})
    assert {"artifact": "a", "from": "a"} in spec["inputs"]
    assert "pipeline_gate" not in spec           # deleted (no dangling)
    assert report["changed"] is True and "a" in report["added_inputs"]

def test_mapping_prereq_shape():
    spec = {"name": "b", "pipeline_gate": {"prerequisites": [{"study": "a"}]}}
    canonicalize_ordering(spec, known_slugs={"a", "b"})
    assert {"artifact": "a", "from": "a"} in spec["inputs"]

def test_dangling_prereq_is_flagged_and_retained():
    spec = {"name": "c1", "pipeline_gate": {"prerequisites": [], "enables": ["c2-does-not-exist"]}}
    # enables is informational; dangling is detected on prerequisites referencing unknown slugs:
    spec2 = {"name": "x", "pipeline_gate": {"prerequisites": ["ghost"]}}
    report = canonicalize_ordering(spec2, known_slugs={"x"})
    assert "dangling_prereq:ghost" in report["flags"]
    assert "pipeline_gate" in spec2  # retained for human fix
    assert "pipeline_gate_retained" in report["flags"]

def test_existing_inputs_not_duplicated():
    spec = {"name": "b", "inputs": [{"artifact": "a", "from": "a"}],
            "pipeline_gate": {"prerequisites": ["a"]}}
    canonicalize_ordering(spec, known_slugs={"a", "b"})
    assert spec["inputs"].count({"artifact": "a", "from": "a"}) == 1

def test_parca_dependency_preserved():
    spec = {"name": "s", "inputs": [{"artifact": "sim_data", "from": "parca"}],
            "pipeline_gate": {"prerequisites": ["parca"]}}
    canonicalize_ordering(spec, known_slugs={"parca", "s"})
    # parca edge already present as sim_data; adding {parca,parca} is allowed but dedup keeps sim_data
    assert {"artifact": "sim_data", "from": "parca"} in spec["inputs"]
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_study_canonicalize.py -k ordering -v`
Expected: FAIL.

- [ ] **Step 3: Implement `canonicalize_ordering`** (append to `study_canonicalize.py`)

```python
def _prereq_slugs(pg) -> list[str]:
    out = []
    for p in (pg.get("prerequisites") or []):
        if isinstance(p, str):
            out.append(p)
        elif isinstance(p, dict) and p.get("study"):
            out.append(str(p["study"]))
    return out


def canonicalize_ordering(spec, known_slugs) -> dict:
    report = {"changed": False, "added_inputs": [], "outputs_declared": False, "flags": []}
    pg = spec.get("pipeline_gate")
    parents = spec.get("parent_studies") or []
    prereqs = (_prereq_slugs(pg) if isinstance(pg, dict) else []) + [str(p) for p in parents]
    if not prereqs:
        return report

    inputs = spec.get("inputs")
    if not isinstance(inputs, list):
        inputs = []; spec["inputs"] = inputs
    existing = {(e.get("artifact"), e.get("from")) for e in inputs if isinstance(e, dict)}

    dangling = False
    for producer in prereqs:
        if producer not in known_slugs:
            report["flags"].append(f"dangling_prereq:{producer}"); dangling = True
            continue
        # skip if any edge already consumes this producer (e.g. parca -> sim_data)
        if any(frm == producer for (_art, frm) in existing):
            continue
        edge = {"artifact": producer, "from": producer}
        inputs.append(edge); existing.add((producer, producer))
        report["added_inputs"].append(producer); report["changed"] = True

    if dangling:
        report["flags"].append("pipeline_gate_retained")
    else:
        for k in ("pipeline_gate",):
            if k in spec:
                del spec[k]; report["changed"] = True
        if "parent_studies" in spec:
            del spec["parent_studies"]; report["changed"] = True
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_study_canonicalize.py -k ordering -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add viva_superpowers/study_canonicalize.py tests/test_study_canonicalize.py
git commit -m "feat(canonicalize): canonicalize_ordering (pipeline_gate/parent_studies -> inputs.from)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task A4: File migrator + CLI + golden comment test

**Files:**
- Modify: `~/code/pbg-superpowers--repro-audit/viva_superpowers/study_canonicalize.py` (add `migrate_study_file`, `main`)
- Modify: `~/code/pbg-superpowers--repro-audit/pyproject.toml` (`[project.scripts]`)
- Test: `~/code/pbg-superpowers--repro-audit/tests/test_study_canonicalize.py`

**Interfaces:**
- Produces: `migrate_study_file(study_dir, known_slugs, write=False) -> dict` — ruamel round-trip load of `<study_dir>/study.yaml`, run `canonicalize_models` then `canonicalize_ordering`, and (only if `write and changed`) write back via `_dump_rt`. `write=False` is a byte-identical dry-run. Returns `{"models": <report>, "ordering": <report>, "written": bool}`.
- Produces: `main(argv=None) -> int` — CLI `viva-canonicalize-studies` with `--workspace`, `--study <slug>` (optional; default = all via `WorkspacePaths.iter_study_dirs`), `--write` (default dry-run). Builds `known_slugs` from `iter_study_dirs`.

- [ ] **Step 1: Write failing golden + dry-run tests**

```python
from viva_superpowers.study_canonicalize import migrate_study_file

STYLE_B_WITH_COMMENTS = """\
# Hand-authored research log — MUST survive migration.
schema_version: 4
name: colonies-x
title: Device run
baseline:
- name: mother-machine-simple
  composite: v2ecoli.composites.ecoli_colony.ecoli_colony
  params:
    seed: 0            # canonical seed
  note: |
    Nominal composite for the workbench catalog.
pipeline_gate:
  prerequisites:
  - colonies-prev
# trailing comment
status: ran
"""

def test_migrate_writes_and_preserves_comments(tmp_path):
    d = tmp_path / "colonies-x"; d.mkdir()
    (d / "study.yaml").write_text(STYLE_B_WITH_COMMENTS)
    report = migrate_study_file(d, known_slugs={"colonies-x", "colonies-prev"}, write=True)
    assert report["written"] is True
    text = (d / "study.yaml").read_text()
    assert "MUST survive migration" in text          # header comment survives
    assert "trailing comment" in text
    assert "Nominal composite for the workbench" in text  # note prose survives
    assert "conditions:" in text and "\nbaseline:" not in text  # moved into conditions
    assert "from: colonies-prev" in text             # ordering -> inputs.from
    assert "pipeline_gate:" not in text              # removed (no dangling)

def test_dry_run_does_not_write(tmp_path):
    d = tmp_path / "colonies-x"; d.mkdir()
    (d / "study.yaml").write_text(STYLE_B_WITH_COMMENTS)
    before = (d / "study.yaml").read_text()
    migrate_study_file(d, known_slugs={"colonies-x", "colonies-prev"}, write=False)
    assert (d / "study.yaml").read_text() == before   # byte-identical

def test_already_canonical_is_noop(tmp_path):
    d = tmp_path / "s"; d.mkdir()
    (d / "study.yaml").write_text(
        "schema_version: 4\nname: s\nconditions:\n  baseline:\n    composite: c.b\n"
        "    params: {}\n  model_settings: []\n")
    before = (d / "study.yaml").read_text()
    migrate_study_file(d, known_slugs={"s"}, write=True)
    assert (d / "study.yaml").read_text() == before   # byte-identical no-op
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_study_canonicalize.py -k "migrate or dry_run or noop" -v`
Expected: FAIL.

- [ ] **Step 3: Implement `migrate_study_file` + `main`** (append to `study_canonicalize.py`)

```python
import argparse
from io import StringIO
from pathlib import Path
from ruamel.yaml import YAML
from viva_superpowers import study_io
from viva_superpowers.workspace_paths import WorkspacePaths


def _ruamel():
    y = YAML(); y.preserve_quotes = True; y.width = 4096
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def migrate_study_file(study_dir, known_slugs, write: bool = False) -> dict:
    study_dir = Path(study_dir)
    path = study_dir / "study.yaml"
    y = _ruamel()
    spec = y.load(path.read_text(encoding="utf-8"))
    if spec is None:
        return {"models": {}, "ordering": {}, "written": False}
    m = canonicalize_models(spec)
    o = canonicalize_ordering(spec, known_slugs)
    changed = bool(m.get("changed") or o.get("changed"))
    written = False
    if write and changed:
        buf = StringIO(); y.dump(spec, buf)
        study_io.atomic_write(path, buf.getvalue()); written = True
    return {"models": m, "ordering": o, "written": written}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="viva-canonicalize-studies")
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--study", default=None, help="slug; default = all")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    wp = WorkspacePaths.load(args.workspace)
    known = {p.name for p in wp.iter_study_dirs()}
    targets = ([wp.study_dir(args.study)] if args.study
               else list(wp.iter_study_dirs()))
    any_flag = False
    for d in targets:
        rep = migrate_study_file(d, known_slugs=known, write=args.write)
        flags = rep["models"].get("flags", []) + rep["ordering"].get("flags", [])
        mark = "WROTE" if rep["written"] else ("would-change" if (rep["models"].get("changed") or rep["ordering"].get("changed")) else "ok")
        print(f"[{mark}] {d.name}" + (f"  flags={flags}" if flags else ""))
        any_flag = any_flag or bool(flags)
    if any_flag:
        print("\nNOTE: flags present (multi_baseline / dangling_prereq) need a human decision.")
    return 0
```

- [ ] **Step 4: Register CLIs** — add to `pyproject.toml` `[project.scripts]`:

```toml
viva-canonicalize-studies = "viva_superpowers.study_canonicalize:main"
pbg-canonicalize-studies = "viva_superpowers.study_canonicalize:main"
```

- [ ] **Step 5: Run tests + confirm the console script resolves**

```bash
.venv/bin/python -m pytest tests/test_study_canonicalize.py -v
uv pip install -e . --no-deps -q && .venv/bin/viva-canonicalize-studies --help
```
Expected: all tests PASS; `--help` prints.

- [ ] **Step 6: Commit**

```bash
git add viva_superpowers/study_canonicalize.py tests/test_study_canonicalize.py pyproject.toml
git commit -m "feat(canonicalize): migrate_study_file + viva-canonicalize-studies CLI (write=False dry-run)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task A5: Investigation-key migrator (`studies:` → `members:`)

**Files:**
- Create: `~/code/pbg-superpowers--repro-audit/viva_superpowers/investigation_canonicalize.py`
- Modify: `~/code/pbg-superpowers--repro-audit/pyproject.toml`
- Test: `~/code/pbg-superpowers--repro-audit/tests/test_investigation_canonicalize.py`

**Interfaces:**
- Produces: `canonicalize_investigation(spec) -> dict` — if `spec` has `studies:` and no `members:`, rename the key to `members:` (preserve the list + comments); if both exist, keep `members:` and flag `both_keys_present`. Report `{"changed": bool, "flags": list[str]}`.
- Produces: `migrate_investigation_file(inv_dir, write=False) -> dict` (ruamel round-trip on `<inv_dir>/investigation.yaml`), `main(argv=None) -> int` CLI `viva-canonicalize-investigations` (`--workspace`, `--investigation`, `--write`).

- [ ] **Step 1: Write failing tests**

```python
from viva_superpowers.investigation_canonicalize import canonicalize_investigation, migrate_investigation_file

def test_studies_renamed_to_members():
    spec = {"name": "inv", "studies": ["a", "b"]}
    report = canonicalize_investigation(spec)
    assert spec["members"] == ["a", "b"] and "studies" not in spec
    assert report["changed"] is True

def test_members_already_present_is_noop():
    spec = {"name": "inv", "members": ["a"]}
    report = canonicalize_investigation(spec)
    assert report["changed"] is False and spec["members"] == ["a"]

def test_both_keys_flagged():
    spec = {"name": "inv", "members": ["a"], "studies": ["a", "b"]}
    report = canonicalize_investigation(spec)
    assert "both_keys_present" in report["flags"] and "studies" in spec

def test_file_dry_run_byte_identical(tmp_path):
    d = tmp_path / "inv"; d.mkdir()
    (d / "investigation.yaml").write_text("# keep\nname: inv\nstudies:\n- a\n- b\n")
    before = (d / "investigation.yaml").read_text()
    migrate_investigation_file(d, write=False)
    assert (d / "investigation.yaml").read_text() == before

def test_file_write_preserves_comments(tmp_path):
    d = tmp_path / "inv"; d.mkdir()
    (d / "investigation.yaml").write_text("# KEEPME\nname: inv\nstudies:\n- a\n- b\n")
    migrate_investigation_file(d, write=True)
    text = (d / "investigation.yaml").read_text()
    assert "KEEPME" in text and "members:" in text and "studies:" not in text
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_investigation_canonicalize.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `investigation_canonicalize.py`**

```python
"""Canonicalize investigation.yaml: the study-list key is `members:` (not `studies:`)."""
from __future__ import annotations
import argparse
from io import StringIO
from pathlib import Path
from ruamel.yaml import YAML
from viva_superpowers import study_io
from viva_superpowers.workspace_paths import WorkspacePaths


def canonicalize_investigation(spec) -> dict:
    report = {"changed": False, "flags": []}
    if "studies" not in spec:
        return report
    if "members" in spec:
        report["flags"].append("both_keys_present")
        return report
    # rename studies -> members, preserving order/comments by moving the node
    spec["members"] = spec.pop("studies")
    report["changed"] = True
    return report


def _ruamel():
    y = YAML(); y.preserve_quotes = True; y.width = 4096
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def migrate_investigation_file(inv_dir, write: bool = False) -> dict:
    path = Path(inv_dir) / "investigation.yaml"
    y = _ruamel()
    spec = y.load(path.read_text(encoding="utf-8"))
    if spec is None:
        return {"changed": False, "flags": [], "written": False}
    rep = canonicalize_investigation(spec)
    written = False
    if write and rep["changed"]:
        buf = StringIO(); y.dump(spec, buf)
        study_io.atomic_write(path, buf.getvalue()); written = True
    rep["written"] = written
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="viva-canonicalize-investigations")
    ap.add_argument("--workspace", default=".")
    ap.add_argument("--investigation", default=None)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)
    wp = WorkspacePaths.load(args.workspace)
    inv_root = wp.dir("investigations")
    targets = ([inv_root / args.investigation] if args.investigation
               else [p for p in sorted(inv_root.iterdir())
                     if (p / "investigation.yaml").is_file()])
    for d in targets:
        rep = migrate_investigation_file(d, write=args.write)
        mark = "WROTE" if rep["written"] else ("would-change" if rep["changed"] else "ok")
        print(f"[{mark}] {d.name}" + (f"  flags={rep['flags']}" if rep["flags"] else ""))
    return 0
```

- [ ] **Step 4: Register CLIs** in `pyproject.toml` `[project.scripts]`:

```toml
viva-canonicalize-investigations = "viva_superpowers.investigation_canonicalize:main"
pbg-canonicalize-investigations = "viva_superpowers.investigation_canonicalize:main"
```

- [ ] **Step 5: Run tests + confirm console script**

```bash
.venv/bin/python -m pytest tests/test_investigation_canonicalize.py -v
uv pip install -e . --no-deps -q && .venv/bin/viva-canonicalize-investigations --help
```
Expected: PASS; `--help` prints.

- [ ] **Step 6: Commit + push the pbg-superpowers branch and open its PR**

```bash
cd ~/code/pbg-superpowers--repro-audit
git add viva_superpowers/investigation_canonicalize.py tests/test_investigation_canonicalize.py pyproject.toml
git commit -m "feat(canonicalize): investigation studies->members migrator + CLI

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git log --oneline origin/main..HEAD   # verify ONLY our commits
git push -u origin feat/study-canonicalization
gh pr create --title "Study canonicalization migrators + top-level-only resolver" --body "Phase 1 tooling for the study reproducibility contract. See docs/superpowers/specs/2026-07-26-study-reproducibility-contract-design.md (v2ecoli)."
```

---

## Group B — Application to the v2ecoli workspace

**Precondition:** Group A merged to `pbg-superpowers` `main` (v2ecoli tracks `branch=main`). To run BEFORE merge, editable-install the tool branch into the v2ecoli venv (Task B0 Step 2). Work in the existing worktree **`~/code/v2ecoli--repro-audit`** (branch `spec/study-reproducibility-audit`, already off `origin/main`).

### Task B0: Prepare the v2ecoli worktree + tool install

**Files:** none (environment)

- [ ] **Step 1: Verify the worktree**

```bash
cd ~/code/v2ecoli--repro-audit && git branch --show-current && git rev-parse --short HEAD
```

- [ ] **Step 2: Editable-install the canonicalization tools into the v2ecoli venv**

```bash
/Users/eranagmon/code/v2ecoli/.venv/bin/python -m pip --version 2>/dev/null || echo "no pip; use uv"
uv pip install --python /Users/eranagmon/code/v2ecoli/.venv/bin/python -e ~/code/pbg-superpowers--repro-audit --no-deps
/Users/eranagmon/code/v2ecoli/.venv/bin/python -c "from viva_superpowers import study_canonicalize, investigation_canonicalize; print('tools ok')"
```

---

### Task B1: Salvage nested comments, then delete the 9 nested `study.yaml`

**Files:**
- Delete: the 9 nested `workspace/investigations/<inv>/studies/<slug>/study.yaml` (list below)
- Possibly modify: their top-level twins in `workspace/studies/<slug>/study.yaml` (only if the nested copy holds unique hand-authored prose/comments)

The 9 duplicated slugs: `colonies-04-device-phenotype-harness`, `colonies-05-mother-machine`, `colonies-06-daughter-machine`, `colonies-08-wcm-daughter-machine`, `colonies-09-wcm-mother-machine`, `ketchup-exchange-comparison`, `ko-and-media`, `pdmp-00-characterization`, `metabolism_redux`.

- [ ] **Step 1: For each pair, diff nested vs top-level and record unique nested content**

```bash
cd ~/code/v2e-main-serve/workspace   # read-only reference checkout is fine for diffing
for slug in colonies-04-device-phenotype-harness colonies-05-mother-machine \
  colonies-06-daughter-machine colonies-08-wcm-daughter-machine colonies-09-wcm-mother-machine \
  ketchup-exchange-comparison ko-and-media pdmp-00-characterization metabolism_redux; do
  nested=$(find investigations -path "*/studies/$slug/study.yaml")
  echo "===== $slug ====="; diff -u "$nested" "studies/$slug/study.yaml" | head -60
done
```

- [ ] **Step 2: Merge any unique hand-authored prose/comments** from a nested copy into its top-level twin in `~/code/v2ecoli--repro-audit/workspace/studies/<slug>/study.yaml`, using the ruamel round-trip (edit by hand or a one-off script; **never** `safe_dump`). If a nested copy has NO unique authored content beyond serialization differences, skip — the top-level copy already wins.

- [ ] **Step 3: Delete the 9 nested study.yaml (and now-empty study dirs)**

```bash
cd ~/code/v2ecoli--repro-audit
for slug in colonies-04-device-phenotype-harness colonies-05-mother-machine \
  colonies-06-daughter-machine colonies-08-wcm-daughter-machine colonies-09-wcm-mother-machine \
  ketchup-exchange-comparison ko-and-media pdmp-00-characterization metabolism_redux; do
  nested=$(find workspace/investigations -path "*/studies/$slug/study.yaml")
  [ -n "$nested" ] && git rm "$nested"
done
# remove any now-empty investigations/<inv>/studies/<slug>/ dirs left behind (keep runs.db sinks if present)
find workspace/investigations -type d -empty -path "*/studies/*" -delete
```

- [ ] **Step 4: Verify the resolver no longer sees nested + commit**

```bash
/Users/eranagmon/code/v2ecoli/.venv/bin/python -c "
from viva_superpowers.workspace_paths import WorkspacePaths
wp = WorkspacePaths.load('workspace')
paths=[str(p) for p in wp.iter_study_dirs()]
assert not any('investigations' in p for p in paths), 'nested still resolved!'
print(len(paths),'studies, none nested')"
git commit -m "chore(workspace): delete 9 nested study.yaml duplicates (single canonical layout)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B2: Canonicalize all studies (dry-run → review → write)

**Files:** the 48 `workspace/studies/<slug>/study.yaml` (in-place)

- [ ] **Step 1: Dry-run over the whole workspace and capture the report**

```bash
cd ~/code/v2ecoli--repro-audit
/Users/eranagmon/code/v2ecoli/.venv/bin/viva-canonicalize-studies --workspace workspace \
  2>&1 | tee /tmp/canon-studies-dryrun.txt
grep -E "flags=" /tmp/canon-studies-dryrun.txt || echo "no flags"
```

- [ ] **Step 2: Triage flags.** For every `multi_baseline_needs_human` or `dangling_prereq:*`, decide by hand (multi-baseline studies stay as-is for now; dangling prereqs are fixed in Task B4). Do NOT `--write` yet if unresolved dangling flags would strand `pipeline_gate` — that is expected and handled in B4.

- [ ] **Step 3: Write**

```bash
/Users/eranagmon/code/v2ecoli/.venv/bin/viva-canonicalize-studies --workspace workspace --write \
  2>&1 | tee /tmp/canon-studies-write.txt
```

- [ ] **Step 4: Spot-check a migrated study renders unchanged in the workbench.** Restart the running workbench (serving `~/code/v2e-main-serve`) is NOT this worktree — instead point a scratch serve at this worktree, or verify via the normalizer directly:

```bash
/Users/eranagmon/code/v2ecoli/.venv/bin/python -c "
import sys; sys.path.insert(0,'/Users/eranagmon/code/vivarium-workbench--serve')
from vivarium_workbench.lib.investigations import load_spec
import pathlib, yaml
p='workspace/studies/colonies-05-mother-machine/study.yaml'
spec=load_spec(yaml.safe_load(open(p)))
bl=spec['baseline']; assert isinstance(bl,list) and bl[0]['composite'], bl
print('baseline[] ok:', [b['name'] for b in bl])"
```

- [ ] **Step 5: Commit**

```bash
git add workspace/studies
git commit -m "migrate(studies): canonicalize model schema + inputs.from ordering (48 studies)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B3: Canonicalize investigations (`studies:` → `members:`)

**Files:** `workspace/investigations/{multiscale-bioprocess,parameter-uq,structural-ecoli,surrogate-modeling}/investigation.yaml`

- [ ] **Step 1: Dry-run**

```bash
/Users/eranagmon/code/v2ecoli/.venv/bin/viva-canonicalize-investigations --workspace workspace
```

- [ ] **Step 2: Write + verify no `studies:` key remains**

```bash
/Users/eranagmon/code/v2ecoli/.venv/bin/viva-canonicalize-investigations --workspace workspace --write
! grep -rl '^studies:' workspace/investigations/*/investigation.yaml && echo "all members: now"
```

- [ ] **Step 3: Commit**

```bash
git add workspace/investigations
git commit -m "migrate(investigations): studies -> members (4 investigations)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B4: Fix the flagged DAG breakage

**Files:** the specific `study.yaml` files the dry-run flagged.

Known breakage from the audit: `colonies-01-hpc-readiness` `enables: colonies-02-hpc-deployment` (slug does not exist; real is `colonies-02-parallel-multigen-perf`); `param-uq-00-screen` requires `param-uq-01-elongation` (numbering inversion). Any `dangling_prereq:*` from Task B2 Step 1.

- [ ] **Step 1: For each `dangling_prereq:<slug>`**, decide: (a) it's a typo → correct the slug in `pipeline_gate.prerequisites`, or (b) the dependency is spurious → remove that prereq. Edit the `study.yaml` (ruamel/by hand, preserve comments).

- [ ] **Step 2: Re-run canonicalize-studies `--write`** so the now-valid prereqs convert to `inputs.from` and `pipeline_gate` is removed:

```bash
/Users/eranagmon/code/v2ecoli/.venv/bin/viva-canonicalize-studies --workspace workspace --write \
  2>&1 | grep -E "flags=" || echo "no remaining flags"
```
Expected: no `dangling_prereq` / `pipeline_gate_retained` flags remain.

- [ ] **Step 3: Commit**

```bash
git add workspace/studies
git commit -m "fix(dag): correct dangling/inverted prerequisites; complete inputs.from migration

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task B5: Interim conformance guard (`lint-workspace.py` + pytest)

**Files:**
- Modify: `~/code/v2ecoli--repro-audit/scripts/lint-workspace.py`
- Create: `~/code/v2ecoli--repro-audit/tests/test_workspace_conformance.py`

**Interfaces:**
- Produces: guard checks (L0/L1-graph): (1) no `study.yaml` under `investigations/*/studies/`; (2) no investigation has a top-level `studies:` key (must be `members:`); (3) every study has `conditions.baseline.composite` (canonical model form) and no top-level `baseline:`/`pipeline_gate:`/`parent_studies:`; (4) the `inputs.from`-derived DAG is acyclic with every `from` referencing a real slug.

- [ ] **Step 1: Write the failing pytest guard** `tests/test_workspace_conformance.py`

```python
import pathlib, yaml, graphlib
WS = pathlib.Path(__file__).resolve().parents[1] / "workspace"

def _studies():
    return {p.parent.name: yaml.safe_load(p.read_text())
            for p in (WS / "studies").glob("*/study.yaml")}

def test_no_nested_studies():
    assert not list(WS.glob("investigations/*/studies/*/study.yaml"))

def test_investigations_use_members():
    for inv in (WS / "investigations").glob("*/investigation.yaml"):
        spec = yaml.safe_load(inv.read_text()) or {}
        assert "studies" not in spec, f"{inv.parent.name} still uses studies:"

def test_studies_canonical_model_form():
    for slug, spec in _studies().items():
        assert "baseline" not in spec, f"{slug} has top-level baseline:"
        assert "pipeline_gate" not in spec, f"{slug} retains pipeline_gate:"
        assert "parent_studies" not in spec, f"{slug} retains parent_studies:"
        cond = spec.get("conditions") or {}
        if slug != "parca":  # parca has no model
            assert (cond.get("baseline") or {}).get("composite"), f"{slug} missing conditions.baseline.composite"

def test_inputs_dag_acyclic_and_resolvable():
    studies = _studies(); slugs = set(studies) | {"parca"}
    ts = graphlib.TopologicalSorter()
    for slug, spec in studies.items():
        deps = []
        for e in (spec.get("inputs") or []):
            frm = e.get("from")
            assert frm in slugs, f"{slug} inputs.from '{frm}' is not a real study"
            deps.append(frm)
        ts.add(slug, *deps)
    ts.prepare()   # raises graphlib.CycleError if cyclic
```

- [ ] **Step 2: Run it**

Run: `cd ~/code/v2ecoli--repro-audit && /Users/eranagmon/code/v2ecoli/.venv/bin/python -m pytest tests/test_workspace_conformance.py -v`
Expected: PASS if Tasks B1–B4 fully applied; any FAIL points at a study still needing migration — fix it, don't weaken the test.

- [ ] **Step 3: Mirror the structural checks into `scripts/lint-workspace.py`** — add a `check_canonical_layout(wp)` that emits the same four assertions as lint errors (so the human-facing linter reports them too). Follow the file's existing error-collection pattern.

- [ ] **Step 4: Run the linter**

Run: `/Users/eranagmon/code/v2ecoli/.venv/bin/python scripts/lint-workspace.py --workspace workspace`
Expected: 0 canonical-layout errors.

- [ ] **Step 5: Commit, push, open the v2ecoli PR**

```bash
git add scripts/lint-workspace.py tests/test_workspace_conformance.py
git commit -m "test(workspace): interim conformance guard (no-nested/members/canonical/acyclic-DAG)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git log --oneline origin/main..HEAD    # verify ONLY our commits
git push -u origin spec/study-reproducibility-audit
gh pr create --title "Phase 1: study data-model canonicalization + conformance guard" \
  --body "Implements docs/superpowers/specs/2026-07-26-study-reproducibility-contract-design.md Phase 1. Depends on pbg-superpowers PR (canonicalization tools + resolver) merged to main."
```

---

## Self-Review

**Spec coverage:** §6.1 canonical model schema → A2 + B2; §6.2 inputs.from ordering → A3 + B2/B4; §6.3 kill dual layout + resolver → A1 + B1; §6.4 investigation key → A5 + B3; §7 implementation surface (viva_superpowers migrators, v2ecoli data, lint-workspace) → all of A + B5; §8 testing (golden/idempotence, guard, resolver) → A2/A4 idempotence + golden, B5 guard, A1 resolver. Phase 2/3 explicitly out of scope. No gaps.

**Placeholder scan:** every code step carries real code; every run step has an exact command + expected result; the one "read first" (A3, `migrate_inputs.py`) is a de-risking instruction, not a deferral (the transform is fully specified regardless). No TBD/TODO.

**Type consistency:** `canonicalize_models(spec) -> report`, `canonicalize_ordering(spec, known_slugs) -> report`, `migrate_study_file(study_dir, known_slugs, write=False) -> {"models","ordering","written"}`, `canonicalize_investigation(spec) -> report`, `migrate_investigation_file(inv_dir, write=False)` — names and signatures match across tasks and the CLIs that call them. Report dicts always carry `changed`/`flags`.

## Notes / known limitations

- **Comment travel on moved keys:** ruamel preserves comments attached to *values* (prose fields like `note:`/`description:` survive), but a comment sitting literally on a moved structural key line (e.g. a `# ...` directly above `baseline:` that becomes `conditions.baseline`) may not travel. The golden test (A4 Step 1) guards prose/header/trailing comments; accept the structural-key-line limitation and note it in the PR.
- **Sequencing:** Group A merges to `pbg-superpowers main` before Group B's PR merges (v2ecoli resolves the tools from `main`). During development, the editable install (B0 Step 2) decouples them so both can be built in one pass.
- **uv.lock refresh:** after the pbg-superpowers PR merges, run `uv lock --upgrade-package pbg-superpowers` in v2ecoli so CI resolves the new resolver; the editable install used during development does not update the lock.
