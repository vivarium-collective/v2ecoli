# Whole-Cell Model Comparison Framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `v2ecoli-vecoli-comparison` into a reusable Vivarium Investigation for comparing whole-cell models — parameterized by a reference repo + a config list, objective-only, with a unified, richer visualization layer — then re-run it against current models.

**Architecture:** The investigation YAML's `comparison:` block becomes the reusable interface: it declares a fixed candidate (v2ecoli), a `reference` engine descriptor, and a `configs[]` list where a config is the single unit of comparison. A new `ReferenceEngine` descriptor replaces hardcoded vEcoli paths in `orchestrator.py`; `study_spec.py` generates one study per config; a shared `theme.py` (dataviz-validated tokens) drives redesigned report cards on both the HTML report and the workbench dashboard.

**Tech Stack:** Python 3.12, process-bigraph Steps, Plotly, PyYAML, pytest; the `dataviz` skill's `scripts/validate_palette.js` (Node) for palette checks.

## Global Constraints

- Worktree: `~/code/v2ecoli--compare-harness`, branch `compare-harness` (off `origin/main` @ `96ee2260`). All commits here; never in canonical `~/code/v2ecoli`.
- Run/verify with the worktree on PATH: `PYTHONPATH=~/code/v2ecoli--compare-harness ~/code/v2ecoli/.venv/bin/python …` (the venv's editable `v2ecoli` finder points at the canonical checkout otherwise).
- Test runner: `~/code/v2ecoli/.venv/bin/python -m pytest`.
- Candidate is always `v2ecoli`; only the reference repo + configs vary.
- Gates unchanged: `{parca, statistical}`. Verdict map: `within_tol→PASS`, `drift→PARTIAL`, `mismatch→FAIL`.
- Objective-only: no before/after, fix, or root-cause prose anywhere in rendered output. Banned keywords (lint): `before`, `after fix`, `root cause`, `we fixed`, `rpoBC`, `exp_free`, `found-and-fix`.
- Engine identity is categorical color, fixed order: candidate = slot 0, reference = slot 1; status palette (within_tol/drift/mismatch) is reserved and always ships glyph+label, never color-alone.
- Commit message trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Phase 1 — Framework generalization

### Task 1: `ReferenceEngine` descriptor

**Files:**
- Create: `scripts/_compare/reference.py`
- Test: `tests/compare/test_reference.py`

**Interfaces:**
- Produces: `ReferenceEngine` dataclass with fields `repo: str`, `kind: str`, and properties `python -> str` (`<repo>/.venv/bin/python`), `parca_cmd(config_path, out_dir, intermediates_dir) -> list[str]`, `sim_cmd(config_path) -> list[str]`, `env() -> dict` (PATH-shimmed). Classmethod `from_spec(block: dict) -> ReferenceEngine` resolving `repo` values of the form `env:VAR` against `os.environ`. Unknown `kind` raises `ValueError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/compare/test_reference.py
import os
import pytest
from scripts._compare.reference import ReferenceEngine


def test_from_spec_resolves_env_indirection(monkeypatch):
    monkeypatch.setenv("V2E_VECOLI_DIR", "/tmp/vEcoli")
    r = ReferenceEngine.from_spec({"repo": "env:V2E_VECOLI_DIR", "kind": "vecoli"})
    assert r.repo == "/tmp/vEcoli"
    assert r.python == "/tmp/vEcoli/.venv/bin/python"


def test_from_spec_literal_repo():
    r = ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"})
    assert r.repo == "/abs/vEcoli"


def test_vecoli_run_commands():
    r = ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"})
    parca = r.parca_cmd("/c.json", "/out", "/out")
    assert parca[0] == "/abs/vEcoli/.venv/bin/python"
    assert "runscripts/parca.py" in parca
    sim = r.sim_cmd("/c.json")
    assert sim[:3] == ["/abs/vEcoli/.venv/bin/python", "-m", "runscripts.workflow"]


def test_env_prepends_venv_to_path():
    r = ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"})
    env = r.env()
    assert env["PATH"].startswith("/abs/vEcoli/.venv/bin:")


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        ReferenceEngine.from_spec({"repo": "/abs/x", "kind": "martian"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_reference.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts._compare.reference`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/_compare/reference.py
"""Reference-engine descriptor: how to run the comparison's reference model.

The candidate is always v2ecoli; the reference engine is declared in the
investigation's `comparison.reference` block. `kind` selects a run-interface
convention. Today only `vecoli` (CovertLab vEcoli-family: ParCa via
`runscripts/parca.py`, sim via `-m runscripts.workflow`) is implemented; the
dispatch is isolated so a new kind is additive.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_KINDS = {"vecoli"}


@dataclass
class ReferenceEngine:
    repo: str
    kind: str

    @classmethod
    def from_spec(cls, block: dict) -> "ReferenceEngine":
        raw = (block or {}).get("repo", "")
        repo = os.environ.get(raw[4:], "") if isinstance(raw, str) and raw.startswith("env:") else raw
        kind = (block or {}).get("kind", "vecoli")
        if kind not in _KINDS:
            raise ValueError(f"unknown reference kind {kind!r}; known: {sorted(_KINDS)}")
        return cls(repo=repo, kind=kind)

    @property
    def python(self) -> str:
        return f"{self.repo}/.venv/bin/python"

    def env(self) -> dict:
        path = os.environ.get("PATH", "")
        return {**os.environ, "PATH": f"{self.repo}/.venv/bin:{path}"}

    def parca_cmd(self, config_path: str, out_dir: str, intermediates_dir: str) -> list:
        return [self.python, "runscripts/parca.py", "--config", config_path,
                "--outdir", out_dir, "--save-intermediates",
                "--intermediates-directory", intermediates_dir]

    def sim_cmd(self, config_path: str) -> list:
        return [self.python, "-m", "runscripts.workflow", "--config", config_path]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_reference.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/reference.py tests/compare/test_reference.py
git commit -m "feat(compare): ReferenceEngine descriptor for the reference model"
```

---

### Task 2: `study_spec` — reference + config-as-unit, retire `from_vecoli_config`

**Files:**
- Modify: `scripts/_compare/study_spec.py`
- Test: `tests/compare/test_study_spec_configs.py`

**Interfaces:**
- Consumes: `ReferenceEngine.from_spec` (Task 1).
- Produces: `StudySpec` gains `config: str` (a condition name or a reference-config path) and `reference: ReferenceEngine`; loses `from_vecoli_config`. `_context(inv_dir)` returns `reference` (parsed) and `configs` (list of dicts). New `specs_from_configs(ctx) -> list[StudySpec]` builds one spec per `comparison.configs[]` entry, applying `defaults`.

- [ ] **Step 1: Write the failing test**

```python
# tests/compare/test_study_spec_configs.py
from scripts._compare.study_spec import specs_from_configs
from scripts._compare.reference import ReferenceEngine


def _ctx(configs):
    return {
        "invest_name": "whole-cell-model-comparison",
        "reference": ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"}),
        "configs": configs,
        "v2_cache": "out/cache_full",
        "ve_cache": "out/compare_harness/vecoli_parca",
        "defaults": {"seeds": 4, "gens": 1, "cards": ["parca", "statistical"]},
        "inv_dir": None,
    }


def test_one_spec_per_config_with_defaults():
    specs = specs_from_configs(_ctx([{"name": "basal", "config": "basal"}]))
    assert len(specs) == 1
    s = specs[0]
    assert s.name == "basal" and s.config == "basal" and s.condition == "basal"
    assert s.seeds == 4 and s.gens == 1 and s.cards == ["parca", "statistical"]


def test_path_config_carries_swap_and_condition_override():
    specs = specs_from_configs(_ctx([
        {"name": "redux_basal", "config": "configs/redux.json", "condition": "basal", "seeds": 6},
    ]))
    s = specs[0]
    assert s.config == "configs/redux.json"    # a swap is just a config path
    assert s.condition == "basal"              # explicit override
    assert s.seeds == 6                          # per-entry override wins over defaults


def test_condition_defaults_to_name_when_absent():
    specs = specs_from_configs(_ctx([{"name": "acetate", "config": "acetate"}]))
    assert specs[0].condition == "acetate"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_study_spec_configs.py -v`
Expected: FAIL — `ImportError: cannot import name 'specs_from_configs'`.

- [ ] **Step 3: Write minimal implementation**

In `scripts/_compare/study_spec.py`:
1. Add `from scripts._compare.reference import ReferenceEngine`.
2. In the `StudySpec` dataclass: add `config: str = ""` and `reference: "ReferenceEngine | None" = None`; delete the `from_vecoli_config` field.
3. In `_context`, parse the new block and return it:

```python
    comp = data.get("comparison") or {}
    return {
        "invest_name": data.get("name", inv_dir.name),
        "reference": ReferenceEngine.from_spec(comp.get("reference") or {}),
        "configs": comp.get("configs") or [],
        "v2_cache": comp.get("v2_cache", _DEFAULT_V2_CACHE),
        "ve_cache": comp.get("ve_cache", _DEFAULT_VE_CACHE),
        "defaults": comp.get("defaults") or {},
        "inv_dir": inv_dir,
    }
```

4. Add the config→spec builder:

```python
def specs_from_configs(ctx: dict) -> list:
    """One StudySpec per comparison.configs[] entry; a config is the unit."""
    defaults = ctx.get("defaults") or {}
    out = []
    for entry in ctx["configs"]:
        name = entry["name"]
        cfg = entry.get("config", name)
        out.append(StudySpec(
            name=name,
            condition=entry.get("condition", name),
            config=cfg,
            seeds=int(entry.get("seeds", defaults.get("seeds", 4))),
            gens=int(entry.get("gens", defaults.get("generations", defaults.get("gens", 1)))),
            cards=list(entry.get("cards") or defaults.get("cards") or list(_DEFAULT_CARDS)),
            invest_name=ctx["invest_name"],
            v2_cache=ctx["v2_cache"],
            ve_cache=ctx["ve_cache"],
            reference=ctx["reference"],
            study_path=str((ctx["inv_dir"] or Path(".")) / "studies" / name / "study.yaml"),
            max_steps_per_gen=int(entry.get("max_steps_per_gen") or 15000),
        ))
    return out
```

5. Remove the `fork` field and its `_context`/`_spec_from_study` uses (superseded by `reference`); update `_spec_from_study` to set `config=data.get("config") or (data.get("comparison") or {}).get("config") or name` and `reference=ctx["reference"]`, dropping `from_vecoli_config`.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_study_spec_configs.py tests/compare/test_config_adapter.py -v`
Expected: PASS (new tests green; config_adapter unaffected).

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/study_spec.py tests/compare/test_study_spec_configs.py
git commit -m "feat(compare): config-is-the-unit study generation; retire from_vecoli_config"
```

---

### Task 3: `orchestrator` — consume `ReferenceEngine`

**Files:**
- Modify: `scripts/_compare/orchestrator.py`
- Test: `tests/compare/test_orchestrator.py` (extend)

**Interfaces:**
- Consumes: `ReferenceEngine` (Task 1).
- Produces: `run_vecoli_parca(*, reference, config_path, out_dir, token=None)` and `run_vecoli_sim(*, reference, config_path, out_dir, token=None, render_only=False)` now take a `reference: ReferenceEngine` instead of the module-level `VECOLI_REPO`/`vecoli_repo=` default. `_vecoli_env` deleted (moves to `ReferenceEngine.env`).

- [ ] **Step 1: Write the failing test**

```python
# add to tests/compare/test_orchestrator.py
from scripts._compare.reference import ReferenceEngine
from scripts._compare import orchestrator


def test_vecoli_parca_uses_reference_commands(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(orchestrator, "_run", lambda cmd, cwd=None, env=None, retries=0: captured.update(cmd=cmd, cwd=cwd, env=env))
    monkeypatch.setattr(orchestrator, "is_stale", lambda *a, **k: True)
    monkeypatch.setattr(orchestrator, "mark_done", lambda *a, **k: None)
    ref = ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"})
    orchestrator.run_vecoli_parca(reference=ref, config_path="/c.json", out_dir=tmp_path)
    assert captured["cmd"][0] == "/abs/vEcoli/.venv/bin/python"
    assert "runscripts/parca.py" in captured["cmd"]
    assert captured["cwd"] == "/abs/vEcoli"
    assert captured["env"]["PATH"].startswith("/abs/vEcoli/.venv/bin:")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_orchestrator.py::test_vecoli_parca_uses_reference_commands -v`
Expected: FAIL — signature mismatch / `run_vecoli_parca` still requires `config_path`+`vecoli_repo`.

- [ ] **Step 3: Write minimal implementation**

In `orchestrator.py`: delete `VECOLI_REPO`, `VECOLI_PYTHON`, `_vecoli_env`. Rewrite the two reference wrappers to take `reference`:

```python
def run_vecoli_parca(*, reference, config_path, out_dir, token=None):
    out_dir = Path(out_dir)
    if not is_stale(out_dir, token):
        return out_dir
    _run(reference.parca_cmd(config_path, str(out_dir), str(out_dir)),
         cwd=reference.repo, env=reference.env())
    mark_done(out_dir, token or "ok")
    return out_dir


def run_vecoli_sim(*, reference, config_path, out_dir, token=None, render_only=False):
    out_dir = Path(out_dir)
    if render_only or not is_stale(out_dir, token):
        return out_dir
    _run(reference.sim_cmd(config_path), cwd=reference.repo,
         env=reference.env(), retries=2)
    mark_done(out_dir, token or "ok")
    return out_dir
```

Keep `run_v2_parca` / `run_v2_sim` (candidate side) unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_orchestrator.py -v`
Expected: PASS (update any pre-existing orchestrator test that referenced `VECOLI_REPO`).

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/orchestrator.py tests/compare/test_orchestrator.py
git commit -m "refactor(compare): orchestrator reads ReferenceEngine, drops hardcoded vEcoli paths"
```

---

### Task 4: `runner` — config-driven engine invocation

**Files:**
- Modify: `scripts/_compare/runner.py`
- Test: `tests/compare/test_runner_configs.py`

**Interfaces:**
- Consumes: `StudySpec.config`, `StudySpec.reference` (Task 2); `specs_from_configs` (Task 2).
- Produces: `_run_engines(spec, out, mode)` passes the spec's single `config` to both engines — a condition name → `--condition`; a path ending `.json` → reference `--config` and candidate `--from-vecoli-config`. `run_investigation` builds specs via `specs_from_configs` (the `configs[]` list), not a `studies:` name list.

- [ ] **Step 1: Write the failing test**

```python
# tests/compare/test_runner_configs.py
from scripts._compare import runner
from scripts._compare.study_spec import specs_from_configs
from scripts._compare.reference import ReferenceEngine


def _spec(config):
    ctx = {"invest_name": "whole-cell-model-comparison",
           "reference": ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"}),
           "configs": [{"name": "s", "config": config, "condition": "basal"}],
           "v2_cache": "vc", "ve_cache": "vec",
           "defaults": {"seeds": 4, "gens": 1, "cards": ["parca"]}, "inv_dir": None}
    return specs_from_configs(ctx)[0]


def test_condition_config_uses_condition_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **k: calls.append(argv) or type("P", (), {"returncode": 0})())
    runner._run_engines(_spec("basal"), out="out/x", mode="serial")
    v2, ve = calls
    assert "--condition" in v2 and "basal" in v2
    assert "--from-vecoli-config" not in v2      # bare condition → no swap flag


def test_path_config_uses_swap_flag_on_both(monkeypatch):
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda argv, **k: calls.append(argv) or type("P", (), {"returncode": 0})())
    runner._run_engines(_spec("configs/redux.json"), out="out/x", mode="serial")
    v2, ve = calls
    assert "--from-vecoli-config" in v2 and "configs/redux.json" in v2
    assert "--from-vecoli-config" in ve
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_runner_configs.py -v`
Expected: FAIL — `_run_engines` still reads `spec.from_vecoli_config`.

- [ ] **Step 3: Write minimal implementation**

In `runner.py`, replace the swap-flag derivation and `run_investigation`'s spec loading:

```python
def _run_engines(spec, out: str, mode: str) -> None:
    out_c = f"{out}/{spec.name}"
    ref_sd = f"{spec.ve_cache}/simData.cPickle"
    per_gen = spec.max_steps_per_gen
    v2_cap = str(spec.gens * per_gen)
    # config is the unit: a path drives a process swap on BOTH engines; a bare
    # condition name is a plain baseline comparison (no swap flag).
    is_path = str(spec.config).endswith(".json")
    swap_flags = ["--from-vecoli-config", spec.config] if is_path else []
    subprocess.run([PY, "scripts/run_comparison_ensemble.py",
                    "--composite", "v2ecoli", "--condition", spec.condition,
                    "--cache-dir", spec.v2_cache, "--n-seeds", str(spec.seeds),
                    "--max-generations", str(spec.gens), "--max-steps", v2_cap,
                    "--chunk", "60", "--mode", mode,
                    "--match-initial-state", "--match-vecoli-simdata", ref_sd,
                    *swap_flags, "--out-root", out_c], cwd=REPO, check=True)
    subprocess.run([PY, "scripts/run_comparison_ensemble.py",
                    "--composite", "vecoli", "--condition", spec.condition,
                    "--cache-dir", spec.ve_cache, "--n-seeds", str(spec.seeds),
                    "--max-generations", str(spec.gens), "--max-steps", str(per_gen),
                    "--chunk", "60", "--mode", mode,
                    "--vecoli-source", "vivarium-process",
                    *swap_flags, "--out-root", out_c], cwd=REPO, check=True)
```

In `run_investigation`, replace `_ctx, specs = load_investigation(inv_ref)` with a build from configs:

```python
    from scripts._compare.study_spec import _context, _invest_dir, specs_from_configs
    ctx = _context(_invest_dir(inv_ref))
    specs = specs_from_configs(ctx)
```

(Keep `load_study`/`run_study` working via `_spec_from_study`, which now sets `config`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_runner_configs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/runner.py tests/compare/test_runner_configs.py
git commit -m "feat(compare): runner drives both engines from the per-study config"
```

---

### Task 5: `v2e-compare init` CLI

**Files:**
- Modify: `scripts/compare_cli.py`
- Create: `scripts/_compare/scaffold.py`
- Test: `tests/compare/test_init_cli.py`

**Interfaces:**
- Produces: `scaffold_investigation(name, reference_repo, configs, out_root) -> Path` writes `<out_root>/<name>/investigation.yaml` with the `comparison:` block (candidate v2ecoli; reference `{repo, kind: vecoli}`; one `configs[]` entry per item — a `.json` path or a bare condition). CLI subparser `init --reference <repo> --configs <c1,c2,…|dir> [-o <name>]` calls it, then materializes studies.

- [ ] **Step 1: Write the failing test**

```python
# tests/compare/test_init_cli.py
import yaml
from scripts._compare.scaffold import scaffold_investigation


def test_scaffold_writes_comparison_block(tmp_path):
    p = scaffold_investigation(
        name="wcm-cmp", reference_repo="/abs/vEcoli",
        configs=["basal", "with_aa", "configs/redux.json"], out_root=tmp_path)
    data = yaml.safe_load(p.read_text())
    comp = data["comparison"]
    assert comp["candidate"] == "v2ecoli"
    assert comp["reference"] == {"repo": "/abs/vEcoli", "kind": "vecoli"}
    names = [c["name"] for c in comp["configs"]]
    assert names == ["basal", "with_aa", "redux"]     # path → basename stem
    redux = [c for c in comp["configs"] if c["name"] == "redux"][0]
    assert redux["config"] == "configs/redux.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_init_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts._compare.scaffold`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/_compare/scaffold.py
"""Scaffold a whole-cell-model-comparison investigation from a reference repo
and a list of configs (condition names or reference-config paths)."""
from __future__ import annotations

from pathlib import Path
import yaml


def _entry(c: str) -> dict:
    if c.endswith(".json"):
        return {"name": Path(c).stem, "config": c}
    return {"name": c, "config": c}


def scaffold_investigation(*, name: str, reference_repo: str, configs: list,
                           out_root) -> Path:
    inv_dir = Path(out_root) / name
    inv_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema_version": 4,
        "name": name,
        "title": "Whole-Cell Model Comparison",
        "question": "Does a candidate whole-cell model reproduce a reference "
                    "implementation across a set of configurations?",
        "comparison": {
            "candidate": "v2ecoli",
            "reference": {"repo": reference_repo, "kind": "vecoli"},
            "defaults": {"seeds": 4, "gens": 1,
                         "cards": ["summary", "parca", "statistical", "standard", "trajectory"]},
            "configs": [_entry(c) for c in configs],
        },
    }
    path = inv_dir / "investigation.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True))
    return path
```

In `compare_cli.py`, add the subparser and dispatch:

```python
    pi = sub.add_parser("init", help="scaffold a comparison investigation from a reference repo + configs")
    pi.add_argument("--reference", required=True, help="reference model repo path")
    pi.add_argument("--configs", required=True, help="comma-separated conditions/paths, or a dir of *.json")
    pi.add_argument("-o", "--name", default="whole-cell-model-comparison")
    pi.add_argument("--out-root", default="workspace/investigations")
```

```python
    if args.cmd == "init":
        from scripts._compare.scaffold import scaffold_investigation
        cfgs = ([str(p) for p in sorted(Path(args.configs).glob("*.json"))]
                if Path(args.configs).is_dir() else args.configs.split(","))
        p = scaffold_investigation(name=args.name, reference_repo=args.reference,
                                   configs=cfgs, out_root=args.out_root)
        _ctx, specs = runner.load_investigation(args.name)
        for spec in specs:
            _materialize(spec)
        print(f"scaffolded {p} ({len(specs)} studies)")
        return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_init_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_cli.py scripts/_compare/scaffold.py tests/compare/test_init_cli.py
git commit -m "feat(compare): v2e-compare init scaffolds a comparison investigation"
```

---

## Phase 2 — Rename + objective cleanup

### Task 6: Study layout fix + rename investigation → `whole-cell-model-comparison`

> **Layout decision (resolved by human):** config-derived studies live in the
> **top-level registry** `workspace/studies/<name>/` (dashboard-visible; matches
> the July registry migration + the dashboard list scanner which only scans root
> `investigations/` + `studies/`). NOT nested under the investigation. This task
> corrects `specs_from_configs` (Task 2 set `study_path` nested) and `init`
> (Task 5 seeded nested stubs) to the top-level registry, then does the rename +
> `configs[]` migration referencing the existing top-level member studies by name.

**Files:**
- Modify: `scripts/_compare/study_spec.py` (`specs_from_configs` → top-level `study_path`)
- Modify: `scripts/compare_cli.py` (`init` seeds stubs at `workspace/studies/<name>/`)
- Rename dir: `workspace/investigations/v2ecoli-vecoli-comparison/` → `workspace/investigations/whole-cell-model-comparison/`
- Rename dir: `docs/report_cards/v2ecoli-vecoli-comparison/` → `docs/report_cards/whole-cell-model-comparison/`
- Modify: the investigation.yaml `name:`/`title:`/`question:` + `comparison:` block; every `workspace/studies/*/study.yaml` `investigation:` back-reference; any other reference to the old id.
- Test: `tests/compare/test_rename_integrity.py`

**Interfaces:**
- Consumes: `specs_from_configs` (Task 2).
- Produces: `specs_from_configs` sets `study_path = REPO / "workspace" / "studies" / <name> / "study.yaml"` (top-level registry). No stale `v2ecoli-vecoli-comparison` id anywhere under `workspace/` or `docs/report_cards/`. The renamed investigation's `comparison.configs[]` names match existing top-level study dirs (`basal`, `with_aa`, `acetate`, `succinate`, `no_oxygen`, `metabolism_redux_*`).

- [ ] **Step 1: Write the failing test**

```python
# tests/compare/test_rename_integrity.py
import subprocess
from pathlib import Path
from scripts._compare.study_spec import specs_from_configs, REPO
from scripts._compare.reference import ReferenceEngine

TOP = Path(__file__).resolve().parents[2]


def test_no_stale_investigation_id():
    hits = subprocess.run(
        ["grep", "-rl", "v2ecoli-vecoli-comparison",
         str(TOP / "workspace"), str(TOP / "docs" / "report_cards")],
        capture_output=True, text=True).stdout.strip()
    assert hits == "", f"stale id remains in:\n{hits}"


def test_new_investigation_loads():
    from scripts._compare.study_spec import _context, _invest_dir
    ctx = _context(_invest_dir("whole-cell-model-comparison"))
    assert ctx["invest_name"] == "whole-cell-model-comparison"
    assert ctx["reference"].kind == "vecoli"


def test_specs_use_top_level_registry_path():
    ctx = {"invest_name": "whole-cell-model-comparison",
           "reference": ReferenceEngine.from_spec({"repo": "/abs/vEcoli", "kind": "vecoli"}),
           "configs": [{"name": "basal", "config": "basal"}],
           "v2_cache": "vc", "ve_cache": "vec",
           "defaults": {"seeds": 4, "gens": 1, "cards": ["parca"]}, "inv_dir": None}
    sp = specs_from_configs(ctx)[0].study_path
    assert sp == str(REPO / "workspace" / "studies" / "basal" / "study.yaml")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD ~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_rename_integrity.py -v`
Expected: FAIL — `study_path` is nested (`inv_dir/studies/...`); stale ids present; new dir absent.

- [ ] **Step 3: Fix the study layout in code**

In `scripts/_compare/study_spec.py`, `specs_from_configs`, change the `study_path` to the top-level registry (drop the `inv_dir/"studies"` nesting):

```python
            study_path=str(REPO / "workspace" / "studies" / name / "study.yaml"),
```

In `scripts/compare_cli.py`, the `init` dispatch: seed any per-config stub study.yaml at `REPO/workspace/studies/<name>/study.yaml` (top-level), not under the investigation dir, so scaffolded studies are registry+dashboard visible and match `specs_from_configs`. (Reuse the study path from `specs_from_configs` rather than recomputing.)

- [ ] **Step 4: Do the rename + `configs[]` migration**

```bash
git mv workspace/investigations/v2ecoli-vecoli-comparison workspace/investigations/whole-cell-model-comparison
git mv docs/report_cards/v2ecoli-vecoli-comparison docs/report_cards/whole-cell-model-comparison
```
Edit `investigation.yaml`: set `name: whole-cell-model-comparison`, `title: "Whole-Cell Model Comparison"`, `question:` (framework framing), and replace the legacy `members:`/`comparison` block with the new schema — `candidate: v2ecoli`, `reference: {repo: env:V2E_VECOLI_DIR, kind: vecoli}`, `defaults: {seeds: 4, gens: 1, cards: [summary, parca, statistical, standard, trajectory]}`, and `configs:` with one entry per **existing** top-level member study dir: `basal`, `with_aa`, `acetate`, `succinate`, `no_oxygen` (bare `config: <name>`), and each `metabolism_redux_*` as `{name: metabolism_redux_<cond>, config: configs/metabolism_redux_<cond>.json, condition: <cond>}` (read the swap path from each existing study.yaml's legacy `from_vecoli_config` before deleting it). Update every `workspace/studies/*/study.yaml` `investigation:` value to the new id. Grep-fix remaining hits:
```bash
grep -rl v2ecoli-vecoli-comparison workspace docs/report_cards
```

- [ ] **Step 5: Run test to verify it passes**

Run: `PYTHONPATH=$PWD ~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_rename_integrity.py tests/compare/ -q`
Expected: PASS (rename-integrity green; suite still green).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(compare): top-level study registry + rename to whole-cell-model-comparison"
```

---

### Task 7: Objective cleanup + narrative lint

**Files:**
- Modify: `workspace/investigations/whole-cell-model-comparison/investigation.yaml` (executive, how_to_read, biological_story, glossary); every `workspace/studies/<member>/study.yaml` narrative field.
- Modify: `scripts/_compare/materialize.py` (audit finding statements).
- Test: `tests/compare/test_objective_narrative.py`

**Interfaces:**
- Produces: no banned keyword (Global Constraints) in any rendered narrative field or materialized finding.

- [ ] **Step 1: Write the failing test**

```python
# tests/compare/test_objective_narrative.py
import re
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
INV = REPO / "workspace" / "investigations" / "whole-cell-model-comparison"
BANNED = re.compile(r"\b(before\b.*\bafter|after fix|root cause|we fixed|rpoBC|exp_free|found-and-fix)\b", re.I)


def _narrative_text():
    docs = [INV / "investigation.yaml"]
    docs += list((REPO / "workspace" / "studies").glob("*/study.yaml"))
    chunks = []
    for p in docs:
        d = yaml.safe_load(p.read_text()) or {}
        for k in ("executive", "how_to_read", "biological_story", "glossary", "narrative", "claim", "findings"):
            v = d.get(k)
            if v is not None:
                chunks.append(yaml.safe_dump(v))
    return "\n".join(chunks)


def test_no_fix_history_in_narrative():
    m = BANNED.search(_narrative_text())
    assert m is None, f"fix-history phrasing found: {m.group(0)!r}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_objective_narrative.py -v`
Expected: FAIL — current executive/studies contain the fix narrative.

- [ ] **Step 3: Rewrite narratives objectively**

Edit the investigation `executive.what_is_this` to describe the framework + current result only (inputs: reference repo + configs; measure: matched-initial-state per-observable deltas over N seeds; grade: parca + statistical). Delete `RESULT (before→after)` tables, `ROOT CAUSE`, and any "we traced/fixed" prose. In each member study.yaml, replace fix narrative with a one-line objective statement (config, observables, verdict). Audit `materialize.py` `_graph_fields` statements — they are already current-state; confirm no banned phrasing is templated.

- [ ] **Step 4: Run test to verify it passes**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_objective_narrative.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "docs(compare): objective-only narrative; strip fix history"
```

---

## Phase 3 — Visualization overhaul

### Task 8: Shared `theme.py` (dataviz-validated tokens)

**Files:**
- Create: `scripts/_compare/theme.py`
- Test: `tests/compare/test_theme.py`

**Interfaces:**
- Produces: `ENGINE = {"candidate": "<hex>", "reference": "<hex>"}` (categorical slots 0/1), `STATUS = {"within_tol": {...}, "drift": {...}, "mismatch": {...}}` each `{color, glyph, label}`, `SURFACE = {"light": {...}, "dark": {...}}` (bg/ink/muted/line), and `css_vars(mode) -> str` emitting a `:root` block. Values seeded from `vivarium_workbench/static/style.css` and snapped to passing steps.

- [ ] **Step 1: Extract workbench tokens (reference read)**

Read the workbench `:root` block for palette seeds:
```bash
WB=$(~/code/v2ecoli/.venv/bin/python -c "import vivarium_workbench,os;print(os.path.dirname(vivarium_workbench.__file__))")
grep -A40 ":root" "$WB/static/style.css" | grep -E "\-\-|#[0-9a-fA-F]{6}" | head -40
```

- [ ] **Step 2: Write the failing test (palette validation)**

```python
# tests/compare/test_theme.py
import subprocess, shutil
from pathlib import Path
import pytest
from scripts._compare import theme


def _validate(hexes, mode, surface):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    script = theme.VALIDATOR_PATH  # vendored tests/fixtures/validate_palette.js (set in Task 8 Step 3)
    out = subprocess.run([node, str(script), ",".join(hexes), "--mode", mode, "--surface", surface],
                         capture_output=True, text=True)
    return out.returncode, out.stdout + out.stderr


def test_engine_pair_validates_light():
    rc, log = _validate(list(theme.ENGINE.values()), "light", theme.SURFACE["light"]["bg"])
    assert rc == 0, log


def test_engine_pair_validates_dark():
    rc, log = _validate(list(theme.ENGINE.values()), "dark", theme.SURFACE["dark"]["bg"])
    assert rc == 0, log


def test_status_is_glyph_plus_label():
    for k, v in theme.STATUS.items():
        assert v["glyph"] and v["label"]      # never color-alone
```

- [ ] **Step 3: Run validator manually + write `theme.py`**

Run the dataviz validator on candidate engine hexes until it passes light + dark:
`node <dataviz>/scripts/validate_palette.js "<c0>,<c1>" --mode light`
Snap to passing steps, then write `theme.py` with the passing `ENGINE`, `STATUS` (glyph+label: ✓ within_tol / ◐ drift / ✗ mismatch), `SURFACE`, `css_vars(mode)`, and `VALIDATOR_PATH` (vendor `validate_palette.js` to `tests/fixtures/validate_palette.js` so tests are hermetic).

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_theme.py -v`
Expected: PASS (or SKIP if node absent — CI must have node).

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/theme.py tests/compare/test_theme.py tests/fixtures/validate_palette.js
git commit -m "feat(compare): shared dataviz-validated theme seeded from workbench tokens"
```

---

### Task 9: Route `report.py` + `plotly_helpers.py` through `theme.py`

**Files:**
- Modify: `scripts/_compare/report.py` (`:root` CSS), `scripts/_compare/plotly_helpers.py` (`VE_COLOR`/`V2_COLOR`)
- Test: `tests/compare/test_theme_wiring.py`

**Interfaces:**
- Consumes: `theme.ENGINE`, `theme.STATUS`, `theme.css_vars` (Task 8).
- Produces: no literal engine/status hex remains in `report.py`/`plotly_helpers.py`; both import from `theme`. Report emits `css_vars("light")` + a `prefers-color-scheme: dark` block from `css_vars("dark")`.

- [ ] **Step 1: Write the failing test**

```python
# tests/compare/test_theme_wiring.py
import re
from pathlib import Path
SRC = Path(__file__).resolve().parents[2] / "scripts" / "_compare"


def test_no_literal_engine_hex_in_helpers():
    txt = (SRC / "plotly_helpers.py").read_text()
    assert "#4f46e5" not in txt and "#d97706" not in txt   # moved to theme
    assert "from scripts._compare.theme import" in txt or "from scripts._compare import theme" in txt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_theme_wiring.py -v`
Expected: FAIL — literal hexes present.

- [ ] **Step 3: Rewire**

In `plotly_helpers.py`: `from scripts._compare import theme`; set `VE_COLOR = theme.ENGINE["reference"]`, `V2_COLOR = theme.ENGINE["candidate"]`. In `report.py`: replace the inline `:root {…}` block with `theme.css_vars("light")` and add a dark block; replace status colors (`--green/--amber/--red`) with `theme.STATUS[*]["color"]` and render each status pill with its glyph+label.

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_theme_wiring.py tests/compare/test_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/report.py scripts/_compare/plotly_helpers.py tests/compare/test_theme_wiring.py
git commit -m "refactor(compare): report + charts consume shared theme (light/dark)"
```

---

### Task 10: Richer per-observable charts (bands + Δ panel + stat annotation)

**Files:**
- Modify: `scripts/_compare/plotly_helpers.py` (add `overlay_band_html`, `delta_panel_html`)
- Test: `tests/compare/test_plotly_helpers.py`

**Interfaces:**
- Consumes: `theme.ENGINE` (Task 8).
- Produces: `overlay_band_html(per_obs, title="") -> str` draws, per observable, a candidate mean line + shaded ±1σ band and a reference mean line + band (one axis; crosshair hover); `delta_panel_html(per_obs, tol, stat=None) -> str` draws median relative Δ vs time with a shaded tolerance band and, if `stat` given, an inline `KS`/`Welch-t` annotation.

- [ ] **Step 1: Write the failing test**

```python
# tests/compare/test_plotly_helpers.py
from scripts._compare.plotly_helpers import overlay_band_html, delta_panel_html

PER = {"cell_mass": {
    "v2": [([0, 1, 2], [1.0, 1.1, 1.2]), ([0, 1, 2], [1.0, 1.05, 1.15])],
    "ve": [([0, 1, 2], [1.0, 1.1, 1.25]), ([0, 1, 2], [1.0, 1.08, 1.2])],
    "gen_bounds": []}}


def test_overlay_band_emits_band_and_both_engines():
    html = overlay_band_html(PER, title="cell")
    assert "cell_mass" in html
    assert html.count("fill") >= 1        # at least one shaded band
    assert "candidate" in html.lower() or "v2ecoli" in html.lower()


def test_delta_panel_shades_tolerance_and_annotates_stat():
    html = delta_panel_html(PER, tol=0.1, stat={"kind": "Welch-t", "p": 0.4})
    assert "Welch-t" in html and "0.4" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_plotly_helpers.py -v`
Expected: FAIL — functions undefined.

- [ ] **Step 3: Implement**

Add both functions to `plotly_helpers.py`. Compute per-timepoint mean±σ across the seed traces (align on the shortest length); draw the band as a filled `scatter` between mean−σ and mean+σ with the engine color at low opacity, mean as a 2px line; single y-axis; `hovermode="x unified"`. In `delta_panel_html`, plot median relative Δ, shade `[-tol, +tol]` with the neutral line token, and add `fig.add_annotation` for the stat.

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_plotly_helpers.py -v`
Expected: PASS.

- [ ] **Step 5: Render-eyeball + commit**

Regenerate one study's trajectory card from cached stores and open it:
`PYTHONPATH=$PWD ~/code/v2ecoli/.venv/bin/python scripts/comparison_report_card.py --investigation whole-cell-model-comparison --study basal --out out/report && open out/report/basal/viz/report_card/trajectory.html`
Check for label collisions/overflow, then:
```bash
git add scripts/_compare/plotly_helpers.py tests/compare/test_plotly_helpers.py
git commit -m "feat(compare): cross-seed band overlays + tolerance delta panels"
```

---

### Task 11: Study-level `summary` card

**Files:**
- Create: `scripts/_compare/report_cards/summary.py`
- Modify: `scripts/_compare/report_cards/__init__.py` (register `summary`)
- Test: `tests/compare/test_summary_card.py`

**Interfaces:**
- Consumes: `theme.STATUS` (Task 8); the per-study verdict dict (`groups[*].axes[*]` with `label`, `verdict`, `detail.median_rel`).
- Produces: a `summary` report-card Step registered as `summary_report_card` that renders a verdict pill strip + a per-observable |Δ| status heat row (status fill + value label, glyph+label legend) + seed count + gate status. Informational (no measure/gate). Pure builder `build_summary_html(verdict: dict, seeds: int) -> str` for unit testing.

- [ ] **Step 1: Write the failing test**

```python
# tests/compare/test_summary_card.py
from scripts._compare.report_cards.summary import build_summary_html

VERDICT = {"overall": "drift", "groups": {"statistical": {"verdict": "drift", "axes": [
    {"label": "cell", "verdict": "within_tol", "detail": {"median_rel": 0.014}},
    {"label": "growth", "verdict": "drift", "detail": {"median_rel": 0.104}},
]}}}


def test_summary_lists_each_observable_with_status_and_value():
    html = build_summary_html(VERDICT, seeds=4)
    assert "cell" in html and "growth" in html
    assert "1.4%" in html and "10.4%" in html       # median_rel as percent
    assert "4 seeds" in html
    # status conveyed by glyph+label, not color alone
    assert "within_tol" in html and "drift" in html


def test_summary_shows_gate_status():
    html = build_summary_html(VERDICT, seeds=4)
    assert "gate" in html.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_summary_card.py -v`
Expected: FAIL — module undefined.

- [ ] **Step 3: Implement**

Write `build_summary_html` (pure) rendering the strip/heat-row from `theme.STATUS`; then wrap it as an `as_step` Step following `trajectory.py`'s pattern, registering `summary_report_card` in `report_cards/__init__.py`. Add `summary` to `_DEFAULT_CARDS` ordering so it renders first. `summary` is NOT in `GRADED` (study_spec) — informational only.

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_summary_card.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/report_cards/summary.py scripts/_compare/report_cards/__init__.py tests/compare/test_summary_card.py
git commit -m "feat(compare): per-study summary card (verdict strip + observable heat row)"
```

---

### Task 12: Cross-config overview (matrix, both surfaces)

**Files:**
- Create: `scripts/_compare/overview.py`
- Modify: `scripts/_compare/report.py` (embed overview on the investigation index)
- Test: `tests/compare/test_overview.py`

**Interfaces:**
- Consumes: `theme.STATUS`; a mapping `config_name -> verdict dict` (Task 11's verdict shape).
- Produces: `build_overview_html(verdicts: dict[str, dict], observables: list[str]) -> str` — a configs × observables matrix, each cell a status fill + median-|Δ| label, row/col headers, glyph+label legend, per-cell hover. Embedded on the HTML report index and exposed for the dashboard investigation view.

- [ ] **Step 1: Write the failing test**

```python
# tests/compare/test_overview.py
from scripts._compare.overview import build_overview_html

VERDICTS = {
  "basal":   {"groups": {"statistical": {"axes": [
      {"label": "cell", "verdict": "within_tol", "detail": {"median_rel": 0.001}},
      {"label": "growth", "verdict": "drift", "detail": {"median_rel": 0.104}}]}}},
  "with_aa": {"groups": {"statistical": {"axes": [
      {"label": "cell", "verdict": "within_tol", "detail": {"median_rel": 0.009}},
      {"label": "growth", "verdict": "within_tol", "detail": {"median_rel": 0.043}}]}}},
}


def test_matrix_has_a_cell_per_config_observable():
    html = build_overview_html(VERDICTS, observables=["cell", "growth"])
    assert "basal" in html and "with_aa" in html
    assert "cell" in html and "growth" in html
    assert "10.4%" in html and "0.1%" in html
    # every (config,observable) rendered → 2x2 status cells
    assert html.count("data-status") == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_overview.py -v`
Expected: FAIL — module undefined.

- [ ] **Step 3: Implement**

Write `build_overview_html` producing an HTML table/grid; each cell carries `data-status="<verdict>"` (for hover + CSS status fill from `theme.STATUS`) and the median-|Δ| percent label. Embed it near the top of `report.py`'s investigation index (a new "Reproduction across configs" section). Expose the same fragment for the dashboard investigation view (write it into the investigation's rendered artifacts alongside the report cards).

- [ ] **Step 4: Run tests to verify they pass**

Run: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_overview.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/overview.py scripts/_compare/report.py tests/compare/test_overview.py
git commit -m "feat(compare): cross-config reproduction matrix on report + dashboard"
```

---

### Task 13: Full compare-suite green + render smoke

**Files:**
- Test: whole `tests/compare/` suite.

- [ ] **Step 1: Run the full suite**

Run: `PYTHONPATH=$PWD ~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/ -v`
Expected: all PASS (E2E test skipped without `COMPARE_E2E=1`).

- [ ] **Step 2: Render-only smoke from any existing cached stores**

Run: `PYTHONPATH=$PWD ~/code/v2ecoli/.venv/bin/python scripts/compare_cli.py run whole-cell-model-comparison --render-only --out out/report || true`
Expected: renders without import/registry errors (data may be absent — that's fine; we're checking wiring, cards, theme, overview all import and render).

- [ ] **Step 3: Commit any fixups**

```bash
git add -A && git commit -m "test(compare): full suite green + render smoke" || echo "nothing to fix"
```

---

## Phase 4 — Re-run current models (execution, not code)

> This phase runs heavy compute; it is not TDD. Do it only after Phases 1–3 land and the suite is green.

### Task 14: Verify one config, then full run + re-render

- [ ] **Step 1: Ensure vEcoli reference is current + reachable**

```bash
git -C /Users/eranagmon/code/vEcoli fetch origin main && git -C /Users/eranagmon/code/vEcoli log -1 --oneline
export V2E_VECOLI_DIR=/Users/eranagmon/code/vEcoli
```

- [ ] **Step 2: Single-config end-to-end verify (local)**

Run: `PYTHONPATH=$PWD ~/code/v2ecoli/.venv/bin/python scripts/compare_cli.py study basal --out out/report`
Expected: engines run, `docs/report_cards/whole-cell-model-comparison/basal/report_card_verdict.json` written, summary/trajectory/overview render objectively. Open `out/report/basal/` and eyeball.

- [ ] **Step 3: Full run on the mini (Ray)**

Sync worktree to the mini, then run all configs in parallel:
`mct v2ecoli "cd <worktree> && PYTHONPATH=\$PWD .venv/bin/python scripts/compare_cli.py run whole-cell-model-comparison --ray --out out/report"`
Monitor via git commits / `mct-log`, not the buffered `-p` log. (Heavy: full ParCa + Nextflow 2-gen lineages per config.)

- [ ] **Step 4: Re-render + verify objective output**

Run: `PYTHONPATH=$PWD ~/code/v2ecoli/.venv/bin/python scripts/compare_cli.py run whole-cell-model-comparison --render-only --out out/report`
Then: `~/code/v2ecoli/.venv/bin/python -m pytest tests/compare/test_objective_narrative.py -v` (still green after materialize refreshes verdicts).

- [ ] **Step 5: Commit refreshed report cards**

```bash
git add docs/report_cards/whole-cell-model-comparison workspace
git commit -m "run(compare): refresh whole-cell-model-comparison against current models"
```

---

## Self-Review Notes

- **Spec coverage:** §1 rename→Task 6; §2 framework spec→Tasks 2/5; §3 reference descriptor→Tasks 1/3; §3 runner→Task 4; §4 cleanup→Task 7; §5 gating→unchanged (asserted by existing tests); §5a viz (unified/richer/summary/cross-config, both surfaces)→Tasks 8–12; §6 CLI→Task 5; §7 re-run + render/run split→Task 14 (+ `--render-only` used in Tasks 10/13); §8 testing→each task's tests + Tasks 7/8/11/12.
- **Render/run split** lets Phase 3 viz iterate on cached stores (Task 10 Step 5, Task 13 Step 2) before the heavy Task 14 run.
- **Dashboard surface:** the workbench renders these cards from the same `docs/report_cards/` + study artifacts; Tasks 9/11/12 write theme-aware HTML the dashboard serves, satisfying "both surfaces" without separate dashboard code.
