# Modular Config-Driven Comparison Harness — Plan A (schema + cards + report)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the seeds/gens-duplicating `comparison_spec.json` with a slim manifest that lists vEcoli configs and assigns modular, registry-based report cards per config, and make the report assemble those assigned cards (overview + per-config sections).

**Architecture:** A new `configs` mode in the stdlib parser reads run shape (seeds/gens) from each referenced vEcoli config instead of the manifest. A new `scripts/_compare/report_cards/` package holds one module per card plus a name→callable registry; cards are thin wrappers over the EXISTING section functions (`runs_section`/`eval_section`/`parca_section`) and Chris Long's on-main card library (`report_card_section.build_report_card` → `v2ecoli.library.report_card`). `comparison_report_card.py` is rewired to render, per config, exactly the cards the manifest assigns. Single-variant only (variant=0); full variant execution is Plan B.

**Tech Stack:** Python 3.12, stdlib `json`/`argparse`, pytest. No new dependencies.

## Global Constraints

- stdlib-only in `scripts/_read_spec.py` (called from bash without a venv) — verbatim from the spec.
- Report cards REUSE the on-main library (`v2ecoli/library/report_card.py`, `card_criteria.py`, `card_plots.py`, `card_vectors.py`); no new grading or plot code.
- `Section = {"title": str, "kind": "content", "html": str, "anchor": str, "verdict": str | None}` — the existing section dict shape; the assembler stays structurally unchanged.
- Rendered output keeps the `docs/report_cards/` convention.
- seeds = a config's `n_init_sims`; gens = a config's `generations` — read from the vEcoli config, never the manifest.
- Run tests with `/Users/eranagmon/code/v2e-cmp-harness/.venv/bin/python -m pytest` (set up the venv first if absent: `cd /Users/eranagmon/code/v2e-cmp-harness && uv sync --no-install-package vivarium-workbench`).

---

## File Structure

- Create `scripts/_compare/report_cards/__init__.py` — registry (`REGISTRY`, `@report_card`, `get`, `all_names`), `CardContext` dataclass, `Section` TypedDict, imports the card modules to register them.
- Create `scripts/_compare/report_cards/standard.py` — `standard_card(ctx)` (runs + eval).
- Create `scripts/_compare/report_cards/statistical.py` — `statistical_card(ctx)` (Chris's graded card).
- Create `scripts/_compare/report_cards/parca.py` — `parca_card(ctx)`.
- Create `scripts/_compare/report_cards/config_diff.py` — `config_diff_card(ctx)`.
- Modify `scripts/_read_spec.py` — add `config_rows()` + a `configs` mode.
- Modify `scripts/comparison_report_card.py` — manifest-driven per-config card assembly.
- Modify `scripts/_compare/config_adapter.py` — add `config_run_shape()` helper.
- Create `comparison.5cond_1x4.json`, `comparison.baseline_4x4_statistical.json`, `configs/cond_basal_4x4.json`.
- Modify `tests/test_read_spec.py`; create `tests/test_report_cards.py`.

---

## Task 1: Read run shape (seeds/gens) from a vEcoli config

**Files:**
- Modify: `scripts/_compare/config_adapter.py` (add `config_run_shape`)
- Test: `tests/test_config_translation.py` (extend)

**Interfaces:**
- Consumes: existing `resolve_vecoli_config_local(config_path, fork_dir) -> dict`.
- Produces: `config_run_shape(config_path: str, fork_dir: str) -> tuple[int, int]` returning `(seeds, gens)` = `(n_init_sims, generations)`, defaulting `n_init_sims` to 1 and `generations` to 1 when absent/None.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_translation.py (append)
from scripts._compare.config_adapter import config_run_shape  # noqa: E402

def test_config_run_shape_reads_n_init_sims_and_generations(tmp_path):
    cfg = tmp_path / "configs"; cfg.mkdir()
    (cfg / "c.json").write_text('{"n_init_sims": 4, "generations": 4}')
    assert config_run_shape("configs/c.json", str(tmp_path)) == (4, 4)

def test_config_run_shape_defaults_when_missing(tmp_path):
    cfg = tmp_path / "configs"; cfg.mkdir()
    (cfg / "c.json").write_text('{"condition": "basal", "generations": null}')
    assert config_run_shape("configs/c.json", str(tmp_path)) == (1, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_translation.py -k run_shape -q`
Expected: FAIL with `ImportError: cannot import name 'config_run_shape'`.

- [ ] **Step 3: Implement**

```python
# scripts/_compare/config_adapter.py (append)
def config_run_shape(config_path: str, fork_dir: str) -> tuple[int, int]:
    """Return (seeds, gens) = (n_init_sims, generations) from a vEcoli config.

    seeds/gens are the config's run shape — the single source of truth. Missing
    or null values default to 1 (one seed, one generation).
    """
    cfg = resolve_vecoli_config_local(config_path, fork_dir)
    seeds = cfg.get("n_init_sims")
    gens = cfg.get("generations")
    return int(seeds) if seeds else 1, int(gens) if gens else 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config_translation.py -k run_shape -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/config_adapter.py tests/test_config_translation.py
git commit -m "feat(harness): read run shape (seeds/gens) from a vEcoli config"
```

---

## Task 2: `configs` mode in `_read_spec.py`

**Files:**
- Modify: `scripts/_read_spec.py`
- Test: `tests/test_read_spec.py`

**Interfaces:**
- Consumes: nothing new (stdlib only — does NOT import config_adapter; the seeds/gens resolution happens in the runner, which has the fork dir).
- Produces:
  - `config_rows(spec) -> Iterator[tuple[str, str]]` yielding `(config_path, cards_csv)` per `spec["configs"]` entry, where `cards_csv` is the entry's `cards` joined by `,`, falling back to `spec["defaults"]["cards"]`, else `"standard"`.
  - CLI `configs` mode printing `config_path<TAB>cards_csv` per row.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_read_spec.py (append)
import scripts._read_spec as rs  # noqa: E402

SPEC = {
    "v2ecoli": {"repo": "r1", "commit": "c1"},
    "vecoli": {"repo": "r2", "commit": "c2"},
    "defaults": {"cards": ["standard"]},
    "configs": [
        {"config": "configs/cond_basal.json", "cards": ["standard", "statistical"]},
        {"config": "configs/cond_with_aa.json"},
    ],
}

def test_config_rows_uses_entry_cards_then_defaults():
    rows = list(rs.config_rows(SPEC))
    assert rows == [
        ("configs/cond_basal.json", "standard,statistical"),
        ("configs/cond_with_aa.json", "standard"),
    ]

def test_config_rows_defaults_to_standard_when_no_defaults():
    spec = {"configs": [{"config": "c.json"}]}
    assert list(rs.config_rows(spec)) == [("c.json", "standard")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_read_spec.py -k config_rows -q`
Expected: FAIL with `AttributeError: module 'scripts._read_spec' has no attribute 'config_rows'`.

- [ ] **Step 3: Implement**

```python
# scripts/_read_spec.py — add near condition_rows()
def config_rows(spec):
    """Yield (config_path, cards_csv) per spec['configs'] entry.

    cards: entry 'cards' -> spec defaults.cards -> ['standard'].
    seeds/gens/variants are NOT here — the runner reads them from each config.
    """
    default_cards = (spec.get("defaults", {}) or {}).get("cards") or ["standard"]
    for entry in spec.get("configs", []):
        config = entry["config"]
        cards = entry.get("cards") or default_cards
        yield config, ",".join(cards)
```

```python
# scripts/_read_spec.py — in main(), add an elif branch before the final else
    elif mode == "configs":
        for config, cards in config_rows(spec):
            print("\t".join([config, cards]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_read_spec.py -k config_rows -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/_read_spec.py tests/test_read_spec.py
git commit -m "feat(harness): configs mode in _read_spec (config -> cards)"
```

---

## Task 3: Report-card registry + `CardContext`

**Files:**
- Create: `scripts/_compare/report_cards/__init__.py`
- Test: `tests/test_report_cards.py`

**Interfaces:**
- Produces:
  - `CardContext` dataclass with fields: `config_name: str`, `variant: int`, `v2_dir: str`, `ve_dir: str`, `seeds: int`, `gens: int`, `per_obs: dict`, `plot_trajs: dict`, `v2_bounds: dict`, `config: dict`.
  - `Section = dict` (keys: `title`, `kind`, `html`, `anchor`, optional `verdict`).
  - `report_card(name)` decorator registering `name -> fn`.
  - `get(name) -> Callable[[CardContext], list[Section]]` raising `KeyError` on unknown name.
  - `all_names() -> list[str]`.
  - `render(name, ctx) -> list[Section]` — calls the card, normalizes a single Section to `[Section]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_cards.py
import pytest
from scripts._compare import report_cards as rc


def test_register_and_get():
    @rc.report_card("dummy_card_xyz")
    def _c(ctx):
        return {"title": "T", "kind": "content", "html": "<p>x</p>", "anchor": "a"}
    assert "dummy_card_xyz" in rc.all_names()
    fn = rc.get("dummy_card_xyz")
    out = rc.render("dummy_card_xyz", _ctx())
    assert out[0]["title"] == "T" and out[0]["html"]


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        rc.get("does_not_exist")


def _ctx():
    return rc.CardContext(config_name="basal", variant=0, v2_dir="", ve_dir="",
                          seeds=1, gens=1, per_obs={}, plot_trajs={}, v2_bounds={},
                          config={})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_report_cards.py -k register -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts._compare.report_cards'`.

- [ ] **Step 3: Implement**

```python
# scripts/_compare/report_cards/__init__.py
"""Registry of modular comparison report cards.

A card is Callable[[CardContext], Section | list[Section]]. Register with the
@report_card("name") decorator; assign by name in the comparison manifest.
Cards are thin wrappers over the existing section functions and Chris Long's
on-main card library (v2ecoli.library.report_card).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

Section = dict  # {title, kind, html, anchor, verdict?}


@dataclass
class CardContext:
    config_name: str
    variant: int
    v2_dir: str
    ve_dir: str
    seeds: int
    gens: int
    per_obs: dict = field(default_factory=dict)
    plot_trajs: dict = field(default_factory=dict)
    v2_bounds: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


Card = Callable[[CardContext], "Section | list[Section]"]
REGISTRY: dict[str, Card] = {}


def report_card(name: str) -> Callable[[Card], Card]:
    def deco(fn: Card) -> Card:
        REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> Card:
    if name not in REGISTRY:
        raise KeyError(f"unknown report card {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def all_names() -> list[str]:
    return sorted(REGISTRY)


def render(name: str, ctx: CardContext) -> list[Section]:
    out = get(name)(ctx)
    return out if isinstance(out, list) else [out]


# Register the built-in cards by importing their modules (each calls
# @report_card at import). Imported at the bottom to avoid circular imports.
from scripts._compare.report_cards import standard, statistical, parca, config_diff  # noqa: E402,F401
```

- [ ] **Step 4: Run test to verify it passes**

(After Task 4-5 create the imported modules, this import resolves. To unblock this task in isolation, temporarily comment the bottom import, run the test, then restore it — or implement Tasks 4-5 first.) Run: `.venv/bin/python -m pytest tests/test_report_cards.py -k "register or unknown" -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/report_cards/__init__.py tests/test_report_cards.py
git commit -m "feat(cards): report-card registry + CardContext"
```

---

## Task 4: `standard` and `statistical` cards

**Files:**
- Create: `scripts/_compare/report_cards/standard.py`, `scripts/_compare/report_cards/statistical.py`
- Test: `tests/test_report_cards.py` (extend)

**Interfaces:**
- Consumes: `report_card`, `CardContext`, `Section` from `scripts._compare.report_cards`; the existing `comparison_report_card.runs_section(cond, per_obs, plot_trajs, v2_bounds)` and `comparison_report_card.eval_section(cond, per_obs)` (both return a Section dict); `scripts._compare.report_card_section.build_report_card(left_by_cell, right_by_cell, *, model_ref, ...) -> (vjson, html)`.
- Produces: `standard_card(ctx) -> list[Section]`, `statistical_card(ctx) -> Section`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_cards.py (append)
def test_builtin_cards_registered():
    for name in ("standard", "statistical", "parca", "config_diff"):
        assert name in rc.all_names()

def test_statistical_card_returns_graded_section():
    ctx = rc.CardContext(config_name="basal", variant=0, v2_dir="", ve_dir="",
                         seeds=2, gens=1,
                         per_obs={"cell_mass": {"ve_cells": [1.0, 1.1],
                                                "v2_cells": [1.0, 1.05]},
                                  "growth_rate": {"ve_cells": [2e-4, 2.1e-4],
                                                  "v2_cells": [2e-4, 2.05e-4]}},
                         plot_trajs={}, v2_bounds={}, config={})
    secs = rc.render("statistical", ctx)
    assert secs[0]["html"] and secs[0]["verdict"] in (
        "within_tol", "drift", "mismatch", "ungraded")
```

NOTE: the exact `per_obs` sub-keys (`ve_cells`/`v2_cells`) must match what `comparison_report_card.build()` stores per observable. Before writing the card, open `scripts/comparison_report_card.py`, find where `per_obs[key]` is populated, and use those real keys in both the card and this test.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_report_cards.py -k "builtin or statistical_card" -q`
Expected: FAIL (cards not yet defined / not registered).

- [ ] **Step 3: Implement**

```python
# scripts/_compare/report_cards/standard.py
"""`standard` card — matched-time run trajectories + evaluation (the lighter
card). Thin wrapper over comparison_report_card.runs_section / eval_section."""
from __future__ import annotations

from scripts._compare.report_cards import report_card, CardContext, Section


@report_card("standard")
def standard_card(ctx: CardContext) -> list[Section]:
    # Imported lazily: comparison_report_card imports heavy deps; importing it at
    # module load would slow registry import and risk a cycle.
    from scripts.comparison_report_card import runs_section, eval_section
    return [
        runs_section(ctx.config_name, ctx.per_obs, ctx.plot_trajs, ctx.v2_bounds),
        eval_section(ctx.config_name, ctx.per_obs),
    ]
```

```python
# scripts/_compare/report_cards/statistical.py
"""`statistical` card — Chris Long's graded equivalence card (violin/strip +
<details> dropdown viz bars + within_tol/drift/mismatch pills). Thin wrapper
over report_card_section.build_report_card -> v2ecoli.library.report_card."""
from __future__ import annotations

from scripts._compare.report_cards import report_card, CardContext, Section
from scripts._compare.report_card_section import build_report_card


@report_card("statistical")
def statistical_card(ctx: CardContext) -> Section:
    # Build per-cell {observable: [scalar per cell]} for vEcoli (reference) and
    # v2ecoli (measured) from ctx.per_obs. The exact per_obs sub-keys come from
    # comparison_report_card.build(); map them here.
    left = {k: list(v.get("ve_cells", [])) for k, v in ctx.per_obs.items()}
    right = {k: list(v.get("v2_cells", [])) for k, v in ctx.per_obs.items()}
    vjson, html = build_report_card(
        left, right, model_ref=f"v2ecoli @ {ctx.config_name} variant {ctx.variant}")
    return {"title": f"{ctx.config_name} — statistical equivalence",
            "kind": "content",
            "anchor": f"{ctx.config_name}-statistical",
            "html": html,
            "verdict": vjson.get("overall")}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_report_cards.py -k "builtin or statistical_card" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/report_cards/standard.py scripts/_compare/report_cards/statistical.py tests/test_report_cards.py
git commit -m "feat(cards): standard + statistical (Chris's graded) cards"
```

---

## Task 5: `parca` and `config_diff` cards

**Files:**
- Create: `scripts/_compare/report_cards/parca.py`, `scripts/_compare/report_cards/config_diff.py`
- Test: covered by `test_builtin_cards_registered` (Task 4).

**Interfaces:**
- Consumes: `comparison_report_card.parca_section(cond_data)` and `comparison_report_card.config_sections_for(cond, v2_dir, ve_dir)` (returns list[Section]).
- Produces: `parca_card(ctx) -> Section`, `config_diff_card(ctx) -> list[Section]`.

- [ ] **Step 1: Implement (registration verified by Task 4's `test_builtin_cards_registered`)**

```python
# scripts/_compare/report_cards/parca.py
"""`parca` card — ParCa / initial-state match for one config."""
from __future__ import annotations

from scripts._compare.report_cards import report_card, CardContext, Section


@report_card("parca")
def parca_card(ctx: CardContext) -> Section:
    from scripts.comparison_report_card import parca_section
    # parca_section takes the cond_data map; build a single-cond slice.
    sec = parca_section({ctx.config_name: (ctx.per_obs, ctx.plot_trajs, ctx.v2_bounds)})
    sec["anchor"] = f"{ctx.config_name}-parca"
    return sec
```

```python
# scripts/_compare/report_cards/config_diff.py
"""`config_diff` card — vEcoli vs v2ecoli config comparison for one config."""
from __future__ import annotations

from scripts._compare.report_cards import report_card, CardContext, Section


@report_card("config_diff")
def config_diff_card(ctx: CardContext) -> list[Section]:
    from scripts.comparison_report_card import config_sections_for
    return config_sections_for(ctx.config_name, ctx.v2_dir, ctx.ve_dir)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_report_cards.py -k builtin -q`
Expected: PASS (all four names registered).

- [ ] **Step 3: Commit**

```bash
git add scripts/_compare/report_cards/parca.py scripts/_compare/report_cards/config_diff.py
git commit -m "feat(cards): parca + config_diff cards"
```

---

## Task 6: Manifest-driven report assembly

**Files:**
- Modify: `scripts/comparison_report_card.py` (`main()`, ~704-830)
- Test: `tests/test_report_cards.py` (assemble-from-manifest unit)

**Interfaces:**
- Consumes: `report_cards.render(name, ctx)`, `report_cards.CardContext`, the existing `build(conds, ...) -> cond_data`, `overview_section(cond_data)`.
- Produces: a `--manifest <path>` arg; when given, assembly = `[overview] + for each config (in manifest order): for each assigned card: render(card, ctx).sections`. The legacy `--only`/`--local-pbg-dir` path is preserved for back-compat.

- [ ] **Step 1: Write the failing test (assembly logic, no full render)**

```python
# tests/test_report_cards.py (append)
def test_assemble_sections_mirrors_manifest(monkeypatch):
    from scripts import comparison_report_card as crc
    # one config, cards ["parca","standard"] -> overview + parca + (runs+eval)
    cond_data = {"basal": ({"cell_mass": {"ve_cells": [1.0], "v2_cells": [1.0]}}, {}, {})}
    manifest = {"configs": [{"config": "configs/cond_basal.json",
                             "cards": ["parca", "standard"]}]}
    secs = crc.assemble_from_manifest(
        manifest, cond_data,
        conds={"basal": ("v2dir", "vedir")},
        config_names={"configs/cond_basal.json": "basal"})
    titles = [s["title"] for s in secs]
    assert titles[0].startswith("Overview")
    assert any("ParCa" in t or "parca" in t.lower() for t in titles)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_report_cards.py -k assemble -q`
Expected: FAIL with `AttributeError: ... has no attribute 'assemble_from_manifest'`.

- [ ] **Step 3: Implement `assemble_from_manifest` + wire `--manifest`**

```python
# scripts/comparison_report_card.py — new function
def assemble_from_manifest(manifest, cond_data, conds, config_names):
    """Overview + per-config assigned-card sections, mirroring the manifest.

    config_names maps a manifest config path -> the condition key used in
    cond_data/conds (the runner names stores by condition).
    """
    from scripts._compare import report_cards as rc
    default_cards = (manifest.get("defaults", {}) or {}).get("cards") or ["standard"]
    overview = overview_section(cond_data); overview["nav_group"] = "Overall"
    sections = [overview]
    for entry in manifest.get("configs", []):
        name = config_names[entry["config"]]
        if name not in cond_data:
            continue
        per_obs, plot_trajs, v2_bounds = cond_data[name]
        v2_dir, ve_dir = conds.get(name, ("", ""))
        ctx = rc.CardContext(config_name=name, variant=0, v2_dir=v2_dir,
                             ve_dir=ve_dir, seeds=0, gens=0, per_obs=per_obs,
                             plot_trajs=plot_trajs, v2_bounds=v2_bounds, config={})
        for card in (entry.get("cards") or default_cards):
            for sec in rc.render(card, ctx):
                sec["nav_group"] = name
                sections.append(sec)
    return sections
```

Then in `main()`: add `p.add_argument("--manifest", default=None)`. When `args.manifest` is set, load it, build `conds`/`cond_data` exactly as the `--local-pbg-dir` branch does (per-config dirs under `args.out`), derive `config_names` by reading each config's `condition` field (via `config_adapter.resolve_vecoli_config_local`), and set `sections = assemble_from_manifest(manifest, cond_data, conds, config_names)` instead of the fixed per-condition loop. Leave the existing branches intact for back-compat.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_report_cards.py -k assemble -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/comparison_report_card.py tests/test_report_cards.py
git commit -m "feat(report): manifest-driven per-config card assembly"
```

---

## Task 7: Example manifests + a 4x4 baseline config

**Files:**
- Create: `comparison.5cond_1x4.json`, `comparison.baseline_4x4_statistical.json`, `configs/cond_basal_4x4.json`
- Test: `tests/test_read_spec.py` (parse the example manifests)

**Interfaces:**
- Consumes: `config_rows` (Task 2).
- Produces: two tracked manifests + a 4x4 baseline config that inherits the basal condition config.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_read_spec.py (append)
import json, pathlib  # noqa: E402
ROOT = pathlib.Path(__file__).resolve().parent.parent

def test_example_manifest_5cond_parses():
    spec = json.loads((ROOT / "comparison.5cond_1x4.json").read_text())
    rows = list(rs.config_rows(spec))
    assert len(rows) == 5
    assert all(cards == "standard" for _, cards in rows)

def test_example_manifest_baseline_statistical_parses():
    spec = json.loads((ROOT / "comparison.baseline_4x4_statistical.json").read_text())
    rows = list(rs.config_rows(spec))
    assert rows == [("configs/cond_basal_4x4.json", "statistical")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_read_spec.py -k example_manifest -q`
Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Create the files**

```json
// comparison.5cond_1x4.json
{
  "v2ecoli": { "repo": "https://github.com/vivarium-collective/v2ecoli", "commit": "" },
  "vecoli":  { "repo": "https://github.com/CovertLab/vEcoli", "commit": "" },
  "defaults": { "cards": ["standard"] },
  "configs": [
    { "config": "configs/cond_basal.json" },
    { "config": "configs/cond_with_aa.json" },
    { "config": "configs/cond_succinate.json" },
    { "config": "configs/cond_no_oxygen.json" },
    { "config": "configs/cond_acetate.json" }
  ],
  "report": { "out": "out/report_5cond_1x4", "title": "v2ecoli ↔ vEcoli — 5 conditions (1×4)" }
}
```

```json
// comparison.baseline_4x4_statistical.json
{
  "v2ecoli": { "repo": "https://github.com/vivarium-collective/v2ecoli", "commit": "" },
  "vecoli":  { "repo": "https://github.com/CovertLab/vEcoli", "commit": "" },
  "defaults": { "cards": ["standard"] },
  "configs": [
    { "config": "configs/cond_basal_4x4.json", "cards": ["statistical"] }
  ],
  "report": { "out": "out/report_baseline_4x4", "title": "v2ecoli ↔ vEcoli — baseline (4×4) statistical" }
}
```

```json
// configs/cond_basal_4x4.json — 4 seeds × 4 gens baseline (inherits the basal condition config)
{
  "inherit_from": ["cond_basal.json"],
  "n_init_sims": 4,
  "generations": 4
}
```

NOTE: confirm `configs/cond_basal.json` (and the other `cond_*.json`) exist in the repo root `configs/` (they are referenced by the legacy `comparison_spec.json`); if the per-condition configs live elsewhere, update the manifest paths to match.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_read_spec.py -k example_manifest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add comparison.5cond_1x4.json comparison.baseline_4x4_statistical.json configs/cond_basal_4x4.json tests/test_read_spec.py
git commit -m "feat(harness): example manifests (5cond 1x4 standard; baseline 4x4 statistical)"
```

---

## Task 8: Retire the duplicated seeds/gens path

**Files:**
- Modify: `scripts/_read_spec.py` (remove `condition_rows`/`_resolve` after the runner is migrated), `scripts/comparison_harness.sh` (call `configs` mode; read seeds/gens from each config via the runner), `comparison_spec.json` (convert to the new `configs[]` schema).

NOTE: This task is the migration cut-over and touches the bash orchestrator + the runner's seed/gen wiring. Do it LAST, only after Tasks 1-7 are green, and verify with a 1-seed smoke run end-to-end before removing the legacy `conditions` mode. Keep `condition_rows` until `comparison_harness.sh` no longer calls the `conditions` mode.

- [ ] **Step 1:** Convert `comparison_spec.json` to `{v2ecoli, vecoli, defaults:{cards}, configs:[{config, cards}]}`.
- [ ] **Step 2:** Update `comparison_harness.sh` to call `_read_spec.py <spec> configs` and, per config, read `(seeds, gens)` from the config (the runner already has the fork; have it print/resolve via `config_run_shape`).
- [ ] **Step 3:** Smoke: `bash scripts/run_local_4x4x5.sh 1 1 200 out/smoke` equivalent driven by `comparison.5cond_1x4.json` produces a report with overview + 5 per-config standard cards.
- [ ] **Step 4:** Remove `condition_rows`/`_resolve`/the `conditions` CLI mode; update `tests/test_read_spec.py`.
- [ ] **Step 5:** Commit `refactor(harness): retire duplicated seeds/gens; configs are source of truth`.

---

## Self-Review

**Spec coverage:** manifest schema (Tasks 2,7,8) ✓; seeds/gens from config (Tasks 1,8) ✓; report-card registry package (Tasks 3,4,5) ✓; statistical=Chris's card (Task 4) ✓; manifest-mirroring report (Task 6) ✓; example manifests (Task 7) ✓; `docs/report_cards/` output convention (unchanged — render_html already writes there; verify in Task 6 smoke). **Variants = Plan B** (explicitly deferred). 

**Placeholder scan:** Task 8 is intentionally coarser (a migration cut-over over bash + runner wiring) and carries explicit NOTEs to verify before deleting; Tasks 1-7 are fully concrete. The `per_obs` sub-key names (`ve_cells`/`v2_cells`) are flagged in Task 4 to confirm against `comparison_report_card.build()` before coding.

**Type consistency:** `CardContext`/`Section` fields match across Tasks 3-6; `config_run_shape -> (int,int)` matches Task 8 usage; `build_report_card` call matches its real signature (Task 4).

## Plan B (separate plan, after A is green)

Full variant execution: read each config's `variants` dict, expand the matrix in `run_comparison_ensemble.py`, run each `(config, variant)` through both engines into `<out>/<config>/variant_<i>/`, and extend `assemble_from_manifest` + `overview_section` with a variant dimension (config → variant → cards).
