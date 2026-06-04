# vEcoli ↔ v2ecoli Comparison Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/compare_harness.py` in v2ecoli that runs both engines (ParCa + a 2-gen sim) from a single vEcoli config and emits one two-column HTML report comparing `sim_data` and simulation dynamics.

**Architecture:** A thin CLI (`scripts/compare_harness.py`) over a `scripts/_compare/` package: `config_adapter` (vEcoli→v2 mapping + schema diff), `orchestrator` (run/cache both engines via subprocess), `parca_section` (wraps existing `parca_compare.py` + a final-`sim_data` diff), `sim_section` (read both parquet outputs, compute observables, stats, plots), `report` (HTML assembly). Pure logic is TDD'd; heavy subprocess runs are exercised by a manually-gated end-to-end test.

**Tech Stack:** Python 3.12, numpy, scipy.stats (KS), matplotlib (Agg), DuckDB/parquet via each repo's `read_stacked_columns`. Two virtualenvs: v2ecoli `.venv` (harness runs here) and vEcoli `/Users/eranagmon/code/vEcoli/.venv` (shelled out to for vEcoli runs).

---

## Confirmed facts (from codebase inspection — use these verbatim)

- **vEcoli python:** `/Users/eranagmon/code/vEcoli/.venv/bin/python` (3.12.9)
- **v2ecoli python:** `/Users/eranagmon/code/v2ecoli/.venv/bin/python`
- **vEcoli ParCa:** `python runscripts/parca.py --config <cfg.json> --outdir <DIR> --save-intermediates --intermediates-directory <DIR>` → `<DIR>/kb/sim_data.cPickle` + intermediates `<DIR>/.../sim_data_<stub>.cPickle`, `cell_specs_<stub>.cPickle`. Run from `cwd=/Users/eranagmon/code/vEcoli`.
- **v2ecoli ParCa:** `v2ecoli-parca --mode full -o <DIR> --cache-dir <CACHE>` → `<DIR>/checkpoint_step_N.pkl` (N=1..9) + `<DIR>/runtimes.json`.
- **Existing comparator:** `scripts/parca_compare.py --v2parca-outdir <DIR> --original-intermediates <DIR> -o <HTML>`. Its `STEPS`, `SCALARS`, `DISTRIBUTIONS` tables (lines ~63–110) enumerate the sim_data attr paths already diffed.
- **vEcoli sim:** `python ecoli/experiments/ecoli_master_sim.py --config <cfg.json> --generations 2 --emitter parquet --emitter_arg out_dir=<DIR> --sim_data_path <sim_data.cPickle>` (run from `cwd=/Users/eranagmon/code/vEcoli`). Emits parquet under `<DIR>/<experiment_id>/history/**/*.pq`.
- **v2ecoli sim:** `v2ecoli-workflow --config <cfg.json> --out <DIR>`; parquet-emitting multigen path is `v2ecoli/library/parquet_run.py:run_multigen_parquet`, output readable via `read_parquet(<DIR>/<experiment_id>/history/**/*.pq)`.
- **Shared reader:** both repos expose `read_stacked_columns` — vEcoli at `ecoli.library.parquet_emitter`, v2ecoli at `v2ecoli.library.parquet_emitter` (re-export of `pbg_emitters`). Same parquet layout on both sides.
- **vEcoli config inheritance:** `runscripts/workflow.py:load_config_with_inheritance(path)`; vEcoli configs may set `inherit_from`. Default sim config keys live in `configs/default.json`.
- **v2ecoli config loader:** `v2ecoli/workflow/config.py:load_config_with_inheritance` already claims to load "vEcoli-style" configs.

## File structure

- Create `scripts/_compare/__init__.py` — package marker.
- Create `scripts/_compare/config_adapter.py` — resolve vEcoli config, `schema_diff`, `translate_vecoli_config`.
- Create `scripts/_compare/cache.py` — `cache_key`, `is_stale`, cache dir helpers.
- Create `scripts/_compare/orchestrator.py` — subprocess wrappers `run_vecoli_parca`, `run_v2_parca`, `run_vecoli_sim`, `run_v2_sim`, each cache-aware.
- Create `scripts/_compare/parca_section.py` — `final_sim_data_diff` + call into `parca_compare.py`.
- Create `scripts/_compare/stats.py` — `compare_series` (tolerance + KS → verdict).
- Create `scripts/_compare/sim_section.py` — `read_observables`, `compare_observables`, trajectory plots.
- Create `scripts/_compare/report.py` — `render_report(sections) -> html`.
- Create `scripts/compare_harness.py` — CLI tying stages together.
- Create tests under `tests/compare/`: `test_config_adapter.py`, `test_cache.py`, `test_stats.py`, `test_report.py`, `test_sim_section_reader.py`, `test_end_to_end.py` (gated).

Run all commands below from `cwd=/Users/eranagmon/code/v2ecoli` unless stated. Python is `.venv/bin/python`.

---

### Task 0: Scaffold package + tests dir

**Files:**
- Create: `scripts/_compare/__init__.py`
- Create: `tests/compare/__init__.py`

- [ ] **Step 1: Create package markers**

`scripts/_compare/__init__.py`:
```python
"""Internal modules for the vEcoli<->v2ecoli comparison harness."""
```

`tests/compare/__init__.py`:
```python
```

- [ ] **Step 2: Commit**

```bash
git add scripts/_compare/__init__.py tests/compare/__init__.py
git commit -m "chore(compare): scaffold comparison-harness package"
```

---

### Task 1: Config adapter — schema diff

**Files:**
- Create: `scripts/_compare/config_adapter.py`
- Test: `tests/compare/test_config_adapter.py`

- [ ] **Step 1: Write the failing test**

`tests/compare/test_config_adapter.py`:
```python
from scripts._compare.config_adapter import schema_diff


def test_schema_diff_partitions_keys():
    vecoli = {"experiment_id": "x", "generations": 2, "emitter": "parquet",
              "analysis_options": {"single": {}}}
    v2 = {"experiment_id": "x", "generations": 2, "cache_dir": "out/cache",
          "analysis_options": {"multiseed": {}}}
    d = schema_diff(vecoli, v2)
    assert d["only_in_vecoli"] == ["emitter"]
    assert d["only_in_v2"] == ["cache_dir"]
    # shared key with differing value is reported with both values
    assert d["different"]["analysis_options"] == (
        {"single": {}}, {"multiseed": {}})
    # shared key with equal value is NOT reported as different
    assert "experiment_id" not in d["different"]
    assert "generations" not in d["different"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/compare/test_config_adapter.py::test_schema_diff_partitions_keys -v`
Expected: FAIL — `ModuleNotFoundError: scripts._compare.config_adapter`

- [ ] **Step 3: Write minimal implementation**

`scripts/_compare/config_adapter.py`:
```python
"""Resolve a vEcoli config and translate/diff it against v2ecoli's schema."""
from __future__ import annotations

from typing import Any


def schema_diff(vecoli: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    """Partition keys: only-in-vEcoli, only-in-v2, shared-but-different.

    ``different`` maps each differing shared key to a (vecoli_value,
    v2_value) tuple. Only top-level keys are compared.
    """
    vkeys, v2keys = set(vecoli), set(v2)
    different = {
        k: (vecoli[k], v2[k])
        for k in (vkeys & v2keys)
        if vecoli[k] != v2[k]
    }
    return {
        "only_in_vecoli": sorted(vkeys - v2keys),
        "only_in_v2": sorted(v2keys - vkeys),
        "different": different,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/compare/test_config_adapter.py::test_schema_diff_partitions_keys -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/config_adapter.py tests/compare/test_config_adapter.py
git commit -m "feat(compare): config schema_diff"
```

---

### Task 2: Config adapter — translate vEcoli → v2ecoli

**Files:**
- Modify: `scripts/_compare/config_adapter.py`
- Test: `tests/compare/test_config_adapter.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/compare/test_config_adapter.py`:
```python
from scripts._compare.config_adapter import translate_vecoli_config


def test_translate_maps_known_keys_and_drops_vecoli_only():
    vecoli = {
        "experiment_id": "two_generations",
        "generations": 2,
        "n_init_sims": 2,
        "single_daughters": True,
        "emitter": "parquet",
        "emitter_arg": {"out_dir": "out"},
        "parca_options": {"cpus": 3, "memory_gb": 6},
        "fail_at_max_duration": True,
        "sim_data_path": None,
        "analysis_options": {"single": {"mass_fraction_summary": {}}},
    }
    v2 = translate_vecoli_config(vecoli)
    # shared keys carried through unchanged
    assert v2["experiment_id"] == "two_generations"
    assert v2["generations"] == 2
    assert v2["n_init_sims"] == 2
    assert v2["single_daughters"] is True
    assert v2["analysis_options"] == {"single": {"mass_fraction_summary": {}}}
    # vEcoli-only keys are dropped from the v2 config body
    for dropped in ("emitter", "emitter_arg", "parca_options",
                    "fail_at_max_duration", "sim_data_path"):
        assert dropped not in v2
    # the mapping is recorded for the report
    assert "emitter" in v2["_dropped_vecoli_keys"]
    assert v2["_dropped_vecoli_keys"]["parca_options"] == {"cpus": 3,
                                                            "memory_gb": 6}


def test_translate_sets_lineage_seed_default_when_absent():
    v2 = translate_vecoli_config({"experiment_id": "x", "generations": 1})
    assert v2["lineage_seed"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/compare/test_config_adapter.py -v`
Expected: FAIL — `ImportError: cannot import name 'translate_vecoli_config'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/_compare/config_adapter.py`:
```python
# Keys vEcoli sets that v2ecoli's workflow config does not consume. Their
# values are preserved under ``_dropped_vecoli_keys`` for the report so the
# mapping is explicit rather than silent.
_VECOLI_ONLY = (
    "emitter",
    "emitter_arg",
    "parca_options",
    "fail_at_max_duration",
    "suffix_time",
    "sim_data_path",
)

# v2ecoli keys with defaults applied when the vEcoli config omits them.
_V2_DEFAULTS = {
    "lineage_seed": 0,
    "single_daughters": True,
}


def translate_vecoli_config(vecoli: dict[str, Any]) -> dict[str, Any]:
    """Map a resolved vEcoli config to a v2ecoli workflow config.

    Shared keys pass through unchanged; vEcoli-only keys are removed from
    the config body and recorded under ``_dropped_vecoli_keys``; missing
    v2ecoli keys get defaults from ``_V2_DEFAULTS``.
    """
    v2: dict[str, Any] = {
        k: v for k, v in vecoli.items() if k not in _VECOLI_ONLY
    }
    v2["_dropped_vecoli_keys"] = {
        k: vecoli[k] for k in _VECOLI_ONLY if k in vecoli
    }
    for k, default in _V2_DEFAULTS.items():
        v2.setdefault(k, default)
    return v2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/compare/test_config_adapter.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/config_adapter.py tests/compare/test_config_adapter.py
git commit -m "feat(compare): translate_vecoli_config adapter"
```

---

### Task 3: Config adapter — resolve vEcoli inheritance via vEcoli venv

**Files:**
- Modify: `scripts/_compare/config_adapter.py`
- Test: `tests/compare/test_config_adapter.py`

This shells out to vEcoli's own loader so `inherit_from` is resolved faithfully.

- [ ] **Step 1: Write the failing test**

Append to `tests/compare/test_config_adapter.py`:
```python
import json
from scripts._compare.config_adapter import resolve_vecoli_config


def test_resolve_vecoli_config_invokes_vecoli_loader(monkeypatch, tmp_path):
    captured = {}

    def fake_check_output(cmd, cwd=None, text=None):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return json.dumps({"experiment_id": "resolved", "generations": 2})

    monkeypatch.setattr(
        "scripts._compare.config_adapter.subprocess.check_output",
        fake_check_output,
    )
    cfg = resolve_vecoli_config("/some/two_generations.json")
    assert cfg == {"experiment_id": "resolved", "generations": 2}
    # runs vEcoli's python from the vEcoli repo
    assert captured["cwd"].endswith("/vEcoli")
    assert captured["cmd"][0].endswith("/vEcoli/.venv/bin/python")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/compare/test_config_adapter.py::test_resolve_vecoli_config_invokes_vecoli_loader -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_vecoli_config'`

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/_compare/config_adapter.py` (add `import subprocess`, `import json` at top):
```python
import json
import subprocess

VECOLI_REPO = "/Users/eranagmon/code/vEcoli"
VECOLI_PYTHON = f"{VECOLI_REPO}/.venv/bin/python"


def resolve_vecoli_config(config_path: str) -> dict[str, Any]:
    """Resolve a vEcoli config (honoring ``inherit_from``) using vEcoli's
    own loader, returning the fully-merged dict."""
    snippet = (
        "import json,sys;"
        "from runscripts.workflow import load_config_with_inheritance;"
        "json.dump(load_config_with_inheritance(sys.argv[1]), sys.stdout)"
    )
    out = subprocess.check_output(
        [VECOLI_PYTHON, "-c", snippet, config_path],
        cwd=VECOLI_REPO,
        text=True,
    )
    return json.loads(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/compare/test_config_adapter.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Verify the real vEcoli loader works end-to-end (no mock)**

Run:
```bash
.venv/bin/python -c "from scripts._compare.config_adapter import resolve_vecoli_config; import json; print(json.dumps(resolve_vecoli_config('/Users/eranagmon/code/vEcoli/configs/two_generations.json'))[:200])"
```
Expected: prints a JSON object starting `{"experiment_id": "two_generations"...`. If `load_config_with_inheritance` is not importable as `runscripts.workflow`, adjust the import path in the snippet to match vEcoli's module layout (check with `cd /Users/eranagmon/code/vEcoli && .venv/bin/python -c "import runscripts.workflow"`).

- [ ] **Step 6: Commit**

```bash
git add scripts/_compare/config_adapter.py tests/compare/test_config_adapter.py
git commit -m "feat(compare): resolve vEcoli config via its own loader"
```

---

### Task 4: Cache keys + staleness

**Files:**
- Create: `scripts/_compare/cache.py`
- Test: `tests/compare/test_cache.py`

- [ ] **Step 1: Write the failing test**

`tests/compare/test_cache.py`:
```python
from scripts._compare.cache import cache_key, is_stale


def test_cache_key_is_deterministic_and_sensitive():
    a = cache_key({"generations": 2}, commit="abc", mode="full")
    b = cache_key({"generations": 2}, commit="abc", mode="full")
    c = cache_key({"generations": 3}, commit="abc", mode="full")
    d = cache_key({"generations": 2}, commit="def", mode="full")
    assert a == b
    assert a != c
    assert a != d
    assert len(a) == 16  # short hex digest


def test_is_stale_true_when_marker_missing(tmp_path):
    assert is_stale(tmp_path / "nope") is True


def test_is_stale_false_when_done_marker_present(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / ".done").write_text("ok")
    assert is_stale(d) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/compare/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts._compare.cache`

- [ ] **Step 3: Write minimal implementation**

`scripts/_compare/cache.py`:
```python
"""Content-addressed caching for expensive engine runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(config: dict[str, Any], *, commit: str, mode: str) -> str:
    """Stable 16-hex-char digest of (config, engine commit, mode)."""
    payload = json.dumps(
        {"config": config, "commit": commit, "mode": mode},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def is_stale(run_dir: Path) -> bool:
    """A run dir is fresh only if it exists and holds a ``.done`` marker."""
    return not (Path(run_dir) / ".done").exists()


def mark_done(run_dir: Path) -> None:
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    (Path(run_dir) / ".done").write_text("ok")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/compare/test_cache.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/cache.py tests/compare/test_cache.py
git commit -m "feat(compare): content-addressed run cache"
```

---

### Task 5: Stats — tolerance + KS verdict

**Files:**
- Create: `scripts/_compare/stats.py`
- Test: `tests/compare/test_stats.py`

- [ ] **Step 1: Write the failing test**

`tests/compare/test_stats.py`:
```python
import numpy as np
from scripts._compare.stats import compare_series


def test_identical_series_within_tol():
    x = np.array([1.0, 2.0, 3.0])
    r = compare_series(x, x.copy(), rel_tol=1e-6)
    assert r["verdict"] == "within_tol"
    assert r["max_rel"] == 0.0


def test_small_drift_flagged_as_drift():
    x = np.array([1.0, 2.0, 3.0])
    y = x * 1.01  # 1% off, above a 1e-3 tol but not wildly different
    r = compare_series(x, y, rel_tol=1e-3)
    assert r["verdict"] == "drift"
    assert 0.009 < r["max_rel"] < 0.011


def test_large_difference_flagged_as_mismatch():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    r = compare_series(x, y, rel_tol=1e-3, mismatch_rel=0.5)
    assert r["verdict"] == "mismatch"


def test_shape_mismatch_returns_not_compared():
    r = compare_series(np.array([1.0, 2.0]), np.array([1.0]), rel_tol=1e-3)
    assert r["verdict"] == "not_compared"
    assert "shape" in r["reason"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/compare/test_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts._compare.stats`

- [ ] **Step 3: Write minimal implementation**

`scripts/_compare/stats.py`:
```python
"""Per-metric comparison: relative error + KS, mapped to a verdict."""
from __future__ import annotations

from typing import Any

import numpy as np

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy optional
    _HAVE_SCIPY = False


def compare_series(
    a,
    b,
    *,
    rel_tol: float,
    mismatch_rel: float = 0.5,
) -> dict[str, Any]:
    """Compare two numeric arrays.

    Verdicts: ``within_tol`` (max relative error <= rel_tol),
    ``mismatch`` (max relative error >= mismatch_rel), ``drift``
    (between the two), ``not_compared`` (shape mismatch / empty).
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        return {"verdict": "not_compared",
                "reason": f"shape {a.shape} != {b.shape}"}
    if a.size == 0:
        return {"verdict": "not_compared", "reason": "empty"}

    denom = np.maximum(np.abs(a), 1e-30)
    rel = np.abs(a - b) / denom
    max_rel = float(np.max(rel))
    max_abs = float(np.max(np.abs(a - b)))

    ks_p = None
    if _HAVE_SCIPY and a.size >= 2:
        ks_p = float(_scipy_stats.ks_2samp(a, b).pvalue)

    if max_rel <= rel_tol:
        verdict = "within_tol"
    elif max_rel >= mismatch_rel:
        verdict = "mismatch"
    else:
        verdict = "drift"

    return {"verdict": verdict, "max_rel": max_rel,
            "max_abs": max_abs, "ks_p": ks_p}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/compare/test_stats.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/stats.py tests/compare/test_stats.py
git commit -m "feat(compare): tolerance+KS verdict for series"
```

---

### Task 6: Report renderer

**Files:**
- Create: `scripts/_compare/report.py`
- Test: `tests/compare/test_report.py`

- [ ] **Step 1: Write the failing test**

`tests/compare/test_report.py`:
```python
from scripts._compare.report import render_report


def _section(title, rows):
    return {"title": title, "rows": rows}


def test_render_report_two_columns_and_badges():
    sections = [
        _section("Config & schema diff", [
            {"label": "emitter", "left": "parquet", "right": "(dropped)",
             "verdict": "drift"},
        ]),
        _section("ParCa / sim_data", [
            {"label": "mass.avg_cell_dry_mass", "left": "2.5e-13",
             "right": "2.5e-13", "verdict": "within_tol"},
        ]),
    ]
    html = render_report(sections, title="vEcoli vs v2ecoli")

    assert "<html" in html.lower()
    # two column headers present
    assert "vEcoli" in html and "v2ecoli" in html
    # each section title rendered
    assert "Config &amp; schema diff" in html or "Config & schema diff" in html
    assert "ParCa / sim_data" in html
    # verdict drives a CSS class
    assert "verdict-within_tol" in html
    assert "verdict-drift" in html
    # self-contained: no external http(s) asset links
    assert "http://" not in html and "https://" not in html


def test_render_report_handles_not_compared_rows():
    sections = [{"title": "Sim", "rows": [
        {"label": "ribosome", "left": "n/a", "right": "n/a",
         "verdict": "not_compared", "reason": "missing on v2 side"}]}]
    html = render_report(sections, title="t")
    assert "missing on v2 side" in html
    assert "verdict-not_compared" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/compare/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts._compare.report`

- [ ] **Step 3: Write minimal implementation**

`scripts/_compare/report.py`:
```python
"""Self-contained two-column HTML report renderer."""
from __future__ import annotations

import html as _html
from typing import Any

_CSS = """
body { font-family: -apple-system, sans-serif; margin: 0; }
nav { position: sticky; top: 0; background: #fff; border-bottom: 1px solid #ccc;
      padding: 8px; }
nav a { margin-right: 12px; }
section { padding: 16px; border-bottom: 1px solid #eee; }
table { border-collapse: collapse; width: 100%; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #f0f0f0;
         vertical-align: top; }
.col-left { width: 40%; } .col-right { width: 40%; }
.badge { padding: 1px 6px; border-radius: 4px; font-size: 11px; color: #fff; }
.verdict-within_tol .badge { background: #2e7d32; }
.verdict-drift .badge { background: #ef6c00; }
.verdict-mismatch .badge { background: #c62828; }
.verdict-not_compared .badge { background: #757575; }
"""


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in s.lower())


def _row_html(row: dict[str, Any]) -> str:
    verdict = row.get("verdict", "not_compared")
    reason = row.get("reason", "")
    label = _html.escape(str(row.get("label", "")))
    left = _html.escape(str(row.get("left", "")))
    right = _html.escape(str(row.get("right", "")))
    reason_html = (f'<div class="reason">{_html.escape(reason)}</div>'
                   if reason else "")
    return (
        f'<tr class="verdict-{verdict}">'
        f'<td>{label}<span class="badge">{verdict}</span>{reason_html}</td>'
        f'<td class="col-left">{left}</td>'
        f'<td class="col-right">{right}</td>'
        f'</tr>'
    )


def _section_html(section: dict[str, Any]) -> str:
    title = section["title"]
    rows = "".join(_row_html(r) for r in section.get("rows", []))
    extra = section.get("html", "")  # for embedded plots (data: URIs)
    return (
        f'<section id="{_slug(title)}">'
        f'<h2>{_html.escape(title)}</h2>'
        f'<table><thead><tr><th>metric</th>'
        f'<th class="col-left">vEcoli</th>'
        f'<th class="col-right">v2ecoli</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>{extra}</section>'
    )


def render_report(sections: list[dict[str, Any]], *, title: str) -> str:
    """Render a list of sections to a single self-contained HTML string."""
    nav = "".join(
        f'<a href="#{_slug(s["title"])}">{_html.escape(s["title"])}</a>'
        for s in sections
    )
    body = "".join(_section_html(s) for s in sections)
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        f'<title>{_html.escape(title)}</title><style>{_CSS}</style></head>'
        f'<body><nav>{nav}</nav><h1>{_html.escape(title)}</h1>'
        f'{body}</body></html>'
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/compare/test_report.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/report.py tests/compare/test_report.py
git commit -m "feat(compare): two-column HTML report renderer"
```

---

### Task 7: Final sim_data diff (ParCa section)

**Files:**
- Create: `scripts/_compare/parca_section.py`
- Test: `tests/compare/test_parca_section.py`

Reuses the `SCALARS`/`DISTRIBUTIONS` attr-path tables from `parca_compare.py` and the existing `_reach`-style attribute walk to diff two final `sim_data` objects into report rows.

- [ ] **Step 1: Write the failing test**

`tests/compare/test_parca_section.py`:
```python
from types import SimpleNamespace

import numpy as np

from scripts._compare.parca_section import final_sim_data_diff


def _fake_sim_data(dry_mass, expression):
    # mirrors the attr paths used by parca_compare.SCALARS / DISTRIBUTIONS
    return SimpleNamespace(
        mass=SimpleNamespace(
            avg_cell_dry_mass=dry_mass,
            avg_cell_dry_mass_init=dry_mass,
            avg_cell_water_mass_init=0.0,
            fitAvgSolubleTargetMolMass=0.0,
        ),
        constants=SimpleNamespace(darkATP=1.0),
        process=SimpleNamespace(
            transcription=SimpleNamespace(
                rna_expression={"basal": expression},
            ),
        ),
    )


def test_final_sim_data_diff_flags_matching_and_drifting():
    left = _fake_sim_data(2.5, np.array([0.1, 0.2, 0.3]))
    right = _fake_sim_data(2.5, np.array([0.1, 0.2, 0.3]) * 1.01)
    rows = final_sim_data_diff(left, right, rel_tol=1e-3)
    by_label = {r["label"]: r for r in rows}
    assert by_label["mass.avg_cell_dry_mass"]["verdict"] == "within_tol"
    assert by_label["RNA expression — basal"]["verdict"] == "drift"


def test_final_sim_data_diff_missing_attr_is_not_compared():
    left = _fake_sim_data(2.5, np.array([0.1]))
    right = SimpleNamespace()  # nothing reachable
    rows = final_sim_data_diff(left, right, rel_tol=1e-3)
    assert all(r["verdict"] == "not_compared" for r in rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/compare/test_parca_section.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts._compare.parca_section`

- [ ] **Step 3: Write minimal implementation**

`scripts/_compare/parca_section.py`:
```python
"""ParCa / sim_data comparison rows for the harness report.

Per-step diffing is delegated to the existing scripts/parca_compare.py;
this module adds a final-sim_data field-by-field diff using the same attr
paths that comparator already curates.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from scripts._compare.stats import compare_series

# Attr paths mirror scripts/parca_compare.py SCALARS + DISTRIBUTIONS.
_SCALARS = [
    ("mass.avg_cell_dry_mass_init", ("mass", "avg_cell_dry_mass_init")),
    ("mass.avg_cell_dry_mass", ("mass", "avg_cell_dry_mass")),
    ("mass.avg_cell_water_mass_init", ("mass", "avg_cell_water_mass_init")),
    ("mass.fitAvgSolubleTargetMolMass", ("mass", "fitAvgSolubleTargetMolMass")),
    ("constants.darkATP", ("constants", "darkATP")),
]
_DISTRIBUTIONS = [
    ("RNA expression — basal",
     ("process", "transcription", "rna_expression", "basal")),
]


def _reach(obj: Any, path: tuple[str, ...]):
    """Follow an attr/key path; return None if any hop is missing."""
    cur = obj
    for p in path:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            cur = getattr(cur, p, None)
    return cur


def _row(label: str, left, right, rel_tol: float) -> dict[str, Any]:
    if left is None or right is None:
        return {"label": label, "left": "n/a", "right": "n/a",
                "verdict": "not_compared",
                "reason": "attribute missing on one side"}
    r = compare_series(np.atleast_1d(left), np.atleast_1d(right),
                       rel_tol=rel_tol)
    return {"label": label,
            "left": np.array2string(np.atleast_1d(left), threshold=4),
            "right": np.array2string(np.atleast_1d(right), threshold=4),
            **r}


def final_sim_data_diff(left, right, *, rel_tol: float) -> list[dict[str, Any]]:
    """Diff curated scalar + distribution fields of two sim_data objects."""
    rows = []
    for label, path in _SCALARS + _DISTRIBUTIONS:
        rows.append(_row(label, _reach(left, path), _reach(right, path),
                         rel_tol))
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/compare/test_parca_section.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/parca_section.py tests/compare/test_parca_section.py
git commit -m "feat(compare): final sim_data field-by-field diff"
```

---

### Task 8: Sim observable reader + comparison

**Files:**
- Create: `scripts/_compare/sim_section.py`
- Test: `tests/compare/test_sim_section.py`

The reader normalizes either engine's parquet output to a dict of named numeric series. Because both engines write the same `history/**/*.pq` layout and both expose `read_stacked_columns`, the same extraction works for both. The unit test exercises the pure comparison (`compare_observables`) on in-memory dicts; the actual parquet read is covered by the gated end-to-end test (Task 10).

- [ ] **Step 1: Write the failing test**

`tests/compare/test_sim_section.py`:
```python
import numpy as np

from scripts._compare.sim_section import compare_observables, OBSERVABLES


def test_observables_cover_four_families():
    families = {o["family"] for o in OBSERVABLES}
    assert families == {"mass_growth", "molecule_counts",
                        "listeners", "division_lineage"}


def test_compare_observables_builds_rows_with_verdicts():
    left = {"dry_mass": np.array([1.0, 2.0, 4.0]),
            "growth_rate": np.array([0.1, 0.1, 0.1])}
    right = {"dry_mass": np.array([1.0, 2.0, 4.0]),
             "growth_rate": np.array([0.2, 0.2, 0.2])}
    rows = compare_observables(left, right,
                               keys=["dry_mass", "growth_rate"],
                               rel_tol=1e-3)
    by = {r["label"]: r for r in rows}
    assert by["dry_mass"]["verdict"] == "within_tol"
    assert by["growth_rate"]["verdict"] == "mismatch"


def test_compare_observables_missing_key_not_compared():
    rows = compare_observables({"a": np.array([1.0])}, {},
                               keys=["a"], rel_tol=1e-3)
    assert rows[0]["verdict"] == "not_compared"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/compare/test_sim_section.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts._compare.sim_section`

- [ ] **Step 3: Write minimal implementation**

`scripts/_compare/sim_section.py`:
```python
"""Read a sim's emitted parquet history and compare observables.

Both engines emit the same ``<out>/<experiment_id>/history/**/*.pq`` layout
and both expose ``read_stacked_columns``; the same reader works for either
side — only the importing module differs.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from scripts._compare.stats import compare_series

# Observables grouped into the four families from the design. ``column`` is
# the emitter field path (dotted) read via read_stacked_columns.
OBSERVABLES: list[dict[str, str]] = [
    {"family": "mass_growth", "key": "dry_mass",
     "column": "listeners__mass__dry_mass"},
    {"family": "mass_growth", "key": "cell_mass",
     "column": "listeners__mass__cell_mass"},
    {"family": "mass_growth", "key": "growth_rate",
     "column": "listeners__mass__instantaneous_growth_rate"},
    {"family": "molecule_counts", "key": "bulk_counts",
     "column": "bulk"},
    {"family": "listeners", "key": "active_ribosomes",
     "column": "listeners__ribosome_data__effective_elongation_rate"},
    {"family": "listeners", "key": "active_rnap",
     "column": "listeners__rnap_data__active_rnap_coordinates"},
    {"family": "division_lineage", "key": "division_time",
     "column": "listeners__mass__cell_mass"},  # divisions inferred from drops
]


def read_observables(
    out_dir: str,
    experiment_id: str,
    reader: Callable[..., Any],
    keys: list[str],
) -> dict[str, np.ndarray]:
    """Read named observable series from a parquet history dir.

    ``reader`` is the engine's ``read_stacked_columns`` (vEcoli or v2ecoli).
    A column that is absent/unreadable is simply omitted from the result so
    the comparison reports it as ``not_compared``.
    """
    import glob
    import os

    history_glob = os.path.join(out_dir, experiment_id, "history", "**", "*.pq")
    files = glob.glob(history_glob, recursive=True)
    by_key = {o["key"]: o for o in OBSERVABLES}
    out: dict[str, np.ndarray] = {}
    for key in keys:
        col = by_key[key]["column"]
        try:
            arr = reader(files, [col])
            out[key] = np.asarray(arr).ravel()
        except Exception:
            continue
    return out


def compare_observables(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    keys: list[str],
    rel_tol: float,
) -> list[dict[str, Any]]:
    """Build report rows comparing each requested observable key."""
    rows = []
    for key in keys:
        l, r = left.get(key), right.get(key)
        if l is None or r is None:
            rows.append({"label": key, "left": "n/a", "right": "n/a",
                         "verdict": "not_compared",
                         "reason": "observable missing on one side"})
            continue
        res = compare_series(l, r, rel_tol=rel_tol)
        rows.append({
            "label": key,
            "left": np.array2string(l, threshold=4),
            "right": np.array2string(r, threshold=4),
            **res,
        })
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/compare/test_sim_section.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Verify observable column names against a real emitter (no mock)**

The `column` strings in `OBSERVABLES` are best-effort field paths. Before the end-to-end run, confirm them against an actual emitted parquet:
```bash
cd /Users/eranagmon/code/vEcoli && .venv/bin/python -c "from ecoli.library.parquet_emitter import field_metadata; print('check field names exist in a real run output')"
```
If a column name is wrong, fix the `column` value in `OBSERVABLES` (the comparison degrades to `not_compared` rather than crashing, so this is safe to defer to first real run). Document any corrected names in a code comment.

- [ ] **Step 6: Commit**

```bash
git add scripts/_compare/sim_section.py tests/compare/test_sim_section.py
git commit -m "feat(compare): sim observable reader + comparison"
```

---

### Task 9: Orchestrator — cache-aware subprocess wrappers

**Files:**
- Create: `scripts/_compare/orchestrator.py`
- Test: `tests/compare/test_orchestrator.py`

The four run functions skip work when the cache dir is fresh and otherwise shell out to the right venv/CLI with the confirmed commands. The unit test mocks `subprocess.run` and verifies the cache short-circuit and the exact argv; real runs happen in Task 10.

- [ ] **Step 1: Write the failing test**

`tests/compare/test_orchestrator.py`:
```python
from pathlib import Path

from scripts._compare import orchestrator


def test_run_v2_parca_skips_when_fresh(tmp_path, monkeypatch):
    out = tmp_path / "v2parca"
    out.mkdir()
    (out / ".done").write_text("ok")
    called = {"n": 0}
    monkeypatch.setattr(orchestrator.subprocess, "run",
                        lambda *a, **k: called.__setitem__("n", called["n"] + 1))
    result = orchestrator.run_v2_parca(out_dir=out, cache_dir=tmp_path / "c",
                                       mode="full")
    assert called["n"] == 0          # cache hit → no subprocess
    assert result == out


def test_run_v2_parca_invokes_cli_when_stale(tmp_path, monkeypatch):
    out = tmp_path / "v2parca"
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        (out).mkdir(parents=True, exist_ok=True)
        class R: returncode = 0
        return R()

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    orchestrator.run_v2_parca(out_dir=out, cache_dir=tmp_path / "c",
                              mode="full")
    assert "v2ecoli-parca" in captured["cmd"]
    assert "--mode" in captured["cmd"] and "full" in captured["cmd"]


def test_run_vecoli_parca_uses_vecoli_python_and_save_intermediates(
        tmp_path, monkeypatch):
    out = tmp_path / "vparca"
    captured = {}

    def fake_run(cmd, cwd=None, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        out.mkdir(parents=True, exist_ok=True)
        class R: returncode = 0
        return R()

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    orchestrator.run_vecoli_parca(config_path="/x/cfg.json", out_dir=out)
    assert captured["cmd"][0].endswith("/vEcoli/.venv/bin/python")
    assert "--save-intermediates" in captured["cmd"]
    assert captured["cwd"].endswith("/vEcoli")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/compare/test_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: scripts._compare.orchestrator`

- [ ] **Step 3: Write minimal implementation**

`scripts/_compare/orchestrator.py`:
```python
"""Cache-aware subprocess wrappers that run each engine's ParCa and sim."""
from __future__ import annotations

import subprocess
from pathlib import Path

from scripts._compare.cache import is_stale, mark_done

VECOLI_REPO = "/Users/eranagmon/code/vEcoli"
VECOLI_PYTHON = f"{VECOLI_REPO}/.venv/bin/python"
V2_PYTHON = ".venv/bin/python"


def _run(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd)
    if getattr(proc, "returncode", 0) != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {cmd}")


def run_v2_parca(*, out_dir: Path, cache_dir: Path, mode: str) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir):
        return out_dir
    _run(["v2ecoli-parca", "--mode", mode, "-o", str(out_dir),
          "--cache-dir", str(cache_dir)])
    mark_done(out_dir)
    return out_dir


def run_vecoli_parca(*, config_path: str, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir):
        return out_dir
    _run([VECOLI_PYTHON, "runscripts/parca.py",
          "--config", config_path,
          "--outdir", str(out_dir),
          "--save-intermediates",
          "--intermediates-directory", str(out_dir)],
         cwd=VECOLI_REPO)
    mark_done(out_dir)
    return out_dir


def run_vecoli_sim(*, config_path: str, sim_data_path: str, out_dir: Path,
                   generations: int = 2) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir):
        return out_dir
    _run([VECOLI_PYTHON, "ecoli/experiments/ecoli_master_sim.py",
          "--config", config_path,
          "--generations", str(generations),
          "--emitter", "parquet",
          "--emitter_arg", f"out_dir={out_dir}",
          "--sim_data_path", sim_data_path],
         cwd=VECOLI_REPO)
    mark_done(out_dir)
    return out_dir


def run_v2_sim(*, config_path: str, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    if not is_stale(out_dir):
        return out_dir
    _run([V2_PYTHON, "-m", "v2ecoli.workflow.run",
          "--config", config_path, "--out", str(out_dir)])
    mark_done(out_dir)
    return out_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/compare/test_orchestrator.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/_compare/orchestrator.py tests/compare/test_orchestrator.py
git commit -m "feat(compare): cache-aware engine run orchestrator"
```

---

### Task 10: CLI wiring + gated end-to-end test

**Files:**
- Create: `scripts/compare_harness.py`
- Test: `tests/compare/test_end_to_end.py`

- [ ] **Step 1: Write the CLI**

`scripts/compare_harness.py`:
```python
#!/usr/bin/env python
"""vEcoli <-> v2ecoli comparison harness.

Runs both engines from a single vEcoli config and emits a two-column HTML
report: config/schema diff, ParCa sim_data comparison, 2-gen sim dynamics.

    .venv/bin/python scripts/compare_harness.py \
        --config /Users/eranagmon/code/vEcoli/configs/two_generations.json \
        -o out/compare/report.html
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._compare import orchestrator
from scripts._compare.config_adapter import (
    resolve_vecoli_config, schema_diff, translate_vecoli_config)
from scripts._compare.parca_section import final_sim_data_diff
from scripts._compare.report import render_report
from scripts._compare.sim_section import (
    OBSERVABLES, compare_observables, read_observables)

# sim_data diffs should be tight; dynamics looser (two engines).
PARCA_REL_TOL = 1e-6
SIM_REL_TOL = 0.05


def _config_section(vecoli_cfg, v2_cfg):
    d = schema_diff(vecoli_cfg, v2_cfg)
    rows = []
    for k in d["only_in_vecoli"]:
        rows.append({"label": k, "left": json.dumps(vecoli_cfg[k]),
                     "right": "(not used by v2ecoli)", "verdict": "drift"})
    for k in d["only_in_v2"]:
        rows.append({"label": k, "left": "(added by adapter)",
                     "right": json.dumps(v2_cfg[k]), "verdict": "drift"})
    for k, (lv, rv) in d["different"].items():
        rows.append({"label": k, "left": json.dumps(lv),
                     "right": json.dumps(rv), "verdict": "drift"})
    return {"title": "Config & schema diff", "rows": rows}


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True,
                   help="Path to a vEcoli JSON config (source of truth).")
    p.add_argument("-o", "--out", default="out/compare/report.html")
    p.add_argument("--workdir", default="out/compare_harness")
    p.add_argument("--mode", default="full", choices=["full", "fast"])
    p.add_argument("--fast-plumbing", action="store_true",
                   help="ParCa --mode fast for wiring iteration ONLY; the "
                        "report is stamped NOT SCIENTIFICALLY VALID.")
    args = p.parse_args(argv)
    mode = "fast" if args.fast_plumbing else args.mode

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    # Stage 1 — config
    vecoli_cfg = resolve_vecoli_config(args.config)
    v2_cfg = translate_vecoli_config(vecoli_cfg)
    v2_cfg_path = work / "v2_config.json"
    v2_cfg_path.write_text(json.dumps(
        {k: v for k, v in v2_cfg.items() if not k.startswith("_")}))
    sections = [_config_section(vecoli_cfg, v2_cfg)]

    # Stage 2 — ParCa (both)
    v_parca = orchestrator.run_vecoli_parca(
        config_path=args.config, out_dir=work / "vecoli_parca")
    v2_parca = orchestrator.run_v2_parca(
        out_dir=work / "v2_parca", cache_dir=work / "parca_cache", mode=mode)

    # Stage 3 — ParCa / sim_data comparison
    v_sim_data = _load_pickle(v_parca / "kb" / "sim_data.cPickle")
    v2_sim_data = _load_pickle(v2_parca / "checkpoint_step_9.pkl")
    sections.append({"title": "ParCa / sim_data",
                     "rows": final_sim_data_diff(v_sim_data, v2_sim_data,
                                                 rel_tol=PARCA_REL_TOL)})

    # Stage 4 — sim (both) + dynamics
    exp_id = vecoli_cfg.get("experiment_id", "default")
    v_sim = orchestrator.run_vecoli_sim(
        config_path=args.config,
        sim_data_path=str(v_parca / "kb" / "sim_data.cPickle"),
        out_dir=work / "vecoli_sim",
        generations=int(vecoli_cfg.get("generations", 2)))
    v2_sim = orchestrator.run_v2_sim(
        config_path=str(v2_cfg_path), out_dir=work / "v2_sim")

    from ecoli.library.parquet_emitter import read_stacked_columns as v_reader  # noqa: E501
    from v2ecoli.library.parquet_emitter import read_stacked_columns as v2_reader  # noqa: E501
    keys = [o["key"] for o in OBSERVABLES]
    left = read_observables(str(v_sim), exp_id, v_reader, keys)
    right = read_observables(str(v2_sim), exp_id, v2_reader, keys)
    sections.append({"title": "2-generation sim dynamics",
                     "rows": compare_observables(left, right, keys=keys,
                                                 rel_tol=SIM_REL_TOL)})

    title = "vEcoli vs v2ecoli"
    if args.fast_plumbing:
        title += "  —  ⚠ NOT SCIENTIFICALLY VALID (fast-plumbing) ⚠"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(render_report(sections, title=title))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI wiring without running engines**

Run:
```bash
.venv/bin/python -c "import scripts.compare_harness as h; print('import ok'); h.main(['--help'])"
```
Expected: prints `import ok` then the argparse help text and exits 0. Fix any import errors before proceeding.

- [ ] **Step 3: Write the gated end-to-end test**

`tests/compare/test_end_to_end.py`:
```python
import os
import shutil
from pathlib import Path

import pytest

# Heavy: runs real ParCa (full) + a 2-gen sim on BOTH engines (hours).
# Opt in with COMPARE_E2E=1.
pytestmark = pytest.mark.skipif(
    os.environ.get("COMPARE_E2E") != "1",
    reason="set COMPARE_E2E=1 to run the full cross-engine harness",
)


def test_harness_produces_report(tmp_path):
    import scripts.compare_harness as h
    out = tmp_path / "report.html"
    h.main([
        "--config", "/Users/eranagmon/code/vEcoli/configs/two_generations.json",
        "-o", str(out),
        "--workdir", str(tmp_path / "work"),
    ])
    html = out.read_text()
    assert "ParCa / sim_data" in html
    assert "2-generation sim dynamics" in html
    assert "Config &amp; schema diff" in html or "Config & schema diff" in html
```

- [ ] **Step 4: Run the full unit suite (fast tests only)**

Run: `.venv/bin/python -m pytest tests/compare/ -v`
Expected: PASS for all non-gated tests; `test_end_to_end.py` SKIPPED (COMPARE_E2E unset).

- [ ] **Step 5: Commit**

```bash
git add scripts/compare_harness.py tests/compare/test_end_to_end.py
git commit -m "feat(compare): harness CLI + gated end-to-end test"
```

---

### Task 11: First real fast-plumbing run + fixes

**Files:**
- Modify: whichever modules surface issues (likely `sim_section.OBSERVABLES` column names, `v2_sim_data` checkpoint key, vEcoli sim driver).

- [ ] **Step 1: Run the harness in fast-plumbing mode**

Run:
```bash
.venv/bin/python scripts/compare_harness.py \
  --config /Users/eranagmon/code/vEcoli/configs/two_generations.json \
  -o out/compare/report_fastplumbing.html \
  --fast-plumbing
```
Expected: completes and writes the report. This validates plumbing only (fast ParCa is NOT scientifically valid — the report says so).

- [ ] **Step 2: Open the report and triage**

Run: `open out/compare/report_fastplumbing.html`
Verify: three sections render; config diff is sensible; note any rows that are wrongly `not_compared` (column-name or attr-path mismatches) and any stage that errored.

- [ ] **Step 3: Fix discovered issues**

Apply the smallest fix per issue. Likely candidates and their fixes:
- vEcoli `ecoli_master_sim.py` does not produce a 2-gen lineage in one process → switch `run_vecoli_sim` to vEcoli's Nextflow workflow runner: replace the command with `[VECOLI_PYTHON, "-m", "runscripts.workflow", "--config", config_path]` and point the reader at its `out/<experiment_id>/.../history` dir. Verify the workflow runner's exact flags first with `cd /Users/eranagmon/code/vEcoli && .venv/bin/python -m runscripts.workflow --help`.
- v2ecoli final sim_data is not `checkpoint_step_9.pkl` shape compatible with `_reach` → adjust `_load_pickle`/the v2 attr access in `final_sim_data_diff` (the v2 checkpoint may wrap state in a dict; unwrap to the sim_data-like object).
- Wrong emitter column names → correct them in `OBSERVABLES` (see Task 8 Step 5).

Re-run Step 1 after each fix until all three sections populate with real comparison rows.

- [ ] **Step 4: Commit fixes**

```bash
git add -A
git commit -m "fix(compare): align harness with real engine outputs"
```

---

### Task 12: Documentation

**Files:**
- Create: `scripts/_compare/README.md`

- [ ] **Step 1: Write the README**

`scripts/_compare/README.md`:
```markdown
# vEcoli ↔ v2ecoli comparison harness

Run both engines from one vEcoli config and produce a two-column HTML report.

## Usage

    .venv/bin/python scripts/compare_harness.py \
        --config /Users/eranagmon/code/vEcoli/configs/two_generations.json \
        -o out/compare/report.html

`--fast-plumbing` runs ParCa in fast mode for wiring iteration only and
stamps the report NOT SCIENTIFICALLY VALID. Omit it for the real comparison
(full ParCa, hours; results cached under `out/compare_harness/`).

## Sections

1. **Config & schema diff** — how the vEcoli config maps to v2ecoli
   (adapter in `config_adapter.py`; v2ecoli core untouched).
2. **ParCa / sim_data** — per-step diff via `scripts/parca_compare.py`
   plus a final-sim_data field-by-field diff (tight tolerance).
3. **2-generation sim dynamics** — mass/growth, molecule counts, listeners,
   division/lineage, compared with per-metric tolerances + KS.

## Tests

    .venv/bin/python -m pytest tests/compare/

The full cross-engine run is gated: `COMPARE_E2E=1 .venv/bin/python -m pytest tests/compare/test_end_to_end.py`.
```

- [ ] **Step 2: Commit**

```bash
git add scripts/_compare/README.md
git commit -m "docs(compare): harness README"
```

---

## Self-review notes

- **Spec coverage:** Stage 1 config adapter → Tasks 1–3; Stage 2 orchestration → Task 9; Stage 3 ParCa/sim_data (reuse parca_compare + final diff) → Task 7 + CLI; Stage 4 sim + 4 observable families → Task 8; Stage 5 report → Task 6; CLI/cache/tolerances → Tasks 4,5,10; fast-plumbing banner → Task 10; the three flagged risks (vEcoli lineage driver, v2 sim_data shape, emitter columns) → resolved in Task 11 with concrete fallback commands.
- **Tolerances:** `PARCA_REL_TOL=1e-6` (tight), `SIM_REL_TOL=0.05` (statistical), `mismatch_rel=0.5` default — centralized in `compare_harness.py` / `stats.py`.
- **Type consistency:** `compare_series` returns `verdict/max_rel/max_abs/ks_p`; report rows everywhere use `label/left/right/verdict[/reason]`; `OBSERVABLES` entries use `family/key/column`; orchestrator run fns all return the `Path`.
- **Reused, not rebuilt:** per-step ParCa diffing stays in `scripts/parca_compare.py`; this harness adds the final-sim_data diff and wraps the rest.
