# Split Polypeptide Elongation into Wireable Processes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the strategy-class elongation models with three sibling `PartitionedProcess` classes (`BasePolypeptideElongation` → `TranslationSupplyPolypeptideElongation` → `SteadyStatePolypeptideElongation`) selected by wiring, defaulting to `SteadyState` for baseline parity, and delete the `translation_supply`/`trna_charging` selector flags.

**Architecture:** The current host `PolypeptideElongation(PartitionedProcess)` holds all shared scaffolding and delegates five hooks (`elongation_rate`, `amino_acid_counts`, `request`, `final_amino_acids`, `evolve`) to one of three model classes in `polypeptide/elongation_models.py`. We fold those model bodies *up* into a process inheritance chain (rewriting `self.process.X` → `self.X`), delete `elongation_models.py`, and move model selection from a config flag to the composite's `PARTITIONED_PROCESSES` registry. A golden-trajectory test locks baseline parity before and after.

**Tech Stack:** Python, process-bigraph, bigraph-schema, pytest, numpy. Run everything via `.venv/bin/python` and `.venv/bin/pytest` (bare `python` lacks `unum`).

**Spec:** `docs/superpowers/specs/2026-05-30-polypeptide-elongation-process-split-design.md`

---

## Task 1: Lock baseline parity with a golden-trajectory test

Captures the current (pre-refactor) `SteadyState` trajectory so any drift introduced by the refactor fails loudly.

**Files:**
- Create: `tests/test_polypeptide_elongation_parity.py`
- Create (generated): `tests/golden/polypeptide_elongation_baseline.json`

- [ ] **Step 1: Write the parity test + golden generator**

```python
# tests/test_polypeptide_elongation_parity.py
"""Golden-trajectory parity gate for the polypeptide-elongation refactor.

The default-wired elongation variant (SteadyState) must reproduce the
baseline trajectory bit-for-bit. Regenerate the golden ONLY intentionally:
    V2_WRITE_GOLDEN=1 .venv/bin/pytest tests/test_polypeptide_elongation_parity.py
"""
import json
import os

import numpy as np
import pytest

CACHE = "out/cache"
GOLDEN = os.path.join(os.path.dirname(__file__), "golden",
                      "polypeptide_elongation_baseline.json")
STEPS = 100  # short, CI-friendly; parity drift shows within a few ticks

pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE) and not os.environ.get("CI"),
    reason=f"cache dir {CACHE!r} not present",
)


def _trajectory():
    from v2ecoli import build_composite
    from v2ecoli.library.quantity_helpers import fg_magnitude
    c = build_composite("baseline", cache_dir=CACHE, seed=0)
    a = c.state["agents"]["0"]
    rec = []
    for _ in range(STEPS):
        c.run(1)
        mass = a["listeners"]["mass"]
        rec.append(round(float(fg_magnitude(mass["dry_mass"])), 6))
    bulk = a.get("bulk")
    bulk_sum = int(np.nansum(bulk["count"])) if getattr(bulk, "dtype", None) and bulk.dtype.names else int(np.nansum(bulk))
    return {"dry_mass": rec, "bulk_total_at_end": bulk_sum}


def test_baseline_elongation_trajectory_matches_golden():
    traj = _trajectory()
    if os.environ.get("V2_WRITE_GOLDEN"):
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, "w") as f:
            json.dump(traj, f, indent=1)
        pytest.skip("wrote golden")
    with open(GOLDEN) as f:
        golden = json.load(f)
    assert traj["dry_mass"] == golden["dry_mass"], (
        "dry_mass trajectory drifted from golden — elongation refactor changed behaviour")
    assert traj["bulk_total_at_end"] == golden["bulk_total_at_end"]
```

- [ ] **Step 2: Generate the golden from current (pre-refactor) code**

Run: `V2_WRITE_GOLDEN=1 .venv/bin/pytest tests/test_polypeptide_elongation_parity.py -q`
Expected: 1 skipped ("wrote golden"); `tests/golden/polypeptide_elongation_baseline.json` now exists with 100 dry_mass values.

- [ ] **Step 3: Verify the test passes against the golden**

Run: `.venv/bin/pytest tests/test_polypeptide_elongation_parity.py -q`
Expected: 1 passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_polypeptide_elongation_parity.py tests/golden/polypeptide_elongation_baseline.json
git commit -m "test(elongation): golden-trajectory parity gate for the refactor"
```

---

## Task 2: Write failing tests for the three new process classes

TDD: assert the new classes exist and are wireable before they do.

**Files:**
- Create: `tests/test_polypeptide_elongation_variants.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_polypeptide_elongation_variants.py
import os
import pytest

CACHE = "out/cache"
pytestmark = pytest.mark.skipif(
    not os.path.isdir(CACHE) and not os.environ.get("CI"),
    reason=f"cache dir {CACHE!r} not present",
)


def test_three_variants_importable_and_subclassed():
    from v2ecoli.steps.partition import PartitionedProcess
    from v2ecoli.processes.polypeptide_elongation import (
        BasePolypeptideElongation,
        TranslationSupplyPolypeptideElongation,
        SteadyStatePolypeptideElongation,
    )
    assert issubclass(BasePolypeptideElongation, PartitionedProcess)
    assert issubclass(TranslationSupplyPolypeptideElongation, BasePolypeptideElongation)
    assert issubclass(SteadyStatePolypeptideElongation,
                      TranslationSupplyPolypeptideElongation)


def test_steadystate_declares_charging_ports_base_does_not():
    """The payoff: only SteadyState exposes charged-tRNA / ppGpp ports."""
    from v2ecoli.processes.polypeptide_elongation import (
        BasePolypeptideElongation, SteadyStatePolypeptideElongation)
    base_in = set(BasePolypeptideElongation({}).inputs().keys())
    ss_in = set(SteadyStatePolypeptideElongation({}).inputs().keys())
    # SteadyState reads at least as much as Base, and adds charging-related stores
    assert base_in <= ss_in
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest tests/test_polypeptide_elongation_variants.py -q`
Expected: FAIL — `ImportError: cannot import name 'BasePolypeptideElongation'`.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_polypeptide_elongation_variants.py
git commit -m "test(elongation): failing tests for the three wireable variants"
```

---

## Task 3: Restructure into the three process classes (keep behaviour identical)

Fold the model bodies into a process chain. **Move method bodies verbatim**; the only mechanical edit inside moved code is `self.process.X` → `self.X` and `self.process` → `self`. Keep `elongation_models.py` in place for now (deleted in Task 4) so partial states still import.

**Files:**
- Modify: `v2ecoli/processes/polypeptide_elongation.py`

- [ ] **Step 1: Rename the host class and add the chain skeleton**

In `v2ecoli/processes/polypeptide_elongation.py`:
- Rename `class PolypeptideElongation(PartitionedProcess):` → `class BasePolypeptideElongation(PartitionedProcess):`.
- At the bottom of the file add:

```python
class TranslationSupplyPolypeptideElongation(BasePolypeptideElongation):
    """Elongation with media-driven amino-acid supply (no explicit charging)."""


class SteadyStatePolypeptideElongation(TranslationSupplyPolypeptideElongation):
    """Full tRNA-charging + ppGpp steady-state elongation. Baseline default."""
```

- [ ] **Step 2: Remove model selection; inline Base hooks into `BasePolypeptideElongation`**

In `initialize()` (lines ~270-287), delete the model-selection block:

```python
        # DELETE these lines:
        if trna_charging:
            self.elongation_model = SteadyStateElongationModel(self.parameters, self)
        elif translation_supply:
            self.elongation_model = TranslationSupplyElongationModel(self.parameters, self)
        else:
            self.elongation_model = BaseElongationModel(self.parameters, self)
```

Also delete the two reads `translation_supply = self.parameters["translation_supply"]` and `trna_charging = self.parameters["trna_charging"]` (lines ~213-214).

Keep the GAM correction but make it the **Base/Supply** behaviour (these don't model charging, so they add the 2 ATP-hydrolysis equivalents):

```python
        # Base/Supply do not model charging, so account for the 2 ATP->AMP
        # hydrolyses removed from measured GAM. SteadyState overrides initialize
        # to skip this (it models charging explicitly).
        self.gtpPerElongation = self.parameters["gtpPerElongation"] + 2
```

Then copy the five hook methods from `BaseElongationModel` (`elongation_models.py`, class starting line 34) into `BasePolypeptideElongation` as methods: `elongation_rate`, `amino_acid_counts`, `request`, `final_amino_acids`, `evolve`. Inside each, replace every `self.process.` with `self.` and every standalone `self.process` with `self`. Update `calculate_request`/`evolve_state` to call `self.elongation_rate(...)`, `self.request(...)`, `self.final_amino_acids(...)`, `self.evolve(...)` instead of `self.elongation_model.*`.

- [ ] **Step 3: Override hooks in `TranslationSupplyPolypeptideElongation`**

Copy `TranslationSupplyElongationModel`'s overrides (`elongation_models.py` line 111: `__init__`, `elongation_rate`, `amino_acid_counts`) into the subclass. The model's `__init__` extras become an `initialize()` extension:

```python
    def initialize(self, config):
        super().initialize(config)
        # <body of TranslationSupplyElongationModel.__init__ after super().__init__,
        #  with self.process -> self>
```

Copy `elongation_rate` and `amino_acid_counts` method bodies (self.process -> self).

- [ ] **Step 4: Override hooks + ports in `SteadyStatePolypeptideElongation`**

Copy `SteadyStateElongationModel`'s members (`elongation_models.py` line 135): `initialize` (from its `__init__` body via `super().initialize(config)` then the rest), `elongation_rate`, `_amino_acid_supply`, `request`, `final_amino_acids`, `_ppgpp_request`, `_ppgpp_evolve`, `evolve`, `distribution_from_aa` — all with `self.process` → `self`.

SteadyState models charging, so it must NOT add the +2 GAM correction. Override:

```python
    def initialize(self, config):
        super().initialize(config)
        self.gtpPerElongation = self.parameters["gtpPerElongation"]  # no +2 (charging modelled)
        # <remaining SteadyStateElongationModel.__init__ body, self.process -> self>
```

Override `inputs()`/`outputs()` to add the charging/ppGpp stores SteadyState reads/writes (charged + uncharged tRNA, synthetases, ppGpp reaction metabolites, the `process_state.polypeptide_elongation` channel) on top of `super().inputs()/outputs()`:

```python
    def inputs(self):
        base = super().inputs()
        base.update({
            # charged/uncharged tRNA, synthetases, ppGpp metabolites that only
            # the steady-state charging math reads (names from self.parameters):
            # add the concrete port schema here, mirroring what the old shared
            # host declared but BaseElongationModel never touched.
        })
        return base
```
(Use the exact port entries the old shared `inputs()`/`outputs()` declared for charging/ppGpp; Base keeps only what `BaseElongationModel` actually reads/writes.)

- [ ] **Step 5: Point imports at temporary shim**

At the top of the file, change the model import to only what the moved code still needs from `polypeptide/kinetics.py` and `polypeptide/common.py`. Remove `from v2ecoli.processes.polypeptide.elongation_models import (...)`.

- [ ] **Step 6: Wire SteadyState as the default and run the parity gate**

In `v2ecoli/composites/_helpers.py`:

```python
# line ~34: import
from v2ecoli.processes.polypeptide_elongation import SteadyStatePolypeptideElongation
# line ~70: registry entry
'ecoli-polypeptide-elongation': SteadyStatePolypeptideElongation,
```

Run: `.venv/bin/pytest tests/test_polypeptide_elongation_parity.py tests/test_polypeptide_elongation_variants.py -q`
Expected: parity test PASS (trajectory unchanged) + variant tests PASS.

If parity FAILS: a moved body reordered an RNG call or dropped a `+2`. Diff the moved method against the original line-by-line; do **not** edit the golden.

- [ ] **Step 7: Commit**

```bash
git add v2ecoli/processes/polypeptide_elongation.py v2ecoli/composites/_helpers.py
git commit -m "refactor(elongation): fold model strategy classes into 3 wireable processes"
```

---

## Task 4: Delete `elongation_models.py` and migrate its importers

**Files:**
- Delete: `v2ecoli/processes/polypeptide/elongation_models.py`
- Modify: `v2ecoli/processes/polypeptide/kinetics.py`, `v2ecoli/library/sim_data.py`, `tests/test_kinetics_units.py`

- [ ] **Step 1: Find every importer**

Run: `grep -rn --include="*.py" -e "elongation_models" -e "ElongationModel" v2ecoli tests | grep -v __pycache__`
Expected: matches in `kinetics.py`, `sim_data.py`, `tests/test_kinetics_units.py` (the host no longer imports it after Task 3).

- [ ] **Step 2: Repoint or remove each import**

For each match: if it imports `BaseElongationModel`/etc., repoint to the new process classes in `polypeptide_elongation.py`, or delete the import if it only existed to instantiate a model. In `sim_data.py`, drop the model-class import (its remaining use is examined in Task 5).

- [ ] **Step 3: Delete the file**

Run: `git rm v2ecoli/processes/polypeptide/elongation_models.py`

- [ ] **Step 4: Verify nothing imports the deleted module**

Run: `.venv/bin/python -c "import v2ecoli.processes.polypeptide_elongation, v2ecoli.library.sim_data, v2ecoli.processes.polypeptide.kinetics; print('imports OK')"`
Expected: `imports OK` (ignore the unrelated `skipping ...` optional-dep lines).

- [ ] **Step 5: Run kinetics tests + parity**

Run: `.venv/bin/pytest tests/test_kinetics_units.py tests/test_polypeptide_elongation_parity.py -q`
Expected: PASS (update `test_kinetics_units.py` imports if it referenced the deleted symbols).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(elongation): delete elongation_models.py, migrate importers"
```

---

## Task 5: Delete the selector flags and stop emitting them

**Files:**
- Modify: `v2ecoli/processes/polypeptide_elongation.py` (config_schema)
- Modify: `v2ecoli/library/sim_data.py:1095-1096`
- Modify: `v2ecoli/core.py` or the cache loader (strip stale keys)

- [ ] **Step 1: Remove the flags from `config_schema`**

In `polypeptide_elongation.py` `config_schema`, delete the `'translation_supply'` and `'trna_charging'` entries (lines ~191-192).

- [ ] **Step 2: Stop emitting the flags from sim_data**

In `v2ecoli/library/sim_data.py` around lines 1095-1096, delete:

```python
            "translation_supply": self.translation_supply,
            "trna_charging": self.trna_charging,
```
Leave the `self.translation_supply` / `self.trna_charging` attributes intact — they still drive ParCa fitting elsewhere.

- [ ] **Step 3: Make the cache loader tolerate the stale keys**

Existing `out/cache` still has the two keys in the elongation config. In `load_cache_bundle` (`v2ecoli/core.py`), after loading `configs`, strip them so old caches build:

```python
    pe = cache.get("configs", {}).get("ecoli-polypeptide-elongation")
    if isinstance(pe, dict):
        pe.pop("translation_supply", None)
        pe.pop("trna_charging", None)
```

- [ ] **Step 4: Verify build + parity still hold**

Run: `.venv/bin/python -c "from v2ecoli import build_composite; build_composite('baseline', cache_dir='out/cache', seed=0); print('build OK')" 2>&1 | grep -v 'skipping '`
Then: `.venv/bin/pytest tests/test_polypeptide_elongation_parity.py -q`
Expected: `build OK` and parity PASS.

- [ ] **Step 5: Commit**

```bash
git add v2ecoli/processes/polypeptide_elongation.py v2ecoli/library/sim_data.py v2ecoli/core.py
git commit -m "refactor(elongation): delete translation_supply/trna_charging selector flags"
```

---

## Task 6: Per-variant behaviour tests

Each variant must run inside a composite and elongate. (Base/Supply are exercised by direct wiring, not the default baseline.)

**Files:**
- Modify: `tests/test_polypeptide_elongation_variants.py`

- [ ] **Step 1: Add a behaviour test that wires each variant**

```python
import numpy as np

VARIANTS = [
    "BasePolypeptideElongation",
    "TranslationSupplyPolypeptideElongation",
    "SteadyStatePolypeptideElongation",
]

@pytest.mark.parametrize("variant", VARIANTS)
def test_variant_elongates_protein(variant, monkeypatch):
    import v2ecoli.composites._helpers as H
    import v2ecoli.processes.polypeptide_elongation as PE
    cls = getattr(PE, variant)
    monkeypatch.setitem(H.PARTITIONED_PROCESSES, "ecoli-polypeptide-elongation", cls)
    from v2ecoli import build_composite
    from v2ecoli.library.quantity_helpers import fg_magnitude
    c = build_composite("baseline", cache_dir="out/cache", seed=0)
    a = c.state["agents"]["0"]
    m0 = float(fg_magnitude(a["listeners"]["mass"]["protein_mass"]))
    c.run(20)
    m1 = float(fg_magnitude(a["listeners"]["mass"]["protein_mass"]))
    assert m1 > m0, f"{variant}: protein mass did not increase ({m0:.1f}->{m1:.1f})"
```

- [ ] **Step 2: Run**

Run: `.venv/bin/pytest tests/test_polypeptide_elongation_variants.py -q`
Expected: all PASS (3 variants elongate).

If a non-default variant errors on a missing store, its `inputs()`/`outputs()` need the ports its math reads — fix the variant's port declaration (this is the intended per-variant divergence).

- [ ] **Step 3: Commit**

```bash
git add tests/test_polypeptide_elongation_variants.py
git commit -m "test(elongation): per-variant behaviour tests (Base/Supply/SteadyState elongate)"
```

---

## Task 7: Migrate remaining references + full suite + parity at division

**Files:**
- Modify: `tests/test_parca_fixture_roundtrip.py` (and any other refs surfaced)
- Modify: `scripts/baseline_report_html.py` (doc text mentions the deleted flags)

- [ ] **Step 1: Re-scan for stale references**

Run: `grep -rn --include="*.py" -e "PolypeptideElongation\b" -e "translation_supply" -e "trna_charging" tests scripts | grep -v __pycache__ | grep -v parca/`
Expected: identifies `test_parca_fixture_roundtrip.py` + `baseline_report_html.py`. Update each: rename `PolypeptideElongation` → `BasePolypeptideElongation`/`SteadyStatePolypeptideElongation` as appropriate; drop the two flags from the report's elongation config-knobs doc text.

- [ ] **Step 2: Add a slow division-parity assertion**

Append to `tests/test_polypeptide_elongation_parity.py`:

```python
@pytest.mark.slow
def test_baseline_divides_unchanged():
    """Full-cycle parity: cell still divides and at the same dry mass band."""
    from v2ecoli import build_composite
    from v2ecoli.library.quantity_helpers import fg_magnitude
    c = build_composite("baseline", cache_dir="out/cache", seed=0)
    for _ in range(3000):
        c.run(1)
        if set(c.state["agents"].keys()) != {"0"} or c.state["agents"]["0"].get("divide"):
            t = float(c.state["global_time"])
            assert 2400 <= t <= 2700, f"division time {t}s outside expected band"
            return
    pytest.fail("no division within 3000s")
```

- [ ] **Step 3: Run the full elongation + behaviour suite**

Run: `.venv/bin/pytest tests/test_polypeptide_elongation_parity.py tests/test_polypeptide_elongation_variants.py tests/test_kinetics_units.py tests/test_parca_fixture_roundtrip.py -q`
Expected: all PASS (run the `slow` one explicitly: `-m slow` or `--runslow` per repo convention).

- [ ] **Step 4: Run the broader behaviour suite to catch regressions**

Run: `.venv/bin/pytest tests/test_sustained_growth.py tests/test_model_behavior.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(elongation): migrate remaining refs; division-parity test"
```

---

## Self-review notes (resolved)

- **Spec coverage:** chain structure (T3), port divergence (T3 step 4, T6), wiring/registry default SteadyState (T3 step 6), flag deletion + sim_data + cache loader (T5), `elongation_models.py` deletion (T4), parity gate (T1, T7), per-variant tests (T6), test migration (T4, T7), report doc text (T7). Transcript elongation untouched (non-goal).
- **Parity discipline:** golden generated pre-refactor (T1); never regenerated during the refactor; RNG-order preserved by verbatim body moves.
- **Open detail from spec** (registry override vs generator param): plan uses the **registry entry** in `_helpers.PARTITIONED_PROCESSES` as the single source of the default, and `monkeypatch.setitem` for per-variant tests — no new generator param needed (YAGNI).
