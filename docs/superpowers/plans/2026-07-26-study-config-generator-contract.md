# Study-Config ↔ Generator-Param Contract — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Make v2ecoli studies runnable through the content-addressed engine by aligning study `conditions.baseline.params` + `composite` refs with what the generators actually accept, and enforce it with the conformance guard.

**Why:** Phase-2 Task-7 real-engine validation found `build_generator` strictly rejects unknown params. Studies declare `params: {condition: <x>}`, but `ecoli_baseline` has no `condition` param (the growth-medium key is `media`; `condition` is a ParCa-cache-build concept in a different namespace). And many studies use unqualified composite names that don't resolve. Neither has ever run through the strict path (a static-view filter silently drops `condition`).

**Tech Stack:** ruamel.yaml (comment-preserving), pytest, the v2ecoli conformance guard.

## Global Constraints

- **Worktree:** `~/code/v2e-condition-media`, branch `feat/study-config-generator-contract` (off `origin/main`). Verify branch/HEAD before commits.
- **Comment preservation mandatory** for study.yaml edits — ruamel round-trip (`YAML(); preserve_quotes=True; width=4096; indent(mapping=2,sequence=4,offset=2)`), never `yaml.safe_dump`.
- **Only touch `conditions.baseline.params.condition` and `conditions.variants[].params.condition`** for the media migration. DO NOT touch: top-level `condition:` metadata field, variant-level `condition`/`fixed_media` *annotations* (siblings of `name`, not under `params`), or any `expected_behavior[].given.condition` descriptors.
- **`media` value map (condition → media id):** `basal→(drop; default minimal)`, `acetate→minimal_acetate`, `succinate→minimal_succinate`, `no_oxygen→minimal_minus_oxygen`, `with_aa→minimal_plus_amino_acids`. (User decision: comparison studies use the lightweight `media:` lever, runnable now; preserve provenance in a note.)
- **Env:** run the guard/checks with `/Users/eranagmon/code/v2ecoli/.venv/bin/python`; generator introspection via `viva_superpowers.composite_generator.discover_generators()` + `_REGISTRY`.
- **Do NOT touch the 2 pdmp `condition`-in-params usages** (`pdmp-00`, `pdmp-01`) — they target different/non-existent composites (`diagnostic`, `utc-process`); out of scope.

---

### Task 1: Migrate `condition` param → `media`/drop (8 ecoli_baseline studies)

**Files (studies with `conditions.baseline.params.condition`):** `acetate`, `basal`, `no_oxygen`, `succinate`, `with_aa`, `parca`, `metabolism_redux`, `statistical` (all `studies/<slug>/study.yaml`).

**Rules (per study, ruamel comment-preserving):**
- If `condition == "basal"` (basal, parca, metabolism_redux, statistical): **remove** the `condition` key from `conditions.baseline.params` (generator default `media: minimal` is correct). Add a one-line ruamel comment on the params block: `# condition:basal was legacy; ecoli_baseline defaults to media:minimal`.
- Else (acetate/succinate/no_oxygen/with_aa): **replace** `condition: <x>` with `media: <mapped-id>` per the value map, and add a comment preserving provenance: `# migrated from condition:<x>; calibrated per-condition ParCa cache lives at out/cache/.regen_<x>_seed00`.
- Apply the same to any `conditions.variants[].params.condition` (check acetate/etc. variants — most have none, but handle if present).

- [ ] **Step 1: Write a verification test first** `tests/test_study_generator_params.py`:
```python
import glob, yaml, pathlib
from viva_superpowers.composite_generator import discover_generators, _REGISTRY
WS = pathlib.Path(__file__).resolve().parents[1] / "workspace"
discover_generators()

def _gen_params(composite_id):
    e = _REGISTRY.get(composite_id)
    return set((getattr(e, "parameters", {}) or {}).keys()) if e else None

def test_baseline_params_are_generator_accepted():
    bad = []
    for p in glob.glob(str(WS / "studies/*/study.yaml")):
        spec = yaml.safe_load(open(p)) or {}
        base = ((spec.get("conditions") or {}).get("baseline") or {})
        comp, params = base.get("composite"), (base.get("params") or {})
        if not comp:
            continue
        gp = _gen_params(comp)
        if gp is None:      # unresolved composite -> Task 2's concern; skip here
            continue
        unknown = set(params) - gp
        if unknown:
            bad.append(f"{p.split('/')[-2]}: {sorted(unknown)} not in {comp} params")
    assert not bad, "studies with non-generator params:\n" + "\n".join(bad)
```
- [ ] **Step 2:** Run it → FAILS (the 8 studies show `condition` as unknown, plus any unresolved composites are skipped).
- [ ] **Step 3:** Apply the migration to the 8 study.yaml files (ruamel; a small script or by hand). Verify with `git diff` that only the `condition`→`media`/drop lines + comments changed and comments elsewhere are intact.
- [ ] **Step 4:** Re-run the test — the `condition`-param failures are gone (only unresolved-composite skips remain, fixed in Task 2).
- [ ] **Step 5:** Commit: `git add workspace/studies tests/test_study_generator_params.py && git commit -m "migrate(studies): condition param -> media (comparison) / drop (basal); ecoli_baseline contract"`

---

### Task 2: Qualify composite names + register `reactor_bird_coupled` aliases

**Files:** ~13 study.yaml with BARE names; `v2ecoli/composites/__init__.py` (alias registration).

**Rules:**
- **Bare `ecoli_baseline`** → `v2ecoli.composites.ecoli_baseline.ecoli_baseline`; **bare `parca`** → `v2ecoli.composites.parca.parca`. Studies: `param-uq-00/01/02/03/04`, `population_phenotype_basal`, `showcase-3-variant-decide`, `showcase-5-next-direction-decide`, `sm-00/01/02/03`, `showcase-1-parca` (parca). Apply to baseline AND variant `composite:` refs (ruamel).
- **`reactor_bird_coupled[_millard]`:** the module-path refs (`v2ecoli.composites.reactor_bird_coupled`/`...reactor_bird_coupled_millard`) don't resolve because those clean aliases aren't registered. FIX in `v2ecoli/composites/__init__.py` — add them to the clean-alias loop (mirror the existing 7: `ecoli_baseline`, `parca`, etc.), so `v2ecoli.composites.reactor_bird_coupled` → `...reactor_bird_coupled.reactor_bird_coupled`. (Studies mbp-03/04/05/06/07 then resolve without study edits.)

- [ ] **Step 1:** Extend the Task-1 test with `test_all_composite_refs_resolve()`:
```python
def test_all_composite_refs_resolve():
    unresolved = []
    for p in glob.glob(str(WS / "studies/*/study.yaml")):
        spec = yaml.safe_load(open(p)) or {}
        cond = spec.get("conditions") or {}
        refs = []
        b = (cond.get("baseline") or {}).get("composite");  refs += [b] if b else []
        for v in (cond.get("variants") or []):
            if v.get("composite"): refs.append(v["composite"])
        for r in refs:
            # skip file-discovered YAML composites (…composite.yaml) — not in _REGISTRY
            if r in _REGISTRY: continue
            if r.endswith("millard2017_metabolism"): continue  # YAML composite, file-discovered
            unresolved.append(f"{p.split('/')[-2]}: {r}")
    assert not unresolved, "unresolved composite refs:\n" + "\n".join(unresolved)
```
- [ ] **Step 2:** Run → FAILS on the bare names + reactor refs.
- [ ] **Step 3:** Register the reactor aliases in `composites/__init__.py`; qualify the bare study refs (ruamel).
- [ ] **Step 4:** Re-run both tests → PASS.
- [ ] **Step 5:** Commit: `git add workspace/studies v2ecoli/composites/__init__.py tests/test_study_generator_params.py && git commit -m "fix(composites): qualify bare study composite refs + register reactor_bird_coupled aliases"`

---

### Task 3: Fold the contract into the conformance guard

**Files:** `tests/test_workspace_conformance.py` (add the two checks from Tasks 1–2 as guard tests), `scripts/lint-workspace.py` (mirror as lint errors).

- [ ] **Step 1:** Move/add `test_baseline_params_are_generator_accepted` + `test_all_composite_refs_resolve` into `tests/test_workspace_conformance.py` (the canonical guard), so a future study with a non-generator param or unresolved composite is caught. (Keep `tests/test_study_generator_params.py` or consolidate — pick one home; the conformance guard is canonical.)
- [ ] **Step 2:** Add a `check_generator_contract(...)` to `scripts/lint-workspace.py` emitting the same as lint errors (follow the file's error-collection pattern).
- [ ] **Step 3:** Run `python -m pytest tests/test_workspace_conformance.py -v` (all pass) + `python scripts/lint-workspace.py` (0 contract errors).
- [ ] **Step 4:** Commit + push + PR.

## Self-Review
Covers: condition→media/drop (T1), composite qualification + reactor aliases (T2), guard enforcement (T3). Excludes pdmp condition/diagnostic (out of scope), variant-annotation/expected_behavior condition (not params). The value map + per-study rules are from the grounding. Provenance preserved via comments.

## Notes
- The comparison studies now run a lightweight `media:` perturbation (user's choice), NOT the calibrated per-condition ParCa re-fit; provenance note points at the `.regen_<x>_seed00` caches for the faithful path (a possible later increment).
- Task 2's reactor-alias fix is v2ecoli code (one place) — cleaner than qualifying every reactor study ref.
