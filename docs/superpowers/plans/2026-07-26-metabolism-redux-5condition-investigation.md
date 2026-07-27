# MetabolismRedux 5-condition Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the `metabolism_redux` study into a 5-condition, 2-seed × 3-generation statistical investigation of v2ecoli-reproduces-vEcoli-under-MetabolismRedux, with four new interactive Plotly report cards, run on the Mac mini and served from its read-only dashboard.

**Architecture:** Both engines swap FBA `ecoli-metabolism` → `ecoli-metabolism-redux`. Per-condition redux fork configs are generated from `cond_<X>.json` + the basal redux swap block. Five studies (one per condition) run via `runner.run_investigation` → `run_comparison_ensemble.py` (v2ecoli matched-initial on `out/cache_full`; genuine vEcoli via vivarium-process on `out/compare_harness/vecoli_parca`). New report cards are `@as_step` process-bigraph Steps that read both engines' v2ecoli-format zarr stores and emit Plotly figures inline. The full ensemble runs on the mini via Ray; the finished investigation is served read-only from the mini dashboard.

**Tech Stack:** Python 3.12, process-bigraph (`as_step`), xarray/zarr, Plotly (`include_plotlyjs="cdn"`), scipy.stats (Welch t-test), Ray, pytest.

## Global Constraints

- Work in worktree `~/code/v2e-redux-invest`, branch `feat/metabolism-redux-5cond-investigation` (based off `fix/inject-vivarium-step-as-step`; the PR #389 as_step fix MUST be present — verify `grep -c _should_inject_as_step scripts/_compare/inject.py` == 2).
- Run tests/renders with `~/code/v2ecoli/.venv/bin/python` and `PYTHONPATH=~/code/v2e-redux-invest` so the worktree's `scripts/` is imported (verify `python -c "import scripts._compare.inject as i; print(i.__file__)"` points at the worktree).
- vEcoli fork: `/Users/eranagmon/code/vEcoli` (ABSOLUTE path; a `~` is NOT expanded and silently falls back to a NON-swapped baseline run). Set `V2E_VECOLI_DIR=/Users/eranagmon/code/vEcoli`.
- All file I/O that touches config/manifest text uses `encoding="utf-8"` (CI runs in an ASCII locale; titles contain `↔`/`×`).
- Report cards return `{card_html, verdict, axes}`; verdict vocab is `within_tol` / `drift` / `mismatch` / `ungraded`; use `scripts/_compare/verdict.worst(...)` to aggregate.
- Plotly figures: `fig.to_html(include_plotlyjs="cdn", full_html=False)`; place the returned string verbatim in a section's `html` field.
- Statistical shape: **2 seeds × 3 generations** per condition. The `distribution`/`statistical` t-tests pool per-cell values across generations (n≈6/condition) AND report per-generation.
- Colours (match existing charts): vEcoli = indigo `#4f46e5`, v2ecoli = amber `#d97706` (`scripts/_compare/charts.py:PALETTE`).
- Do NOT merge any PR or push to `main`. Commit to the build branch. PR #389 merge is a separate user action.

---

## File Structure

**Create:**
- `scripts/gen_redux_condition_configs.py` — generator: `cond_<X>.json` + basal redux block → `metabolism_redux_<X>.json` in the fork.
- `/Users/eranagmon/code/vEcoli/configs/metabolism_redux_{with_aa,succinate,no_oxygen,acetate}.json` — generated (committed to the fork repo).
- `workspace/investigations/v2ecoli-vecoli-comparison/studies/metabolism_redux_{basal,with_aa,succinate,no_oxygen,acetate}/study.yaml` — 5 studies.
- `scripts/_compare/report_cards/trajectory.py` — interactive trajectory-overlay card.
- `scripts/_compare/report_cards/distribution.py` — violin/strip distribution card.
- `scripts/_compare/report_cards/metabolism.py` — growth-law / metabolism card.
- `scripts/_compare/report_cards/composition.py` — composition + cell-cycle + perf card.
- `scripts/_compare/plotly_helpers.py` — shared Plotly builders (overlay, violin, bars) returning `to_html` fragments.
- `tests/test_redux_condition_configs.py`, `tests/test_report_card_trajectory.py`, `tests/test_report_card_distribution.py`, `tests/test_report_card_metabolism.py`, `tests/test_report_card_composition.py`, `tests/fixtures/redux_cards/` (tiny fixture state + zarrs).
- `scripts/run_redux_investigation_mini.sh` — mini run driver.

**Modify:**
- `scripts/_compare/report_cards/__init__.py:71-72` — import the 4 new card modules so they self-register.
- `scripts/_compare/report_card_section.py:22` and `scripts/comparison_report_card.py:95-106` — set `plot="violin"` on the statistical axes.
- `scripts/_compare/study_spec.py` — if a per-study `generations`/`max_steps_per_gen` override is not already read, ensure 3-gen studies resolve (verify first).
- `workspace/investigations/v2ecoli-vecoli-comparison/investigation.yaml` — register the 5 studies as members; update executive summary after the run.

---

## Phase A — Harness hardening

### Task A1: Per-condition redux config generator

**Files:**
- Create: `scripts/gen_redux_condition_configs.py`
- Test: `tests/test_redux_condition_configs.py`

**Interfaces:**
- Produces: `build_redux_config(cond_config: dict, basal_redux: dict) -> dict` and `main(fork_dir: str) -> list[str]` (writes `metabolism_redux_<X>.json`, returns paths).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_redux_condition_configs.py
import json
from scripts.gen_redux_condition_configs import build_redux_config

def test_build_redux_config_merges_swap_block_and_condition():
    cond = {"experiment_id": "cond_acetate", "condition": "acetate"}
    basal_redux = {
        "experiment_id": "metabolism_redux_basal", "condition": "basal",
        "swap_processes": {"ecoli-metabolism": "ecoli-metabolism-redux"},
        "exclude_processes": ["exchange_data"],
        "flow": {"ecoli-metabolism-redux": [["ecoli-chromosome-structure"]]},
        "strip_pint_ports": {"ecoli-metabolism-redux": ["listeners"]},
        "attach_pint_ports": {"ecoli-metabolism-redux": {"boundary": "mM"}},
        "output_ports": {"ecoli-metabolism-redux": ["bulk", "environment"]},
    }
    out = build_redux_config(cond, basal_redux)
    # condition comes from the cond config; swap block from basal redux
    assert out["condition"] == "acetate"
    assert out["swap_processes"] == {"ecoli-metabolism": "ecoli-metabolism-redux"}
    assert out["flow"]["ecoli-metabolism-redux"] == [["ecoli-chromosome-structure"]]
    assert out["strip_pint_ports"] and out["attach_pint_ports"] and out["output_ports"]
    assert out["experiment_id"] == "metabolism_redux_acetate"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=~/code/v2e-redux-invest ~/code/v2ecoli/.venv/bin/python -m pytest tests/test_redux_condition_configs.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the generator**

```python
# scripts/gen_redux_condition_configs.py
"""Generate per-condition MetabolismRedux fork configs from cond_<X>.json + the
basal redux swap block. Deterministic; committed to the vEcoli fork."""
from __future__ import annotations
import json, os

CONDITIONS = ["with_aa", "succinate", "no_oxygen", "acetate"]  # basal already exists
SWAP_KEYS = ["swap_processes", "exclude_processes", "flow", "raw_output",
             "strip_pint_ports", "attach_pint_ports", "output_ports"]

def build_redux_config(cond_config: dict, basal_redux: dict) -> dict:
    out = dict(cond_config)  # start from the condition config (media/condition/nutrients)
    for k in SWAP_KEYS:
        if k in basal_redux:
            out[k] = basal_redux[k]
    cond = out.get("condition") or cond_config.get("condition")
    out["condition"] = cond
    out["experiment_id"] = f"metabolism_redux_{cond}"
    return out

def main(fork_dir: str = "/Users/eranagmon/code/vEcoli") -> list[str]:
    cfg = os.path.join(fork_dir, "configs")
    with open(os.path.join(cfg, "metabolism_redux_basal.json"), encoding="utf-8") as f:
        basal_redux = json.load(f)
    written = []
    for c in CONDITIONS:
        with open(os.path.join(cfg, f"cond_{c}.json"), encoding="utf-8") as f:
            cond_config = json.load(f)
        out = build_redux_config(cond_config, basal_redux)
        p = os.path.join(cfg, f"metabolism_redux_{c}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)
        written.append(p)
    return written

if __name__ == "__main__":
    for p in main():
        print("wrote", p)
```

- [ ] **Step 4: Run the test — expect PASS**

Run: `PYTHONPATH=~/code/v2e-redux-invest ~/code/v2ecoli/.venv/bin/python -m pytest tests/test_redux_condition_configs.py -q`

- [ ] **Step 5: Generate the 4 configs and sanity-check they resolve**

Run:
```bash
cd ~/code/v2e-redux-invest
~/code/v2ecoli/.venv/bin/python scripts/gen_redux_condition_configs.py
V2E_VECOLI_DIR=/Users/eranagmon/code/vEcoli PYTHONPATH=~/code/v2e-redux-invest \
  ~/code/v2ecoli/.venv/bin/python -c "
from scripts._compare.config_adapter import resolve_vecoli_config_local
for c in ['with_aa','succinate','no_oxygen','acetate']:
    r = resolve_vecoli_config_local(f'configs/metabolism_redux_{c}.json','/Users/eranagmon/code/vEcoli')
    print(c, '->', r.get('condition'), r.get('swap_processes'))
"
```
Expected: each prints the right condition + `{'ecoli-metabolism': 'ecoli-metabolism-redux'}`.

- [ ] **Step 6: Commit** (v2ecoli generator + test; and commit the 4 generated configs in the fork repo)

```bash
cd ~/code/v2e-redux-invest && git add scripts/gen_redux_condition_configs.py tests/test_redux_condition_configs.py && git commit -m "feat(compare): per-condition MetabolismRedux config generator"
cd /Users/eranagmon/code/vEcoli && git add configs/metabolism_redux_*.json && git commit -m "configs: generated per-condition metabolism_redux configs" || echo "fork commit optional; ensure mini gets configs"
```

---

### Task A2: Five redux studies + investigation members

**Files:**
- Create: `workspace/investigations/v2ecoli-vecoli-comparison/studies/metabolism_redux_<cond>/study.yaml` (5)
- Modify: `workspace/investigations/v2ecoli-vecoli-comparison/investigation.yaml`
- Test: `tests/test_redux_studies_load.py`

**Interfaces:**
- Consumes: `scripts._compare.study_spec.load_investigation(inv_ref) -> (ctx, specs)`; each spec has `.name, .condition, .seeds, .gens, .from_vecoli_config, .cards`.

- [ ] **Step 1: Verify study_spec reads per-study `generations`.** Read `scripts/_compare/study_spec.py` around the `StudySpec` construction. Confirm `gens` is read from `comparison.generations` (the basal study sets `comparison.generations: 1`). If it is hard-coded, add reading `comparison.get("generations", 1)`. Record the finding in the commit message.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_redux_studies_load.py
from scripts._compare.study_spec import load_investigation
def test_five_redux_studies_load_with_2x3_shape():
    _ctx, specs = load_investigation("v2ecoli-vecoli-comparison")
    by = {s.name: s for s in specs}
    for c in ["basal","with_aa","succinate","no_oxygen","acetate"]:
        s = by[f"metabolism_redux_{c}"]
        assert s.seeds == 2 and s.gens == 3
        assert s.condition == c
        assert s.from_vecoli_config == f"configs/metabolism_redux_{c}.json"
        for card in ["config","parca","standard","statistical","trajectory","distribution","metabolism","composition"]:
            assert card in s.cards
```

- [ ] **Step 3: Run to verify it fails.** Run the pytest; expect KeyError/missing studies.

- [ ] **Step 4: Author the 5 study.yaml files.** Copy the existing `metabolism_redux/study.yaml` to `metabolism_redux_basal/study.yaml`; for each condition set `name: metabolism_redux_<cond>`, `condition: <cond>`, `comparison: {seeds: 2, generations: 3, max_steps_per_gen: 15000}`, `from_vecoli_config: configs/metabolism_redux_<cond>.json`, `comparison.cards: [config, parca, standard, statistical, trajectory, distribution, metabolism, composition]`, and a `runs:`/`tests:`/`conditions:` block mirroring the basal study (update the `swap`/condition params). Keep the resolved `metabolism_redux_<cond>_basal.json` sidecar only if referenced; otherwise omit. Remove the old single `metabolism_redux/` dir (its content is now `metabolism_redux_basal/`).

- [ ] **Step 5: Register members in investigation.yaml.** Ensure `load_investigation` discovers the 5 studies (verify how the current single study is discovered — directory scan vs explicit member list — and match that mechanism). Update `comparison.defaults` if needed (`seeds: 2`).

- [ ] **Step 6: Run the test — expect PASS.**

- [ ] **Step 7: Commit** `git add workspace/... tests/test_redux_studies_load.py && git commit -m "feat(compare): 5 MetabolismRedux condition studies (2 seeds x 3 gens)"`

---

### Task A3: Multi-generation redux smoke gate (RISK GATE)

**Files:** Create: `scripts/smoke_redux_multigen.sh` (throwaway-but-committed driver)

**Interfaces:** none (verification task).

- [ ] **Step 1: Run 1 seed × 2 gens redux (basal), both engines, locally.**

```bash
cd ~/code/v2e-redux-invest
OUT=out/smoke_redux; rm -rf $OUT; mkdir -p $OUT
for eng in v2ecoli vecoli; do
  cache=$([ $eng = v2ecoli ] && echo out/cache_full || echo out/compare_harness/vecoli_parca)
  env PYTHONPATH=/Users/eranagmon/code/v2e-redux-invest V2E_VECOLI_DIR=/Users/eranagmon/code/vEcoli \
    ~/code/v2ecoli/.venv/bin/python scripts/run_comparison_ensemble.py \
    --composite $eng --condition basal --cache-dir $cache \
    --n-seeds 1 --max-generations 2 --max-steps 4000 --chunk 60 --mode serial \
    --from-vecoli-config configs/metabolism_redux_basal.json \
    $([ $eng = v2ecoli ] && echo "--match-vecoli-simdata out/compare_harness/vecoli_parca/simData.cPickle") \
    --out-root $OUT 2>&1 | tee $OUT/$eng.log
done
```

- [ ] **Step 2: Verify both engines produced 2 generations without hang/crash.** Check each run's JSON summary has `"generations": [1, 2]` and non-null wall time; confirm no `Traceback`. Confirm the v2 log shows `+injected ['ecoli-metabolism']` (swap active).

- [ ] **Step 3: DECISION GATE.**
  - PASS (both engines reach gen 2, no crash) → record masses per gen in the commit message; proceed to Phase B.
  - FAIL (division crash / hang / redux error at gen ≥2) → STOP. Invoke `superpowers:systematic-debugging`; fix the multi-gen redux path before any further Phase C work. Do NOT start the mini run.

- [ ] **Step 4: Commit the smoke driver + result note** `git add scripts/smoke_redux_multigen.sh && git commit -m "test(compare): multi-gen redux smoke gate — <PASS/FAIL + per-gen masses>"`

---

## Phase B — New report cards

### Task B0: Pin card state shape + build a test fixture

**Files:** Create: `tests/fixtures/redux_cards/README.md`, `tests/fixtures/redux_cards/state.json` (+ small zarr stores or a builder).

**Interfaces:**
- Produces: a `make_card_state()` test helper (in `tests/conftest.py` or the fixture dir) returning a valid `state` dict matching `CARD_INPUTS`, with `v2_dir`/`ve_dir` pointing at real (tiny) zarr stores and populated `observables`/`plot_trajs`.

- [ ] **Step 1: Document the exact shapes.** Read `scripts/comparison_report_card.py` state-building block (~lines 861-866) and `runs_section`/`eval_section` (~780-830), plus `statistical.py`, and write into `tests/fixtures/redux_cards/README.md` the exact structure of `state["observables"]` (per-observable per-seed stat records) and `state["plot_trajs"]`.

- [ ] **Step 2: Create the fixture from the smoke run.** Use `out/smoke_redux/{v2ecoli,vecoli}_seed00.zarr` (from A3) as `v2_dir`/`ve_dir`; write `make_card_state()` returning `{name:"metabolism_redux_basal", condition:"basal", seeds:1, generations:2, variant:0, observables:..., plot_trajs:..., v2_bounds:[...], config:{}, v2_dir:"out/smoke_redux", ve_dir:"out/smoke_redux"}`. Build `observables`/`plot_trajs` by calling the same loader `comparison_report_card` uses (factor it out or replicate the call).

- [ ] **Step 3: Commit** `git add tests/fixtures/redux_cards tests/conftest.py && git commit -m "test(compare): report-card state fixture from smoke run"`

---

### Task B1: `trajectory` card (Plotly overlay)

**Files:** Create: `scripts/_compare/report_cards/trajectory.py`, `scripts/_compare/plotly_helpers.py`; Modify: `scripts/_compare/report_cards/__init__.py:72`; Test: `tests/test_report_card_trajectory.py`

**Interfaces:**
- Consumes: `make_card_state()` (B0); `scripts.compare_matched_trajectories.read_v2ecoli_trajectory/read_vecoli_pbg_trajectory/OBSERVABLES`.
- Produces: `plotly_helpers.overlay_html(per_obs_traces: dict, title: str) -> str`; card `update_trajectory_report_card(state) -> {card_html, verdict, axes}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report_card_trajectory.py
from tests.conftest import make_card_state
from scripts._compare.report_cards.trajectory import update_trajectory_report_card

def test_trajectory_card_emits_plotly_and_ungraded():
    out = update_trajectory_report_card(make_card_state())
    assert "plotly" in out["card_html"].lower()
    assert out["verdict"] in ("ungraded", "within_tol", "drift", "mismatch")
    assert isinstance(out["axes"], list)
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `plotly_helpers.overlay_html`** — one figure per observable, v2 (amber) vs vEcoli (indigo) value-vs-time lines with per-seed spread as a shaded band; generation boundary vlines; unified hover; `fig.to_html(include_plotlyjs="cdn", full_html=False)`. Concatenate per-observable divs; load plotly.js CDN once (first fig `include_plotlyjs="cdn"`, rest `include_plotlyjs=False`).

```python
# scripts/_compare/plotly_helpers.py  (core of overlay_html)
import plotly.graph_objects as go
V2="#d97706"; VE="#4f46e5"
def overlay_html(per_obs, title=""):
    parts=[]
    for i,(obs, d) in enumerate(per_obs.items()):
        fig=go.Figure()
        fig.add_scatter(x=d["ve_t"], y=d["ve_v"], name="vEcoli", line=dict(color=VE))
        fig.add_scatter(x=d["v2_t"], y=d["v2_v"], name="v2ecoli", line=dict(color=V2))
        fig.update_layout(title=f"{title} {obs}", height=280, margin=dict(l=40,r=10,t=30,b=30),
                          hovermode="x unified", template="simple_white")
        parts.append(fig.to_html(include_plotlyjs=("cdn" if i==0 else False), full_html=False))
    return "".join(parts)
```

- [ ] **Step 4: Implement the card** — read each seed's trajectory from `state["v2_dir"]/v2ecoli_seed{n}.zarr` and `.../vecoli_seed{n}.zarr` for `OBSERVABLES`, average/collect per-seed arrays into the `per_obs` dict `overlay_html` expects, call it, wrap in a section, return `{card_html, verdict:"ungraded", axes:[]}`. Register `REPORT_CARD_STEPS["trajectory_report_card"]=...` and add to `__init__.py:72` import line.

- [ ] **Step 5: Run the test — expect PASS.**

- [ ] **Step 6: Commit** `git commit -m "feat(cards): interactive Plotly trajectory-overlay card"`

---

### Task B2: `distribution` card (violin + Welch)

**Files:** Create: `scripts/_compare/report_cards/distribution.py`; Modify: `plotly_helpers.py` (add `violin_html`), `__init__.py:72`; Test: `tests/test_report_card_distribution.py`

**Interfaces:**
- Consumes: `v2ecoli.library.card_criteria.grade_axis` (ttest); per-cell values pooled across seeds×gens from `state["observables"]`.
- Produces: `plotly_helpers.violin_html(axis_records: list, title) -> str`; card `update_distribution_report_card(state)`.

- [ ] **Step 1: Write the failing test** — assert card_html contains "violin", verdict ∈ vocab, and `axes` each carry `p`/`cohens_d`/`delta_rel` details.

```python
# tests/test_report_card_distribution.py
from tests.conftest import make_card_state
from scripts._compare.report_cards.distribution import update_distribution_report_card
def test_distribution_card_violin_and_graded():
    out = update_distribution_report_card(make_card_state())
    assert "violin" in out["card_html"].lower()
    assert out["verdict"] in ("within_tol","drift","mismatch","ungraded")
    assert all("detail" in a for a in out["axes"])
```

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement `violin_html`** — per axis a Plotly violin (v2 amber, vEcoli indigo) with the per-cell points overlaid (`points="all"`), annotated with `Δ% · p · d` from `grade_axis`. `to_html` fragments as in B1.

- [ ] **Step 4: Implement the card** — for each of the 7 axes, pull per-cell values (pooled across seeds×gens) for both engines from `state["observables"]`, call `grade_axis({"values":...}, {"type":"ttest","ref_values":...,"within_pct":0.05,"mismatch_pct":0.10,"p_min":0.05})`, build `axes` with the returned verdict/detail, render `violin_html`, return `{card_html, verdict: worst(...), axes}`. Register + import.

- [ ] **Step 5: Run the test — expect PASS.**

- [ ] **Step 6: Commit** `git commit -m "feat(cards): distribution/violin card with pooled Welch t-test"`

---

### Task B3: `metabolism` card (growth law + biomass)

**Files:** Create: `scripts/_compare/report_cards/metabolism.py`; Modify: `plotly_helpers.py` (add `grouped_bar_html`/`scatter_html`), `__init__.py:72`; Test: `tests/test_report_card_metabolism.py`

**Interfaces:**
- Consumes: per-condition final growth_rate + cell/dry mass from `state`; NOTE this card is per-study (one condition) — the cross-condition "growth law" figure is assembled at render time from all studies OR the card renders this condition's growth-rate vs its matched vEcoli and links siblings. Decide in Step 1.
- Produces: card `update_metabolism_report_card(state)`.

- [ ] **Step 1: Check flux availability + decide cross-condition assembly.** Inspect a smoke-run zarr (`out/smoke_redux/v2ecoli_seed00.zarr`) for any flux/exchange leaf. Record what's present. Decide: (a) growth-rate-vs-nutrient across conditions is assembled in `assemble_from_studies` (a small addition that collects each study's growth axis) OR (b) the per-study card shows this condition's growth-rate + biomass v2-vs-vEcoli only, with a note. Default to (b) to keep the card self-contained; log fluxes as follow-up if absent.

- [ ] **Step 2: Write the failing test** — assert card_html has a Plotly div and a growth-rate comparison; verdict ∈ vocab.

- [ ] **Step 3: Implement** the per-study metabolism card: growth-rate v2-vs-vEcoli trace + a biomass/mass-yield bar; if a flux leaf exists, add an exchange-flux scatter. Register + import.

- [ ] **Step 4: Run the test — expect PASS.**

- [ ] **Step 5: Commit** `git commit -m "feat(cards): metabolism growth-law/biomass card"`

---

### Task B4: `composition` card (mass-fraction + cell-cycle + perf)

**Files:** Create: `scripts/_compare/report_cards/composition.py`; Modify: `plotly_helpers.py`, `__init__.py:72`; Test: `tests/test_report_card_composition.py`

**Interfaces:**
- Consumes: protein/rna/cell/dry mass from `state`; division/gen labels from the zarr; wall time from the run summary JSON (optional perf panel).
- Produces: card `update_composition_report_card(state)`.

- [ ] **Step 1: Write the failing test** — assert card_html has a Plotly div (bars), verdict ∈ vocab, and axes for mass-fraction match.

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** — grouped-bar of proteome vs RNA vs "other" mass fractions (v2 vs vEcoli) computed from protein_mass/rna_mass/dry_mass; a small doubling-time / division-tick-per-gen readout from the gen labels; an optional steps/s perf line from the run summary. Grade the mass-fraction match with `rel_tol` (within 5% / drift 10%). Register + import.

- [ ] **Step 4: Run the test — expect PASS.**

- [ ] **Step 5: Commit** `git commit -m "feat(cards): composition + cell-cycle + perf card"`

---

### Task B5: Enable violin on the existing statistical card

**Files:** Modify: `scripts/_compare/report_card_section.py:22`, `scripts/comparison_report_card.py:95-106`; Test: extend `tests/test_report_card_distribution.py` or add `tests/test_statistical_violin.py`

- [ ] **Step 1: Write/extend the failing test** — render the statistical card via `core.link_registry['statistical_report_card']` on the fixture state and assert the HTML now contains a violin/strip SVG (`card_plots.violin_strip` output marker).

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Set `plot="violin"`** on `CARD_AXES` (`report_card_section.py:22`) and `EXTRA_AXES` (`comparison_report_card.py:95-106`) so `_axis_plot_svg` renders violins.

- [ ] **Step 4: Run the test — expect PASS.**

- [ ] **Step 5: Full local render smoke** — render the basal study from the smoke run and open the assembled report; confirm all 4 new cards + statistical violins appear.

```bash
cd ~/code/v2e-redux-invest
env PYTHONPATH=/Users/eranagmon/code/v2e-redux-invest V2E_VECOLI_DIR=/Users/eranagmon/code/vEcoli \
  ~/code/v2ecoli/.venv/bin/python scripts/comparison_report_card.py \
  --investigation v2ecoli-vecoli-comparison --out out/smoke_report \
  --study metabolism_redux_basal --local-pbg-seeds 1
```

- [ ] **Step 6: Commit** `git commit -m "feat(cards): render violins on the statistical card + local render smoke"`

---

## Phase C — Run on mini + serve

### Task C1: Verify + sync the mini

**Files:** Create: `scripts/run_redux_investigation_mini.sh`

- [ ] **Step 1: Verify mini reachability + state.** `ssh -o ConnectTimeout=8 mini 'uname -m; git -C ~/code/v2ecoli branch --show-current; ls ~/code/v2ecoli/out/cache_full ~/code/v2ecoli/out/compare_harness >/dev/null && echo CACHES_OK; ls ~/code/vEcoli/configs/metabolism_redux_*.json'`. If SSH hangs (as at design time), diagnose connectivity first (Tailscale up? `ssh mini` interactive?).

- [ ] **Step 2: Get the build branch + fork configs onto the mini.** `git fetch` the branch into the mini's v2ecoli (or `~/code/sync-pbg-to-mini.sh`); copy the 4 fork configs; verify `grep -c _should_inject_as_step` == 2 on the mini and the 5 studies load there.

- [ ] **Step 3: Commit the mini driver** (written in C2). 

### Task C2: Run the 5-condition redux ensemble on the mini (Ray)

- [ ] **Step 1: Write `scripts/run_redux_investigation_mini.sh`** — loops the 5 conditions, runs both engines via `run_comparison_ensemble.py --mode ray` with `V2E_RAY_THREADS=4` (≈3 concurrent on 12 cores), `--n-seeds 2 --max-generations 3`, ABSOLUTE `V2E_VECOLI_DIR`, `--match-vecoli-simdata` for v2, into `out/redux_5cond/<cond>/`. Verify-progress note: watch on-disk zarr, not the buffered `-p` log.
- [ ] **Step 2: Launch on the mini** headless (`mct` with the driver content, or detached ssh + nohup). Confirm the swap is active per condition (`+injected ['ecoli-metabolism']`).
- [ ] **Step 3: Monitor to completion** via zarr store presence per condition/seed; re-run any failed condition. Expected artifacts: `out/redux_5cond/<cond>/{v2ecoli,vecoli}_seed0{0,1}.zarr`.

### Task C3: Render the investigation

- [ ] **Step 1: Run `runner.run_investigation`** (render path) on the mini over the 5 studies → per-study cards + verdicts + assembled `standardized_comparison_report.html` + Plotly embeds; materialize verdicts.
- [ ] **Step 2: Verify** every condition graded on statistical + parca; all 4 new cards render with real multi-seed/gen data; no missing-observable errors.

### Task C4: Serve read-only from the mini dashboard

- [ ] **Step 1: Point the mini's read-only dashboard** (`vdash-ro` / `mdash`) at the investigation output; verify report cards, Plotly figures, and result stores are reachable over the tunnel (`mdash` from the laptop).
- [ ] **Step 2: Confirm** the investigation renders end-to-end in the browser: overview verdict matrix, per-condition cards, interactive Plotly.

### Task C5: Update investigation executive + findings

- [ ] **Step 1: Update** `investigation.yaml` executive summary + each study's finding with the 5-condition redux reproduction verdicts (per-condition statistical result). Use ruamel round-trip; verify parse.
- [ ] **Step 2: Final commit + push the build branch**; open/update the PR. Do NOT merge.

---

## Self-Review

**Spec coverage:** Section A (hardening) → A1 (configs), A2 (studies), A3 (smoke gate). Section B (4 cards) → B1 trajectory, B2 distribution, B3 metabolism, B4 composition, B5 statistical-violin (+ B0 fixture). Section C (mini + serve) → C1 verify/sync, C2 run, C3 render, C4 serve, C5 executive. Statistical-power pooling → B2 Step 4. Risks → A3 gate, B3 Step 1 (flux), C1 Step 1 (connectivity). All spec sections covered.

**Placeholder scan:** No "TBD"/"handle edge cases". Two deliberate in-plan investigation steps (A2 Step 1 verify `generations` reading; B3 Step 1 flux availability) produce a recorded decision, not deferred code. B0 requires reading the exact `observables`/`plot_trajs` shape before card code — this is setup, not a placeholder.

**Type consistency:** Card contract `update_<name>_report_card(state) -> {card_html, verdict, axes}` and `REPORT_CARD_STEPS["<name>_report_card"]` used consistently across B1–B5. `overlay_html`/`violin_html` live in `plotly_helpers.py`, consumed by their cards. `build_redux_config`/`main` (A1) consistent. `load_investigation` return `(ctx, specs)` consistent with A2.
