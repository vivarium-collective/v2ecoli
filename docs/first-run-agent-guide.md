# v2ecoli + vivarium-workbench — first-run agent prompt

> **You are a Claude coding agent. A first-time user handed you this file to get
> them onto the v2ecoli + vivarium-workbench stack.** Your job: take them from
> zero to a *running* workbench with the `baseline` cell open in the loom viewer,
> then orient them and offer next steps. Execute the **Agent runbook** below
> gate-by-gate — run the commands yourself where the environment allows, verify
> each gate's success condition, and tell the user what you confirmed. After it's
> running, use the reference sections (authoring, contributing, troubleshooting)
> as their goals require.
>
> **Operating rules:**
> - Run project code through the venv (`.venv/bin/…`) — **never** a bare `python`.
> - Confirm each runbook gate before advancing; if one fails, consult §9 first,
>   don't improvise a workaround.
> - Ask before anything destructive (deleting `out/`, force-push, merging a PR).
> - Commands assume macOS/Linux. Reflects `main` of both repos as of mid-2026; if
>   a UI detail below doesn't match, trust the running app and update this file.

---

## What these things are

- **[v2ecoli](https://github.com/vivarium-collective/v2ecoli)** — the Covert lab's
  whole-cell *E. coli* model ([vEcoli](https://github.com/CovertLab/vEcoli))
  rebuilt on the [process-bigraph](https://github.com/vivarium-collective/process-bigraph)
  engine + [bigraph-schema](https://github.com/vivarium-collective/bigraph-schema)
  type system. Instead of one monolithic simulation, it's a set of **typed,
  independently-wireable processes** you *compose*. A whole cell is one
  `build_composite("ecoli_baseline")` call. The repo is also a **research workspace**:
  *investigations* (a question) and *studies* (the simulations answering it).

- **[vivarium-workbench](https://github.com/vivarium-collective/vivarium-workbench)** —
  a local web UI over any process-bigraph workspace. Browse the registry,
  **explore composites visually** (the embedded *bigraph-loom* graph viewer),
  view/run studies, and read investigation reports. Every action commits to a git
  branch in the workspace, so there's a full audit trail.

- **[viva-superpowers](https://github.com/vivarium-collective/viva-superpowers)** — a
  **Claude Code plugin** whose `/viva-*` skills drive the workbench's HTTP API so an
  AI assistant can author and run the workspace conversationally (add composites,
  studies, investigations; run sims; regenerate reports). This is *the* intended way
  an agent adds content. See §6.

---

## 0. Try it with zero install

To just *look* (no clone needed):

- **Interactive dashboard:** <https://vivarium-collective.github.io/v2ecoli/dashboard/>
- **Baseline Showcase report** (best starting point), **3D cell**, and the **report
  gallery** — all linked from the repo README.

---

## 1. Prerequisites

- [`uv`](https://docs.astral.sh/uv/) (Python/dep manager — provisions its own Python 3.12).
- A C compiler: **Xcode command-line tools** on macOS (`xcode-select --install`),
  `build-essential` on Linux (v2ecoli compiles vendored Cython extensions).
- `git`. For AI-assisted authoring (§6): **Claude Code**.

---

## ▶ Agent runbook — zero → live workbench (execute this first, in order)

This is your primary task. Work top to bottom; **do not skip a gate**. Once the
venv exists, run everything through it. After each gate, tell the user what you
verified. Sections §2–§4 below are the detailed reference for these steps.

- [ ] **G1 — Prereqs.** `uv --version` and `git --version` succeed; on macOS
  `xcode-select -p` resolves. If `uv` is missing, stop and have the user install
  it (§1) — don't hand-roll a Python/venv.
- [ ] **G2 — Clone + build.** `git clone https://github.com/vivarium-collective/v2ecoli.git && cd v2ecoli && uv sync`.
  Provisions Python 3.12, installs all deps, builds the Cython extensions.
  **Gate:** `.venv/bin/python -c "import v2ecoli"` exits 0.
- [ ] **G3 — ParCa cache.** The `baseline` composite needs an on-disk cache at
  `out/cache`. Build once: `.venv/bin/python scripts/build_cache.py`.
  **Gate:** `out/cache` exists. (Skip if already present.)
- [ ] **G4 — Install the workbench.** It is **not** a v2ecoli dependency — install
  it into the same venv from `main`:
  `uv pip install "vivarium-workbench @ git+https://github.com/vivarium-collective/vivarium-workbench.git@main"`.
  ⚠️ **Order + re-sync trap:** any later `uv sync` **removes** the workbench (it's
  not in v2ecoli's lockfile). If you re-sync, re-add it with
  `uv pip install -e <path-or-git-url> --no-deps`.
  **Gate:** `.venv/bin/python -c "import vivarium_workbench.server"` exits 0.
- [ ] **G5 — Serve.** The workspace root **is the repo root** (`workspace.yaml`
  lives at `./`, not in a `workspace/` subdir):
  `.venv/bin/vivarium-workbench serve --workspace . --port 8080`.
  **Gate:** `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8080/` → `200`.
  Report the URL to the user and leave the server running.
- [ ] **G6 — First look.** Have them open the URL → **Composites → `baseline` →
  Explore** (loom viewer; orientation in §3). Then offer to run a first sim (§4)
  or author new science (§6).

If a gate fails, go to §9 Troubleshooting before improvising.

---

## 2. Get v2ecoli running

```bash
git clone https://github.com/vivarium-collective/v2ecoli.git
cd v2ecoli
uv sync                      # provisions Python 3.12, installs all deps, builds Cython
```

**Golden rule:** run project code through the venv — `.venv/bin/python …`,
`.venv/bin/pytest`, `.venv/bin/v2ecoli-workflow …`. A bare `python` on your PATH is
missing `unum` and other deps and fails confusingly.

### ParCa (the parameter data the model reads)

The model reads a fitted `sim_data` blob produced by **ParCa** (the Parameter
Calculator). You do **not** need to re-run ParCa to start:

- A pre-computed `sim_data` ships in the repo (`models/parca/parca_state.pkl.gz`).
- The **baseline** composite (and derivatives) needs an on-disk **ParCa cache** at
  `out/cache`. Build it once: `.venv/bin/python scripts/build_cache.py` (fingerprinted).
- Full re-fit from scratch (rarely needed): `.venv/bin/v2ecoli-parca --mode fast`
  (~70 min). See `docs/generate_full_parca.md`.

> If a composite fails to resolve with `build failed: out/cache`, the ParCa cache is
> missing — build it with `scripts/build_cache.py`.

---

## 3. Launch the interactive workbench

The workbench isn't a hard dependency of v2ecoli — install it into the same venv from
`main`, then serve the repo *as a workspace*:

```bash
uv pip install "vivarium-workbench @ git+https://github.com/vivarium-collective/vivarium-workbench.git@main"
vivarium-workbench serve --workspace .          # picks a free port, prints the URL
# or pin one:  vivarium-workbench serve --workspace . --port 8080
```

(If you use viva-superpowers, `/viva-workbench` starts/opens this for you — §6.)

> ⚠️ The workbench is **not** a declared v2ecoli dependency, so a later `uv sync`
> will uninstall it. If that happens, re-add it with
> `uv pip install -e <checkout-path> --no-deps` (or the `git+…@main` URL). Point
> editable installs at an **up-to-date `main`** checkout, not a stale worktree.

Open the printed URL. The UI has side-rail tabs: **Workspace, Registry, Composites,
Investigations, Visualizations, Studies**. Then:

- **Composites** → pick one (e.g. `baseline`) → **Explore** opens the **bigraph-loom**
  graph viewer.
- **In the viewer**, zoom is *semantic*: zoomed out shows glyphs; zoom in and cards
  reveal ports → types → the process contract → its config (parameter values + types).
  Rectangles are **processes**, rounded nodes are **stores** (shared state); wires are
  port→store connections.
- **Processes panel** — group by **Subsystem** (transcription, translation, …),
  Connection, or Location; search; click a row to focus/center a process; ★ keeps a
  card open at full detail.
- **Adjust ▾** — one-shot layout helpers (stack stores by depth, center on a locked
  process, spread nodes to remove overlaps).
- Click a card's **config** box for every parameter with its value and type; click a
  **port** to see where it connects.

---

## 4. Run a simulation

All sweeps (single run, multi-seed, multi-generation, multi-variant) are one CLI + a
JSON config:

```bash
.venv/bin/v2ecoli-workflow --config v2ecoli/configs/default.json          # 1 seed, 1 generation
.venv/bin/v2ecoli-workflow --config v2ecoli/configs/two_generations.json  # a lineage across 2 divisions
```

- Example configs live in `v2ecoli/configs/` and support inheritance
  (`"inherit_from": ["default.json"]`).
- Knobs: `n_init_sims` (seeds), `generations` (multigen depth), `variants` (a parameter
  grid). The sweep expands to `variants × seeds × generations` lineages.
- Results land under `out/…` and can be browsed back in the workbench.

---

## 5. The mental model (internalize before editing)

| Concept | What it is |
|---|---|
| **Composite** | A wiring of processes + stores. `build_composite("ecoli_baseline")` = a whole cell. A colony embeds many cells via one bridge process; a kinetic-metabolism variant is a *different wiring of the same parts*. |
| **Process** | A unit of dynamics. Declares typed **input/output ports** via `inputs()` / `outputs()` + a `TOPOLOGY`. Its **config** = parameters (mostly ParCa-hydrated at build time). |
| **Store** | Shared state a process reads/writes through a port. |
| **Contract** | What a process advertises: which ports it reads, what it computes, what it writes. |
| **Study** | A self-contained research unit: one-or-more baseline composites, variants (parameter perturbations), interventions, runs, and conclusions. Lifecycle: **Design → Build → Simulate → Evaluate → Decide**. |
| **Investigation** | A named collection of Studies under one research question. Lives at `workspace/investigations/<inv>/`; studies at `…/studies/<slug>/study.yaml`. |

Composition is the point: adding/swapping a subsystem is *wiring*, not forking. New
science is a **new study**, not a patch to the model.

---

## 6. Authoring with viva-superpowers (the AI way to add content)

`viva-superpowers` is a Claude Code plugin (renamed from `pbg-superpowers` in the
pbg→viva rebrand). Its `/viva-*` skills call the workbench's API so your agent can add
composites, studies, and investigations and have them show up in the workbench
immediately.

> **Rebrand notes:** `/viva-*` are the canonical skill names; the old `/pbg-*` names
> still work as **deprecated aliases**. The old marketplace/repo paths redirect. The
> **PyPI/dependency package is still named `pbg-superpowers`** for now (that's why
> v2ecoli's `pyproject.toml` lists `pbg-superpowers`), and the Python import works
> under both `viva_superpowers` and the legacy `pbg_superpowers`.

### Install (one time)

In Claude Code:

```
/plugin marketplace add vivarium-collective/viva-superpowers
/plugin install viva-superpowers
/viva-init          # installs the /viva-* skills so they're invocable
```

Sanity-check you're in a workspace: `/viva-status` (detects `workspace.yaml`, reports
server liveness, study count).

### The skills, mapped to what you want to do

| Goal | Skill |
|---|---|
| Start / stop / open the workbench UI | **`/viva-workbench`** |
| See "is this a viva workspace?" + server status | **`/viva-status`** |
| Scaffold a *new* workspace (not needed for v2ecoli) | **`/viva-workspace`** |
| Browse / install / uninstall workspace modules | **`/viva-catalog`** |
| Open a composite in the loom Explorer | **`/viva-explore <composite-id>`** |
| Test-run a composite for N steps, see observables | **`/viva-run`** |
| Create & manage a **Study** (baselines, variants, runs, conclusions) | **`/viva-study`** |
| Create & manage an **Investigation** (group studies) | **`/viva-investigation`** |
| Generate an interactive **Visualization** | **`/viva-viz`** |
| Regenerate the dashboard + investigation reports | **`/viva-report`** |
| Read-only navigate the workspace graph / "decisions needed" | **`/viva-navigate`** |

### A typical authoring flow (composite → study → investigation → display)

1. **Add a composite.** In v2ecoli, composites are `@composite_generator` functions in
   `v2ecoli/composites/*.py` (a *different wiring of the same processes*). Write one
   there; it auto-registers and appears under the workbench **Composites** tab. New
   *processes* it needs go in `v2ecoli/processes/` (or use `/viva-expert` to draft
   one). Verify it resolves: **`/viva-run`** or **`/viva-explore <id>`**.
2. **Create a study around it:**
   `/viva-study new <study-name> <composite-id>` → writes
   `workspace/investigations/<inv>/studies/<slug>/study.yaml`. Then flesh it out with
   subcommands: `set-objective`, `baseline-add`, `variant-add` /
   `variant-set-params`, `run-baseline` / `run-variant`, `set-conclusion` /
   `set-verdicts`. (The **Build** phase — new process code — is `/viva-expert` or a
   hand edit in `v2ecoli/processes/`.)
3. **Group studies under a question:**
   `/viva-investigation new <name>` then `/viva-investigation add-study <study>` (also
   `open`, `list`, `set-overview`, `set-status`, `close`).
4. **Display / publish:** `/viva-workbench` to view live; `/viva-report` regenerates the
   dashboard + each investigation's self-contained HTML report. Everything you added
   shows in the workbench tabs immediately (it reads the workspace files); the
   published read-only site updates via the publish workflow (§8).

> You can do all of this by hand in the workbench UI too — the skills just let your
> agent do it conversationally and script multi-step authoring.

---

## 7. Contributing well

**Read [`AGENTS.md`](AGENTS.md) first** (and `CONTRIBUTING.md`). If you touch process
code, composite wiring, or the type system, AGENTS.md documents the checks your change
**must** pass — schema round-trip, port-contract, units, conservation — plus PR
conventions. Don't skip it.

**Workflow:**

1. Branch off `main` (protected — no direct pushes; changes land via PR).
2. Make the change; keep files focused.
3. **Test:** `.venv/bin/pytest` (or a targeted file: `… tests/test_x.py -x`).
   `uv sync --extra dev` pulls the test-only deps.
4. **Parity gate (critical for any model change):** the baseline simulation must stay
   **byte-identical**:
   ```bash
   .venv/bin/python scripts/parity_check.py --seconds 10 \
     --compare tests/golden/baseline_parity_signature.json --build-check
   ```
   If it mismatches, first check whether *current `main`* also mismatches the golden
   (the committed golden can lag real main) — build a signature from a clean `main`
   checkout and diff against yours. A true regression is a mismatch vs. current main;
   a stale golden is not your bug (but flag it).
5. Commit → push → open a PR to `main`. **Never** force-push or auto-merge; a human
   approves merges.

**Principles:** keep the biology faithful to upstream vEcoli (v2ecoli tracks dry mass
to a fraction of a percent through the full cell cycle). One clear responsibility per
file. Add new science as a study/investigation.

---

## 8. If you change how composites *display* (workbench / env-worker)

The published read-only dashboard's viewer reads **static snapshots** committed at
`reports/composite-state/*.json` — not a live server. So if you change how
composite-state is serialized (config, port types, contracts — this lives in the
workbench's `env_worker`), **regenerate + commit those snapshots** or the published
viewer keeps showing the old content:

```bash
# needs the ParCa cache (out/cache) AND a venv whose process-bigraph has composite_spec
.venv/bin/python scripts/regenerate_composite_states.py --only baseline
git add -f reports/composite-state/v2ecoli.composites.*.json   # reports/ is gitignored
```

Merging then bumps the `publish-dashboard` GitHub Action, which rebuilds the viewer
bundle from `vivarium-workbench@main` and ships the new snapshots.

> **CDN caveat:** GitHub Pages caches for a few minutes after a publish. Right after
> it "succeeds," the public URL may still serve the old bundle. Verify the real branch
> content via `raw.githubusercontent.com/<repo>/gh-pages/…`, not the Pages URL.

---

## 9. Troubleshooting quick-reference

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: unum` (or similar) | Ran a bare `python`. Use `.venv/bin/python`. |
| `serve` fails: `not a workspace (no workspace.yaml)` | `workspace.yaml` is at the **repo root** — pass `--workspace .`, not `--workspace workspace`. |
| `ModuleNotFoundError: vivarium_workbench` after a `uv sync` | `uv sync` dropped it (not a v2ecoli dep). Re-add: `uv pip install -e <path> --no-deps` (§3). |
| Composite won't resolve; `build failed: out/cache` | ParCa cache missing → `.venv/bin/python scripts/build_cache.py`. |
| Workbench lists only non-v2ecoli composites | The workspace venv has a stale `bigraph-schema` (missing `bigraph_schema.contract`) so `import v2ecoli` fails silently → re-run `uv sync` in the workspace. |
| `/viva-*` skills not found | Run `/viva-init` (or reinstall the plugin: `/plugin install viva-superpowers`). |
| Editing the workbench/schema has no effect | Editable installs must point at an **up-to-date `main`** checkout, not a stale worktree — reinstall from current `main`. |
| Published dashboard shows old data after a change | It serves static snapshots + a CDN — regenerate snapshots (§8) and allow for Pages CDN lag. |

---

## Where to read more

- `README.md` — architectures, pipelines, ParCa, performance/validation.
- `AGENTS.md` — the mandatory checks + PR conventions for model/type changes.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`.
- `docs/` — deep dives (ParCa, comparison pipeline, reports, native analyses, …).
- viva-superpowers repo — full `/viva-*` skill reference and workspace concepts.
