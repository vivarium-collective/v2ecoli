# Config → Composite translator (v1 vEcoli config → v2 executable document)

- **Date:** 2026-08-26
- **Status:** design (approved in brainstorming; pending spec review)
- **Owner:** Eran Agmon
- **Repos:** authored in `vivarium-collective/v2ecoli` (`v2ecoli/library/`); reaches
  sms-ecoli by the normal `scripts/sync_upstream.sh` merge. Workbench-side glue is
  generic and lives in `vivarium-workbench`.

## 1. Problem

The bigraph-loom "Composite Explorer" has a **Config → Apply** control. Today
`Apply` *builds* the target composite with the given parameter overrides
(`resolve_composite → build_generator`) and renders the resulting wiring. For a
native per-process composite (`ecoli_baseline`) that yields a rich graph. For the
`vecoli` composite — which wraps genuine upstream vEcoli as **one opaque
process-bigraph node** (`agents/<id>/vivarium_ecoli`, running an entire
vivarium-core `Engine`+`EcoliSim` inside its `update()`) — `Apply` can only show
that single node. The config's declared processes (`permeability`, `gillespie`,
`pg-shape`, a metabolism `swap`, …) live *inside* the wrapped engine and are never
visible as nodes.

We want a single notion of `Apply` that behaves like `initialize()`: from a
config, produce a bigraph document that carries the topology **and** initial
state — one artifact that is both **viewable** in the loom and **executable** by
`process_bigraph.Composite`.

## 2. Approach (decisions locked in brainstorming)

1. **Complement, don't replace.** Keep the `vecoli` wrapper as the
   "genuine vEcoli reference" arm. Add a *translator* that emits a native
   per-process composite from a config. Two composites, two purposes.
2. **Standalone mapper.** The translator maps config JSON → composite JSON
   **without running `EcoliSim`**. (Contrast: `vecoli_pbg_upstream.py`
   `build_upstream_cell_document` already does the *reuse-EcoliSim* whole-cell
   translation — see §6. We deliberately do not rebuild that.)
3. **Faithful JSON first.** The output is a faithful composite *document*
   (topology + declared structure), `Composite`-executable once its referenced
   stores are provided. Full-cell running still routes through the existing
   `vecoli`/`build_upstream_cell_document` path; pbg-native whole-cell execution
   is an explicit follow-on.
4. **Scope = the config's declared layer.** The translator renders only what the
   config declares — `add_processes` / `swap_processes` / `exclude_processes` /
   `spatial_environment_config` / `variants` — **not** the ~55 baseline processes
   or `sim_data`-derived state. This is the boundary that makes a standalone,
   pure-JSON, `Composite`-addressable document feasible (see §5).
5. **The translator supersedes the earlier "drill-in decoration" idea.** Because
   the config becomes a real per-process composite, we do not graft a `_draft`
   view onto the opaque `vecoli` node. Document-level unification is achieved by
   the translated document itself.

### Non-goals

- Reproducing the whole cell (baseline processes, `sim_data` configs, bulk/unique
  numpy state, partitioned Requester/Evolver identity, RNG, division) as
  standalone JSON. Those are the gaps in §5; they belong to the reuse-EcoliSim
  whole-cell path, not this translator.
- Changing how genuine-vEcoli reference runs (the `vecoli` wrapper is untouched).

## 3. Components & interfaces (`v2ecoli/library/`)

### 3.1 `config_to_composite.py` — the translator

```python
def config_to_composite(config: dict, *, fork_dir: str = "") -> dict:
    """vEcoli-style config → a Composite-executable v2 document of the
    config's DECLARED layer. Returns {"schema": {...}, "state": {...}}."""
```

- One `{"_type": "process", "address": "local:<ClassName>", "config": {...},
  "inputs": {port:[path]}, "outputs": {...}, "interval": <float>}` node per
  `add_processes` and per `swap_processes` value (the new process; annotated with
  what it replaces).
- **`config`** taken verbatim from the config's own `process_configs[name]`
  (plain dicts — no `sim_data`). Absent → `{}`.
- **`inputs`/`outputs`** from the config's `topology[name]` (authoritative),
  falling back to the fork's `topology_registry.access(name)` for processes the
  config leaves un-wired (e.g. `pg-shape`, `pg-maturation`). Path normalization:
  strip leading `..` walk-ups to a root store path; a nested sub-port dict (with
  `_path`) is expanded via the existing `_resolve_port_wires` helper where flat,
  else the port renders un-wired.
  - **Port direction.** A vivarium port is bidirectional (the process reads and
    writes the same store), so for **executability** each declared port is wired
    into **both** `inputs` and `outputs`, matched to the adapter-derived
    `inputs()`/`outputs()` schema (`translate_ports`) — a port the adapter classes
    as read-only stays out of `outputs`. This is the one place the executable
    translator differs from the Phase-1a *viewer*, which collapsed every port into
    `inputs` for rendering only. Where the adapter is unavailable (no fork), fall
    back to the viewer behavior (wire as `inputs`, un-executable but viewable).
- **Annotations:** `exclude_processes` → an `excluded_processes` store node;
  `spatial_environment_config` → an `environment` store node summarizing the
  reaction-diffusion field; `variants` → a `variants` store node listing the grid.
- **Store nodes** created for every wire target so edges have endpoints; a store
  never clobbers a process node of the same name.
- `fork_dir` / `$V2E_VECOLI_DIR` selects the vEcoli checkout for class-address /
  `topology_registry` lookup; the graph shape needs no fork.

### 3.2 `build_core` registration hook

The finding is that no standing `local:` address exists for an arbitrary
adapter-wrapped vivarium process — the existing paths inject a live `instance`.
To make the **pure-JSON `address` document executable**, the workspace's
`build_core` registers, for each declared process class, its adapter-wrapped
class under `local:<ClassName>`:

```python
core.register_process("<ClassName>", wrap_vivarium_process(v1_cls, name=...))
```

`wrap_vivarium_process` (`v2ecoli/library/vivarium_bridge.py:382`) already turns
an unmodified vivarium-core `Process` into a pbg `Process`
(`EcoliProcess`/`EcoliStep` base), deriving typed `inputs()`/`outputs()` from
`ports_schema()` via `translate_ports`. Registration follows the existing
`core.register_process(...)` pattern (e.g. `v2ecoli/composites/parca/composite.py`).
This closes the "executable" gap: `address: local:<ClassName>` then resolves via
`core.link_registry` at `Composite` realize time
(`bigraph_schema/protocols.py:local_lookup`).

### 3.3 `config_bigraph.py` (Phase-1a) folds in

The already-written structural transform (`config_to_document`, `_process_node`,
`_normalize_path`, `_resolve_process_meta`, the `topology_registry` fallback,
tuple-path handling) becomes the shared node/wiring core; the executable
translator subsumes the `_draft` viewer. The Phase-1a commit currently on an
*sms-ecoli* worktree is re-homed here (v2ecoli) under this branch.

### 3.4 Workbench integration (generic; `vivarium-workbench`)

- **env-worker method** `config_to_composite` — imports the workspace package's
  `config_to_composite` and calls it (same dispatch shape as
  `resolve_composite_state` / `registry_catalog`). Workbench hardcodes no
  workspace name; a workspace that provides the module opts in.
- **Route** `POST /api/config-to-composite {config_json, fork_dir?}` →
  `{state, schema, kind: "config-composite"}`. Loom renders it via the existing
  static-state / `composite:load` path.
- **Apply unification.** When a config is applied to a config-backed composite,
  `Apply` returns the translated document (viewable + executable) rather than an
  opaque build. The `vecoli` wrapper's opaque build remains available as the
  genuine-reference arm.

## 4. Data flow

```
config JSON
   │  config_to_composite(config, fork_dir)        (v2ecoli, standalone)
   ▼
{schema, state}  ── one document, two uses ──────────────────────────────
   │                                              │
   ▼ loom render (viewable)                       ▼ Composite(document, core)
   per-process nodes, ports, wiring               realize via local:<Class>
   (Explorer / Apply)                             (executable once referenced
                                                   stores are provided)
```

Whole-cell running remains `build_upstream_cell_document` → `Composite`
(reuse-EcoliSim); this translator's document is the declared-layer view + a
source-of-truth spec.

## 5. v1 → v2 mapping and the deferred gaps

Clean (in scope):

| v1 element | v2 JSON |
|---|---|
| process class (`process_registry`) | node `address: local:<ClassName>` (adapter-wrapped, registered §3.2) |
| topology tuples / `_path` / `..` | `inputs`/`outputs` wires (existing expansion helper) |
| config from the config's `process_configs` | node `config` (plain dict) |
| `add`/`swap`/`exclude`/`spatial`/`variants` | process nodes + annotation store nodes |

Deferred (out of scope — belong to the whole-cell / reuse path):

| v1 element | why deferred |
|---|---|
| config from `sim_data` (`get_config_by_name`) | numpy arrays / pint Quantities / serializer tags — not JSON-clean |
| PartitionedProcess (Requester+Evolver, shared instance) | shared identity needs a live instance / `shared_process` type |
| bulk/unique initial-state numpy arrays | must be bundled; large |
| RNG state, Division/MarkDPeriod | no clean v2 JSON path (division disabled even in `build_upstream_cell_document`) |

The declared antibiotic/swap processes avoid every deferred row (plain-dict
configs, not partitioned, no `sim_data`), which is exactly why the standalone
scope is feasible.

## 6. Reuse

- **`v2ecoli/library/vivarium_bridge.py`** — `wrap_vivarium_process` /
  `wrap_vivarium_instance`, `translate_ports`. The vivarium→pbg adapter.
- **`v2ecoli/library/vecoli_pbg_upstream.py::build_upstream_cell_document`** — the
  worked *reuse-EcoliSim* whole-cell translator; our reference for the mapping and
  the complement for whole-cell runs.
- **`build_core` `local:` registration** — existing pattern (`parca/composite.py`).
- **Loom renderer** (`vivarium_workbench/loom/src/convert.ts::stateToReactFlow`) —
  renders the document unchanged; ports render from `_inputs`/wires, edges derived.

## 7. Testing

- **Pure-logic** (no fork): synthetic config → assert document shape — process
  nodes for add/swap, ports from topology, flat paths wired, `..` normalized,
  nested sub-ports render un-wired, annotations present, store-never-clobbers-
  process. (Mirrors the Phase-1a tests, extended for `address`/`config`.)
- **Fork-backed**: assert each declared process's `local:<ClassName>` address
  resolves after the registration hook, and that `Composite({schema, state}, core)`
  **realizes** the declared-layer document without error (executability check).
- **Live**: `final_mec.json` (5 procs, 11 wired ports) and `mecillinam_shape.json`
  (7 procs, incl. `pg-shape`) render in the loom and realize.

## 8. Rollout

1. Author in the `v2ecoli` worktree `feat/config-to-composite-translator`
   (off `origin/main`); re-home the Phase-1a transform here.
2. Workbench glue on a `vivarium-workbench` worktree (env-worker method + route +
   Apply wiring).
3. PR v2ecoli upstream; sms-ecoli picks it up via `scripts/sync_upstream.sh`
   (no `descope/extensions.yaml` entry needed — it is upstream code).
4. Settle the workbench editable install (repoint at canonical main, drop the
   throwaway worktree) once the loom glue lands.

## 9. Open questions / follow-ons

- **Pbg-native whole-cell run** (materialize `sim_data` state + partitioned
  processes as pure JSON) — the big deferred item; may extend
  `build_upstream_cell_document` to optionally emit address-based nodes.
- **Initial-state values** for the declared layer — referenced now; whether to
  bundle a minimal store seed so the declared-layer document runs standalone.
- **Config inheritance** — `v2ecoli.workflow.config.load_config_with_inheritance`
  has a pre-existing crash on `final_mec.json` (`_merge_configs` set-dedupes a
  list-of-lists); the translator reads the raw config for now. Fix is a small,
  separate upstream change.
