"""Multi-generation, multi-seed ensemble driver for the v2ecoli↔vEcoli comparison.

Runs EITHER engine through the IDENTICAL process-bigraph + XArray path so the
comparison is apples-to-apples (same framework, same compact emitter, same
compute), with NO Nextflow:

  --composite v2ecoli  → v2ecoli's ported ``baseline()`` composite
  --composite vecoli   → the PRISTINE upstream ``CovertLab/vEcoli`` model, run as
                         a process-bigraph composite via the EXTERNAL wrapper
                         (``v2ecoli.library.vecoli_pbg_upstream`` +
                         ``upstream_division``) with ZERO edits to the upstream
                         checkout. Opt back into the legacy composite-softfloor
                         fork with ``--vecoli-source composite-softfloor``.

Each seed runs ``max_generations`` past divisions (``run_multigen_xarray``,
daughter-following) and emits ONLY the compact comparison ``view`` (8 scalar
observables — no bulk/unique arrays) to a per-seed zarr store, which the Ray
backend ships to S3. Seeds run in parallel via ``run_seeds_parallel`` (Ray).

    python scripts/run_comparison_ensemble.py --composite v2ecoli \
        --condition basal --n-seeds 16 --max-generations 16 \
        --out-root s3://.../vecoli-output/<exp> [--chunk 60]

The compact view is the storage-minimizing piece the comparison needs: just the
report-card axes + a few diagnostics.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Compact comparison observables (confirmed set): report-card axes + diagnostics.
# Dotted paths under the followed agent; mapped to an XArray ``view`` below.
COMPARISON_PATHS = [
    "listeners.mass.cell_mass",
    "listeners.mass.dry_mass",
    "listeners.mass.protein_mass",
    "listeners.mass.rna_mass",
    "listeners.mass.instantaneous_growth_rate",
    "listeners.unique_molecule_counts.active_RNAP",
    "listeners.unique_molecule_counts.active_ribosome",
    "listeners.rna_synth_prob.total_rna_init",
]


def _spec_from_vecoli_config() -> str | None:
    """``from_vecoli_config`` from the comparison_spec.json baked into the image.

    ``COPY . .`` in the Dockerfile puts the repo (incl. comparison_spec.json) at
    REPO_ROOT, so the default Ray route picks up the spec's fork config WITHOUT
    any per-job API/sms-api parameter. Returns None if absent/unreadable.
    """
    spec_path = REPO_ROOT / "comparison_spec.json"
    if not spec_path.exists():
        return None
    try:
        spec = json.loads(spec_path.read_text())
    except Exception:  # noqa: BLE001
        return None
    return spec.get("from_vecoli_config") or None


def _build_v2ecoli(seed: int, condition: str, cache_dir: str,
                   overrides: dict | None = None):
    """v2ecoli ported composite (baseline) for the given media condition.

    The condition MUST be threaded through: the v2ecoli builder selects the
    condition-specific initial state (growth rate / doubling time / saved media
    from the ParCa state's condition_to_doubling_time — no refit) exactly as
    _build_vecoli does via sim.config["condition"]. Omitting it (the old bug)
    silently ran BASAL for every condition, so non-basal v2ecoli↔vEcoli rows
    compared a basal v2ecoli cell against a condition-specific vEcoli cell.

    ``overrides`` (PART 3, opt-in) supplies extra ``baseline`` generator
    parameters translated from the vEcoli config (see ``_translated_v2_overrides``).
    Default None keeps the working runs on the baseline defaults.

    NOTE: ``condition`` is NOT a ``build_composite`` parameter (that raises
    "unknown parameter(s) for ...baseline: ['condition']"). It is applied at
    CACHE-BUILD time — ``save_sim_input(sim_data, condition=...)`` →
    ``LoadSimData(condition=...)`` selects the condition-specific initial state
    (growth rate / doubling time / saved media from the ParCa state's
    ``condition_to_doubling_time`` — no refit). The sms-api ParCa builds a BASAL
    cache, so for non-basal we regenerate a condition+seed-specific bundle from
    the raw SimulationData (``simData.cPickle``, also dumped into the cache) into
    a per-condition subdir and build from that. Validated: basal 1267 /
    with_aa 2687 / acetate 350 fg initial cell_mass (vEcoli 1272 / 3013 / 346).
    """
    from v2ecoli import build_composite
    eff_cache = cache_dir
    if condition and condition != "basal":
        import pickle
        from v2ecoli.core import save_sim_input
        cond_cache = os.path.join(cache_dir, f"cond_{condition}_seed{seed:02d}")
        marker = os.path.join(cond_cache, "sim_data_cache.dill")
        sd_path = os.path.join(cache_dir, "simData.cPickle")
        if not os.path.exists(marker) and os.path.exists(sd_path):
            with open(sd_path, "rb") as f:
                sim_data = pickle.load(f)
            # save_sim_input's generate_initial_state spins a default emitter that
            # writes to a FIXED relative path (.pbg/parquet-runs/default/…), so
            # parallel seeds collide → FileExistsError (+ a secondary KeyError
            # 'emitter'). Run it in a unique cwd per (condition, seed) so that
            # side-effect is isolated; bundle_dir (absolute) is unaffected.
            _prev = os.getcwd()
            _iso = os.path.join(cache_dir, f".regen_{condition}_seed{seed:02d}")
            os.makedirs(_iso, exist_ok=True)
            os.chdir(_iso)
            try:
                save_sim_input(sim_data, bundle_dir=cond_cache, seed=seed,
                               condition=condition)
            finally:
                os.chdir(_prev)
        if os.path.exists(marker):
            eff_cache = cond_cache
    # emitter="null": the real comparison output is captured OUT OF BAND by the
    # external XArrayEmitter that run_multigen_xarray drives (the compact 8-path
    # view → zarr). The composite's INTERNAL parquet sink is pure redundancy
    # here, writes to ephemeral container-local disk that's never collected, and
    # — critically — its daughter instances collide on a shared default
    # partition at division and crash gen 2 (see v2ecoli/steps/division.py).
    # Disabling it leaves only the global_time RAMEmitter internally.
    kwargs: dict = {"cache_dir": eff_cache, "seed": seed, "emitter": "null"}
    if overrides:
        kwargs.update(overrides)
    return build_composite("baseline", **kwargs)


# --------------------------------------------------------------------------- #
# PART 1 — emit the RESOLVED v2ecoli build config as an S3 sidecar.
#
# The built Composite carries the resolved model under
# ``state['agents'][<id>]``: every process/step keyed by name with its
# ``address`` (which class ran), ``interval`` (per-step time_step), ``config``
# (per-process config), and ``inputs``/``outputs`` (the topology wiring). We
# summarize that into a small JSON-serializable dict — process set, per-process
# config KEYS + topology, time_step, condition/seed, and the build options —
# and write it next to the run's zarr so the report can diff it against vEcoli's
# resolved workflow_config.json. Best-effort: never crash a run.
# --------------------------------------------------------------------------- #
def extract_v2_build_config(composite, *, seed: int, condition: str,
                            cache_dir: str, options: dict) -> dict:
    """Pull the resolved v2ecoli config out of the built Composite document."""
    state = getattr(composite, "state", {}) or {}
    agents = state.get("agents", {}) or {}
    agent_id = next(iter(agents), None)
    agent = agents.get(agent_id, {}) if agent_id is not None else {}
    processes: list[dict] = []
    topology: dict[str, dict] = {}
    for name, node in sorted(agent.items()):
        if not isinstance(node, dict):
            continue
        ntype = node.get("_type")
        if ntype not in ("process", "step"):
            continue
        cfg = node.get("config")
        processes.append({
            "name": name, "type": ntype, "address": node.get("address"),
            "interval": node.get("interval"),
            "config_keys": sorted(cfg.keys()) if isinstance(cfg, dict) else [],
        })
        topology[name] = {"inputs": node.get("inputs") or {},
                          "outputs": node.get("outputs") or {}}
    return {
        "engine": "v2ecoli", "source": "composite_document",
        "condition": condition, "seed": seed, "cache_dir": str(cache_dir),
        "time_step": agent.get("timestep", state.get("timestep")),
        "global_time": state.get("global_time"), "agent_id": agent_id,
        "options": options, "n_processes": len(processes),
        "processes": processes, "topology": topology,
    }


def _write_json_sidecar(path: str, obj: dict) -> None:
    """Write ``obj`` as JSON to ``path`` (local or s3://) via fsspec.

    Uses the SAME storage layer the zarr writer relies on (fsspec/s3fs), so the
    sidecar lands alongside the per-seed stores under the run's out_root.
    """
    import fsspec
    text = json.dumps(obj, default=str, indent=2)
    so: dict = {}
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
    if str(path).startswith("s3://") and region:
        so = {"client_kwargs": {"region_name": region}}
    with fsspec.open(path, "w", **so) as f:
        f.write(text)


# --------------------------------------------------------------------------- #
# PART 3 — advance the goal that vEcoli configs AUTO-TRANSLATE to v2ecoli.
#
# End goal: v2ecoli configured FROM the vEcoli config (equivalence by
# construction) rather than from two independent default sets. Today the gap is
# a NAMESPACE one: ``translate_vecoli_config`` yields a vEcoli-WORKFLOW-shaped
# dict (lineage_seed, single_daughters, max_duration_per_gen, condition, …),
# but ``build_composite("baseline")`` accepts a DISJOINT set of generator
# parameters (seed, cache_dir, transcript/polypeptide_initiation_mode,
# config_overrides, the feature toggles, emitter, injected_processes). The two
# share almost no key names, so a straight pass-through maps very little.
#
# What is default-vs-translatable today:
#   * DEFAULT-only (set inside build_composite("baseline"), NOT from vEcoli cfg):
#       process set + topology, time_step (1 s), the feature toggles
#       (ppgpp_regulation on, trna/supercoiling/mass off), the initiation modes,
#       and emitter — these come from the generator defaults.
#   * Cache-driven (NOT a build_composite arg): condition / media and every
#       ParCa parameter — they enter via the cache_dir ParСa build, not baseline().
#   * Translatable NOW: only keys whose NAME already matches a baseline generator
#       parameter (e.g. config_overrides). seed is owned by the per-seed loop and
#       is deliberately NOT overridden.
#
# What fully closing the loop requires: a NAME-MAPPING layer (vEcoli key ->
# baseline generator param OR cache/ParCa knob), e.g. {fixed_media/condition ->
# the cache build, mar_regulon -> a feature toggle, ppgpp* -> ppgpp_regulation,
# process_configs/swap_processes -> injected_processes/config_overrides}, plus
# driving the ParCa cache build itself from parca_options. Until that table
# exists, this path is gated OFF by default (--translate-vecoli-config) so it
# cannot destabilize the working runs; when on it applies only the safely
# name-matched subset and records the full translation in the sidecar.
# --------------------------------------------------------------------------- #
def _baseline_param_names() -> set:
    """The declared ``baseline`` generator parameter names (for override filtering)."""
    import v2ecoli.composites  # noqa: F401  (register generators)
    from pbg_superpowers.composite_generator import _REGISTRY
    for e in _REGISTRY.values():
        if e.name == "baseline":
            return set(e.parameters)
    return set()


def _overrides_from_resolved(resolved: dict) -> tuple[dict, dict]:
    """Translate an ALREADY-resolved vEcoli config into baseline overrides.

    Returns ``(overrides, translated_full)``. ``overrides`` keeps only keys that
    are declared ``baseline`` generator parameters (minus ``seed``, owned by the
    per-seed loop). For typical vEcoli configs this is nearly empty — that
    emptiness IS the namespace finding above.
    """
    from scripts._compare.config_adapter import translate_vecoli_config
    valid = _baseline_param_names()
    translated = translate_vecoli_config(resolved)
    overrides = {k: v for k, v in translated.items()
                 if k in valid and k != "seed"}
    return overrides, translated


def _translated_v2_overrides(vecoli_config_path: str) -> tuple[dict, dict]:
    """Resolve+translate a vEcoli config; return (overrides, translated_full).

    Resolves via the fork's own loader (``resolve_vecoli_config``), then delegates
    to :func:`_overrides_from_resolved`.
    """
    from scripts._compare.config_adapter import resolve_vecoli_config
    return _overrides_from_resolved(resolve_vecoli_config(vecoli_config_path))


# Cache for the once-per-process vEcoli import + global registrations. Re-importing
# ``ecoli.*`` per seed would re-trigger module-level emitter registration (e.g.
# ``parquet``) into the process-global registry → "registry already contains an
# entry for parquet" on the 2nd seed. So do the path-fix + purge + imports +
# resolve-dispatch registration EXACTLY ONCE, then reuse the cached symbols.
_VECOLI: dict = {}


def _ensure_vecoli() -> dict:
    """Import the vEcoli composite stack once per process; cache its symbols.

    The v2ecoli image bakes the vEcoli composite-softfloor checkout at
    ``/app/vEcoli``, but an installed vEcoli release (lacking
    ``ecoli.composites.ecoli_composite``) may already be cached in sys.modules
    transitively. ``sys.path.insert`` does NOT re-resolve an already-imported
    package, so we drop every ``ecoli*`` module ONCE to force a fresh import
    from the composite-softfloor branch. Mirrors scripts/run_vecoli_composite.py
    (incl. the SharedProcess resolve dispatches the composite branch needs).
    """
    if _VECOLI:
        return _VECOLI
    vecoli_dir = os.environ.get(
        "V2E_VECOLI_DIR", str(REPO_ROOT.parent / "vEcoli"))

    # Force-cache the INSTALLED (Cython-COMPILED) wholecell tree BEFORE vecoli_dir
    # goes on sys.path. The composite checkout ships wholecell SOURCE only (its
    # Cython _build_sequences etc. are unbuilt), so once vecoli_dir is first on
    # the path an uncached `import wholecell.utils.polymerize` would resolve to
    # the unbuilt copy and raise "Failed to import Cython module". Importing the
    # whole installed tree now pins every wholecell.* submodule to the compiled
    # build; ecoli (loaded from vecoli_dir below) then reuses those cached modules.
    import importlib as _importlib
    import pkgutil as _pkgutil
    import wholecell as _wholecell
    for _mi in _pkgutil.walk_packages(_wholecell.__path__, "wholecell."):
        try:
            _importlib.import_module(_mi.name)
        except Exception:
            pass  # optional/unbuildable submodules; ecoli imports only the core ones

    sys.path.insert(0, vecoli_dir)

    # ecoli/__init__.py registers parquet/updaters/dividers/serializers into
    # vivarium's PROCESS-GLOBAL registries as an import side effect. A shadow
    # ecoli release may already have registered those keys, and vivarium's
    # Registry.register RAISES on a duplicate key whose item object differs —
    # which re-importing ecoli guarantees (new class/function identities). Make
    # register idempotent (skip keys already present) so re-importing ecoli from
    # the composite checkout below doesn't blow up on "registry already contains
    # an entry for parquet". The freshly imported ecoli.composites.* and the
    # soft-floor transcription still come from vecoli_dir; only the shared
    # registry helpers fall back to whatever registered first (functionally
    # equivalent — the soft-floor lives in transcription, not in these helpers).
    import vivarium.core.registry as _vreg
    if not getattr(_vreg.Registry, "_idempotent_patched", False):
        _orig_register = _vreg.Registry.register

        def _idempotent_register(self, key, item, alternate_keys=tuple()):
            if key in self.registry:
                return None
            return _orig_register(self, key, item, alternate_keys)

        _vreg.Registry.register = _idempotent_register
        _vreg.Registry._idempotent_patched = True

    # The composite-softfloor vEcoli expects cloud-URI helpers that the v2ecoli
    # venv's INSTALLED wholecell.utils.filepath lacks (is_cloud_uri /
    # cloud_path_join, added in the composite branch for S3 output). Inject them
    # onto the installed module rather than swapping wholecell wholesale —
    # wholecell is SHARED with the v2ecoli infra (parallel_seeds / xarray_run)
    # and must stay the installed version for the v2ecoli engine to keep working.
    import posixpath as _posixpath
    import wholecell.utils.filepath as _wfp
    if not hasattr(_wfp, "is_cloud_uri"):
        _wfp.is_cloud_uri = lambda path: str(path).startswith(("s3://", "gs://", "gcs://"))
    if not hasattr(_wfp, "cloud_path_join"):
        _wfp.cloud_path_join = lambda *parts: _posixpath.join(*parts)

    # Drop any cached (possibly shadow) ecoli modules so the imports below
    # resolve from vecoli_dir (the composite-softfloor checkout).
    for _m in [m for m in sys.modules if m == "ecoli" or m.startswith("ecoli.")]:
        del sys.modules[_m]

    from ecoli.experiments.ecoli_master_sim import EcoliSim
    from ecoli.composites.ecoli_composite import build_composite_native
    from ecoli.library.bigraph_types import ECOLI_TYPES, SharedProcess
    from process_bigraph import Composite
    from bigraph_schema import allocate_core
    import v2ecoli.types  # noqa: F401  (register v2ecoli types/dispatches)

    # SharedProcess (new in the composite branch) needs resolve dispatches.
    from bigraph_schema.methods.resolve import resolve as _resolve
    from bigraph_schema.schema import Tuple as _Tuple
    from v2ecoli.types.process import ProcessInstance

    # This file uses `from __future__ import annotations` (PEP 563), so the
    # dispatch annotations below are stored as STRINGS and resolved lazily by
    # beartype from the function's __module__ globals (which is "__main__" when
    # the script runs directly, e.g. on a Ray worker). Defined inside this
    # function, _Tuple/SharedProcess/ProcessInstance would be locals that vanish
    # on return → "Forward reference '_Tuple' unimportable from module
    # '__main__'". Publish them into the module (and __main__) globals so the
    # forward-ref resolution succeeds, mirroring run_vecoli_composite.py's
    # module-level definitions.
    _main = sys.modules.get("__main__")
    for _nm, _ty in (("_Tuple", _Tuple), ("SharedProcess", SharedProcess),
                     ("ProcessInstance", ProcessInstance)):
        globals()[_nm] = _ty
        if _main is not None:
            setattr(_main, _nm, _ty)

    @_resolve.dispatch
    def _resolve_tuple_shared(current: _Tuple, update: SharedProcess, path=None):
        return update

    @_resolve.dispatch
    def _resolve_shared_tuple(current: SharedProcess, update: _Tuple, path=None):
        return current

    @_resolve.dispatch
    def _resolve_pi_shared(current: ProcessInstance, update: SharedProcess, path=None):
        return update

    @_resolve.dispatch
    def _resolve_shared_pi(current: SharedProcess, update: ProcessInstance, path=None):
        return current

    _VECOLI.update(
        vecoli_dir=vecoli_dir, EcoliSim=EcoliSim,
        build_composite_native=build_composite_native,
        ECOLI_TYPES=ECOLI_TYPES, Composite=Composite, allocate_core=allocate_core,
    )
    return _VECOLI


# Known-good upstream-master-compatible sim_data fallback (vEcoli-built ParCa;
# the v2ecoli-built out/kb pickle predates upstream master's TCS
# modified_molecules and won't unpickle cleanly under master).
_UPSTREAM_SIMDATA_FALLBACK = (
    "/Users/eranagmon/code/v2ecoli/out/compare_harness/vecoli_parca/"
    "kb/simData.cPickle")


def _build_vecoli(seed: int, condition: str, cache_dir: str,
                  source: str = "upstream", native_kwargs: dict | None = None):
    """Build the ``--composite vecoli`` engine.

    ``source="upstream"`` (DEFAULT) builds the **pristine, unmodified
    CovertLab/vEcoli** model via the external wrapper
    (:func:`v2ecoli.library.upstream_division.build_upstream_agents_composite`):
    upstream ``EcoliSim`` is imported + each vivarium process wrapped externally
    (ZERO edits to the upstream checkout), placed under a colony ``('agents',)``
    map with a harness-side pbg division step so it runs multi-generationally
    through ``run_multigen_xarray``.

    ``native_kwargs`` (from ``config_adapter.vecoli_native_kwargs`` on the
    resolved vEcoli config) threads the ORIGINAL config's run knobs straight into
    the wrapper — ``time_step`` and ``exclude_processes`` — so the vEcoli side
    runs the SAME source config the v2ecoli side was translated from. ``condition``
    stays CLI-driven (the harness launches one job per media condition), so the
    config's own ``condition`` does not override it.

    ``source="composite-softfloor"`` (opt-in) builds the legacy
    vivarium-collective/vEcoli ``composite-softfloor`` fork via
    ``build_composite_native`` (see :func:`_build_vecoli_composite_softfloor`).
    """
    native = native_kwargs or {}
    if source == "upstream":
        from v2ecoli.library.upstream_division import (
            build_upstream_agents_composite)
        sim_data_path = os.path.abspath(os.path.join(cache_dir, "simData.cPickle"))
        if not os.path.exists(sim_data_path):
            sim_data_path = _UPSTREAM_SIMDATA_FALLBACK
        # monomer_counts_listener is a read-only listener that trips on a TCS
        # sim_data provenance skew; excluded per the wrapper smoke-driver
        # precedent. Does not affect the comparison observables. Merge it with
        # any exclude_processes the source config requested.
        exclude = list(native.get("exclude_processes") or [])
        if "monomer_counts_listener" not in exclude:
            exclude.append("monomer_counts_listener")
        extra: dict = {}
        if "time_step" in native:
            extra["time_step"] = float(native["time_step"])
        composite, _info = build_upstream_agents_composite(
            seed=seed,
            condition=condition,
            sim_data_path=sim_data_path,
            exclude_processes=exclude,
            verbose=True,
            **extra,
        )
        return composite
    if source == "composite-softfloor":
        return _build_vecoli_composite_softfloor(seed, condition, cache_dir)
    raise ValueError(f"unknown vecoli source {source!r}")


def _build_vecoli_composite_softfloor(seed: int, condition: str, cache_dir: str):
    """Legacy: vivarium-collective/vEcoli ``composite-softfloor`` fork composite.

    Mirrors scripts/run_vecoli_composite.py: ``build_composite_native`` from the
    vEcoli composite-softfloor branch (carries the ppGpp soft-floor). The import
    + global registration happens once via ``_ensure_vecoli``; this builds a
    fresh per-seed composite on its own ``core``. Opt-in via
    ``--vecoli-source composite-softfloor``.
    """
    v = _ensure_vecoli()
    prev = os.getcwd()
    os.chdir(v["vecoli_dir"])
    try:
        core = v["allocate_core"]()
        core.register_types(v["ECOLI_TYPES"])
        # EcoliSim.from_cli() parses sys.argv, which here still holds the
        # comparison driver's args (e.g. --composite vecoli). That collides with
        # EcoliSim's own --composite_checkpoint_* options ("ambiguous option:
        # --composite") → argparse sys.exit(2), which inside a Ray worker shows
        # up as WorkerCrashedError. Strip argv to just the program name (as
        # run_vecoli_composite.py does) so from_cli() uses defaults; the
        # condition/seed are applied via sim.config below.
        _saved_argv = sys.argv
        sys.argv = sys.argv[:1]
        try:
            sim = v["EcoliSim"].from_cli()
        finally:
            sys.argv = _saved_argv
        # condition → media is set via the vEcoli sim config; lineage_seed=seed.
        sim.config["condition"] = condition
        sim.config["seed"] = seed
        # Consume the SAME ParCa output as the v2ecoli composite: the ParCa step
        # (v2ecoli build_cache → _write_sim_input_bundle) now also dumps the raw
        # SimulationData as <cache_dir>/simData.cPickle. Point the vEcoli
        # composite's LoadSimData at it instead of the unbuilt default
        # /app/vEcoli/out/kb/simData.cPickle.
        sim.config["sim_data_path"] = os.path.join(cache_dir, "simData.cPickle")
        sim.processes = sim._retrieve_processes(
            sim.processes, sim.add_processes, sim.exclude_processes, sim.swap_processes)
        sim.topology = sim._retrieve_topology(
            sim.topology, sim.processes, sim.swap_processes, sim.log_updates)
        sim.process_configs = sim._retrieve_process_configs(sim.process_configs, sim.processes)
        state = v["build_composite_native"](core, sim.config)
        # NB: do NOT clear composite.to_run here. run_vecoli_composite.py clears
        # it because it drives ecoli.run(chunk) MANUALLY; our driver
        # (run_multigen_xarray) calls composite.run(1)/composite.run(chunk)
        # itself, so an emptied to_run means zero steps execute → an empty zarr
        # (group hierarchy only, no observable arrays). The v2ecoli composite
        # leaves to_run at its default and emits correctly; mirror that.
        composite = v["Composite"](dict(schema=dict(), state=state), core=core)
        return composite
    finally:
        os.chdir(prev)


def make_run_one(*, composite_kind: str, condition: str, cache_dir: str,
                 max_generations: int, max_steps: int, chunk: int,
                 out_root: str, seed_start: int = 0,
                 vecoli_config: str | None = None,
                 translate_config: bool = False,
                 vecoli_source: str = "upstream",
                 from_vecoli_config: str | None = None,
                 vecoli_dir: str | None = None):
    """Return a ``run_one(seed)`` closure for ``run_seeds_parallel``."""
    from v2ecoli.library.xarray_run import run_multigen_xarray, view_from_emit_paths

    # PART 3 (opt-in): translate the vEcoli config into baseline overrides ONCE.
    v2_overrides: dict | None = None
    v2_translated: dict | None = None
    vecoli_native: dict | None = None

    # NEW (headline capability): drive BOTH engines from ONE vEcoli FORK config.
    # Resolve it ONCE (using v2ecoli's own loader so the fork needs no venv),
    # then derive each engine's form: the v2ecoli side gets the TRANSLATED
    # baseline overrides; the vEcoli side gets the ORIGINAL config's native run
    # knobs (time_step / exclude_processes). One source config → both engines.
    if from_vecoli_config:
        try:
            from scripts._compare.config_adapter import (
                resolve_vecoli_config_local, vecoli_native_kwargs)
            fork_dir = vecoli_dir or os.environ.get(
                "V2E_VECOLI_DIR", str(REPO_ROOT.parent / "vEcoli"))
            resolved = resolve_vecoli_config_local(from_vecoli_config, fork_dir)
            if composite_kind == "v2ecoli":
                v2_overrides, v2_translated = _overrides_from_resolved(resolved)
                print(f"[from-vecoli-config] {from_vecoli_config} → v2 overrides "
                      f"{sorted(v2_overrides)} ({len(v2_translated)} translated keys)")
            else:
                vecoli_native = vecoli_native_kwargs(resolved)
                print(f"[from-vecoli-config] {from_vecoli_config} → vecoli native "
                      f"kwargs {sorted(vecoli_native)} (run on the ORIGINAL config)")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] from-vecoli-config resolve failed: {type(e).__name__} {e}")

    # Legacy opt-in: translate the vEcoli config into baseline overrides ONCE
    # (only when --from-vecoli-config did not already populate them).
    if (composite_kind == "v2ecoli" and translate_config and vecoli_config
            and v2_overrides is None):
        try:
            v2_overrides, v2_translated = _translated_v2_overrides(vecoli_config)
            print(f"[translate] vEcoli→v2 overrides applied: {sorted(v2_overrides)} "
                  f"({len(v2_translated)} translated keys total)")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] vEcoli→v2 config translation failed: {type(e).__name__} {e}")

    def run_one(seed: int) -> dict:
        t0 = time.time()
        store_path = f"{out_root.rstrip('/')}/{composite_kind}_seed{seed:02d}.zarr"
        # local stores: clear stale
        if "://" not in str(store_path) and Path(store_path).exists():
            shutil.rmtree(store_path)

        # vivarium-process engine: genuine vEcoli as a SINGLE pbg node with vivarium's
        # own Engine inside (faithful by construction — no re-implemented partition/
        # reconcile/division). It runs its OWN single-lineage multigen + XArrayEmitter
        # (vivarium handles division internally; no pbg _add), emitting the SAME
        # v2ecoli-format zarr, so it bypasses the pbg-division run_multigen_xarray below.
        if composite_kind == "vecoli" and vecoli_source == "vivarium-process":
            from v2ecoli.library.vivarium_ecoli_engine import run_vivarium_ecoli_pbg_multigen
            sim_data_path = os.path.abspath(os.path.join(cache_dir, "simData.cPickle"))
            if not os.path.exists(sim_data_path):
                sim_data_path = _UPSTREAM_SIMDATA_FALLBACK
            res = run_vivarium_ecoli_pbg_multigen(
                store_path=store_path, sim_data_path=sim_data_path, condition=condition,
                seed=seed, max_generations=max_generations, max_steps_per_gen=max_steps,
                chunk=chunk, exclude_processes=["monomer_counts_listener"],
                fork_dir=os.environ.get("V2E_VECOLI_DIR"),
                experiment_id=f"cmp-vecoli-{condition}-seed{seed:02d}",
                variant=0, lineage_seed=seed)
            return {"seed": seed, "wall_seconds": round(time.time() - t0, 1),
                    "store": str(store_path), "steps": None,
                    "generations": list(range(1, res.get("generations", 0) + 1))}

        if composite_kind == "v2ecoli":
            composite = _build_v2ecoli(seed, condition, cache_dir,
                                       overrides=v2_overrides)
            # PART 1: emit the resolved v2ecoli build config ONCE (lowest seed)
            # as a sidecar next to the zarr stores. Best-effort — never crash.
            if seed == seed_start:
                try:
                    cfg = extract_v2_build_config(
                        composite, seed=seed, condition=condition,
                        cache_dir=cache_dir,
                        options={"overrides": v2_overrides or {},
                                 "translated_from_vecoli": vecoli_config
                                 if translate_config else None,
                                 "translated": v2_translated})
                    _write_json_sidecar(
                        f"{out_root.rstrip('/')}/v2ecoli_build_config.json", cfg)
                    print(f"[config] wrote v2ecoli_build_config.json "
                          f"({cfg['n_processes']} processes) under {out_root}")
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] v2 build-config sidecar emit failed: "
                          f"{type(e).__name__} {e}")
        elif composite_kind == "vecoli":
            composite = _build_vecoli(seed, condition, cache_dir,
                                      source=vecoli_source,
                                      native_kwargs=vecoli_native)
        else:
            raise ValueError(f"unknown composite_kind {composite_kind!r}")
        # include_vectors=True (the xarray_run default): the scalar counts
        # listeners.unique_molecule_counts.active_RNAP / active_ribosome share a
        # LEAF NAME with the unique-molecule coordinate vectors, so the legacy
        # include_vectors=False skipped the counts by name too — dropping them
        # from the comparison. The view is still only the 8 COMPARISON_PATHS, so
        # this keeps the two counts (scalars) without emitting any coord vectors.
        view = view_from_emit_paths(COMPARISON_PATHS, include_vectors=True)
        metadata_base = {
            "experiment_id": f"cmp-{composite_kind}-{condition}-seed{seed:02d}",
            "engine": composite_kind,
            "condition": condition,
            "variant": 0,
            "lineage_seed": seed,
            "time_step": 1.0,
            "max_duration": float(max_steps),
            "agent_id": "0",
        }
        result = run_multigen_xarray(
            composite,
            store_path=store_path,
            view=view,
            metadata_base=metadata_base,
            max_steps=max_steps,
            max_generations=max_generations,
            chunk=chunk,
        )
        return {"seed": seed, "wall_seconds": round(time.time() - t0, 1),
                "store": str(store_path), **{k: result.get(k) for k in
                ("steps", "generations")}}

    return run_one


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--composite", required=True, choices=["v2ecoli", "vecoli"])
    p.add_argument("--condition", default="basal")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "out" / "cache"),
                   help="v2ecoli condition cache (v2ecoli engine only).")
    p.add_argument("--n-seeds", type=int, default=16)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--max-generations", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=60000,
                   help="hard tick cap across all generations (safety stop).")
    p.add_argument("--chunk", type=int, default=60)
    p.add_argument("--out-root", required=True,
                   help="dir or s3:// prefix for per-seed zarr stores.")
    p.add_argument("--mode", default="ray", help="run_seeds_parallel mode (ray/serial).")
    p.add_argument("--vecoli-source", default="upstream",
                   choices=["upstream", "composite-softfloor", "vivarium-process"],
                   help="--composite vecoli engine: 'upstream' (default) = the "
                        "pristine CovertLab/vEcoli external wrapper; "
                        "'composite-softfloor' = the legacy vivarium-collective "
                        "fork via build_composite_native.")
    p.add_argument("--vecoli-config", default=None,
                   help="vEcoli config path to translate into v2ecoli build "
                        "overrides (PART 3; only used with --translate-vecoli-config).")
    p.add_argument("--translate-vecoli-config", action="store_true",
                   help="OPT-IN: configure the v2ecoli build FROM the translated "
                        "vEcoli config (default off — keeps the working runs on "
                        "baseline defaults).")
    p.add_argument("--from-vecoli-config", default=None,
                   help="Path WITHIN the vEcoli fork (under V2E_VECOLI_DIR), e.g. "
                        "configs/default.json, to drive BOTH engines from: the "
                        "v2ecoli side gets the TRANSLATED overrides, the vecoli "
                        "side runs the ORIGINAL config. If omitted, falls back to "
                        "$V2E_FROM_VECOLI_CONFIG then comparison_spec.json's "
                        "from_vecoli_config (baked into the image) — so the default "
                        "Ray route is spec-driven with no per-job flag.")
    args = p.parse_args(argv)

    # Resolve --from-vecoli-config: CLI flag > env > baked comparison_spec.json.
    from_vc = (args.from_vecoli_config
               or os.environ.get("V2E_FROM_VECOLI_CONFIG")
               or _spec_from_vecoli_config())
    vecoli_dir = os.environ.get("V2E_VECOLI_DIR", str(REPO_ROOT.parent / "vEcoli"))

    from v2ecoli.library.parallel_seeds import run_seeds_parallel
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    run_one = make_run_one(
        composite_kind=args.composite, condition=args.condition,
        cache_dir=args.cache_dir, max_generations=args.max_generations,
        max_steps=args.max_steps, chunk=args.chunk, out_root=args.out_root,
        seed_start=args.seed_start, vecoli_config=args.vecoli_config,
        translate_config=args.translate_vecoli_config,
        vecoli_source=args.vecoli_source,
        from_vecoli_config=from_vc, vecoli_dir=vecoli_dir)
    parallel = run_seeds_parallel(seeds, run_one, mode=args.mode)
    summaries = getattr(parallel, "results", parallel)
    ensemble = {"composite": args.composite, "condition": args.condition,
                "n_seeds": len(seeds), "max_generations": args.max_generations,
                "wall_s": getattr(parallel, "wall_s", None),
                "seeds": list(summaries)}
    print(json.dumps(ensemble, indent=2))


if __name__ == "__main__":
    main()
