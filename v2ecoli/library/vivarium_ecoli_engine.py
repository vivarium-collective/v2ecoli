"""Single-process, faithful-by-construction vEcoli engine for the comparison harness.

Instead of decomposing upstream vEcoli into ~50 individual process-bigraph steps and
reimplementing vivarium's Engine semantics (partition/allocation, update reconciliation,
division) in process-bigraph — which introduced a class of faithfulness bugs (mass
explosion from a missing bulk reconcile; a division-handoff crash from `_add` dropping
live process instances) — this module runs the GENUINE vEcoli on its OWN vivarium Engine
inside one wrapper, advancing it incrementally with ``Engine.run_for``.

Faithful by construction: partition, allocation, and per-tick update reconciliation are
all handled by vivarium-core exactly as upstream intends. Multi-generation is single-
lineage (matching v2ecoli's ``run_multigen_xarray``): each generation runs a genuine
vEcoli Engine to its division point, vEcoli's own ``divide_cell`` splits the state, and
the followed daughter seeds the next generation's Engine. No process-bigraph division.

Fork-robust: the vEcoli fork is selected by ``V2E_VECOLI_DIR`` (or the ``fork_dir`` arg),
imported through the same compiled-``wholecell``-pinning shim the step-decomposed wrapper
uses (:func:`v2ecoli.library.vecoli_pbg_upstream._ensure_upstream`). Swapping forks is a
one-knob change: point ``V2E_VECOLI_DIR`` at a different checkout + its matching ParCa
``sim_data``.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from v2ecoli.library.vecoli_pbg_upstream import _ensure_upstream
from v2ecoli.library.upstream_division import _n_chromosomes, _inc_to_fg
from v2ecoli.library.division import divide_cell


# Path to a FULL vEcoli config file the wrapped ``EcoliSim`` should load natively
# (add_processes / process_configs / topology / spatial_environment_config /
# variants). None => build the default baseline sim (``swap_processes``/``flow``
# only). Set for the duration of a run by ``run_vivarium_ecoli_pbg_multigen``
# (which restores it in a ``finally``), so every per-generation ``build_vivarium_
# ecoli`` rebuild picks it up without threading a param through 3 call sites, and
# a whole-config run cannot leak into a later default run in the same process.
_ECOLISIM_CONFIG_FILE: str | None = None


def set_ecolisim_config_file(path: str | None) -> None:
    """Point the wrapped single-node ``EcoliSim`` at a full vEcoli config file to
    load natively (or clear it with ``None``). Faithful-by-construction: vEcoli's
    own composer applies ``add_processes``/``process_configs``/``topology``/
    ``spatial_environment_config``/``variants``/etc. exactly as upstream, so a
    config whose model content can't be expressed as ``swap_processes``/``flow``
    (a multi-agent / spatial-environment model) still runs correctly as one pbg
    node — for ANY fork selected by ``$V2E_VECOLI_DIR``."""
    global _ECOLISIM_CONFIG_FILE
    _ECOLISIM_CONFIG_FILE = path


def _select_variant_params(variants_config: dict, variant_index: int):
    """Resolve a 1-based ``variant_index`` against a config ``variants`` block.

    Mirrors the fork's ``runscripts.create_variants`` convention: ``parse_variants``
    returns ``param_dicts`` and the fork maps ``param_dicts[i]`` to variant ``i+1``,
    reserving index 0 for the unperturbed baseline. Returns ``(None, None)`` for the
    baseline, else ``(variant_name, params_dict)``. Delegates the grid expansion
    (op prod/zip/add, value/linspace/arange) entirely to the fork.
    """
    if variant_index <= 0:
        return None, None                       # baseline: strict no-op
    if not variants_config:
        # variant_index >= 1 but nothing to select from → fail loud rather than
        # silently running the unperturbed baseline (Finding #3 in spirit).
        raise ValueError(
            f"variant index {variant_index} requested but config has no "
            f"'variants' block to select from")
    if len(variants_config) != 1:
        raise ValueError(
            f"expected exactly one variant in config, got {sorted(variants_config)}")
    (name, cfg), = variants_config.items()
    from runscripts.create_variants import parse_variants  # fork-bound
    param_dicts = parse_variants(cfg)
    idx = variant_index - 1
    if idx >= len(param_dicts):
        raise IndexError(
            f"variant index {variant_index} out of range: {len(param_dicts)} "
            f"grid point(s) (valid 1..{len(param_dicts)})")
    return name, param_dicts[idx]


def _apply_config_variant(sim_data, variants_config: dict, variant_index: int):
    """Apply the selected config variant to ``sim_data`` via the fork's own
    ``ecoli.variants.<name>.apply_variant``. Returns ``(sim_data, meta|None)``.
    ``sim_data`` must already be a fresh (non-shared) object."""
    name, params = _select_variant_params(variants_config, variant_index)
    if name is None:
        return sim_data, None
    import importlib
    mod = importlib.import_module(f"ecoli.variants.{name}")  # fork-bound
    sim_data = mod.apply_variant(sim_data, params)
    return sim_data, {"variant_name": name, "variant_index": int(variant_index),
                      "params": params}


# ---------------------------------------------------------------------------
# Build a genuine vEcoli vivarium Engine (fork-parameterized)
# ---------------------------------------------------------------------------

@dataclass
class EngineHandle:
    """A built genuine-vEcoli vivarium Engine plus the metadata needed to drive it."""
    engine: object
    sim: object
    dry_mass_inc_dict: dict = field(default_factory=dict)
    condition: str = "basal"
    media_id: str = "minimal"
    time_step: float = 1.0


def build_vivarium_ecoli(
    *,
    sim_data_path: str,
    condition: str = "basal",
    seed: int = 0,
    time_step: float = 1.0,
    exclude_processes: list | None = None,
    swap_processes: dict | None = None,
    flow: dict | None = None,
    fork_dir: str | None = None,
    initial_overlay: dict | None = None,
    variant: int = 0,
) -> EngineHandle:
    """Build the genuine upstream vEcoli composite and wrap its vivarium Engine.

    ``fork_dir`` (or ``$V2E_VECOLI_DIR``) selects the vEcoli checkout; ``sim_data_path``
    is its matching upstream ParCa ``simData.cPickle``. ``initial_overlay`` (a daughter's
    divided ``bulk``/``unique``/``environment``/``boundary``) seeds a non-founder
    generation; ``None`` builds a fresh founder. ``variant`` selects a 1-based grid
    point from the loaded config's ``variants`` block (0 = baseline, no-op); only
    applies when a full config file (``set_ecolisim_config_file``) is in effect.
    """
    if fork_dir:
        os.environ["V2E_VECOLI_DIR"] = fork_dir
    up = _ensure_upstream()
    EcoliSim = up["EcoliSim"]
    from vivarium.core.engine import Engine

    _cfgfile = _ECOLISIM_CONFIG_FILE
    _argv = sys.argv
    _cwd = os.getcwd()
    _fork = fork_dir or os.environ.get("V2E_VECOLI_DIR")
    _saved_sers: dict = {}
    if _cfgfile:
        # Fork-bind vivarium's serializers before ``EcoliSim`` deserializes the
        # config's ``!ParameterSerializer[...]`` / ``!units[...]`` tags. A
        # Serializer binds ``param_store`` at its module's import time; the
        # INSTALLED vEcoli (registered into vivarium's GLOBAL serializer_registry
        # first) lacks fork-only params -> "No parameter found at path". Replace
        # each registered serializer with a FRESH instance imported from the
        # fork's ``ecoli.library.serialize`` (now first on sys.path after
        # ``_ensure_upstream``), and RESTORE the originals afterward so this does
        # not leak into a later default run in the same process (determinism).
        try:
            import importlib as _il
            from vivarium.core.registry import serializer_registry as _sreg, Serializer as _Ser
            _fs = _il.import_module("ecoli.library.serialize")
            for _nm, _obj in list(_sreg.registry.items()):
                _forkcls = getattr(_fs, type(_obj).__name__, None)
                if isinstance(_forkcls, type) and issubclass(_forkcls, _Ser) and _forkcls is not _Ser:
                    _saved_sers[_nm] = _obj
                    _sreg.registry[_nm] = _forkcls()
        except Exception as _se:  # noqa: BLE001 — fall back to registry serializers
            print(f"[build_vivarium_ecoli] serializer fork-bind skipped "
                  f"({type(_se).__name__}: {_se})")
    # A full config file loads via ``--config`` (EcoliSim resolves its
    # ``inherit_from`` chain relative to the fork's own configs dir, so run from
    # cwd = the fork checkout). Empty argv => the default baseline sim.
    sys.argv = ([sys.argv[0], "--config", _cfgfile] if _cfgfile else sys.argv[:1])
    try:
        if _cfgfile and _fork and os.path.isdir(_fork):
            os.chdir(_fork)
        sim = EcoliSim.from_cli()
    finally:
        sys.argv = _argv
        os.chdir(_cwd)
        if _saved_sers:
            from vivarium.core.registry import serializer_registry as _sreg2
            _sreg2.registry.update(_saved_sers)

    sim.config["condition"] = condition
    sim.config["seed"] = int(seed)
    sim.config["sim_data_path"] = sim_data_path
    sim.config["time_step"] = float(time_step)
    # Apply the CONDITION's media. genuine vEcoli's LoadSimData defaults
    # media_timeline to ((0,'minimal'),) and `condition` alone never updates it
    # (the "have to change both" footgun), so without this the runner runs every
    # condition on 'minimal' media — e.g. no_oxygen ran AEROBIC instead of
    # minimal_minus_oxygen (O2=0). Set the condition's nutrients as fixed_media
    # (flows through Ecoli(config) -> LoadSimData(**config)). Hand the loaded
    # sim_data in too so EcoliSim reuses it (skips a 2nd ~300MB load).
    # Preload sim_data ONCE (reuse it so EcoliSim skips a 2nd ~300MB load). The
    # preload itself is best-effort — on failure EcoliSim just loads
    # sim_data_path natively — EXCEPT when a variant is requested: we must not
    # silently run the unperturbed baseline, so a preload failure there is loud.
    _sd_obj = None
    _variant_simdata_tmp = None   # temp pickle holding variant-mutated sim_data
    try:
        import pickle as _pickle
        with open(sim_data_path, "rb") as _sdf:
            _sd_obj = _pickle.load(_sdf)
    except Exception as _loaderr:  # noqa: BLE001 — sim_data reuse is best-effort
        if _cfgfile and int(variant):
            raise        # a requested variant cannot be applied → fail loud
        print(f"[build_vivarium_ecoli] sim_data preload skipped: "
              f"{type(_loaderr).__name__} {_loaderr}")

    if _sd_obj is not None:
        # Variant application must fail LOUD for ALL exception types — it is
        # deliberately NOT inside the best-effort media block below. A swallowed
        # ImportError (fork off path) / KeyError / AttributeError from
        # parse_variants/apply_variant would silently run the UNPERTURBED
        # baseline while the caller believes variant=k applied.
        if _cfgfile and int(variant):
            _variants_cfg = sim.config.get("variants") or {}
            _sd_obj, _vmeta = _apply_config_variant(_sd_obj, _variants_cfg, int(variant))
            if _vmeta:
                print(f"[build_vivarium_ecoli] applied config variant "
                      f"'{_vmeta['variant_name']}' #{variant}: {_vmeta['params']}")
                # PERSIST the variant-mutated sim_data and repoint sim_data_path.
                # The composer's ``LoadSimData`` ALWAYS reloads sim_data from
                # ``sim_data_path`` and IGNORES a handed-in ``sim.config['sim_data']``
                # object, so an in-memory-only variant (e.g. a ``field_timeline``
                # that ``LoadSimData.get_field_timeline_config`` reads off
                # ``external_state``) is silently discarded and the run reverts to
                # the unperturbed baseline. Writing the mutated sim_data to a temp
                # pickle and pointing sim_data_path at it is what actually delivers
                # the variant to the composer.
                import tempfile as _tf
                _vfd, _variant_simdata_tmp = _tf.mkstemp(
                    prefix="v2e_variant_simdata_", suffix=".cPickle")
                os.close(_vfd)
                with open(_variant_simdata_tmp, "wb") as _vf:
                    _pickle.dump(_sd_obj, _vf)
                sim.config["sim_data_path"] = _variant_simdata_tmp
        # The condition's nutrients (fixed_media) IS best-effort — a lookup miss
        # just means EcoliSim keeps its default media; never fatal.
        try:
            _nutrients = (_sd_obj.conditions.get(condition, {}) or {}).get("nutrients")
            if _nutrients:
                sim.config["fixed_media"] = _nutrients
        except Exception as _mediaerr:  # noqa: BLE001 — media reuse is best-effort
            print(f"[build_vivarium_ecoli] media-from-condition skipped: "
                  f"{type(_mediaerr).__name__} {_mediaerr}")
        # Hand the loaded (possibly variant-mutated) sim_data to EcoliSim so it
        # reuses THIS object (must always run when the preload succeeded).
        sim.config["sim_data"] = _sd_obj
    # Division is handled by THIS module's single-lineage loop (genuine divide_cell
    # between generations), not vivarium's in-Engine Division roundtrip — so each
    # generation is one clean genuine-vEcoli Engine.
    sim.config["divide"] = False
    sim.config["d_period"] = False
    sim.config["generations"] = None
    sim.config["progress_bar"] = False
    sim.config["emit_paths"] = []
    if exclude_processes:
        existing = list(sim.config.get("exclude_processes", []) or [])
        sim.config["exclude_processes"] = existing + list(exclude_processes)
    # Process swap (e.g. FBA metabolism -> MetabolismRedux): EcoliSim applies
    # ``swap_processes`` natively at build_ecoli() the same way vEcoli's own
    # configs/metabolism_redux.json does, and ``flow`` reorders the swapped
    # process's dependents. Merge so a caller-supplied swap composes with any
    # config default rather than clobbering it.
    if swap_processes:
        merged_swap = dict(sim.config.get("swap_processes", {}) or {})
        merged_swap.update(swap_processes)
        sim.config["swap_processes"] = merged_swap
    if flow:
        merged_flow = dict(sim.config.get("flow", {}) or {})
        merged_flow.update(flow)
        sim.config["flow"] = merged_flow

    # Capture the per-media expectedDryMassIncreaseDict (drives the division
    # threshold) the same way upstream_division does — via the composer's sim_data.
    dry_mass_inc_dict: dict = {}
    import ecoli.composites.ecoli_master as _em
    _orig_init = _em.Ecoli.__init__
    _captured: dict = {}

    def _capturing_init(self, config, *a, **k):
        _orig_init(self, config, *a, **k)
        _captured["composer"] = self

    _em.Ecoli.__init__ = _capturing_init
    try:
        sim.build_ecoli()
    finally:
        _em.Ecoli.__init__ = _orig_init
        # The composer has now loaded sim_data from sim_data_path; the temp
        # pickle (if any) has served its purpose — remove it so per-generation
        # variant builds don't leak ~300MB files.
        if _variant_simdata_tmp:
            try:
                os.unlink(_variant_simdata_tmp)
            except OSError:
                pass
    composer = _captured.get("composer")
    if composer is not None:
        try:
            dry_mass_inc_dict = dict(
                composer.load_sim_data.sim_data.expectedDryMassIncreaseDict)
        except Exception:
            dry_mass_inc_dict = {}

    init_state = sim.generated_initial_state
    if initial_overlay:
        for key in ("bulk", "unique", "environment", "boundary"):
            if key in initial_overlay:
                init_state[key] = initial_overlay[key]

    engine = Engine(
        processes=sim.ecoli.processes,
        steps=sim.ecoli.steps,
        flow=sim.ecoli.flow,
        topology=sim.ecoli.topology,
        initial_state=init_state,
        emitter={"type": "null"},
        progress_bar=False,
    )

    media_id = "minimal"
    try:
        env = init_state.get("environment", {})
        media_id = env.get("media_id", "minimal") if isinstance(env, dict) else "minimal"
    except Exception:
        pass

    return EngineHandle(
        engine=engine, sim=sim, dry_mass_inc_dict=dry_mass_inc_dict,
        condition=condition, media_id=media_id, time_step=float(time_step))


# ---------------------------------------------------------------------------
# Read observables out of a live Engine state
# ---------------------------------------------------------------------------

def _state(engine):
    return engine.state.get_value()


# The 7 report-card observables (scripts/comparison_report_card.py OBSERVABLES), at the
# SAME paths v2ecoli emits (run_comparison_ensemble COMPARISON_PATHS) so both engines read
# identically: 5 in listeners.mass, 2 in listeners.unique_molecule_counts.
MASS_OBS = ("cell_mass", "dry_mass", "protein_mass", "rna_mass",
            "instantaneous_growth_rate")
COUNT_OBS = ("active_RNAP", "active_ribosome")
OBSERVABLES = MASS_OBS + COUNT_OBS


def cell_observables(engine) -> dict:
    """Pull the 7 comparison observables from the live Engine state. Single-cell, no
    agents wrapper (divide=False). Scalar axes + the raw bulk/unique for division.

    Spatial-aware: a whole-config run with a ``spatial_environment_config`` builds
    vEcoli's own colony composite whose top-level state is
    ``{agents:{<id>:cell}, multibody, reaction_diffusion, fields}`` — the cell's
    ``listeners``/``bulk``/``unique`` live under ``agents/<id>``, not the top level.
    Follow the first agent so the emit + the division gate read the real cell (else
    every observable reads 0 off an empty top level)."""
    st = _state(engine)
    _agents = st.get("agents")
    if isinstance(_agents, dict) and _agents:
        st = _agents.get("0") or next(iter(_agents.values()))
    listeners = st.get("listeners", {}) or {}
    mass = listeners.get("mass", {}) or {}
    umc = listeners.get("unique_molecule_counts", {}) or {}
    obs = {k: float(mass.get(k, 0.0) or 0.0) for k in MASS_OBS}
    obs.update({k: float(umc.get(k, 0.0) or 0.0) for k in COUNT_OBS})
    obs.update({
        "bulk": st.get("bulk"),
        "unique": st.get("unique"),
        "environment": st.get("environment", {}),
        "boundary": st.get("boundary", {}),
    })
    return obs


def _select_bulk_observables(obs_bulk, ids: list) -> dict:
    """Pick ``ids`` out of the inner cell's bulk store as floats.

    Handles BOTH representations the wrapped engine can hand us:
    - a numpy structured array (the REAL vEcoli ``bulk`` store: fields ``id``
      and ``count`` — see ``ecoli.library.schema`` / ``initial_conditions``),
      resolved id->count via a name->index map built once per call;
    - a plain ``Mapping`` (name->count), the shape the dict-based tests use.

    A requested id absent from the store yields ``0.0`` (a species absent this
    tick), never a KeyError/IndexError. Empty ``ids`` -> ``{}``. ``None`` or any
    unexpected value falls back to the mapping path -> all ``0.0`` (no crash).

    PURE: operates only on the passed array/dict — no fork import. Never uses
    truthiness on ``obs_bulk`` (a multi-element ndarray raises ValueError on
    ``bool``); emptiness is gated solely on ``ids``.
    """
    if not ids:
        return {}
    # numpy structured array (has a compound dtype with the expected fields)
    dtype = getattr(obs_bulk, "dtype", None)
    if dtype is not None and dtype.names is not None \
            and "id" in dtype.names and "count" in dtype.names:
        name_to_idx = {str(n): i for i, n in enumerate(obs_bulk["id"])}
        counts = obs_bulk["count"]
        return {i: (float(counts[name_to_idx[i]]) if i in name_to_idx else 0.0)
                for i in ids}
    # Mapping (dict-like) fallback: name->count, missing -> 0.0.
    from collections.abc import Mapping
    src = obs_bulk if isinstance(obs_bulk, Mapping) else {}
    return {i: float(src.get(i, 0.0)) for i in ids}


def _select_exchange_fluxes(environment, fluxes: dict) -> dict:
    """Pick named metabolic exchange fluxes out of the cell's environment store.

    ``fluxes`` maps ``leaf_name -> exchange_key`` (e.g.
    ``{"violacein_exchange": "VIOLACEIN[c]", "glucose_exchange": "GLC[p]"}``).
    The exchange dmdt lives at ``environment["exchange"]`` (keyed by metabolite
    id, uptake negative / secretion positive — the same store #547 measured with
    175 keys). A key absent this tick yields ``0.0`` so the leaf stays a
    continuous trace. Sign is preserved verbatim; consumers decide on ``abs``.

    Deliberately generic: no molecule is special-cased here. GENERIC/violacein-
    agnostic by design — the flux map is supplied by config, so this stays out of
    the shared model's knowledge of any particular pathway."""
    if not fluxes:
        return {}
    from v2ecoli.steps.derivers.exchange_flux_listener import resolve_exchange_key
    env = environment if isinstance(environment, dict) else {}
    exchange = env.get("exchange")
    exchange = exchange if isinstance(exchange, dict) else {}
    out = {}
    for leaf, key in fluxes.items():
        v = resolve_exchange_key(exchange, key)
        out[leaf] = float(v) if v is not None else 0.0
    return out


# ---------------------------------------------------------------------------
# pbg Process: genuine vEcoli as ONE process-bigraph node (vivarium Engine inside)
# ---------------------------------------------------------------------------

from process_bigraph import Process


class VivariumEcoliProcess(Process):
    """Genuine upstream vEcoli as a SINGLE process-bigraph node, with vivarium-core's
    own Engine running inside.

    This PRESERVES the process-bigraph design: v2ecoli and vEcoli compare on the SAME
    pbg runtime (one pbg node each at ``agents/0``), so the comparison stays a true
    pbg-vs-pbg test — but vivarium handles partition / update-reconciliation / division
    *inside* this node, so none of the Engine-reimplementation bugs (the bulk-reconcile
    mass explosion; the ``_add`` division crash) can arise.

    ``update(state, interval)`` advances the inner vivarium Engine by ``interval`` via
    ``Engine.run_for`` and writes the cell's mass observables back to the pbg store
    (``listeners.mass.*``), so the standard ``XArrayEmitter`` view reads them exactly as
    it does for v2ecoli. REST-ready: the in-process Engine can later be swapped for an
    out-of-process vivarium service behind this same port interface.
    """

    config_schema = {
        "sim_data_path": "string",
        "condition": {"_type": "string", "_default": "basal"},
        "seed": {"_type": "integer", "_default": 0},
        "time_step": {"_type": "float", "_default": 1.0},
        "exclude_processes": {"_type": "list[string]", "_default": []},
        "fork_dir": {"_type": "string", "_default": ""},
        "variant": {"_type": "integer", "_default": 0},
        "observable_bulk_ids": {"_type": "list[string]", "_default": []},
        # {leaf_name: exchange_key} — metabolic exchange fluxes to emit under
        # listeners.exchange_flux.<leaf> (generic; the caller names the keys).
        "exchange_fluxes": {"_type": "map[string]", "_default": {}},
    }

    # Set by build_vivarium_ecoli_composite to inject a pre-built (possibly daughter-
    # overlaid) engine so the process doesn't rebuild EcoliSim. Consumed once at
    # construction. Single build is sequential; Ray seeds are separate processes.
    _PENDING_HANDLE = None

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        if VivariumEcoliProcess._PENDING_HANDLE is not None:
            self._handle = VivariumEcoliProcess._PENDING_HANDLE
            VivariumEcoliProcess._PENDING_HANDLE = None
            self._obs_bulk_ids = list(self.config.get("observable_bulk_ids") or [])
        else:
            self._handle = build_vivarium_ecoli(
                sim_data_path=self.config["sim_data_path"],
                condition=self.config["condition"],
                seed=int(self.config["seed"]),
                time_step=float(self.config["time_step"]),
                exclude_processes=list(self.config.get("exclude_processes") or []) or None,
                fork_dir=(self.config.get("fork_dir") or None),
                variant=int(self.config.get("variant") or 0),
            )
            self._obs_bulk_ids = list(self.config.get("observable_bulk_ids") or [])
        self._exchange_fluxes = dict(self.config.get("exchange_fluxes") or {})

    def inputs(self):
        return {}

    def outputs(self):
        # Recomputed-absolute each tick → 'set' semantics (overwrite), matching
        # vivarium's listener _updater='set'.
        out = {"listeners": {
            "mass": {k: "overwrite[float]" for k in MASS_OBS},
            "unique_molecule_counts": {k: "overwrite[float]" for k in COUNT_OBS},
        }}
        if self._obs_bulk_ids:
            out["bulk"] = {i: "overwrite[float]" for i in self._obs_bulk_ids}
        if self._exchange_fluxes:
            out["listeners"]["exchange_flux"] = {
                leaf: "overwrite[float]" for leaf in self._exchange_fluxes}
        return out

    def update(self, state, interval):
        self._handle.engine.run_for(float(interval))
        obs = cell_observables(self._handle.engine)
        upd = {"listeners": {
            "mass": {k: obs[k] for k in MASS_OBS},
            "unique_molecule_counts": {k: obs[k] for k in COUNT_OBS},
        }}
        if self._exchange_fluxes:
            upd["listeners"]["exchange_flux"] = _select_exchange_fluxes(
                obs.get("environment"), self._exchange_fluxes)
        if self._obs_bulk_ids:
            upd["bulk"] = _select_bulk_observables(obs.get("bulk", {}), self._obs_bulk_ids)
        return upd

    def divide(self) -> dict:
        """Split the inner cell with vEcoli's faithful ``divide_cell``; return
        daughter-0's overlay (bulk/unique/environment/boundary) to seed the next
        generation's Engine. The split is vivarium-native — no pbg ``_add``."""
        obs = cell_observables(self._handle.engine)
        d1, _d2 = divide_cell({
            "bulk": obs["bulk"], "unique": obs["unique"],
            "environment": obs["environment"], "boundary": obs["boundary"]})
        return d1

    def division_signals(self) -> tuple:
        """``(dry_mass_fg, n_chromosomes)`` for the lineage driver's division gate."""
        obs = cell_observables(self._handle.engine)
        return obs["dry_mass"], _n_chromosomes(obs["unique"])


def build_vivarium_ecoli_composite(
    *,
    sim_data_path: str,
    condition: str = "basal",
    seed: int = 0,
    time_step: float = 1.0,
    exclude_processes: list | None = None,
    swap_processes: dict | None = None,
    flow: dict | None = None,
    fork_dir: str | None = None,
    core=None,
    agent_id: str = "0",
    initial_overlay: dict | None = None,
    variant: int = 0,
    observable_bulk_ids: list | None = None,
    exchange_fluxes: dict | None = None,
):
    """Wrap a single :class:`VivariumEcoliProcess` as a one-node pbg Composite under
    ``agents/<agent_id>`` — the genuine-vEcoli analogue of the v2ecoli agent composite,
    so the SAME ``run_multigen_xarray`` / ``XArrayEmitter`` path serves both engines.

    ``initial_overlay`` (a daughter's divided bulk/unique/env/boundary) seeds a non-
    founder generation. Returns ``(composite, info)``. The process writes
    ``listeners.mass.*`` (overwrite/set semantics) into the agent store each tick.
    """
    from process_bigraph import Composite
    if core is None:
        from v2ecoli.core import build_core
        core = build_core()

    # Build the (optionally daughter-seeded) engine once and inject it so the process
    # doesn't rebuild EcoliSim.
    VivariumEcoliProcess._PENDING_HANDLE = build_vivarium_ecoli(
        sim_data_path=sim_data_path, condition=condition, seed=int(seed),
        time_step=float(time_step), exclude_processes=list(exclude_processes or []) or None,
        swap_processes=swap_processes or None, flow=flow or None,
        fork_dir=fork_dir or None, initial_overlay=initial_overlay, variant=int(variant))
    proc = VivariumEcoliProcess(config={
        "sim_data_path": sim_data_path, "condition": condition, "seed": int(seed),
        "time_step": float(time_step),
        "exclude_processes": list(exclude_processes or []),
        "fork_dir": fork_dir or "",
        "variant": int(variant),
        "observable_bulk_ids": list(observable_bulk_ids or []),
        "exchange_fluxes": dict(exchange_fluxes or {}),
    }, core=core)
    iface = proc.interface()

    # Wire the process's output ports to the agent's stores. The process only
    # DECLARES a ``bulk`` output port when observable ids are configured, so only
    # then do we wire ``bulk`` -> ``["bulk"]`` (agents/<id>/bulk); otherwise pbg
    # would drop an unmapped-but-declared port and the observables never land.
    _outputs = {"listeners": ["listeners"]}
    if list(observable_bulk_ids or []):
        _outputs["bulk"] = ["bulk"]
    cell_state = {
        "vivarium_ecoli": {
            "_type": "process",
            "instance": proc,
            "_inputs": iface.get("inputs", {}),
            "_outputs": iface.get("outputs", {}),
            "inputs": {},
            "outputs": _outputs,
            "interval": float(time_step),
        }
    }
    state = {"agents": {agent_id: cell_state}, "global_time": 0.0}
    composite = Composite(dict(schema=dict(), state=state), core=core)
    return composite, {"core": core, "agent_root": "agents",
                       "agent_id": agent_id, "process": proc}


def _dperiod_should_divide(handle) -> tuple[bool, int]:
    """Genuine vEcoli's DEFAULT division criterion (``d_period=True``), faithfully:
    the wcEcoli D-period mechanism (``ecoli.processes.cell_division.MarkDPeriod``).

    Divide as soon as the cell's time reaches a full chromosome's ``division_time``
    — an attribute set DURING replication (= replication-complete + D_period, where
    D_period comes from sim_data) — for a chromosome that has not yet triggered
    division, once there are >= 2 full chromosomes. The dry-mass threshold is NOT
    used (real vEcoli ignores it under d_period=True; v2's MarkDPeriod is identical).

    Returns ``(should_divide, n_full_chromosomes)``. Reads the genuine-vEcoli inner
    Engine directly, so ``division_time`` flows straight from sim_data → vEcoli's
    replication → here — no re-derivation, no mass approximation.
    """
    eng = handle.engine
    gt = float(getattr(eng, "global_time", 0.0) or 0.0)
    obs = cell_observables(eng)
    u = obs.get("unique") or {}
    fc = u.get("full_chromosome") if isinstance(u, dict) else None
    if fc is None or not hasattr(fc, "dtype") or fc.dtype.names is None:
        return False, 0
    names = fc.dtype.names
    active = (fc["_entryState"].view(np.bool_) if "_entryState" in names
              else np.ones(len(fc), dtype=bool))
    nchrom = int(active.sum())
    if nchrom < 2 or "division_time" not in names:
        return False, nchrom
    dt = np.asarray(fc["division_time"])[active]
    htd = (np.asarray(fc["has_triggered_division"]).astype(bool)[active]
           if "has_triggered_division" in names else np.zeros(nchrom, dtype=bool))
    untriggered = dt[~htd]
    if untriggered.size == 0:
        return False, nchrom
    return bool(gt >= float(untriggered.min())), nchrom


def _vecoli_config_summary(handle, *, condition: str, seed: int,
                           time_step: float, exclude_processes) -> dict:
    """JSON-able summary of the resolved vEcoli config the run actually used.

    Pulls the process/step NAMES the genuine-vEcoli Engine built plus a sanitized
    copy of ``EcoliSim.config`` (scalars/lists/small dicts only — the ~300MB
    ``sim_data`` object and other non-serializable values are dropped). Written
    next to the vEcoli zarr as ``vecoli_build_config.json`` so the comparison
    report shows vEcoli's OWN full config alongside v2ecoli's. Best-effort."""
    sim = getattr(handle, "sim", None)
    ecoli = getattr(sim, "ecoli", None) if sim is not None else None
    processes = sorted((getattr(ecoli, "processes", {}) or {}).keys()) if ecoli else []
    steps = sorted((getattr(ecoli, "steps", {}) or {}).keys()) if ecoli else []
    safe: dict = {}
    for k, v in (getattr(sim, "config", {}) or {}).items():
        if k in ("sim_data",):                 # huge object — never serialize
            continue
        if isinstance(v, (str, int, float, bool, type(None))):
            safe[k] = v
        elif isinstance(v, (list, tuple)) and len(v) <= 64:
            try:
                json.dumps(list(v))
                safe[k] = list(v)
            except Exception:  # noqa: BLE001
                safe[k] = f"<{type(v).__name__}[{len(v)}]>"
        elif isinstance(v, dict) and len(v) <= 32:
            try:
                json.dumps(v)
                safe[k] = v
            except Exception:  # noqa: BLE001
                safe[k] = f"<dict[{len(v)} keys]>"
    return {
        "engine": "vecoli", "source": "EcoliSim.config",
        "condition": condition, "seed": int(seed), "time_step": float(time_step),
        "media_id": getattr(handle, "media_id", None),
        "n_processes": len(processes), "processes": processes, "steps": steps,
        "exclude_processes": list(exclude_processes or []),
        "config": safe,
    }


def run_vivarium_ecoli_pbg_multigen(
    *,
    store_path,
    sim_data_path: str,
    condition: str = "basal",
    seed: int = 0,
    max_generations: int = 2,
    max_steps_per_gen: int = 9000,
    time_step: float = 1.0,
    chunk: int = 20,
    exclude_processes: list | None = None,
    swap_processes: dict | None = None,
    flow: dict | None = None,
    fork_dir: str | None = None,
    mass_multiplier: float = 1.0,
    core=None,
    experiment_id: str = "vecoli",
    variant: int = 0,
    lineage_seed: int = 0,
    whole_config: str | None = None,
    exchange_fluxes: dict | None = None,
) -> dict:
    """Single-lineage multigen for the vEcoli **pbg node**, emitting the v2ecoli-format zarr.

    ``whole_config`` (a full vEcoli config-file path) makes the wrapped ``EcoliSim``
    load that config NATIVELY instead of the default baseline — so a config whose
    model content can't be expressed as ``swap_processes``/``flow`` (one declaring
    ``add_processes`` and/or a ``spatial_environment_config``) runs faithfully as
    one node. Scoped to this call (restored in ``finally``) for deterministic
    isolation.

    Each generation is a one-node pbg ``Composite`` (``VivariumEcoliProcess``) driven by
    ``composite.run``; a per-generation ``XArrayEmitter`` writes a ``generation=N``
    partition into the shared store (the SAME emitter v2ecoli uses). At the division
    criterion (dry_mass ≥ birth + expectedDryMassIncrease AND ≥2 chromosomes) vEcoli's own
    ``divide_cell`` splits the inner cell and the followed daughter seeds the next
    generation's Composite. No pbg ``_add`` — the division-handoff crash cannot occur.
    """
    import shutil
    from pathlib import Path
    from v2ecoli.library.xarray_run import _build_emitter, _filter_agent_state
    from v2ecoli.library.upstream_division import daughter_phylogeny_id

    # Scope the whole-config selection to this run so every per-generation
    # ``build_vivarium_ecoli`` rebuild sees it and a later default run is unaffected.
    set_ecolisim_config_file(whole_config)

    if core is None:
        from v2ecoli.core import build_core
        core = build_core()
    store_path = str(store_path)
    if Path(store_path).exists():
        shutil.rmtree(store_path)

    exchange_fluxes = dict(exchange_fluxes or {})
    _view_vars = {
        "mass": {k: [{"path": k, "dtype": "<f8"}] for k in MASS_OBS},
        "unique_molecule_counts": {k: [{"path": k, "dtype": "<f8"}] for k in COUNT_OBS},
    }
    if exchange_fluxes:
        _view_vars["exchange_flux"] = {
            leaf: [{"path": leaf, "dtype": "<f8"}] for leaf in exchange_fluxes}
    view = [{"root": ("listeners",), "variables": _view_vars}]
    metadata_base = {
        "experiment_id": experiment_id, "variant": int(variant),
        "lineage_seed": int(lineage_seed), "time_step": float(time_step),
        "max_duration": float(max_generations * max_steps_per_gen),
    }

    overlay = None
    composite_agent_id = "0"            # the inner cell's key in the pbg agents map
    partition_agent_id = "0"            # the emitter's phylogeny key ("0"->"00"->...),
                                        # distinct per generation so each writes its own
                                        # zarr partition (avoids a same-store collision).
    done_global = 0
    divisions = 0
    gens_done = 0
    final_cell_mass = None
    build_config = None

    for gen in range(max_generations):
        # gen 0 is a fresh founder (overlay=None); later generations seed the inner
        # Engine from the previous generation's daughter (overlay set below).
        comp, info = build_vivarium_ecoli_composite(
            sim_data_path=sim_data_path, condition=condition, seed=seed + gen,
            time_step=time_step, exclude_processes=exclude_processes,
            swap_processes=swap_processes, flow=flow,
            fork_dir=fork_dir, core=core, agent_id=composite_agent_id,
            initial_overlay=overlay, variant=variant,
            exchange_fluxes=exchange_fluxes)
        proc = info["process"]
        comp.run(1)  # warm-up tick so listeners materialise
        if gen == 0:                       # capture vEcoli's OWN resolved config once
            try:
                build_config = _vecoli_config_summary(
                    proc._handle, condition=condition, seed=seed,
                    time_step=time_step, exclude_processes=exclude_processes)
            except Exception as _cfgerr:  # noqa: BLE001 — never block the run
                print(f"[vecoli-config] summary skipped: "
                      f"{type(_cfgerr).__name__} {_cfgerr}")
        em = _build_emitter(
            core=core, store_path=store_path, view=view, metadata_base=metadata_base,
            generation=gen + 1,  # 1-indexed to match run_multigen_xarray (v2ecoli side)
            agent_id=partition_agent_id, output_metadata={}, buffer_size=3)

        steps = 1
        divided = False
        while steps < max_steps_per_gen:
            comp.run(chunk)
            steps += chunk
            done_global += chunk
            agent_state = comp.state["agents"][composite_agent_id]
            payload = _filter_agent_state(agent_state, view)
            # Relabel the payload to the emitter's phylogeny key (the emitter strips
            # the agent prefix via get_in(data, ("agents", partition_agent_id))).
            em.update({"time": float(done_global), "global_time": float(done_global),
                       "agents": {partition_agent_id: payload}})
            mass = agent_state["listeners"]["mass"]
            final_cell_mass = float(mass.get("cell_mass", 0.0) or 0.0)
            # Divide by genuine vEcoli's D-period criterion (the d_period=True
            # default), NOT a dry-mass threshold. The old mass-based rule diverged
            # from real vEcoli on fast-growth media (e.g. with_aa, multifork
            # replication): D-period fires when replication+D_period elapses, well
            # before the mass threshold, so the mass rule divided ~40% too late.
            should_divide, _nchrom = _dperiod_should_divide(proc._handle)
            if should_divide:
                divided = True
                break

        try:
            em.close(success=True)
        except AssertionError:
            pass  # F5: trailing-buffer include_static assert; generation already on disk
        gens_done += 1
        if not divided:
            break
        overlay = proc.divide()
        partition_agent_id = daughter_phylogeny_id(partition_agent_id)[0]
        divisions += 1

    set_ecolisim_config_file(None)  # reset for the next run (deterministic isolation)
    return {"generations": gens_done, "divisions": divisions,
            "store": store_path, "final_cell_mass": final_cell_mass,
            "build_config": build_config}


# ---------------------------------------------------------------------------
# Single-lineage multi-generation driver (standalone, non-pbg — local utility)
# ---------------------------------------------------------------------------

def run_vivarium_ecoli_multigen(
    *,
    sim_data_path: str,
    condition: str = "basal",
    seed: int = 0,
    max_generations: int = 2,
    max_steps_per_gen: int = 9000,
    time_step: float = 1.0,
    chunk: int = 20,
    exclude_processes: list | None = None,
    fork_dir: str | None = None,
    mass_multiplier: float = 1.0,
    on_emit=None,
) -> dict:
    """Drive genuine vEcoli single-lineage for ``max_generations`` generations.

    Each generation: build a genuine-vEcoli Engine (seeded with the previous daughter's
    divided state), advance with ``run_for(chunk)`` until the division criterion is met
    (dry_mass >= birth_dry_mass + expectedDryMassIncrease AND >=2 chromosomes), then split
    with vEcoli's ``divide_cell`` and carry one daughter into the next generation.

    ``on_emit(gen, t_global, obs)`` is called once per chunk with the live observables
    (the caller writes them wherever it likes — e.g. a matched-timepoint zarr).

    Returns a summary dict ``{generations, divisions, steps_per_gen, final_cell_mass}``.
    """
    overlay = None
    divisions = 0
    steps_per_gen: list[int] = []
    t_global = 0.0
    last_obs = None

    for gen in range(max_generations):
        h = build_vivarium_ecoli(
            sim_data_path=sim_data_path, condition=condition, seed=seed + gen,
            time_step=time_step, exclude_processes=exclude_processes,
            fork_dir=fork_dir, initial_overlay=overlay)
        engine = h.engine
        threshold = None
        gen_steps = 0
        divided = False

        while gen_steps < max_steps_per_gen:
            engine.run_for(float(chunk))
            gen_steps += chunk
            obs = cell_observables(engine)
            last_obs = obs
            if on_emit is not None:
                on_emit(gen, t_global + gen_steps, obs)

            if threshold is None and obs["dry_mass"] > 0:
                inc = _inc_to_fg(h.dry_mass_inc_dict.get(h.media_id))
                if inc is None:
                    inc = _inc_to_fg(h.dry_mass_inc_dict.get("minimal"))
                if inc is None:
                    inc = obs["dry_mass"]  # fallback: mass doubling
                threshold = obs["dry_mass"] + inc * mass_multiplier

            n_chrom = _n_chromosomes(obs["unique"])
            if threshold is not None and obs["dry_mass"] >= threshold and n_chrom >= 2:
                divided = True
                break

        steps_per_gen.append(gen_steps)
        t_global += gen_steps

        if not divided:
            # ran out of steps before dividing — stop the lineage here
            break
        # vEcoli's faithful split; follow daughter 0 into the next generation
        d1, _d2 = divide_cell({
            "bulk": last_obs["bulk"], "unique": last_obs["unique"],
            "environment": last_obs["environment"], "boundary": last_obs["boundary"]})
        overlay = d1
        divisions += 1

    return {
        "generations": len(steps_per_gen),
        "divisions": divisions,
        "steps_per_gen": steps_per_gen,
        "final_cell_mass": (last_obs or {}).get("cell_mass"),
    }
