"""Multi-generation, multi-seed ensemble driver for the v2ecoli↔vEcoli comparison.

Runs EITHER engine through the IDENTICAL process-bigraph + XArray path so the
comparison is apples-to-apples (same framework, same compact emitter, same
compute), with NO Nextflow:

  --composite v2ecoli  → v2ecoli's ported ``baseline()`` composite
  --composite vecoli   → the PRISTINE upstream ``CovertLab/vEcoli`` model, run as
                         a SINGLE process-bigraph node with vivarium-core's own
                         Engine inside (``vivarium_ecoli_engine``; the ONLY
                         supported vEcoli loader — faithful by construction, ZERO
                         edits to the upstream checkout). The old colony-wrapper
                         (``upstream``) and ``composite-softfloor`` loaders were
                         removed; ``--vecoli-source`` accepts only vivarium-process.

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

# Local fallback for the genuine-vEcoli ParCa simData when a run's --cache-dir
# has no simData.cPickle (the vivarium-process vEcoli loader + matched-initial-
# state reference both resolve to this as a last resort).
_UPSTREAM_SIMDATA_FALLBACK = (
    "/Users/eranagmon/code/v2ecoli/out/compare_harness/vecoli_parca/"
    "kb/simData.cPickle")


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
    the vEcoli side does via sim.config["condition"]. Omitting it (the old bug)
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
    # Regenerate a per-(condition, seed) initial-state bundle from simData. This
    # MUST include basal: the base cache_full snapshot is a SINGLE seed's basal
    # initial state, so without regen every basal seed starts IDENTICAL (e.g.
    # active_ribosome fixed at 12441) while genuine vEcoli resamples per seed
    # (seed2=9947). Regenerating basal per seed reproduces vEcoli's seed-specific
    # draw to <0.1% (9944 vs 9947), so the v2 basal ENSEMBLE varies by seed like
    # vEcoli's. (Non-basal already regenerated; basal was the lone exception.)
    # Falls back to the snapshot when simData is absent (e.g. a minimal cache).
    _sd_probe = os.path.abspath(os.path.join(cache_dir, "simData.cPickle"))
    if condition and (condition != "basal" or os.path.exists(_sd_probe)):
        import pickle
        from v2ecoli.core import save_sim_input
        # ABSOLUTE paths: the regen below does os.chdir(_iso) for emitter isolation,
        # so a RELATIVE bundle_dir would be written nested under the changed cwd
        # (".regen_.../out/cache_full/cond_X") and the marker check — back at the
        # original cwd — would never find it → silent basal fallback for EVERY
        # condition. (The old comment claimed "bundle_dir (absolute)" but cache_dir
        # is relative when --cache-dir is relative, as it always is here.)
        cond_cache = os.path.abspath(os.path.join(cache_dir, f"cond_{condition}_seed{seed:02d}"))
        marker = os.path.join(cond_cache, "sim_data_cache.dill")
        sd_path = os.path.abspath(os.path.join(cache_dir, "simData.cPickle"))

        # ROBUSTNESS GUARD (all conditions): a per-condition bundle bakes media_id
        # into its configs at generation. If a STALE bundle (built before a
        # condition->media fix) baked the wrong media, blindly reusing it runs the
        # WRONG media — e.g. with_aa on 'minimal' => no amino-acid uptake =>
        # methionine starvation => RelA/ppGpp runaway (5-13x) => RNAP active
        # fraction halved => ~14% RNA / ~7% growth-rate deficit that reads as a
        # bogus "port divergence". The marker-exists-only check never caught this
        # because the cache fingerprint is computed in a chdir'd dir (all hashes
        # MISSING) and was never verified here. Derive the REQUIRED media straight
        # from sim_data (single source of truth) and regenerate any bundle whose
        # recorded media disagrees, so every condition self-corrects.
        sim_data = None
        expected_media = None
        if os.path.exists(sd_path):
            with open(sd_path, "rb") as f:
                sim_data = pickle.load(f)
            expected_media = (sim_data.conditions.get(condition, {}) or {}).get("nutrients")

        def _bundle_media() -> str | None:
            try:
                with open(os.path.join(cond_cache, "metadata.json")) as mf:
                    return json.load(mf).get("media_id")
            except (OSError, ValueError):
                return None

        if (os.path.exists(marker) and expected_media is not None
                and _bundle_media() != expected_media):
            print(
                f"  [media-guard] {condition} seed{seed:02d}: bundle media "
                f"{_bundle_media()!r} != required {expected_media!r}; regenerating",
                flush=True,
            )
            shutil.rmtree(cond_cache, ignore_errors=True)

        if not os.path.exists(marker) and os.path.exists(sd_path):
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
        elif condition == "basal":
            # The base cache IS basal, so a failed/absent basal regen safely
            # degrades to the snapshot (seed-fixed) rather than erroring.
            eff_cache = cache_dir
        else:
            # FAIL LOUD instead of silently falling back to the base (basal) cache.
            # The base ParCa is the shipped --mode FAST fixture (reduced TF condition
            # set), so the per-condition regen produces no condition-specific bundle —
            # and a silent basal fallback made EVERY non-basal condition run basal,
            # which read as a 44-336% v2-vs-vEcoli "divergence" that was pure setup
            # artifact. A condition that cannot be applied must error, not run basal.
            raise RuntimeError(
                f"v2ecoli condition '{condition}' could not be applied: the per-"
                f"condition regen produced no bundle at {cond_cache} (likely because "
                f"the ParCa cache {cache_dir!r} is the --mode fast fixture, which omits "
                f"non-basal conditions). Build a FULL condition-complete ParCa "
                f"(scripts/build_comparison_caches.sh) and point --cache-dir at it."
            )
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
    comp = build_composite("baseline", **kwargs)

    # FAIL-LOUD media assertion (all conditions): the composite must actually run
    # on the media the condition requires. Anything else silently mis-models the
    # whole comparison (see the methionine-starvation chain in the media-guard
    # note above). Cheap O(1) check straight off the built state — never trust
    # that the cache plumbing did the right thing.
    if expected_media is not None:
        _ag = comp.state.get("agents", {}).get("0", comp.state) if hasattr(comp, "state") else {}
        _got = (_ag.get("environment", {}) or {}).get("media_id")
        if _got is not None and _got != expected_media:
            raise RuntimeError(
                f"v2ecoli {condition} seed{seed:02d} built on media {_got!r} but "
                f"the condition requires {expected_media!r}. The per-condition "
                f"bundle at {eff_cache} is stale/mislabelled — delete it and "
                f"rebuild (the media-guard should have regenerated it)."
            )
    return comp


# --------------------------------------------------------------------------- #
# Matched-initial-state seeding.
#
# v2ecoli and genuine vEcoli draw their initial bulk molecule counts by
# independent stochastic sampling, and the two codebases consume their RNG in
# different order, so the SAME seed yields different draws. For HIGH-copy
# species this is negligible, but for LOW-copy regulators it is not: SpoT
# (ppGpp hydrolase, ~1-12 molecules) drove a >35% apparent acetate "divergence"
# at seed 0 (v2 drew 12, vEcoli drew 1) that vanished once the counts matched —
# the elongation+ppGpp models are otherwise equivalent. To make single-seed
# v2-vs-vEcoli comparisons reflect genuine DYNAMICS rather than sampling luck,
# overlay genuine vEcoli's initial bulk onto v2 so both engines start identical.
# (Unique molecules — ribosomes/RNAP/chromosome — are NOT overlaid; they are
# higher-copy and their initial counts already agree to a few percent.)
# --------------------------------------------------------------------------- #
def _vecoli_reference_bulk(sim_data_path: str, condition: str, seed: int,
                           fork_dir: str | None) -> dict[str, int]:
    """Genuine vEcoli initial bulk ``{molecule_id: count}`` for (condition, seed).

    Builds the real upstream vEcoli vivarium Engine and reads its PRE-run bulk
    state — the reference the v2ecoli run is seeded from.
    """
    import numpy as np
    from v2ecoli.library.vivarium_ecoli_engine import build_vivarium_ecoli
    h = build_vivarium_ecoli(
        sim_data_path=sim_data_path, condition=condition, seed=seed,
        exclude_processes=["monomer_counts_listener"], fork_dir=fork_dir)
    bulk = np.asarray(h.engine.state.get_value()["bulk"])
    return {str(i): int(c) for i, c in zip(bulk["id"], bulk["count"])}


def _apply_bulk_overlay(composite, ref_bulk: dict[str, int]) -> dict:
    """Overwrite the v2 composite's initial bulk counts IN PLACE from
    ``ref_bulk`` (keyed by molecule id), for molecules present in BOTH engines.
    Returns a stats dict for logging. Raises if the composite has no bulk.
    """
    import numpy as np
    state = getattr(composite, "state", {}) or {}
    agents = state.get("agents")
    if isinstance(agents, dict) and agents:
        agent = agents.get("0") or next(iter(agents.values()))
    else:
        agent = state
    bulk = agent.get("bulk") if isinstance(agent, dict) else None
    if bulk is None:
        raise RuntimeError("matched-initial-state: composite has no 'bulk' to overlay")
    arr = np.asarray(bulk)
    ids = [str(x) for x in arr["id"]]
    counts_new = arr["count"].copy()
    matched = changed = 0
    for i, name in enumerate(ids):
        ref = ref_bulk.get(name)
        if ref is not None:
            matched += 1
            if int(counts_new[i]) != ref:
                changed += 1
            counts_new[i] = ref
    arr["count"][:] = counts_new  # whole-field assign → writes through to composite
    return {"v2_bulk": len(ids), "ref_bulk": len(ref_bulk), "matched": matched,
            "changed": changed, "v2_only": len(ids) - matched,
            "ref_only": len(ref_bulk) - matched}


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


def _injected_from_resolved(resolved: dict, fork_repo: str,
                            fork_sim_data: str | None) -> dict | None:
    """Assemble a baseline ``injected_processes`` block from a resolved vEcoli
    config, so the v2 side converts+injects the fork's add/swap processes.

    Returns None when the config declares no injection. Carries every per-port
    knob the bridge understands (output/defer/strip/attach), so a swap is fully
    config-described; ``fork_sim_data`` lets a swapped process pull its full
    config from the fork's own LoadSimData.
    """
    if not (resolved.get("add_processes") or resolved.get("swap_processes")
            or resolved.get("exclude_processes")):
        return None
    inj = {
        "fork_repo": fork_repo,
        "add_processes": resolved.get("add_processes") or [],
        "swap_processes": resolved.get("swap_processes") or {},
        "exclude_processes": resolved.get("exclude_processes") or [],
        "process_configs": resolved.get("process_configs") or {},
        "topology": resolved.get("topology") or {},
        "time_step": float(resolved.get("time_step", 1.0)),
        "output_ports": resolved.get("output_ports") or {},
        "defer_ports": resolved.get("defer_ports") or {},
        "strip_pint_ports": resolved.get("strip_pint_ports") or {},
        "attach_pint_ports": resolved.get("attach_pint_ports") or {},
    }
    if fork_sim_data:
        inj["fork_sim_data"] = fork_sim_data
    return inj


def _translated_v2_overrides(vecoli_config_path: str) -> tuple[dict, dict]:
    """Resolve+translate a vEcoli config; return (overrides, translated_full).

    Resolves via the fork's own loader (``resolve_vecoli_config``), then delegates
    to :func:`_overrides_from_resolved`.
    """
    from scripts._compare.config_adapter import resolve_vecoli_config
    return _overrides_from_resolved(resolve_vecoli_config(vecoli_config_path))


def make_run_one(*, composite_kind: str, condition: str, cache_dir: str,
                 max_generations: int, max_steps: int, chunk: int,
                 out_root: str, seed_start: int = 0,
                 vecoli_config: str | None = None,
                 translate_config: bool = False,
                 vecoli_source: str = "vivarium-process",
                 from_vecoli_config: str | None = None,
                 vecoli_dir: str | None = None,
                 match_initial_state: bool = False,
                 match_vecoli_simdata: str | None = None):
    """Return a ``run_one(seed)`` closure for ``run_seeds_parallel``."""
    from v2ecoli.library.xarray_run import run_multigen_xarray, view_from_emit_paths

    # PART 3 (opt-in): translate the vEcoli config into baseline overrides ONCE.
    v2_overrides: dict | None = None
    v2_translated: dict | None = None

    # Drive the v2ecoli side from a vEcoli FORK config: resolve it ONCE (using
    # v2ecoli's own loader so the fork needs no venv) and translate into baseline
    # overrides. The genuine vEcoli side (vivarium-process) reads the fork's config
    # directly via EcoliSim, so it needs no native-kwargs threading here.
    if from_vecoli_config and composite_kind == "v2ecoli":
        try:
            from scripts._compare.config_adapter import resolve_vecoli_config_local
            fork_dir = vecoli_dir or os.environ.get(
                "V2E_VECOLI_DIR", str(REPO_ROOT.parent / "vEcoli"))
            resolved = resolve_vecoli_config_local(from_vecoli_config, fork_dir)
            v2_overrides, v2_translated = _overrides_from_resolved(resolved)
            # Build the injected_processes block so the v2 side actually
            # converts+injects the fork's add/swap processes (translate alone
            # passes the raw keys through but never assembles the block).
            inj = _injected_from_resolved(
                resolved, fork_dir,
                os.path.abspath(match_vecoli_simdata) if match_vecoli_simdata else None)
            if inj:
                v2_overrides = dict(v2_overrides or {})
                v2_overrides["injected_processes"] = inj
            print(f"[from-vecoli-config] {from_vecoli_config} → v2 overrides "
                  f"{sorted(v2_overrides)} ({len(v2_translated)} translated keys"
                  f"{'; +injected '+str(inj.get('add_processes') or list(inj.get('swap_processes') or {})) if inj else ''})")
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

        # The genuine vEcoli side ALWAYS runs as a SINGLE pbg node with vivarium's
        # own Engine inside (vecoli_source="vivarium-process" — the ONLY supported
        # vEcoli loader; faithful by construction, no re-implemented partition/
        # reconcile/division). It runs its OWN single-lineage multigen + XArrayEmitter
        # (vivarium handles division internally; no pbg _add), emitting the SAME
        # v2ecoli-format zarr, so it bypasses the pbg-division run_multigen_xarray below.
        if composite_kind == "vecoli":
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
            # Matched-initial-state seeding (opt-in): overlay genuine vEcoli's
            # initial bulk onto v2 so both engines start from identical molecule
            # counts — removing the stochastic low-copy sampling divergence
            # (e.g. SpoT) that otherwise dominates single-seed comparisons.
            if match_initial_state:
                # Resolve the genuine-vEcoli reference simData: explicit flag >
                # cache_dir/simData.cPickle (present in the cloud image's ParCa,
                # same as the vivarium-process branch) > upstream local fallback.
                if match_vecoli_simdata:
                    ref_sd = os.path.abspath(match_vecoli_simdata)
                else:
                    cand = os.path.abspath(os.path.join(cache_dir, "simData.cPickle"))
                    ref_sd = cand if os.path.exists(cand) else _UPSTREAM_SIMDATA_FALLBACK
                if not os.path.exists(ref_sd):
                    raise RuntimeError(
                        f"--match-initial-state needs the genuine vEcoli simData; "
                        f"{ref_sd!r} not found. Pass --match-vecoli-simdata <path>.")
                ref_bulk = _vecoli_reference_bulk(
                    ref_sd, condition, seed,
                    vecoli_dir or os.environ.get("V2E_VECOLI_DIR"))
                stats = _apply_bulk_overlay(composite, ref_bulk)
                print(f"[match-initial-state] seed{seed:02d} {condition}: overlaid "
                      f"vEcoli bulk onto v2 — matched {stats['matched']}/"
                      f"{stats['v2_bulk']} ({stats['changed']} counts changed); "
                      f"v2-only {stats['v2_only']}, ref-only {stats['ref_only']}")
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
        else:
            # vecoli is handled by the vivarium-process early-return above; the
            # only other composite_kind is v2ecoli. Anything else is a bug.
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
            # Follow a single lineage (prune non-followed daughters each
            # division) so EVERY generation — including the last — runs to its
            # own division, matching genuine vEcoli. Without this, kept siblings
            # trigger spurious division signals that truncate the final
            # generation (v2ecoli would stop one generation short of vEcoli).
            single_daughters=True,
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
    p.add_argument("--vecoli-source", default="vivarium-process",
                   choices=["vivarium-process"],
                   help="How the --composite vecoli side runs. The ONLY supported "
                        "loader is 'vivarium-process': genuine vEcoli as a single "
                        "pbg node with vivarium-core's Engine inside (faithful by "
                        "construction). The old 'upstream'/'composite-softfloor' "
                        "wrappers were removed — vivarium-process is canonical.")
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
    p.add_argument("--match-initial-state", action="store_true",
                   help="Seed v2ecoli's initial bulk from genuine vEcoli's (same "
                        "condition+seed) so both engines start from IDENTICAL "
                        "molecule counts. Removes the stochastic low-copy sampling "
                        "divergence (e.g. SpoT) that dominates single-seed "
                        "comparisons. No-op for the vecoli run (it IS the reference).")
    p.add_argument("--match-vecoli-simdata", default=None,
                   help="Path to the genuine vEcoli simData.cPickle used as the "
                        "matched-initial-state reference (default: the upstream "
                        "fallback). Required when that fallback is absent.")
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
        from_vecoli_config=from_vc, vecoli_dir=vecoli_dir,
        match_initial_state=args.match_initial_state,
        match_vecoli_simdata=args.match_vecoli_simdata)
    # V2E_RAY_THREADS caps Ray concurrency: each worker requests this many CPUs,
    # so concurrency = cores // threads. Use it to bound memory (a v2ecoli 4-gen
    # seed is ~16GB; on the 12-core/69GB mini set 4 → 3 concurrent ≈ 48GB, safe).
    _ray_threads = int(os.environ.get("V2E_RAY_THREADS", "0") or 0) or None
    parallel = run_seeds_parallel(seeds, run_one, mode=args.mode,
                                  num_threads=_ray_threads)
    summaries = getattr(parallel, "results", parallel)
    ensemble = {"composite": args.composite, "condition": args.condition,
                "n_seeds": len(seeds), "max_generations": args.max_generations,
                "wall_s": getattr(parallel, "wall_s", None),
                "seeds": list(summaries)}
    print(json.dumps(ensemble, indent=2))


if __name__ == "__main__":
    main()
