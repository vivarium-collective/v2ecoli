"""Multi-generation baseline parquet sim for one condition.

Adapts run_acetate_multigen.py's daughter-bundle pattern + the per-gen
ctx mgr pattern in studies/stage1_sanity_check/run_sim.py for parquet.
Each generation wraps in its own ``parquet_emitter()`` context so the
metadata (generation, agent_id) is fresh per gen.

Agent IDs follow vEcoli's lineage convention: gen N uses agent_id "0"*N
("0", "00", "000", ...). This way ``generation = len(agent_id)`` matches
vEcoli's parquet partitioning semantics.

Full per-gen history retention (the vEcoli-aligned mechanism):
  * The runner-driven cell emits its WHOLE cycle to its own partition
    ``generation=N/agent_id="0"*N``. The in-composite Division step
    closes that emitter with ``success=True`` BEFORE it ``_remove``-s the
    agent (see Division.next_update), so the trailing partial batch + the
    success sentinel land on disk — mirroring vEcoli's pre-divide
    ``emitter.finalize()`` hook in ecoli_master_sim.py.
  * Transient daughter emitters spawned inside the composite at division
    time are re-pointed to their OWN slot
    ``generation=N+1/agent_id="0"*N+{0,1}`` (Division re-points the parquet
    override during the daughter build) so their ``_write_configuration``
    can NOT wipe the parent's fully-written partition. The d?0 daughter
    slot is later re-wiped + rewritten cleanly by the next runner-driven
    generation; the d?1 slot leaves a 1-tick birth stub (ignore it — the
    canonical lineage is the all-zeros agent_id chain).

Usage:
    python scripts/run_condition_multigen_parquet.py \\
        --cache-dir out/cache_glycerol_stage1_all_knobs \\
        --out-dir out/glycerol_stage1_no_crp_5gen_parquet \\
        --experiment-id glycerol_stage1_no_crp_5gen \\
        --generations 5 \\
        --max-min 200
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time

import dill

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# The cache config key whose ``perturbations`` dict TranscriptInitiation reads;
# perturbation values are fixed RNA synthesis probabilities keyed by RNA id
# (e.g. ``TU00259[c]`` — the dnaA operon TU).
_TI_CONFIG_KEY = "ecoli-transcript-initiation"

from process_bigraph import Composite

from v2ecoli import build_composite
from v2ecoli.composites._helpers import parquet_emitter
from v2ecoli.composites.ecoli_baseline import baseline as baseline_doc
from v2ecoli.core import build_core
from v2ecoli.library.division import divide_cell
from v2ecoli.library.parquet_emitter import ParquetEmitter
from v2ecoli.library.run_provenance import (
    build_run_identity,
    write_run_identity_record,
)


SNAPSHOT_INTERVAL = 60


def _last_cell_state(comp: Composite) -> dict | None:
    """Snapshot the parent agent's biological state for divide_cell()."""
    cell = comp.state.get("agents", {}).get("0")
    if cell is None:
        return None
    keys = {"bulk", "unique", "listeners", "environment", "boundary",
            "global_time", "timestep", "divide", "division_threshold",
            "process_state", "allocator_rng"}
    return {k: v for k, v in cell.items()
            if k in keys or k.startswith("request_") or k.startswith("allocate_")}


def _fg(q) -> float:
    """dry_mass in fg as a plain float. Post-merge with origin/main the mass
    listener emits a pint Quantity (femtogram); strip units if present."""
    return float(getattr(q, "magnitude", q))


def parse_config_override(spec: str) -> tuple[str, object]:
    """Parse a ``PROCESS.KEY=VALUE`` override into ``(dotted_key, value)``.

    VALUE is parsed as JSON so booleans/numbers/null arrive typed
    (``...mechanistic_replisome=true`` -> ``True``, not ``"true"``);
    anything JSON cannot parse is kept as a literal string.
    """
    if "=" not in spec:
        raise ValueError(
            f"--config-override {spec!r} must be PROCESS.KEY=VALUE "
            "(e.g. 'ecoli-chromosome-replication.mechanistic_replisome=true')")
    key, _, raw = spec.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"--config-override {spec!r}: empty KEY")
    if "." not in key:
        raise ValueError(
            f"--config-override {spec!r}: KEY must be PROCESS.KEY, got {key!r}")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        value = raw
    return key, value


def parse_generator_param(spec: str) -> tuple[str, object]:
    """Parse a ``KEY=VALUE`` generator parameter into ``(key, value)``.

    Distinct from ``--config-override``: that targets ONE process's config via
    a dotted ``PROCESS.KEY``, whereas these are top-level keyword arguments of
    the composite generator itself (``ecoli_baseline.baseline``) -- the build
    switches that decide which biology is assembled at all, e.g.
    ``ppgpp_regulation=false``. A dotted key is therefore rejected as a
    probable ``--config-override`` typed into the wrong flag.

    VALUE is parsed as JSON so booleans/numbers/null arrive typed
    (``ppgpp_regulation=false`` -> ``False``, not the truthy string
    ``"false"``); anything JSON cannot parse is kept as a literal string.
    """
    if "=" not in spec:
        raise ValueError(
            f"--generator-param {spec!r} must be KEY=VALUE "
            "(e.g. 'ppgpp_regulation=false')")
    key, _, raw = spec.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"--generator-param {spec!r}: empty KEY")
    if "." in key:
        raise ValueError(
            f"--generator-param {spec!r}: KEY must be a bare generator kwarg, "
            f"got dotted {key!r} -- did you mean --config-override?")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        value = raw
    return key, value


def parse_perturbation(spec: str) -> tuple[str, float]:
    """Parse a ``KEY=VALUE`` perturbation spec into ``(rna_id, synth_prob)``.

    e.g. ``"TU00259[c]=1.7e-3"`` -> ``("TU00259[c]", 0.0017)``. KEY is an RNA
    id; VALUE is a fixed synthesis probability (float). Raises ValueError on a
    malformed spec.
    """
    if "=" not in spec:
        raise ValueError(
            f"--perturbation {spec!r} must be KEY=VALUE (e.g. 'TU00259[c]=1.7e-3')")
    key, _, value = spec.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"--perturbation {spec!r}: empty KEY")
    try:
        val = float(value)
    except ValueError as e:
        raise ValueError(
            f"--perturbation {spec!r}: VALUE {value!r} is not a float") from e
    return key, val


def apply_perturbations(configs: dict, perturbations: dict[str, float]) -> dict:
    """Explicitly set each ``rna_id -> synth_prob`` perturbation into the cache
    config TranscriptInitiation reads, OVERRIDING whatever the cache baked in.

    Mutates ``configs`` in place (caller deep-copies first). Returns a record
    ``{rna_id: {"applied": v, "cache_baked": prev_or_None, "overrode": bool}}``
    so the run_config can note both the applied value and any pre-baked one.
    """
    record: dict = {}
    if not perturbations:
        return record
    ti = configs.setdefault(_TI_CONFIG_KEY, {})
    baked = ti.get("perturbations")
    if not isinstance(baked, dict):
        baked = {}
    new = dict(baked)
    for rna_id, val in perturbations.items():
        prev = baked.get(rna_id)
        new[rna_id] = val
        record[rna_id] = {
            "applied": val,
            "cache_baked": prev,
            "overrode": prev is not None and prev != val,
        }
    ti["perturbations"] = new
    return record


def cache_fingerprint(cache_dir: str) -> dict:
    """Cheap, verifiable fingerprint of the cache: sim_data_cache.dill
    mtime + size + a sha256 of (size, mtime_ns) so a swapped cache is
    detectable without hashing the whole multi-hundred-MB dill."""
    path = os.path.join(cache_dir, "sim_data_cache.dill")
    if not os.path.exists(path):
        return {"path": path, "exists": False}
    st = os.stat(path)
    h = hashlib.sha256(f"{st.st_size}:{st.st_mtime_ns}".encode()).hexdigest()[:16]
    return {
        "path": path,
        "exists": True,
        "size_bytes": st.st_size,
        "mtime": st.st_mtime,
        "fingerprint": h,
    }


def read_synth_probs(cache: dict, rna_ids: list[str]) -> dict[str, float | None]:
    """Read back the ParCa-derived basal synthesis probability for each
    ``rna_id`` from the cache's TranscriptInitiation config — the verifiable
    ground-truth the perturbation is meant to override. Returns
    ``{rna_id: basal_prob_or_None}``."""
    out: dict[str, float | None] = {}
    if not rna_ids:
        return out
    try:
        import numpy as np
        ti = cache.get("configs", {}).get(_TI_CONFIG_KEY, {})
        ids = np.asarray(ti.get("rna_data", {})["id"])
        basal = np.asarray(ti.get("basal_prob"))
        for rid in rna_ids:
            hit = np.where(ids == rid)[0]
            out[rid] = float(basal[hit[0]]) if len(hit) else None
    except Exception:
        for rid in rna_ids:
            out.setdefault(rid, None)
    return out


def build_run_config(args, *, perturbations: dict[str, float],
                     applied_record: dict, cache: dict,
                     generator_params: dict | None = None) -> dict:
    """Assemble the decision-relevant run-config provenance dict.

    Captures the knobs that determine what the run actually computed:
    applied perturbations (+ whether they overrode a cache-baked value), the
    cache dir + cheap fingerprint, seed, generations, max_min, resume_dill,
    and the verifiable ParCa-derived synth prob read back from the cache for
    each perturbed RNA.

    ``cache_fingerprint`` is the plain short-hash string (``None`` if the
    cache file is missing), not the detailed dict — this is
    reproducible-rerun-spine Task 2's env-threading key: a downstream
    manifest builder (``vivarium_workbench.lib.composite_runs.build_run_manifest``)
    sniffs a plain-string ``params["cache_fingerprint"]`` straight onto
    ``manifest["env"]["cache_fingerprint"]``, so it must land here as a
    scalar, not a dict. The full diagnostic dict (path/exists/size/mtime)
    is preserved under ``cache_fingerprint_detail`` for anyone reading the
    run_config directly.
    """
    cache_fp = cache_fingerprint(args.cache_dir)
    return {
        "experiment_id": args.experiment_id,
        "perturbations": dict(perturbations),
        "perturbations_detail": applied_record,
        "cache_dir": args.cache_dir,
        # Generator build switches (e.g. ppgpp_regulation=False). Recorded
        # because they change WHICH biology was assembled, not merely a
        # parameter value -- two runs sharing a cache fingerprint are still
        # different models if these differ.
        "generator_params": dict(generator_params or {}),
        "cache_fingerprint": cache_fp.get("fingerprint"),
        "cache_fingerprint_detail": cache_fp,
        "seed": args.seed,
        "generations": args.generations,
        "start_gen": args.start_gen,
        "max_min": args.max_min,
        "resume_dill": args.resume_dill,
        "dnaA_synth_prob_from_cache": read_synth_probs(
            cache, list(perturbations.keys())),
        "out_dir": args.out_dir,
        # v2ecoli#472/#473: code identity + cache-content fingerprint + the
        # design/grid metadata already available here. Also written as its
        # own canonical `run_identity.json` sidecar on a real (non-dry-run)
        # completion — see the end of main() — so sim_vector_cache._run_commit
        # can read it without knowing this runner's summary shape.
        "run_identity": build_run_identity(
            cache_dir=args.cache_dir,
            design={
                "experiment_id": args.experiment_id,
                "seed": args.seed,
                "start_gen": args.start_gen,
                "generations": args.generations,
                "perturbations": dict(perturbations),
            },
        ),
    }


def register_run_config(args, run_config: dict, *, status: str) -> str | None:
    """Register the run into the per-study runs.db via pbg-superpowers'
    ``register_run`` (run_id = experiment_id). Best-effort: returns the
    runs.db path on success, None if viva_superpowers is unavailable or
    ``--study-dir`` is not resolvable. Never raises into the sim path."""
    try:
        from viva_superpowers.run_registry import register_run
    except Exception:
        return None
    study_dir = args.study_dir
    if not study_dir:
        return None
    runs_db = os.path.join(study_dir, "runs.db")
    try:
        register_run(
            runs_db, args.experiment_id,
            spec_id=args.spec_id or args.experiment_id,
            status=status,
            params=run_config,
        )
        return runs_db
    except Exception as e:  # provenance is best-effort; don't kill the sim
        print(f"    [provenance] register_run failed (non-fatal): {e}")
        return None


def _run_gen(comp: Composite, max_duration: int, gen_idx: int) -> tuple[float, bool, dict | None]:
    total = 0.0
    divided = False
    last_state = _last_cell_state(comp)
    cell0 = comp.state["agents"]["0"]
    dry0 = _fg(cell0["listeners"]["mass"].get("dry_mass", 0))
    print(f"    gen {gen_idx}: initial dry_mass={dry0:.1f} fg")

    while total < max_duration:
        chunk = min(SNAPSHOT_INTERVAL, max_duration - total)
        try:
            comp.run(chunk)
        except Exception as e:
            err = str(e)
            if "divide" in err.lower() or "_add" in err or "_remove" in err:
                total += chunk
                divided = True
                break
            if comp.state.get("agents", {}).get("0") is None:
                divided = True
                break
            raise
        total += chunk
        cur = comp.state.get("agents", {}).get("0")
        if cur is None:
            divided = True
            break
        last_state = _last_cell_state(comp)
        if int(total) % 600 == 0 or total >= max_duration - 60:
            dry = _fg(cur["listeners"]["mass"].get("dry_mass", 0))
            print(f"    gen {gen_idx}: t={total/60:5.1f} min  dry_mass={dry:.1f} fg")

    # When divided=True the Division step already close(success=True)-d the
    # parent emitter (via the per-agent registry) BEFORE tearing the agent
    # subtree down, so the full per-gen history + success sentinel are on
    # disk. flush_all_in_composite is the belt-and-suspenders path for the
    # non-divided edge case (last gen): it closes any still-live emitter so
    # the trailing batch is durable. On divide it's a no-op (agent gone).
    n_closed = ParquetEmitter.flush_all_in_composite(comp, success=divided)
    if n_closed:
        print(f"    gen {gen_idx}: flushed {n_closed} ParquetEmitter instance(s)")

    return total, divided, last_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="Parquet out-dir root (under <root>/<experiment-id>/...)")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--max-min", type=float, default=200.0,
                    help="Max duration per generation, minutes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dill-dir", default=None,
                    help="Optional dir to dump per-gen daughter dills "
                         "(default: out/<experiment-id>/gen_dills)")
    ap.add_argument("--resume-dill", default=None,
                    help="Resume the lineage from a previously-dumped "
                         "gen{N}.dill instead of cold-starting gen 1 from "
                         "the cache. The dill is divided to seed this run's "
                         "first generation.")
    ap.add_argument("--start-gen", type=int, default=1,
                    help="Generation index (and agent_id length) for the "
                         "first gen of THIS run. Default 1.")
    ap.add_argument("--perturbation", action="append", default=[],
                    metavar="RNA_ID=SYNTH_PROB",
                    help="Explicitly set a fixed RNA synthesis probability at "
                         "run time, e.g. 'TU00259[c]=1.7e-3' for the dnaA "
                         "operon. Repeatable. OVERRIDES any value baked into "
                         "the cache; the override is recorded in the run "
                         "provenance (run_config.perturbations).")
    ap.add_argument("--config-override", action="append", default=[],
                    metavar="PROCESS.KEY=VALUE",
                    help="Declarative composite config override, forwarded to "
                         "the generator's `config_overrides` parameter, e.g. "
                         "'ecoli-chromosome-replication.mechanistic_replisome=true'. "
                         "Repeatable. Applied to EVERY generation. Values are "
                         "parsed as JSON, falling back to the literal string.")
    ap.add_argument("--generator-param", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="Top-level composite-generator keyword argument, e.g. "
                         "'ppgpp_regulation=false' to build the lineage WITHOUT "
                         "ppGpp-dependent transcription regulation. Unlike "
                         "--config-override (which targets one process's "
                         "config), these select which biology is assembled. "
                         "Repeatable; applied to EVERY generation. Values are "
                         "parsed as JSON, falling back to the literal string.")
    ap.add_argument("--study-dir", default=None,
                    help="Study dir whose runs.db records this run's config "
                         "provenance (run_id = experiment_id). If omitted, "
                         "the run_config is still written to the summary JSON "
                         "but not registered.")
    ap.add_argument("--spec-id", default=None,
                    help="spec/study id to record for this run "
                         "(default: experiment_id).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Assemble + print + register the run_config WITHOUT "
                         "running any simulation. Use to verify provenance "
                         "wiring and --perturbation parsing.")
    args = ap.parse_args()

    # Parse perturbations up front (fail fast on a malformed spec).
    perturbations = dict(parse_perturbation(s) for s in args.perturbation)

    # Declarative composite config overrides, forwarded to the generator's
    # `config_overrides` parameter. Kept as a kwargs dict so the three
    # composite-construction sites below stay unchanged when no override is
    # given (empty dict -> no kwarg -> byte-identical call).
    config_overrides = dict(
        parse_config_override(s) for s in args.config_override)
    _cfg_over_kwarg = {"config_overrides": config_overrides} if config_overrides else {}
    if config_overrides:
        print(f"  config overrides: {config_overrides}")

    # Generator build switches, merged into the SAME kwargs dict so all three
    # composite-construction sites (gen-1 perturbed, gen-1 plain, gen-2+
    # daughter) receive them identically -- ppgpp_regulation et al. are in
    # ecoli_baseline's per-cell threaded kwarg set, so the switch holds across
    # division rather than silently reverting after generation 1.
    generator_params = dict(
        parse_generator_param(s) for s in args.generator_param)
    _cfg_over_kwarg.update(generator_params)
    if generator_params:
        print(f"  generator params: {generator_params}")

    max_duration = int(args.max_min * 60)
    dill_dir = args.dill_dir or f"out/{args.experiment_id}/gen_dills"
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(dill_dir, exist_ok=True)

    print("=" * 60)
    print(f"Multi-gen lineage — {args.generations} gens, seed={args.seed}")
    print(f"  cache:        {args.cache_dir}")
    print(f"  out_dir:      {args.out_dir}")
    print(f"  experiment:   {args.experiment_id}")
    print(f"  max/gen:      {args.max_min:.0f} min")
    print("=" * 60)

    # Load configs/unique_names/dry_mass_inc once — same cache feeds every gen.
    with open(os.path.join(args.cache_dir, "sim_data_cache.dill"), "rb") as f:
        cache = dill.load(f)
    try:
        from v2ecoli.library.unit_bridge import rebind_cache_quantities
        rebind_cache_quantities(cache)
    except ImportError:
        pass

    # Apply explicit run-time perturbations BEFORE building any composite, so
    # they are a recorded run-time decision (not an opaque cache). Deep-copy
    # the configs subtree so we don't mutate the lru_cache-shared cache dict.
    applied_record: dict = {}
    if perturbations:
        cache["configs"] = copy.deepcopy(cache.get("configs", {}))
        applied_record = apply_perturbations(cache["configs"], perturbations)
        for rid, info in applied_record.items():
            tag = ("OVERRODE cache-baked " + repr(info["cache_baked"])
                   if info["overrode"] else
                   ("set (no cache value)" if info["cache_baked"] is None
                    else "set (== cache value)"))
            print(f"  perturbation: {rid} = {info['applied']:g}  ({tag})")

    # Assemble + register the run-config provenance (run_id = experiment_id).
    run_config = build_run_config(
        args, perturbations=perturbations,
        applied_record=applied_record, cache=cache,
        generator_params=generator_params)
    registered_db = register_run_config(
        args, run_config, status="dry-run" if args.dry_run else "running")
    run_config["registered_runs_db"] = registered_db
    print("  run_config:")
    print("    " + json.dumps(run_config, indent=2, default=str).replace(
        "\n", "\n    "))
    if registered_db:
        print(f"  registered run '{args.experiment_id}' -> {registered_db}")

    if args.dry_run:
        # Write the run_config to a *_run_config.json sidecar too, then stop
        # WITHOUT running any simulation.
        cfg_path = os.path.join(
            args.out_dir, f"{args.experiment_id}_run_config.json")
        with open(cfg_path, "w") as f:
            json.dump(run_config, f, indent=2, default=str)
        print(f"\n[dry-run] no simulation executed. run_config: {cfg_path}")
        return

    configs = cache.get("configs", {})
    unique_names = cache.get("unique_names", [])
    dry_mass_inc = cache.get("dry_mass_inc_dict", {})

    core = build_core()
    t_pipeline = time.time()
    summary = []
    prev_cell_data = None

    # Resume mode: seed prev_cell_data from a saved dill so the first gen of
    # this run is a daughter of that state (divide path), not a cold cache
    # start.
    if args.resume_dill:
        with open(args.resume_dill, "rb") as f:
            prev_cell_data = dill.load(f)
        print(f"  resume_dill:  {args.resume_dill} "
              f"(first gen seeded from this state)")

    for gen_idx in range(args.start_gen, args.start_gen + args.generations):
        agent_id = "0" * gen_idx
        gen_label = f"gen {gen_idx} (agent_id={agent_id})"
        print(f"\n[{time.strftime('%H:%M:%S')}] {gen_label}")
        t0 = time.time()

        with parquet_emitter(
            out_dir=args.out_dir,
            experiment_id=args.experiment_id,
            lineage_seed=args.seed,
            agent_id=agent_id,
            generation=gen_idx,
        ):
            t_build = time.time()
            if prev_cell_data is None:
                if perturbations:
                    # Build gen 1 from the perturbed in-memory bundle so the
                    # run-time --perturbation override actually reaches
                    # TranscriptInitiation. (build_composite re-reads the cache
                    # from disk, which would silently drop the override.)
                    from v2ecoli.core import load_cache_bundle
                    bundle = dict(load_cache_bundle(args.cache_dir))
                    bundle["configs"] = configs  # perturbed configs
                    doc = baseline_doc(core=core, seed=args.seed,
                                       cache_dir=args.cache_dir, bundle=bundle,
                                       **_cfg_over_kwarg)
                    comp = Composite(doc, core=core)
                else:
                    comp = build_composite("ecoli_baseline", cache_dir=args.cache_dir,
                                                           **_cfg_over_kwarg)
            else:
                d1_state, _d2_state = divide_cell(prev_cell_data)
                bundle = {
                    "initial_state": d1_state,
                    "configs": configs,
                    "unique_names": unique_names,
                    "dry_mass_inc_dict": dry_mass_inc,
                }
                doc = baseline_doc(core=core, seed=args.seed + gen_idx,
                                   cache_dir=args.cache_dir, bundle=bundle,
                                   **_cfg_over_kwarg)
                comp = Composite(doc, core=core)
            print(f"    composite built in {time.time()-t_build:.1f}s")

            duration, divided, last_state = _run_gen(comp, max_duration, gen_idx)

        dry_f = _fg((last_state or {}).get("listeners", {}).get("mass", {}).get("dry_mass", 0))
        wall = time.time() - t0
        result = {
            "gen": gen_idx,
            "agent_id": agent_id,
            "duration_min": duration / 60.0,
            "divided": divided,
            "final_dry_mass_fg": dry_f,
            "wall_seconds": wall,
        }
        summary.append(result)
        print(f"    gen {gen_idx} summary: tau={duration/60:.1f} min  "
              f"final_dry={dry_f:.1f} fg  divided={divided}  wall={wall:.0f}s")

        # Persist daughter state for the next gen / for restart.
        prev_cell_data = last_state
        if last_state is not None:
            dill_path = os.path.join(dill_dir, f"gen{gen_idx}.dill")
            with open(dill_path, "wb") as f:
                dill.dump(last_state, f)

        if not divided:
            print(f"    gen {gen_idx} did not divide — stopping lineage.")
            break

    pipeline_wall = time.time() - t_pipeline
    out_summary = {
        "experiment_id": args.experiment_id,
        "cache_dir": args.cache_dir,
        "generations_requested": args.generations,
        "generations_completed": len(summary),
        "pipeline_wall_seconds": pipeline_wall,
        "run_config": run_config,
        "gens": summary,
    }
    summary_path = os.path.join(args.out_dir, f"{args.experiment_id}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(out_summary, f, indent=2, default=str)

    # Canonical run_identity.json sidecar (v2ecoli#472/#473) — reuses the
    # record already computed into run_config["run_identity"] rather than
    # recomputing (a second git/cache-fingerprint round trip).
    write_run_identity_record(args.out_dir, run_config["run_identity"])

    # Mark the run complete in the registry (preserves the recorded params).
    register_run_config(args, run_config, status="complete")

    print("\n" + "=" * 60)
    print(f"DONE — {len(summary)} gens, {pipeline_wall/60:.1f} min wall")
    for s in summary:
        print(f"  gen {s['gen']}: tau={s['duration_min']:5.1f} min  "
              f"final_dry={s['final_dry_mass_fg']:7.1f} fg  "
              f"divided={s['divided']}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
