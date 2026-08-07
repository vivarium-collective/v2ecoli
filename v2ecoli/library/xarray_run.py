"""Multi-generation XArray run helper for v2ecoli composites.

Wraps the validated emitter-per-generation pattern (see
``docs/superpowers/notes/2026-05-22-xarray-multigen-emit-validated.py``) so the
three non-obvious gotchas of driving ``XArrayEmitter`` manually are handled
once, in one place:

  1. Buffer must flush before close, else ``flush`` asserts. We force a small
     buffer + ``subsample(1)`` so every update produces an emit-step and a
     flush happens within a few ticks per generation.

  2. Input must EXACTLY match the view subtree. We filter the agent's state
     down to the leaves declared in the view config before each ``update``.

  3. vEcoli daughters can re-use the parent's id at division (e.g. ``00`` ->
     ``00``, ``01``). We detect division by agent-count growth in the colony,
     not by ``followed in agents`` (which keeps being True across some
     divisions). When a division is seen, we close the current emitter and
     open a new one for the next generation, following a daughter.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

# NB: XArrayEmitter is imported lazily inside _build_emitter — it requires the
# [xarray] extra, but build_emitter_config (a pure dict builder) and the
# view/coord helpers must remain importable without it (e.g. in CI fast-tests).


#: v2ecoli observables known to be vectors/arrays.
#:
#: Vectors need a coord array in ``output_metadata`` so XArrayEmitter can
#: allocate them as (buf_size, N)-shape buffers instead of scalar slots.
#: :func:`extract_output_metadata_from_state` discovers coord arrays from
#: the composite's live state.
#:
#: ``view_from_emit_paths`` previously SKIPPED these leaves entirely
#: (so vector-only studies fell back to SQLite). Setting
#: ``include_vectors=True`` (the default since 2026-05-23) keeps them
#: in the view; the caller supplies output_metadata.
_KNOWN_VECTOR_LEAVES: set[str] = {
    "monomer_counts",
    "mRna_counts",
    "fork_coordinates",
    "RNAP_coordinates",
    "ribosome_coordinates",
}


# Env-gated unique-store emit (V2ECOLI_EMIT_UNIQUE=1): mirror of the
# parquet_run path. The chromosome-state renderer needs per-molecule genomic
# coordinates that live in the agent's structured ``unique`` numpy arrays --
# which the plain view filter cannot reach. When enabled we extract the named
# attribute arrays of the ACTIVE entries (``_entryState`` mask) and emit them
# as nested keys so the emitter flattens them to ``active_RNAP__coordinates``
# etc. ``chromosome_domain__child_domains`` (the (n_domain, 2) parent->child
# domain tree) is FLATTENED row-major to a 1-D list<int> aligned to
# ``chromosome_domain__domain_index`` so the renderer can place daughter-strand
# RNAPs on the replication bubbles, not just the rim.
_EMIT_UNIQUE = os.environ.get("V2ECOLI_EMIT_UNIQUE", "") not in ("", "0", "false", "False")
_UNIQUE_EMIT_SPEC = {
    "active_RNAP": ["coordinates", "domain_index"],
    "active_replisome": ["coordinates", "domain_index"],
    "full_chromosome": ["unique_index", "domain_index"],
    "chromosome_domain": ["domain_index", "child_domains"],
}


def _flatten_attr(col):
    """Flatten a unique-store attribute column to a 1-D python list.

    1-D attributes pass through; 2-D ones (child_domains is (n_active, 2)) are
    flattened row-major so they serialize as a plain list<int> column aligned
    to domain_index.
    """
    arr = np.asarray(col)
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.tolist()


def _extract_unique_attrs(agent_state: dict) -> dict:
    """Pull active-entry attribute arrays out of an agent's ``unique`` store.

    Returns ``{mol: {attr: [values...]}}`` for the active entries of each
    configured unique molecule, ready to merge into the emit payload (flattens
    to ``<mol>__<attr>`` columns).
    """
    out: dict = {}
    unique = (agent_state or {}).get("unique") or {}
    for mol, attrs in _UNIQUE_EMIT_SPEC.items():
        arr = unique.get(mol)
        if arr is None or not hasattr(arr, "dtype") or arr.dtype.names is None:
            out[mol] = {a: [] for a in attrs}
            continue
        names = set(arr.dtype.names)
        if "_entryState" in names:
            active = arr[arr["_entryState"].view(np.bool_)]
        else:
            active = arr
        out[mol] = {
            a: (_flatten_attr(active[a]) if a in names else [])
            for a in attrs
        }
    return out


def view_from_emit_paths(
    emit_paths: list[str], *, default_dtype: str = "<f8",
    skip_leaves: set[str] | None = None,
    include_vectors: bool = True,
) -> list[dict]:
    """Build a v2ecoli-shaped XArray view config from a flat list of dotted emit paths.

    For dashboard-driven runs, each study declares its observables as dotted
    paths like ``"listeners.replication_data.number_of_oric"``. The XArray
    view config is nested by root + variable path. This helper builds it.

    Only handles ``listeners.<...>`` paths for v0 (the v2ecoli convention).
    Drops other roots (``bulk``, etc.) with a warning.

    Default dtype is ``<f8`` (float64); pass an explicit ``dtype_overrides``
    map at the call site if int dtypes are needed for specific leaves.
    """
    # Behaviour: by default INCLUDE known vector leaves (caller must supply
    # output_metadata coord arrays); old callers using include_vectors=False
    # keep the legacy scalar-only behaviour.
    if skip_leaves is None:
        skip = set() if include_vectors else _KNOWN_VECTOR_LEAVES
    else:
        skip = skip_leaves
    variables: dict = {}
    for p in emit_paths:
        # Accept dotted or slash-separated paths; strip optional agents/<id>/ prefix.
        parts = [x for x in p.replace(".", "/").split("/") if x]
        if parts and parts[0] == "agents" and len(parts) >= 3:
            parts = parts[2:]
        if len(parts) < 2 or parts[0] != "listeners":
            continue
        rest = parts[1:]
        leaf_name = rest[-1]
        if leaf_name in skip:
            continue
        cursor = variables
        for k in rest[:-1]:
            cursor = cursor.setdefault(k, {})
        cursor[leaf_name] = [{"path": leaf_name, "dtype": default_dtype}]
    if not variables:
        return []
    return [{"root": ("listeners",), "variables": variables}]


def _flatten_keys(d: dict, prefix: str = "") -> Any:
    for k, v in d.items():
        full = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            yield from _flatten_keys(v, full)
        else:
            yield full


def extract_output_metadata_from_state(
    state: dict,
    view: list[dict],
    named_metadata: dict | None = None,
) -> dict:
    """Inspect a live composite state + a view config; build XArrayEmitter
    ``output_metadata`` with coord arrays for vector leaves.

    XArrayEmitter consumes ``config["output_metadata"]`` at construction time
    to learn the per-leaf coord arrays (which determine each variable's
    additional dimension beyond the time axis). Scalar leaves get no coord
    (and the storage allocates a scalar slot per emit step). Vector leaves
    need a coord array.

    When ``named_metadata`` is provided (typically the return value of
    ``v2ecoli.library.output_metadata.output_metadata(composite.state)``),
    element names from annotated listener ``outputs()`` schemas are preferred
    over the default integer-index ``range(N)`` fallback.  Un-annotated vector
    leaves still get ``range(N)`` so nothing breaks for them.

    The returned dict mirrors the view's path structure (so
    ``make_coords`` can resolve each leaf via ``get_in``).

    Reads from ``state['agents']['0']`` first; falls back to top-level state.
    Returns ``{}`` if no vector leaves found (XArrayEmitter then renders all
    leaves as scalars, the legacy behaviour).

    Output is shaped RELATIVE to the view root: e.g. for view
    ``root=('listeners',)`` with a vector leaf ``monomer_counts``, the result
    is ``{'monomer_counts': [0..N-1]}`` (integers, fallback) or
    ``{'monomer_counts': [name1, name2, ...]}`` (names, when annotated).
    TreeView.make_coords does ``get_in(coords, root.metadata_path)`` where
    metadata_path is ``()`` for plain listener roots, then walks the leaf
    paths from there. So the coords dict must be flat under the view root.

    Args:
        state: Composite state dict.
        view: XArrayEmitter view config (list of root/variables dicts).
        named_metadata: Optional store-relative metadata dict from
            ``output_metadata(state)``.  When a leaf's full store path
            (root_path + leaf_input_path) is present in this dict, its value
            is used as the coord array instead of ``range(N)``.  Pass ``None``
            (default) to keep the legacy ``range(N)`` behaviour for all leaves.
    """
    cell = (state.get("agents") or {}).get("0") or state
    if not isinstance(cell, dict):
        return {}

    out: dict = {}
    for entry in view:
        root_path = tuple(entry.get("root") or ())
        # Walk the state to the view root.
        cursor: Any = cell
        for k in root_path:
            cursor = (cursor or {}).get(k, {})
        if not isinstance(cursor, dict):
            continue
        # For each declared leaf, check if the value at that path is a vector.
        for leaf_input_path, leaf_output_name in _view_leaves([entry]):
            value: Any = cursor
            for k in leaf_input_path:
                value = (value or {}).get(k)
                if value is None:
                    break
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim == 0 or arr.size <= 1:
                continue  # scalar — no coord needed

            # Prefer element names from named_metadata when available.
            # Navigate named_metadata by root_path + leaf_input_path.
            named_coord: list | None = None
            if named_metadata is not None:
                nm: Any = named_metadata
                for k in root_path:
                    if not isinstance(nm, dict):
                        nm = None
                        break
                    nm = nm.get(k)
                for k in leaf_input_path:
                    if not isinstance(nm, dict):
                        nm = None
                        break
                    nm = nm.get(k)
                if isinstance(nm, (list, tuple)) and len(nm) == arr.shape[0]:
                    named_coord = list(nm)

            coord = named_coord if named_coord is not None else list(range(int(arr.shape[0])))

            # Nest coord under leaf path RELATIVE to root (see docstring).
            cursor_out = out
            for k in leaf_input_path[:-1]:
                cursor_out = cursor_out.setdefault(k, {})
            cursor_out[leaf_input_path[-1] if leaf_input_path else leaf_output_name] = coord
    return out


def _view_leaves(view: list[dict]) -> list[tuple[tuple[str, ...], str]]:
    """Walk a view config; return [(input_path_under_root, output_name), ...]."""
    leaves: list[tuple[tuple[str, ...], str]] = []

    def walk(node: Any, prefix: tuple[str, ...]) -> None:
        if isinstance(node, list):
            for spec in node:
                leaves.append((prefix, spec["path"]))
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, prefix + (k,))

    for entry in view:
        walk(entry.get("variables", {}), ())
    return leaves


def filter_view_to_existing_leaves(state: dict, view: list[dict]) -> list[dict]:
    """Drop view leaves that aren't produced by the composite's current state.

    XArrayEmitter raises ``KeyError: Missing emit paths`` when the view
    declares a leaf the composite doesn't emit. SQLite was lenient and
    silently captured only what was there. This mirrors that leniency
    for the xarray path: walks each leaf path in ``state['agents']['0']``,
    drops any leaf whose value is ``None`` (path doesn't exist), and
    returns a new view with the same shape minus the missing leaves.

    Empty variable subtrees collapse out — if a TreeView ends up with no
    leaves it is removed entirely (TreeView raises ``Missing arguments``
    on an empty variables dict).
    """
    cell = (state.get("agents") or {}).get("0") or state
    if not isinstance(cell, dict):
        return view

    out_view: list[dict] = []
    for entry in view:
        root_path = tuple(entry.get("root") or ())
        cursor: Any = cell
        for k in root_path:
            cursor = (cursor or {}).get(k, {})
        if not isinstance(cursor, dict):
            continue

        # Rebuild the variables dict, keeping only present leaves.
        # A leaf is "present" iff the composite emits a scalar/array at it.
        # Missing OR empty-container values (None, {}, []) are dropped — they
        # show up when ``inject_emitter_for_declared_paths`` registers a path
        # whose listener Step isn't in this particular composite variant
        # (the schema default for the slot is an empty container).
        # XArrayEmitter raises ``KeyError: Missing emit paths`` on those at
        # tick 1 because ``dict_to_paths`` yields nothing for an empty
        # container, so the path never gets discarded from ``emit_queue``.
        new_vars: dict = {}
        kept_any = False
        for leaf_input_path, leaf_output_name in _view_leaves([entry]):
            value: Any = cursor
            for k in leaf_input_path:
                value = (value or {}).get(k)
                if value is None:
                    break
            if value is None:
                continue
            if isinstance(value, (dict, list, str)) and len(value) == 0:
                continue
            # Re-attach this leaf at its original path in new_vars.
            # Need to find the original list-spec from entry["variables"].
            orig_spec = entry.get("variables", {})
            for k in leaf_input_path[:-1]:
                orig_spec = orig_spec.get(k, {})
            if not isinstance(orig_spec, dict):
                continue
            spec_list = orig_spec.get(leaf_input_path[-1] if leaf_input_path else leaf_output_name)
            if not isinstance(spec_list, list):
                continue
            cursor_out = new_vars
            for k in leaf_input_path[:-1]:
                cursor_out = cursor_out.setdefault(k, {})
            cursor_out[leaf_input_path[-1] if leaf_input_path else leaf_output_name] = spec_list
            kept_any = True

        if kept_any:
            new_entry = dict(entry)
            new_entry["variables"] = new_vars
            out_view.append(new_entry)

    return out_view


def _filter_agent_state(agent_state: dict, view: list[dict]) -> dict:
    """Filter agent state to exactly the leaves declared by the view.

    Returns a dict shaped like ``{<root[0]>: {<root[1]>: ...{<leaf>: value}...}}``
    for each view entry. Unrelated keys (other listeners, bulk, etc.) are dropped.
    """
    out: dict[str, Any] = {}
    for entry in view:
        root: tuple[str, ...] = tuple(entry["root"])
        # Start at the agent and walk down to the root.
        cur: Any = agent_state
        for k in root:
            cur = (cur or {}).get(k, {})
        # Now `cur` is the subtree at the view root. Filter it to declared paths.
        for leaf_path, _out_name in _view_leaves([entry]):
            value: Any = cur
            for k in leaf_path:
                value = (value or {}).get(k)
                if value is None:
                    break
            # Place value into out at full path: root/.../leaf_path
            cursor = out
            for k in root + leaf_path[:-1]:
                cursor = cursor.setdefault(k, {})
            cursor[leaf_path[-1]] = value
    return out


def build_emitter_config(
    *,
    store_path: Path,
    view: list[dict],
    metadata_base: dict,
    generation: int,
    agent_id: str,
    buffer_size: int = 4,
    output_metadata: dict | None = None,
    writer: dict | None = None,
    predicate: list | None = None,
    provenance: dict | None = None,
) -> dict:
    """Pure builder for an XArrayEmitter config dict (no IO).

    ``writer`` settings (buffers_per_chunk, backend_config, ...) merge over the
    zarr defaults — the store path is always forced. ``predicate`` overrides the
    default subsample(1). Leaf-spec ``unit`` / ``codecs`` keys in ``view`` pass
    through verbatim to LeafView.
    """
    md = dict(metadata_base)
    md["agent_id"] = agent_id
    md["generation"] = generation
    writer_cfg = {
        "backend": "zarr",
        "store": str(store_path),
        "buffers_per_chunk": 1,
        "backend_config": {"format": 3},
    }
    if writer:
        merged = {**writer_cfg, **writer}
        if isinstance(writer.get("backend_config"), dict):
            merged["backend_config"] = {
                **writer_cfg["backend_config"], **writer["backend_config"]}
        merged["store"] = str(store_path)
        writer_cfg = merged
    return {
        "emit": {"global_time": "node"},
        "out_uri": str(store_path),
        # Drive the (pbg-emitters >=0.2.0) generic XArrayEmitter Step with the
        # v2ecoli colony partition config so the multi-cell/lineage zarr layout
        # is preserved:
        #   * ``strategy="colony"`` makes ``extract_partition`` read ``agent_id``
        #     from metadata (``generation == len(agent_id)``, with a ``parent``
        #     cell) instead of the degenerate ``"flat"`` single partition.
        #   * ``emit_root=["agents", <agent_id>]`` tells the transducer to strip
        #     the ``{"agents": {<id>: ...}}`` envelope from each ``update()``
        #     payload (the Step default ``emit_root == ()`` would instead try to
        #     emit the whole wired state and raise ``Unexpected emit path:
        #     ('agents', <id>, ...)``). Keyed to ``agent_id`` so every
        #     generation's emitter strips its own lineage key.
        "strategy": "colony",
        "emit_root": ["agents", agent_id],
        "transducer": {
            "predicate": predicate or [[{"subsample": {"interval": 1}}]],
            "buffer": {"size": buffer_size},
        },
        "view": view,
        "writer": writer_cfg,
        "metadata": md,
        "metadata_keys": [],
        "metadata_validators": {},
        "output_metadata": output_metadata or {},
        # Self-describing run provenance ({composite, config, run_id}) written to
        # the zarr store's root attrs by the XArrayEmitter, so a saved simulation
        # on disk knows what produced it. Empty -> emitter writes no attr.
        "provenance": provenance or {},
        "debug": False,
    }


def _build_emitter(*, core: Any, **kwargs):
    from v2ecoli.library.xarray_emitter import XArrayEmitter  # requires [xarray]
    return XArrayEmitter(config=build_emitter_config(**kwargs), core=core)


def run_multigen_xarray(
    composite: Any,
    *,
    store_path: str | Path,
    view: list[dict],
    metadata_base: dict,
    max_steps: int,
    max_generations: int = 3,
    chunk: int = 60,
    initial_agent_id: str = "0",
    overwrite: bool = True,
    buffer_size: int = 3,
    single_daughters: bool = False,
    division_detector: Callable[[set[str], set[str]], tuple[bool, str | None]] | None = None,
    provenance: dict | None = None,
) -> dict:
    """Run a v2ecoli composite past divisions, swapping XArrayEmitters per generation.

    Args:
      composite: a process-bigraph ``Composite`` with v2ecoli agent state.
      store_path: zarr store path; partitions accumulate by generation.
      view: XArray view config (list of ``{"root": tuple, "variables": dict}``).
      metadata_base: base metadata for the emitter; ``agent_id`` and
        ``generation`` are filled per-generation. Must contain ``experiment_id``,
        ``variant``, ``lineage_seed``, ``time_step``, ``max_duration``.
      max_steps: stop after this many composite ticks.
      max_generations: cap on how many generations to follow.
      chunk: how many ticks between emitter updates.
      initial_agent_id: agent_id to start following.
      overwrite: if True, delete ``store_path`` before starting.
      buffer_size: XArrayEmitter transducer buffer size. Default 3 (NOT the
        config builder's 4): the installed pbg-emitters trips an
        ``assert not include_static`` in its ``flush(final=True)`` path when the
        buffer is exactly full at close (i.e. ``n_updates % buffer_size == 0``);
        3 is the value the single-generation runner uses to dodge it. Override
        only if you know the emit count won't land on a multiple of it.
      division_detector: optional ``(prev_ids, curr_ids) -> (divided?, daughter_id|None)``.
        Default: detect division when ``len(curr) > len(prev)`` and pick the
        first new agent_id sorted.

    Returns: ``{"steps": int, "generations": list[int], "store": str}``.
    """
    store_path = Path(store_path)
    if overwrite and store_path.exists():
        shutil.rmtree(store_path)
    core = composite.core

    if division_detector is None:
        def division_detector(prev, curr):
            new = sorted(curr - prev)
            if len(curr) > len(prev) and new:
                return True, new[0]
            return False, None

    # Discover output_metadata (coord arrays) for any vector leaves in the
    # view by inspecting the composite's live state. Without this, vector
    # observables (e.g. listeners.monomer_counts) crash the emitter with
    # "shape mismatch: value array of shape (N,) could not be broadcast to
    # indexing result with 0 dimensions."
    #
    # At t=0 v2ecoli's listeners are empty dicts — the listener Steps
    # populate them only on first run-tick. Warm the composite with 1 tick
    # so the listener vectors materialise and we can read their length.
    # Cost: we lose tick 0 from the capture, which is fine (the emit
    # predicate's subsample interval is usually > 1 anyway).
    try:
        composite.run(1)
    except Exception as _e:
        print(f"[xarray_run] warm-up run failed: {_e!r}")
    # Filter out view leaves the composite doesn't actually emit. XArrayEmitter
    # is strict on missing-emit-paths (raises KeyError), so cross-variant views
    # that target listeners only some composites have would otherwise sink the
    # run. We mirror SQLite's lenient behaviour: keep what's there, drop what
    # isn't.
    state_after_warmup = composite.state or {}
    filtered_view = filter_view_to_existing_leaves(state_after_warmup, view)
    if not filtered_view:
        raise RuntimeError(
            "[xarray_run] no view leaves remain after filtering against "
            "composite state — the composite emits none of the declared "
            f"observables. Declared view: {view}"
        )
    dropped = sum(len(_view_leaves([e])) for e in view) - \
              sum(len(_view_leaves([e])) for e in filtered_view)
    if dropped:
        print(f"[xarray_run] dropped {dropped} view leaf(s) absent from composite state")
    view = filtered_view
    # Harvest element-name labels from listener outputs() schemas via the
    # registry walker, then feed them to extract_output_metadata_from_state
    # so XArrayEmitter uses real IDs (e.g. monomer names) as coord arrays
    # instead of the integer-index fallback.
    from v2ecoli.library.output_metadata import output_metadata as _get_named_metadata
    named_metadata = _get_named_metadata(state_after_warmup)
    output_metadata = extract_output_metadata_from_state(
        state_after_warmup, view, named_metadata=named_metadata)
    if output_metadata:
        print(f"[xarray_run] discovered vector coord arrays for: "
              f"{list(_flatten_keys(output_metadata))}")

    from v2ecoli.steps.division import daughter_phylogeny_id

    # ``followed`` = the inner-composite key for the tracked cell. The inner
    # Division step always names its mother "0", so EVERY division yields
    # daughters "00"/"01" regardless of lineage depth — the followed key can be
    # REUSED by a daughter ("00" -> "00"/"01"). Detecting division by
    # ``followed in agents`` (which stays True on reuse) folded the last
    # generation's post-division daughter into the parent partition. Detect
    # division structurally (a NEW agent id appeared = an ``_add`` this chunk)
    # and carry a SEPARATE ``partition_agent_id`` along the true phylogeny
    # ("0" -> "00" -> "000") for the zarr generation/agent_id metadata.
    followed = initial_agent_id
    partition_agent_id = initial_agent_id
    gen = 1
    em = _build_emitter(
        core=core, store_path=store_path, view=view,
        metadata_base=metadata_base, generation=gen, agent_id=partition_agent_id,
        output_metadata=output_metadata, buffer_size=buffer_size,
        provenance=provenance,
    )
    # ``done`` counts ticks already advanced past tick 0 (we used 1 for the
    # coord-discovery warm-up so first chunk completes at done = 1 + chunk).
    done = 1
    gens_seen = [gen]
    prev_ids = set(((composite.state or {}).get("agents") or {}).keys())

    def _emit_followed(emitter, agents_map, key):
        if key not in agents_map:
            return
        payload = _filter_agent_state(agents_map[key], view)
        if _EMIT_UNIQUE:
            payload = {**payload, **_extract_unique_attrs(agents_map[key])}
        # CRITICAL: emit the payload under the emitter's OWN agent_id
        # (``partition_agent_id``), NOT the inner-composite key ``key``
        # (``followed``). The XArrayEmitter transducer strips the agent prefix
        # via ``get_in(data, ("agents", partition.agent_id))`` where
        # ``partition.agent_id`` is the metadata agent_id this generation's
        # emitter was built with (== ``partition_agent_id``). When the inner
        # composite REUSES a key so ``followed`` diverges from the true
        # phylogeny ``partition_agent_id`` (happens at deeper divisions, gen 4+),
        # emitting under ``followed`` makes ``get_in`` miss → ``dict_to_paths``
        # yields the empty-tuple path → ``KeyError: 'Unexpected emit path: ()'``.
        # Relabelling the payload to ``partition_agent_id`` here is exact: the
        # emitter only uses the key to locate the cell subtree, and the
        # generation/agent_id partition metadata is already keyed to
        # ``partition_agent_id``. (``partition_agent_id`` is read by late binding
        # so each call uses the current generation's value.)
        emitter.update({
            "time": float(done),
            "global_time": float(done),
            "agents": {partition_agent_id: payload},
        })

    while done < max_steps and gen <= max_generations:
        try:
            composite.run(chunk)
        except Exception as e:
            # FAIL LOUD when a crash TRUNCATES the requested run. A swallowed
            # ``break`` here returns normally → the process exits 0 → the batch
            # job reports SUCCESS while having written only partial, often
            # unreadable data (the 4x4x5 crash that masqueraded as success).
            #
            # The ONLY tolerable raise is one that fires AFTER all requested
            # generations have already completed (``gen >= max_generations`` and
            # we are past the warm-up tick) — post-run noise, e.g. a cell that
            # can no longer divide. Everything else — a crash mid-generation,
            # before the requested generation count — is a real failure that
            # must surface as a non-zero exit, never a silent truncation.
            import traceback as _tb
            tb = _tb.format_exc()
            reached_target = gen >= max_generations and done > 1
            print(f"[multigen_xarray] composite raised at {done}s "
                  f"(gen {gen}/{max_generations}, target_reached={reached_target}):\n{tb}",
                  flush=True)
            if reached_target:
                break
            # Finalize whatever data exists (for diagnosis) then re-raise so the
            # batch job FAILS visibly instead of exiting 0 with a partial zarr.
            try:
                em.close(success=False)
            except Exception:
                pass
            raise RuntimeError(
                f"[multigen_xarray] composite CRASHED at {done}s in generation "
                f"{gen} (requested {max_generations} generations) — failing loud; "
                f"see traceback above") from e
        done += chunk
        agents = (composite.state or {}).get("agents") or {}
        curr_ids = set(agents.keys())

        divided, _detected_daughter = division_detector(prev_ids, curr_ids)
        new_ids = curr_ids - prev_ids
        if not divided and new_ids:
            divided = True

        if not divided:
            # Same pbg-emitters quirk guarded at the per-gen and final closes
            # below: a per-tick emit that triggers an internal buffer-full flush
            # (buffer_size overflow) can hit zarr_writer's flush assert MID-RUN.
            # Unguarded it propagates out of the Ray task and fails the whole run
            # BEFORE reaching division — an emitter-layer flakiness orthogonal to
            # the biology. The overflowed rows are already on disk; swallow the
            # trailing-buffer assert and keep going, exactly as the closes do.
            try:
                _emit_followed(em, agents, followed)
            except AssertionError:
                print(f"[multigen_xarray] tick {done}: pbg-emitters buffer-flush "
                      "assert (buffer full mid-run); flushed rows retained, "
                      "continuing.")
            prev_ids = curr_ids
            continue

        # --- DIVISION: end this generation here; the parent partition must NOT
        # receive the post-division daughter row. ---
        if gen >= max_generations:
            # Generation cap reached: stop (don't fold the daughter in).
            break

        survivors = sorted(curr_ids)
        inner_next = next((i for i in survivors if i.endswith("0")), None)
        if inner_next is None:
            inner_next = survivors[0] if survivors else None
        if inner_next is None:
            break

        # Same pbg-emitters quirk as the final close below: flush(final=True)
        # asserts when the buffer is exactly full at close. This PER-GENERATION
        # close lands on a buffer multiple for some conditions (alt-media gens
        # have different step counts), and an unguarded assert here propagates
        # out of the Ray task → the whole run FAILS (the 4/5 alt-media failures).
        # The completed generation's buffers are already on disk; swallow the
        # trailing-buffer assert and keep going, exactly as the final close does.
        try:
            em.close(success=True)
        except AssertionError:
            print(f"[multigen_xarray] gen-{gen} close hit pbg-emitters final-flush "
                  "assert (buffer full at division); flushed data retained.")
        followed = inner_next
        # Prune the non-followed daughters so ONLY the followed lineage survives
        # in the inner composite. Kept siblings keep growing and dividing, each
        # surfacing a NEW agent id that the division detector reads as a division
        # — and once gen == max_generations the very next such signal hits the
        # `gen >= max_generations` break below, TRUNCATING the last generation to
        # a few ticks (the "v2ecoli runs one generation fewer than vEcoli"
        # artifact). With pruning, only the followed cell's own division ends a
        # generation, so the last generation runs to completion like vEcoli.
        # Mirrors run_multigen_parquet / sqlite_run's single_daughters prune.
        if single_daughters:
            import gc
            from v2ecoli.library.sqlite_run import prune_to_followed_lineage
            dropped = prune_to_followed_lineage(composite, followed)
            gc.collect()
            print(f"[multigen_xarray] gen {gen + 1} → following {followed!r} at "
                  f"tick {done} (single_daughters: dropped {dropped} sibling(s))")
        partition_agent_id = daughter_phylogeny_id(partition_agent_id)[0]
        gen += 1
        gens_seen.append(gen)
        em = _build_emitter(
            core=core, store_path=store_path, view=view,
            metadata_base=metadata_base, generation=gen,
            agent_id=partition_agent_id, output_metadata=output_metadata,
            buffer_size=buffer_size,
        )
        # Emit the daughter's birth row into the NEW generation's store.
        _emit_followed(em, (composite.state or {}).get("agents") or {}, followed)
        prev_ids = set(((composite.state or {}).get("agents") or {}).keys())

    try:
        em.close(success=True)
    except AssertionError:
        # Known pbg-emitters quirk: flush(final=True) asserts when the buffer
        # is exactly full at close. Buffers flushed mid-run are already on
        # disk; only the (full, just-flushed) trailing buffer is at risk — and
        # with buffer_size=3 the full-buffer flush already wrote it. Log + keep
        # the run rather than losing every generation's data to the assert.
        print("[multigen_xarray] close hit pbg-emitters final-flush assert "
              "(buffer full at close); flushed data retained.")
    except Exception as e:  # noqa: BLE001
        print(f"[multigen_xarray] close failed: {type(e).__name__}: {str(e)[:80]}")

    return {"steps": done, "generations": gens_seen, "store": str(store_path)}
