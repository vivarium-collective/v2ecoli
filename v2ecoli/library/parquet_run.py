"""Multi-generation Parquet run helper for v2ecoli composites.

Sibling to :mod:`v2ecoli.library.sqlite_run`. Same external-emitter driving
model (own the emitter, run the composite in chunks, push the followed
agent's state per-chunk), but the persistence target is a hive-partitioned
parquet directory instead of a sqlite db.

Storage: ~3-5× smaller than the equivalent sqlite run for v2ecoli-shaped
sims — Parquet is column-oriented and dictionary-encodes repeated listener
fields. Trade-off: the dashboard's Simulations-DB tab cannot read parquet
yet (vivarium-dashboard follow-up); for now use this runner when downstream
analysis is DuckDB/Polars-based, ``sqlite_run`` when dashboard inspection
is required.

Lineage layout mirrors vEcoli's Parquet convention::

    <out_dir>/<experiment_id>/history/
      experiment_id=<quoted>/variant=<v>/lineage_seed=<s>/generation=<g>/agent_id=<id>/N.pq

Every division advances the partitioning's ``generation`` and ``agent_id``
hive levels — so each daughter lineage gets its own subtree and can be
queried independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from v2ecoli.library.sqlite_run import (
    _normalize_emit_paths,
    _filter_agent_state,
    _build_emit_schema,
    prune_to_followed_lineage,
)


def run_multigen_parquet(
    composite: Any,
    *,
    experiment_id: str,
    out_dir: str | Path,
    emit_paths: list[str],
    max_steps: int,
    max_generations: int = 1,
    chunk: int = 100,
    initial_agent_id: str = "0",
    initial_variant: int = 0,
    initial_lineage_seed: int = 0,
    initial_generation: int = 1,
    division_detector: Callable[[set[str], set[str]], tuple[bool, str | None]] | None = None,
    core: Any = None,
    single_daughters: bool = False,
    batch_size: int = 400,
    threaded: bool = True,
    study_slug: str | None = None,
    investigation_slug: str | None = None,
) -> dict:
    """Run a v2ecoli composite across divisions, externally-driven ParquetEmitter.

    Args mostly mirror :func:`v2ecoli.library.sqlite_run.run_multigen_sqlite`.
    Differences:

      * ``out_dir`` (vs ``db_file``): root directory for the parquet hive.
      * ``experiment_id``: top-level partition key. Quoted via ``parse.quote_plus``
        internally so it survives any path-unsafe characters.
      * ``initial_variant`` / ``initial_lineage_seed`` / ``initial_generation``:
        hive partition values seeded on the first generation. Each subsequent
        generation increments ``generation`` and uses the daughter agent_id.
      * ``batch_size`` / ``threaded``: forwarded to ParquetEmitter.

    Returns: ``{"steps": int, "generations": list[int], "out_dir": str}``.

    Per-generation ParquetEmitter lifecycle: each new generation rotates the
    emitter (close + re-create with new partition keys). Without this each
    generation would overwrite the prior one's history dir on next config
    write — the per-generation rotation is what lets ``read_parquet(out_dir/
    experiment_id/history/**/*.pq)`` pick up all generations in one read.
    """
    # Imported directly from pbg-emitters (the upstream library);
    # ``v2ecoli.library.parquet_emitter`` is just a re-export shim.
    from pbg_emitters import ParquetEmitter

    if division_detector is None:
        def division_detector(prev: set[str], curr: set[str]) -> tuple[bool, str | None]:
            new = sorted(curr - prev)
            if len(curr) > len(prev) and new:
                return True, new[0]
            return False, None

    leaves = _normalize_emit_paths(emit_paths)
    emit_schema = _build_emit_schema(leaves)
    out_dir = str(Path(out_dir).resolve())

    def _make_emitter(agent_id: str, generation: int) -> ParquetEmitter:
        metadata: dict[str, Any] = {
            "experiment_id": experiment_id,
            "variant": initial_variant,
            "lineage_seed": initial_lineage_seed,
            "generation": generation,
            "agent_id": agent_id,
        }
        if study_slug:
            metadata["study_slug"] = study_slug
        if investigation_slug:
            metadata["investigation_slug"] = investigation_slug

        return ParquetEmitter(
            config={
                "emit": emit_schema,
                "out_dir": out_dir,
                "batch_size": batch_size,
                "threaded": threaded,
                "flatten_separator": "__",
                "partitioning_keys": [
                    "experiment_id", "variant", "lineage_seed",
                    "generation", "agent_id",
                ],
                "metadata": metadata,
            },
            core=core or composite.core,
        )

    import gc

    max_steps = int(max_steps)
    followed = initial_agent_id
    gen = int(initial_generation)
    done = 0
    gens_seen = [gen]
    prev_ids = set(((composite.state or {}).get("agents") or {}).keys())

    em = _make_emitter(followed, gen)

    try:
        while done < max_steps:
            n = min(chunk, max_steps - done)
            try:
                composite.run(n)
            except Exception as e:
                print(f"[multigen_parquet] composite stopped at tick {done}: "
                      f"{type(e).__name__}: {str(e)[:120]}")
                break
            done += n
            agents = (composite.state or {}).get("agents") or {}
            curr_ids = set(agents.keys())

            if followed in agents:
                payload = _filter_agent_state(agents[followed], leaves)
                # ParquetEmitter takes the flat tick state directly — no
                # `agents/<id>/` wrapper (which is sqlite-only convention).
                try:
                    em.update({
                        "global_time": float(done),
                        **payload,
                    })
                except Exception as e:
                    print(f"[multigen_parquet] emit failed at tick {done}: "
                          f"{type(e).__name__}: {str(e)[:120]}")
            else:
                # Followed agent disappeared — division. Pick daughter if we
                # have generations left, else stop.
                if gen >= max_generations:
                    break
                divided, daughter = division_detector(prev_ids, curr_ids)
                if not divided or daughter is None:
                    remaining = sorted(curr_ids)
                    if not remaining:
                        break
                    daughter = remaining[0]

                # Rotate the emitter: close the current generation's hive,
                # open a fresh one keyed on the new generation + daughter.
                em.close(success=True)
                followed = daughter
                gen += 1
                gens_seen.append(gen)
                em = _make_emitter(followed, gen)

                if single_daughters:
                    dropped = prune_to_followed_lineage(composite, followed)
                    gc.collect()
                    print(f"[multigen_parquet] gen {gen} → following agent "
                          f"{followed!r} at tick {done} (single_daughters: "
                          f"dropped {dropped} sibling agent(s) + ran gc)")
                else:
                    print(f"[multigen_parquet] gen {gen} → following agent "
                          f"{followed!r} at tick {done}")

                # Emit the gen-handoff marker.
                if followed in agents:
                    payload = _filter_agent_state(agents[followed], leaves)
                    try:
                        em.update({
                            "global_time": float(done),
                            **payload,
                        })
                    except Exception:
                        pass
            prev_ids = curr_ids
            if single_daughters and n >= 50:
                gc.collect()
    finally:
        em.close(success=True)

    return {"steps": done, "generations": gens_seen, "out_dir": out_dir}
