"""``parca_prep`` composite — ParCa's pull-or-compute contract as an ordinary
prerequisite study on the investigation-as-composite substrate (Gap 1).

**Why this exists** (design doc
``docs/superpowers/specs/2026-08-02-phase-b-comparison-on-composite-substrate-design.md``,
Gap 1): ``v2ecoli.workflow.parca_study.resolve_or_build_parca`` already
implements the REUSE-vs-REBUILD decision for both engines' ParCa caches (see
that module's docstring and ``tests/test_parca_study_pull_or_compute.py``),
but nothing wires it into the ``@composite_generator`` convention the general
``vivarium-workbench`` runner drives every other study through. Without a
registered composite, a study whose baseline is "make sure ParCa is fresh"
has no ``build_composite("parca_prep", ...)`` entry point — the runner can
build/run every OTHER study uniformly except this one.

This module closes that gap with a thin wrapper: it calls
``resolve_or_build_parca`` for the candidate engine (v2ecoli) and, when
``reference_cache_dir`` is given, the reference engine (vEcoli) too,
escalating a ``"stale"`` check to a real rebuild (``build=True``) only when
``build_if_stale`` is set. It performs NO ParCa science itself — the 9-step
fit lives in ``v2ecoli/processes/parca/`` (see ``composites/parca.py``); this
is purely the pull-or-compute *gate* a comparison investigation's later
studies (``ecoli_baseline``, ``vecoli``, ...) depend on.

The returned document is a minimal state (no process/step nodes) recording
each engine's resolved ``{status, path}`` under ``state["parca_prep"]`` —
enough for a ``run_study`` harvest to persist a ``runs.db`` row for this
study, matching the brief's "prep marker, not a sim" framing. It is a valid
``@composite_generator`` return (a bare document dict); ``Composite``
construction of a plain-data-only tree needs no processes/steps to be valid.
"""
from __future__ import annotations

from typing import Any

from viva_superpowers.composite_generator import composite_generator

from v2ecoli.workflow.parca_study import (
    CANDIDATE_ENGINE,
    REFERENCE_ENGINE,
    STATUS_STALE,
    resolve_or_build_parca,
)

# Imported (not called eagerly) so tests can monkeypatch
# ``v2ecoli.composites.parca_prep.resolve_or_build_parca`` directly — the
# module-level name this function actually looks up at call time.


def _resolve(engine: str, cache_dir: str, *, reference_repo: str = "",
             build_if_stale: bool) -> dict:
    """Resolve one engine's ParCa cache, escalating a stale check to a real
    rebuild iff ``build_if_stale``. Returns the ``resolve_or_build_parca``
    result dict (``{status, path, reason}``) from whichever call was last."""
    result = resolve_or_build_parca(engine, cache_dir, reference_repo=reference_repo)
    if result["status"] == STATUS_STALE and build_if_stale:
        result = resolve_or_build_parca(
            engine, cache_dir, reference_repo=reference_repo, build=True)
    return result


@composite_generator(
    name="parca_prep",
    description=(
        "ParCa pull-or-compute prerequisite, as an ordinary study on the "
        "investigation-as-composite substrate. Resolves (and, when stale, "
        "rebuilds) the candidate (v2ecoli) ParCa cache and, when "
        "reference_cache_dir is given, the reference (vEcoli) cache too, via "
        "v2ecoli.workflow.parca_study.resolve_or_build_parca. No ParCa "
        "science of its own -- a prep marker, not a sim."
    ),
    parameters={
        "candidate_cache_dir": {
            "type": "string",
            "default": "",
            "description": "Cache dir checked/rebuilt for the candidate (v2ecoli) ParCa sim_data.",
        },
        "reference_cache_dir": {
            "type": "string",
            "default": "",
            "description": (
                "Cache dir checked/rebuilt for the reference (vEcoli) ParCa "
                "sim_data. Empty (default) = skip the reference engine "
                "entirely (candidate-only prep)."
            ),
        },
        "reference_repo": {
            "type": "string",
            "default": "",
            "description": "vEcoli fork checkout path, forwarded to the reference engine's HEAD check.",
        },
        "build_if_stale": {
            "type": "boolean",
            "default": True,
            "description": (
                "When a resolve reports \"stale\", immediately re-resolve with "
                "build=True (a real rebuild). False leaves a stale result as "
                "\"stale\" without rebuilding."
            ),
        },
    },
)
def parca_prep(
    core: Any = None,
    *,
    candidate_cache_dir: str,
    reference_cache_dir: str = "",
    reference_repo: str = "",
    build_if_stale: bool = True,
) -> dict:
    """Resolve (and, when stale + ``build_if_stale``, rebuild) the ParCa
    cache(s) for the candidate engine and, when ``reference_cache_dir`` is
    given, the reference engine too.

    Args:
        core: bigraph-schema core. Unused -- this composite builds no
            processes/steps, so no core registration is needed; accepted
            only to match the ``@composite_generator`` calling convention.
        candidate_cache_dir: cache dir for the candidate (v2ecoli) engine.
        reference_cache_dir: cache dir for the reference (vEcoli) engine.
            Empty (default) skips the reference engine entirely.
        reference_repo: vEcoli fork checkout path, forwarded to the
            reference engine's HEAD-provenance check.
        build_if_stale: escalate a "stale" check to a real rebuild
            (``build=True``) when True (default).

    Returns:
        ``{"schema": {}, "state": {"parca_prep": {"candidate": {status,
        path}, "reference": {status, path}?}}}`` -- a minimal document (no
        process/step nodes) so a ``run_study`` harvest has each engine's
        resolved status/path to record.
    """
    candidate = _resolve(
        CANDIDATE_ENGINE, candidate_cache_dir, build_if_stale=build_if_stale)

    parca_prep_state: dict[str, Any] = {
        "candidate": {"status": candidate["status"], "path": candidate["path"]},
    }

    if reference_cache_dir:
        reference = _resolve(
            REFERENCE_ENGINE, reference_cache_dir,
            reference_repo=reference_repo, build_if_stale=build_if_stale)
        parca_prep_state["reference"] = {
            "status": reference["status"], "path": reference["path"]}

    return {"schema": {}, "state": {"parca_prep": parca_prep_state, "global_time": 0.0}}
