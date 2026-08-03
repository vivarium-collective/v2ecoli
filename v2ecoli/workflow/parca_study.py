"""Comparison convergence Phase 2, ParCa study: pull-or-compute prerequisite
for the comparison investigation.

**Why this exists** (`.superpowers/sdd/2026-08-01-comparison-convergence-phase-2/
gate-e2e-report.md` §A): the Task 5 e2e gate ran the general `vivarium-workbench`
runner all the way through genuine upstream vEcoli — real build, real engine,
real process code — and hit a real wall: the only locally available
vEcoli-native ParCa cache (`out/compare_harness/vecoli_parca/simData.cPickle`)
was built against an OLDER `~/code/vEcoli` fork commit than the one currently
checked out. The fork's `ecoli/processes/listeners/monomer_counts.py` had since
resized its two-component-system molecule list (commit `14f04a3f`), so the
stale simData's arrays disagreed in length with what the current listener code
expects: ``ValueError: operands could not be broadcast together with shapes
(45,) (41,) (45,)``, several frames deep inside a real sim step — an opaque
failure mode for what is really just "the cache is stale."

Both `comparison_materialize.py` (candidate ``ecoli_baseline``, reference
``vecoli``) and the legacy `scripts/_compare/compare_harness.py` previously
ASSUMED both engines' sim_data caches already exist and are compatible ("no
ParCa is re-run — each engine uses a prebuilt, cached sim_data", per
``compare_harness.py``'s own module docstring). This module removes that
assumption for the general-runner path: it is a pull-or-compute contract per
engine —

- **Candidate (v2ecoli):** REUSE iff ``verify_cache_version(cache_dir)``
  (``v2ecoli/library/cache_version.py``, PR #446) passes — hashes
  ``INPUT_FILES`` + runtime context + build_params, raises ``StaleCacheError``
  on any drift (including a wholly-missing/pre-versioning cache). Else REBUILD
  via ``v2ecoli-parca --mode full -o <cache_dir> --cache-dir <cache_dir>``.
- **Reference (vEcoli):** REUSE iff ``<cache_dir>/simData.cPickle`` exists AND
  a sidecar (``PROVENANCE_FILENAME``) records the vEcoli fork commit that
  built it, AND that commit equals ``git -C <reference_repo> rev-parse HEAD``
  right now. A mismatch is exactly the gate-e2e-report §A failure mode, caught
  BEFORE a sim run instead of several frames into a listener's ``next_update``.
  Else REBUILD via the vEcoli-native ``runscripts/parca.py``
  (``scripts._compare.orchestrator.run_vecoli_parca`` — the same command
  ``scripts/run_comparison_ensemble.py``/``compare_harness.py`` use for the
  reference engine).

The CHECK half (``resolve_or_build_parca`` with ``build=False``, the default)
is pure and hermetically testable — it does file/sidecar reads and one
subprocess call to ``git rev-parse HEAD``, never a ParCa build. The heavy
REBUILD half only runs when a caller explicitly passes ``build=True``; no test
in this repo does that (see ``tests/test_parca_study_pull_or_compute.py``'s
module docstring).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from v2ecoli.library.cache_version import StaleCacheError, verify_cache_version

#: Study name the ParCa prerequisite is materialized under (matches
#: `MaterializedInvestigation.parca.name` / `PARCA_STUDY_NAME` usage in
#: `comparison_materialize.py`).
PARCA_STUDY_NAME = "parca"

CANDIDATE_ENGINE = "candidate"
REFERENCE_ENGINE = "reference"
_ENGINES = (CANDIDATE_ENGINE, REFERENCE_ENGINE)

#: Sidecar recording which vEcoli fork commit produced a reference cache's
#: simData.cPickle — the provenance record `_reference_cache_ready` compares
#: against the fork's CURRENT HEAD.
PROVENANCE_FILENAME = "vecoli_build_provenance.json"
SIMDATA_FILENAME = "simData.cPickle"

STATUS_REUSED = "reused"
STATUS_STALE = "stale"
STATUS_REBUILT = "rebuilt"


def prerequisite_edge(name: str = PARCA_STUDY_NAME, relation: str = "leads-to") -> dict:
    """A `pipeline_gate.prerequisites` item — the SAME `{study, relation}`
    shape the installed `vivarium_workbench.lib.study_seed` module writes for
    a study->study DAG edge (verified against the installed package, not
    guessed). Reusing this shape means the general runner's existing DAG
    machinery (`investigation_graph_views.py`'s edge renderer,
    `investigations_index.py`'s `_normalize_parents`/`_condition_satisfied`)
    recognizes the ParCa study as a real prerequisite with zero new
    `vivarium_workbench` wiring."""
    return {"study": name, "relation": relation}


def _current_vecoli_commit(reference_repo: str) -> str | None:
    """`git rev-parse HEAD` of the vEcoli fork checkout, or None if
    unresolvable (no repo given, not a git checkout, git missing, timeout,
    ...) — never raises."""
    if not reference_repo:
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", reference_repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    commit = proc.stdout.strip()
    return commit or None


def write_producing_commit(cache_dir: str, commit: str) -> str:
    """Write the `PROVENANCE_FILENAME` sidecar recording `commit` as the
    vEcoli fork commit that built `cache_dir`'s simData. Called by the real
    rebuild (`_build_reference`) and by tests staging a fixture cache."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, PROVENANCE_FILENAME)
    with open(path, "w") as f:
        json.dump({"vecoli_commit": commit}, f, indent=2)
    return path


def read_producing_commit(cache_dir: str) -> str | None:
    """The commit recorded by `write_producing_commit`, or None if the
    sidecar is absent/unreadable/malformed."""
    path = os.path.join(cache_dir, PROVENANCE_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f).get("vecoli_commit")
    except Exception:
        return None


def _candidate_cache_ready(cache_dir: str) -> tuple[bool, str]:
    """Candidate contract: `verify_cache_version(cache_dir)` passes. Covers
    "cache dir absent", "cache_version.json absent" (pre-versioning cache),
    and "inputs_hash mismatch" (code/fixture/context drift) uniformly — all
    three raise `StaleCacheError` from the SAME #446 machinery, so this
    module doesn't reimplement any of that staleness logic."""
    try:
        verify_cache_version(cache_dir)
    except StaleCacheError as exc:
        return False, str(exc)
    return True, "verify_cache_version passed"


def _reference_cache_ready(cache_dir: str, reference_repo: str) -> tuple[bool, str]:
    """Reference contract: simData.cPickle exists AND its recorded producing
    commit (the `PROVENANCE_FILENAME` sidecar) equals the fork's CURRENT
    HEAD. This is the check that would have caught gate-e2e-report.md §A's
    version-skew (a cache built against an older `monomer_counts.py` than the
    currently-checked-out fork) up front, instead of failing several frames
    into a listener's `next_update` with an opaque array-shape mismatch.

    Deliberately does NOT unpickle simData.cPickle to verify internal shape
    compatibility — that would require a full ParCa-scale load (expensive,
    and still only a partial substitute for actually running the fork's own
    process code). The commit-provenance check is the cheap, hermetic proxy:
    gate-e2e-report's §A traceback root-caused to exactly a
    producing-commit/HEAD mismatch, so comparing commits catches the same
    staleness class without a heavy load.
    """
    simdata_path = os.path.join(cache_dir, SIMDATA_FILENAME)
    if not os.path.exists(simdata_path):
        return False, f"{simdata_path} does not exist"

    producing_commit = read_producing_commit(cache_dir)
    if producing_commit is None:
        return False, (
            f"{os.path.join(cache_dir, PROVENANCE_FILENAME)} missing or "
            f"unreadable -- can't prove which vEcoli commit built this simData")

    current_commit = _current_vecoli_commit(reference_repo)
    if current_commit is None:
        return False, f"could not resolve current HEAD of reference_repo={reference_repo!r}"

    if producing_commit != current_commit:
        return False, (
            f"producing commit {producing_commit!r} != current vEcoli HEAD "
            f"{current_commit!r} -- version-skew (gate-e2e-report.md §A)")

    return True, f"simData.cPickle built by current HEAD ({current_commit[:12]})"


def resolve_or_build_parca(engine: str, cache_dir: str, *,
                           reference_repo: str | None = None,
                           build: bool = False) -> dict:
    """Pull-or-compute resolver for one engine's ParCa cache.

    Pure CHECK logic by default (``build=False``): reads ``cache_dir`` (+, for
    the reference engine, ``reference_repo``'s git HEAD) and decides REUSE vs
    REBUILD without doing any heavy computation itself — this is what every
    test in ``tests/test_parca_study_pull_or_compute.py`` exercises. The
    actual rebuild (the compute branch, ``_build_candidate``/
    ``_build_reference``) only runs when a caller explicitly passes
    ``build=True``.

    Returns ``{"status": "reused" | "stale" | "rebuilt", "path": cache_dir,
    "reason": <human-readable>}``:
    - ``"reused"``  — cache is available and correct; use it as-is.
    - ``"stale"``   — cache is unavailable/incorrect and needs a rebuild that
      did NOT run (``build=False``, the hermetic default).
    - ``"rebuilt"`` — cache needed a rebuild AND it just ran (``build=True``).
    """
    if engine == CANDIDATE_ENGINE:
        ok, reason = _candidate_cache_ready(cache_dir)
    elif engine == REFERENCE_ENGINE:
        ok, reason = _reference_cache_ready(cache_dir, reference_repo or "")
    else:
        raise ValueError(f"unknown engine {engine!r}; expected one of {_ENGINES}")

    if ok:
        return {"status": STATUS_REUSED, "path": cache_dir, "reason": reason}
    if not build:
        return {"status": STATUS_STALE, "path": cache_dir, "reason": reason}

    if engine == CANDIDATE_ENGINE:
        _build_candidate(cache_dir)
    else:
        _build_reference(cache_dir, reference_repo or "")
    return {"status": STATUS_REBUILT, "path": cache_dir, "reason": reason}


def _build_candidate(cache_dir: str) -> None:
    """REAL rebuild: ``v2ecoli-parca --mode full -o <cache_dir> --cache-dir
    <cache_dir>`` (module ``v2ecoli.cli.parca``), via
    ``scripts._compare.orchestrator.run_v2_parca`` (the same wrapper
    ``scripts/compare_harness.py``'s sibling tooling uses). Full mode takes on
    the order of minutes (see project memory
    ``reference_v2ecoli_full_parca_runtime``); gated behind
    ``resolve_or_build_parca(..., build=True)``, never invoked by the
    hermetic test suite."""
    from scripts._compare import orchestrator
    orchestrator.run_v2_parca(out_dir=Path(cache_dir), cache_dir=Path(cache_dir),
                              mode="full")


def _build_reference(cache_dir: str, reference_repo: str) -> None:
    """REAL rebuild: vEcoli-native ParCa (``runscripts/parca.py`` inside
    ``reference_repo``), via ``scripts._compare.orchestrator.run_vecoli_parca``
    + ``scripts._compare.reference.ReferenceEngine`` — the exact
    command/env-isolation convention ``scripts/compare_harness.py`` already
    uses for the reference engine's other stages (sim), extended here to
    ParCa (which no existing caller in this repo runs — every prior comparison
    tool assumed the reference cache pre-exists; this function is what removes
    that assumption). Tens of minutes+; gated behind
    ``resolve_or_build_parca(..., build=True)``, never invoked by the hermetic
    test suite.

    Config is resolved via the fork's own ``configs/default.json`` (read
    through ``scripts._compare.config_adapter.resolve_vecoli_config``, the
    same fork-native loader ``compare_harness.py`` uses) with ``outdir``
    pointed at ``cache_dir``. After a successful build, normalizes the
    produced ``kb/simData.cPickle`` to ``<cache_dir>/simData.cPickle`` — the
    flat path ``v2ecoli.composites.vecoli._resolve_sim_data_path`` expects —
    and writes the ``PROVENANCE_FILENAME`` sidecar recording the fork commit
    that built it, so the NEXT resolve sees a matching provenance record.
    """
    import shutil

    from scripts._compare import orchestrator
    from scripts._compare.config_adapter import resolve_vecoli_config
    from scripts._compare.reference import ReferenceEngine

    os.makedirs(cache_dir, exist_ok=True)
    reference = ReferenceEngine(repo=reference_repo, kind="vecoli")
    default_config = os.path.join(reference_repo, "configs", "default.json")
    vecoli_cfg = resolve_vecoli_config(default_config, vecoli_repo=reference_repo)
    vecoli_cfg["outdir"] = cache_dir
    config_path = os.path.join(cache_dir, "_parca_build_config.json")
    with open(config_path, "w") as f:
        json.dump(vecoli_cfg, f)

    orchestrator.run_vecoli_parca(reference=reference, config_path=config_path,
                                  out_dir=Path(cache_dir))

    produced = Path(cache_dir) / "kb" / SIMDATA_FILENAME
    target = Path(cache_dir) / SIMDATA_FILENAME
    if produced.exists():
        shutil.copy2(produced, target)

    commit = _current_vecoli_commit(reference_repo)
    if commit:
        write_producing_commit(cache_dir, commit)
