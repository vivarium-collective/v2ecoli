"""Turn the comparison investigation into a composite the engine can trigger.

`workspace/investigations/v2ecoli-vecoli-comparison/` already declares the
right data model — members, per-condition configs, and the ParCa dependency
edge each study carries. This is a re-expression of that into a
process-bigraph document whose member regions carry the ``_study`` metadata
``process_bigraph.templates.trigger`` reads, so "run one condition" and
"rerun a condition reusing ParCa" become the same call.

Three things the declared data does *not* say, which this module decides and
records:

- **Two ParCa roots, not one.** The investigation declares a single ``parca``
  member, but its context names two caches — ``v2_cache`` and ``ve_cache`` —
  because v2ecoli and vEcoli need separate fits and v2 cannot load the
  upstream one. Each declared ``from: parca`` edge is therefore expanded onto
  *both* roots.
- **The vEcoli pin.** The context's ``commit`` is empty and the checkout is
  found through an environment variable. This emits an address site instead,
  so the compared implementation is pinned in the document.
- **Unresolved members.** A declared member with no study of its own is
  reported, never silently dropped.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

INVESTIGATION = 'v2ecoli-vecoli-comparison'

#: The declared ParCa member each condition depends on.
PARCA_MEMBER = 'parca'

#: The two fits it expands into, and the context key naming each one's cache.
PARCA_ROOTS = {
    'parca_v2': 'v2_cache',
    'parca_vecoli': 've_cache'}

SIM_DATA = 'sim_data'
TRAJECTORY = 'trajectory'

#: What a ParCa cache directory calls its calibration object.
SIM_DATA_FILE = 'simData.cPickle'


# ── reading the declared workspace ──────────────────────────────────

def _read_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def load_investigation(workspace: Path, name: str = INVESTIGATION) -> dict:
    """Read an investigation's declaration.

    Reads ``members:`` and top-level ``workspace/studies/<name>/`` — the
    layout the migration moved to.

    ``scripts/_compare/study_spec.load_investigation`` already reads the same
    thing (``members`` with a legacy ``studies`` fallback, resolved against
    ``<workspace>/studies/``), so there is nothing stale to repair there. It
    does differ in one way worth knowing: it *silently drops* a member whose
    ``study.yaml`` is missing, so its list can be shorter than ``members``.
    This one reports the gap instead.
    """
    path = Path(workspace) / 'investigations' / name / 'investigation.yaml'
    if not path.is_file():
        raise FileNotFoundError(f'no investigation at {path}')
    return _read_yaml(path)


def study_path(workspace: Path, member: str) -> Path | None:
    """Where a member's study lives, or None if it has none."""
    candidate = Path(workspace) / 'studies' / member / 'study.yaml'
    return candidate if candidate.is_file() else None


def member_family(workspace: Path, member: str) -> list:
    """Studies that look like per-condition instances of a member.

    ``metabolism_redux`` is declared as one member but exists on disk as
    ``metabolism_redux_basal``, ``metabolism_redux_acetate``, … — a family,
    not a study. Returned so the caller can report it rather than lose it.
    """
    studies = Path(workspace) / 'studies'
    if not studies.is_dir():
        return []
    return sorted(
        entry.name for entry in studies.iterdir()
        if entry.is_dir() and entry.name.startswith(f'{member}_')
        and (entry / 'study.yaml').is_file())


def study_declaration(workspace: Path, member: str) -> dict:
    path = study_path(workspace, member)
    return _read_yaml(path) if path else {}


# ── the pieces a member contributes to the document ─────────────────

def study_config(declaration: dict) -> dict:
    """The knobs that identify a study's *result* — its content address.

    Deliberately narrow: the condition and the comparison knobs that change
    what is computed. Prose, status and recorded outcomes are not inputs to
    the result and must not perturb the address.
    """
    comparison = declaration.get('comparison') or {}
    config = {'condition': declaration.get('condition')}
    for key in ('seeds', 'generations', 'max_steps_per_gen'):
        if key in comparison:
            config[key] = comparison[key]
    if comparison.get('cards'):
        config['cards'] = list(comparison['cards'])
    return {key: value for key, value in config.items() if value is not None}


def study_inputs(declaration: dict, *, parca_roots=tuple(PARCA_ROOTS)) -> list:
    """The member's dependency edges, with ``parca`` expanded to both fits.

    A declared ``{artifact: sim_data, from: parca}`` becomes one edge per
    ParCa root, because the two engines are fit separately.
    """
    edges = []
    for edge in declaration.get('inputs') or []:
        source = edge.get('from')
        artifact = edge.get('artifact')
        if source == PARCA_MEMBER:
            for root in parca_roots:
                edges.append({'artifact': SIM_DATA, 'from': root})
        elif source:
            edges.append({
                'artifact': TRAJECTORY if artifact != SIM_DATA else SIM_DATA,
                'from': source})
    return edges


def compute_node(address: str, config: dict) -> dict:
    """The node that computes a member's results when it is not pulled.

    ``trigger`` swaps this for a ``CachedResults`` when the artifact already
    exists, so every member carries one and the document says nothing about
    run-vs-pull.
    """
    return {
        '_type': 'step',
        'address': address,
        'config': config,
        'inputs': {},
        'outputs': {'results': ['results']}}


def parca_fit_inputs() -> dict:
    """Inputs that change a ParCa fit but are not declared anywhere.

    ParCa takes **no seed** — its CLI has none and its internal RNGs are
    seeded by loop index, so the fit is deterministic by construction. But
    ``V2PARCA_N_SEEDS`` changes how many samples
    ``calculateBulkDistributions`` draws, and so changes the fit, and it is
    recorded in no manifest: two caches built with different values are
    indistinguishable on disk. Folding it into the address closes that hole —
    a fit made with a different sample count gets a different address instead
    of silently answering for one it is not.
    """
    return {'n_seeds': int(os.environ.get('V2PARCA_N_SEEDS', '10'))}


#: The entrypoint the vEcoli address resolves to. It is a **top-level**
#: module name, not ``v2ecoli.comparison.vecoli_adapter``: the git: worker
#: imports it inside vEcoli's own venv, where the ``v2ecoli`` package (and
#: everything its ``__init__`` pulls in) does not exist. Put
#: ``v2ecoli/comparison/`` on PYTHONPATH, not the repo root.
VECOLI_ENTRY = 'vecoli_adapter:make_process'


def vecoli_address(commit: str, repo: str = 'CovertLab/vEcoli',
                   entry: str = VECOLI_ENTRY) -> str:
    """The pinned implementation the comparison runs against.

    Replaces the ``V2E_VECOLI_DIR`` environment variable and the empty
    ``commit:`` in the declaration — the address names *which* vEcoli, and
    which entrypoint in it, rather than whatever happens to be checked out.
    """
    pinned = f'git:{repo}@{commit}' if commit else f'git:{repo}'
    return f'{pinned}#{entry}' if entry else pinned


# ── the document ────────────────────────────────────────────────────

def investigation_document(
        workspace: Path,
        *,
        name: str = INVESTIGATION,
        commit: str = '',
        vecoli_commit: str = '',
        parca_address: str = 'local:ParcaStep',
        study_address_: str = 'local:ComparisonStudy',
        cache_root: Path | None = None,
) -> tuple[dict, dict]:
    """Build the composite document, and a report of what it could not model.

    Returns ``(document, report)`` where report carries ``members``,
    ``unresolved`` (declared members with no study) and ``families``.
    """
    workspace = Path(workspace)
    declaration = load_investigation(workspace, name)
    context = declaration.get('comparison') or {}
    members = list(declaration.get('members') or [])

    document: dict = {}
    unresolved: dict = {}
    modelled: list = []

    # -- the two ParCa roots -----------------------------------------
    for root, cache_key in PARCA_ROOTS.items():
        cache = context.get(cache_key) or ''
        config = {'fit': root, 'cache': cache, **parca_fit_inputs()}
        document[root] = {
            '_study': {
                'id': root,
                'config': config,
                'inputs': [],
                'kind': SIM_DATA},
            'results': {},
            'sim': compute_node(parca_address, {'name': root, **config})}
        modelled.append(root)

    # -- the shared, pinned implementation under comparison ----------
    document['_vecoli'] = {
        'address': vecoli_address(vecoli_commit),
        'repo': (context.get('vecoli') or {}).get('repo', ''),
        'declared_commit': (context.get('vecoli') or {}).get('commit', ''),
        'legacy_env': context.get('vecoli_dir_env', '')}

    # -- one node per condition study --------------------------------
    for member in members:
        if member == PARCA_MEMBER:
            continue                      # expanded into the two roots above

        study = study_declaration(workspace, member)
        if not study:
            family = member_family(workspace, member)
            unresolved[member] = {
                'reason': 'no study.yaml for this member',
                'family': family}
            continue

        config = study_config(study)
        document[member] = {
            '_study': {
                'id': member,
                'config': config,
                'inputs': study_inputs(study),
                'kind': TRAJECTORY},
            'results': {},
            'sim': compute_node(
                study_address_, {'name': member, **config})}
        modelled.append(member)

    report = {
        'investigation': name,
        'members': members,
        'modelled': modelled,
        'unresolved': unresolved,
        'vecoli_address': document['_vecoli']['address'],
        'parca_roots': list(PARCA_ROOTS),
        'cache_root': str(cache_root) if cache_root else None}

    return document, report


# ── resolving ParCa artifacts from the real caches ──────────────────

def register_sim_data_loader() -> None:
    """Teach the artifact registry how to read a ParCa cache.

    A ParCa artifact's store holds the calibration object under
    ``simData.cPickle`` — the name the existing caches already use, so a
    content-addressed store entry can be a directory that *is* (or points at)
    one of those caches.
    """
    from process_bigraph.artifacts import register_artifact_loader

    def load(ref):
        import pickle
        store = Path(ref.store)
        candidate = store / SIM_DATA_FILE
        if not candidate.is_file():
            found = sorted(store.glob('*.cPickle')) + sorted(store.glob('*.pkl'))
            if not found:
                raise FileNotFoundError(
                    f'no {SIM_DATA_FILE} under {store} for artifact '
                    f'{ref.hash!r}')
            candidate = found[0]
        with open(candidate, 'rb') as handle:
            return pickle.load(handle)

    register_artifact_loader(SIM_DATA, load)


def publish_parca_cache(document: dict, root: str, cache_dir: Path, *,
                        store_root: str, commit: str = '') -> str:
    """Register an existing ParCa cache as this root's artifact.

    The cache stays where it is; the content-addressed store entry points at
    it. That is what retires the worktree symlink dance — the address is a
    function of the fit's config and the commit, so any worktree at that
    commit resolves the same entry.

    Returns the address.
    """
    from process_bigraph.templates import study_address
    from process_bigraph.artifacts import artifact_store

    address = study_address(document, root, commit=commit)
    entry = Path(artifact_store(address, store_root))
    entry.parent.mkdir(parents=True, exist_ok=True)

    if not entry.exists():
        target = Path(cache_dir)
        try:
            entry.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            entry.mkdir(parents=True, exist_ok=True)
            source = target / SIM_DATA_FILE
            if source.is_file():
                os.link(source, entry / SIM_DATA_FILE)

    return address
