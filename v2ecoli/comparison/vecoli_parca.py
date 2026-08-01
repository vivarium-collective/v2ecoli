"""Compute vEcoli's ParCa **inside vEcoli's own venv**, and hand back a ref.

This is the ``#entry`` of the comparison's vEcoli ParCa root::

    git:CovertLab/vEcoli@<sha>#vecoli_parca:build_sim_data

It is the *compute* half of a pull-or-compute root, and it is deliberately
the same subprocess/venv boundary the tick adapter already uses — running a
build step rather than a simulation step. Only the **handle** crosses the RPC
boundary; the 60-90 MB ``simData.cPickle`` stays in the fetched checkout.

**Why computing here is load-bearing.** The blocker to a live comparison was
never the protocol: a `simData.cPickle` built on 2026-07-21 unpickles with
``ModuleNotFoundError: scipy._lib.array_api_compat`` in a freshly-resolved
vEcoli venv, because it was written by an older scipy. Building the pickle in
the *same per-SHA venv that will read it* makes that incompatibility
impossible by construction — and content-addressing over the resolved vEcoli
commit means a repo or env change invalidates the artifact and recomputes it
here, rather than serving something the current environment cannot open.

Runs under the same rules as the tick adapter: standard library only at module
scope, vEcoli imported lazily, addressed as the top-level module
``vecoli_parca`` (put ``v2ecoli/comparison/`` on PYTHONPATH, not the repo
root).
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys
import time
from typing import Any, Dict

#: What vEcoli's ParCa calls its calibration object, and where it puts it.
KB_DIR = 'kb'
SIM_DATA_FILE = 'simData.cPickle'


def sim_data_path(outdir: str) -> str:
    """Where ``run_parca`` leaves the calibration object."""
    return os.path.join(outdir, KB_DIR, SIM_DATA_FILE)


def _fingerprint(path: str, sim_data_hash: str = '') -> str:
    """A stable fingerprint of the produced artifact.

    ParCa itself returns a SHA256 of the pickled bytes, which is exactly the
    right thing — it is a fingerprint of the *result*, not of the inputs, so a
    nondeterministic refit at the same address is detectable. When it is
    unavailable the file is hashed directly.
    """
    if sim_data_hash:
        return sim_data_hash[:16]
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def build_sim_data(config: Any = None) -> 'VEcoliParcaBuild':
    """The ``#entry`` the vEcoli ParCa root resolves to."""
    return VEcoliParcaBuild(config)


class VEcoliParcaBuild:
    """A build step shaped like an edge, so the git: worker can drive it.

    ``update()`` runs the fit and returns an **artifact reference** — the
    store path, the fingerprint, and the environment it is valid for. The
    caller turns that into an ``ArtifactRef``; nothing bulky crosses.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = dict(config or {})
        self.result: Dict[str, Any] = {}

    # -- the face -----------------------------------------------------
    def inputs(self) -> dict:
        return {}

    def outputs(self) -> dict:
        # A build produces a reference, not a trajectory.
        return {'artifact_ref': 'quote'}

    def interface(self) -> dict:
        return {'inputs': self.inputs(), 'outputs': self.outputs()}

    # -- the build ----------------------------------------------------
    def _outdir(self) -> str:
        outdir = self.config.get('outdir')
        if not outdir:
            raise ValueError(
                'the vEcoli ParCa build needs `outdir` — where to write the '
                'fit. It must be writable from inside the fetched checkout.')
        return outdir

    def _parca_config(self) -> dict:
        """vEcoli's ParCa config, defaults filled from its own default.json."""
        from configs import CONFIG_DIR_PATH
        from ecoli.experiments.ecoli_master_sim import SimConfig

        overrides = dict(self.config.get('parca_config') or {})
        config = SimConfig()
        config.update_from_json(
            os.path.join(CONFIG_DIR_PATH, 'default.json'))
        parca = dict(config.to_dict().get('parca_options') or {})
        parca.update(overrides)
        outdir = self._outdir()
        parca['outdir'] = outdir
        # `runscripts/parca.py:main` sets these two itself before calling
        # `run_parca`; calling `run_parca` directly (which is how the fit is
        # reached at all — `main` skips it whenever the config carries a
        # `sim_data_path`, as vEcoli's own default.json does) means setting
        # them here.
        parca['cache_dir'] = os.path.join(outdir, 'cache')
        pathlib.Path(parca['cache_dir']).mkdir(parents=True, exist_ok=True)
        parca.setdefault('cpus', int(self.config.get('cpus', 1)))
        return parca

    def update(self, state: Any = None, interval: float = 0.0) -> dict:
        """Run the fit, or report the one already there.

        Idempotent on a warm checkout: if the calibration object already
        exists at ``outdir`` it is fingerprinted and returned rather than
        refit, so a repeated pull-or-compute costs nothing.
        """
        outdir = self._outdir()
        target = sim_data_path(outdir)

        recompute = bool(self.config.get('force', False))
        if os.path.isfile(target) and not recompute:
            return {'artifact_ref': self._ref(target, '', rebuilt=False)}

        from runscripts.parca import run_parca

        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        sim_data_hash = run_parca(self._parca_config())
        elapsed = time.monotonic() - started

        if not os.path.isfile(target):
            raise RuntimeError(
                f'ParCa reported success but wrote no {SIM_DATA_FILE} at '
                f'{target!r}')

        return {'artifact_ref': self._ref(
            target, sim_data_hash or '', rebuilt=True, seconds=elapsed)}

    def _ref(self, target: str, sim_data_hash: str, *, rebuilt: bool,
             seconds: float = 0.0) -> dict:
        """The handle that crosses the RPC boundary — never the bulk."""
        return {
            'kind': 'sim_data',
            'store': os.path.dirname(target),
            'file': os.path.basename(target),
            'bytes': os.path.getsize(target),
            'fingerprint': _fingerprint(target, sim_data_hash),
            'rebuilt': rebuilt,
            'seconds': round(seconds, 1),
            # The artifact is only valid for the environment that wrote it —
            # this is what makes the stale-pickle class of failure impossible.
            'context': {
                'built_in_venv': True,
                # `sys.prefix`, not `$VIRTUAL_ENV`: the worker inherits the
                # host's environment, so the env var names the *caller's*
                # venv, not the per-SHA one this actually ran in.
                'python': sys.prefix,
                'scipy': _module_version('scipy'),
                'numpy': _module_version('numpy')}}


def _module_version(name: str) -> str:
    try:
        import importlib.metadata as metadata
        return metadata.version(name)
    except Exception:
        return ''
