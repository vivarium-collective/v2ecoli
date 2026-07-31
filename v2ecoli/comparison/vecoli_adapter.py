"""Wrap CovertLab/vEcoli's entrypoint as an Edge the comparison can drive.

This module is the ``#entry`` of the comparison's vEcoli address::

    git:CovertLab/vEcoli@<sha>#vecoli_adapter:make_process

**It runs inside vEcoli's own venv, not v2ecoli's.** The ``git:`` protocol
launches its worker with the per-SHA venv's interpreter and ``cwd`` set to the
checkout, so this file may import only the standard library at module scope,
plus vEcoli's own stack (vivarium-core, ecoli) lazily inside the adapter.
It deliberately does **not** import anything from the ``v2ecoli`` package —
that package's ``__init__`` pulls process-bigraph and viva_superpowers, none of
which exist in vEcoli's venv. For the same reason it is imported as the
top-level module ``vecoli_adapter``: put *this directory* on ``PYTHONPATH``,
not the repo root.

Why an adapter is needed at all: vEcoli has no process-bigraph surface. It is
built on vivarium-core's older API — ``ports_schema()`` and
``next_update(timestep, states)`` — so nothing in the repo answers
``inputs()``/``outputs()``/``update(state, interval)``. This translates
between the two.
"""

from __future__ import annotations

import os
from typing import Any, Dict

#: The observables the comparison grades on — the whole-cell face. Declared
#: here as plain data so the face can be read without importing vEcoli.
WHOLE_CELL_FACE: Dict[str, Dict[str, str]] = {
    'inputs': {},
    'outputs': {
        'cell_mass': 'float',
        'dry_mass': 'float',
        'protein_mass': 'float',
        'rna_mass': 'float',
        'growth_rate': 'float'}}

#: Where each observable lives in vEcoli's mass listener.
MASS_LISTENER = ('listeners', 'mass')
OBSERVABLE_PATHS = {
    'cell_mass': 'cell_mass',
    'dry_mass': 'dry_mass',
    'protein_mass': 'proteinMass',
    'rna_mass': 'rnaMass',
    'growth_rate': 'growth'}


class VEcoliProcess:
    """A pbg-shaped edge over a vivarium ``EcoliSim``.

    ``inputs()``/``outputs()`` answer the whole-cell face without touching
    vEcoli, so an address can be *conformance-checked before a run* — which is
    the point of checking conformance at all. The simulation is built lazily,
    on the first ``update``, because building it unpickles the whole ParCa
    ``sim_data`` and instantiates every process.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = dict(config or {})
        self.sim = None
        self._built = False

    # -- the face -----------------------------------------------------
    def inputs(self) -> dict:
        return dict(WHOLE_CELL_FACE['inputs'])

    def outputs(self) -> dict:
        return dict(WHOLE_CELL_FACE['outputs'])

    def interface(self) -> dict:
        return {'inputs': self.inputs(), 'outputs': self.outputs()}

    # -- the bridge ---------------------------------------------------
    def _build(self):
        """Construct the vEcoli simulation. Heavy: unpickles sim_data."""
        sim_data_path = self.config.get('sim_data_path')
        if not sim_data_path:
            raise ValueError(
                'the vEcoli adapter needs `sim_data_path` — the ParCa pickle '
                'this run compares against. vEcoli builds nothing without it.')
        if not os.path.isfile(sim_data_path):
            raise FileNotFoundError(
                f'no vEcoli sim_data at {sim_data_path!r}; build it with '
                f'vEcoli\'s runscripts/parca.py first.')

        # Imported only once the config is known good: this module is also
        # imported on the host (to read the declared face), where vEcoli's
        # stack does not exist.
        from ecoli.experiments.ecoli_master_sim import EcoliSim

        config_path = self.config.get('config_path')
        sim = (EcoliSim.from_file(config_path) if config_path
               else EcoliSim.from_file())

        sim.sim_data_path = sim_data_path
        for key in ('emitter', 'seed', 'max_duration', 'time_step'):
            if key in self.config:
                setattr(sim, key, self.config[key])
        sim.build_ecoli()

        self.sim = sim
        self._built = True
        return sim

    def _observables(self) -> dict:
        """Read the graded observables out of vEcoli's mass listener."""
        experiment = getattr(self.sim, 'ecoli_experiment', None)
        if experiment is None:
            return {name: 0.0 for name in WHOLE_CELL_FACE['outputs']}

        state = experiment.state.get_value()
        node: Any = state
        for step in MASS_LISTENER:
            node = (node or {}).get(step, {})

        return {
            name: float(node.get(source, 0.0) or 0.0)
            for name, source in OBSERVABLE_PATHS.items()}

    def update(self, state: Any, interval: float) -> dict:
        """Advance vEcoli by ``interval`` and report the observables.

        vivarium's engine is driven by ``update(dt)``; the mass listener is
        read afterwards. That is the whole translation — vivarium's
        ``next_update(timestep, states)`` contract never surfaces here because
        the engine, not a single process, is what is being wrapped.
        """
        if not self._built:
            self._build()
            self.sim.ecoli_experiment = getattr(
                self.sim, 'ecoli_experiment', None)

        experiment = getattr(self.sim, 'ecoli_experiment', None)
        if experiment is None:
            raise RuntimeError(
                'EcoliSim.build_ecoli() did not produce an experiment to '
                'advance; vEcoli changed its construction path.')

        experiment.update(float(interval))
        return self._observables()


def make_process(config: Any = None) -> VEcoliProcess:
    """The ``#entry`` the comparison's vEcoli address resolves to."""
    return VEcoliProcess(config)


def face() -> dict:
    """The declared whole-cell face, for a conformance check that does not
    need vEcoli installed."""
    return {'inputs': dict(WHOLE_CELL_FACE['inputs']),
            'outputs': dict(WHOLE_CELL_FACE['outputs'])}
