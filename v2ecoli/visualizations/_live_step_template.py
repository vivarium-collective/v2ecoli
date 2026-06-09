"""Template + reference for live-sim Visualization Steps.

A live Visualization Step runs INSIDE a composite simulation and emits
a self-contained HTML/SVG artifact at sim end. It complements:

  * **probes** — scripts run OUTSIDE the sim that rebuild a composite,
    re-execute, and write side-by-side artifacts.
  * **per-study charts** — pre-rendered SVGs checked in under
    ``workspace/studies/<name>/charts/`` that the dashboard surfaces.

| pattern  | per-tick cost              | sees emitted-only data | persists with sim |
|----------|----------------------------|------------------------|-------------------|
| probe    | composite rebuild + dt loop | no — reads live state  | no (probe writes side artifacts) |
| Step     | accumulate(state) per tick  | yes (same stream the emitter writes) | yes (HTML is an emitter output) |

The base class is ``pbg_superpowers.visualization.Visualization``. It
ships ``config_schema`` keys ``title``, ``render_mode``, and
``sample_every`` out of the box; subclasses extend with their own keys.

This file is documentation-as-example only. Nothing here is imported
by the composite layer; copy/adapt for new live Visualization Steps.

Pattern (minimal):

    from pbg_superpowers.visualization import Visualization


    class MyLiveVisualization(Visualization):
        '''One-line summary of what gets rendered.'''

        # ── 1. Extend config_schema with your own keys ──────────────
        config_schema = {
            **Visualization.config_schema,
            # 'my_threshold': {'_type': 'float', '_default': 0.5},
        }

        # ── 2. Declare typed input ports ────────────────────────────
        # Inputs should match a slice of the composite state tree —
        # only what render() needs to see. Use registered v2ecoli
        # type names (bulk_array, listener_store, etc.) where possible
        # so the bigraph-schema registry validates the wiring.
        def inputs(self):
            return {
                'global_time': {'_type': 'float', '_default': 0.0},
                # 'bulk': {'_type': 'bulk_array', '_default': []},
                # 'listeners': {...},
            }

        # ── 3. Initialise per-tick accumulator buffers ──────────────
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._times: list[float] = []
            # self._series_a: list[float] = []
            # self._series_b: list[float] = []

        # ── 4. Per-tick callback ────────────────────────────────────
        # Called once per simulation tick with the current state slice.
        # Respect self.config['sample_every'] to downsample expensive
        # per-tick work (e.g. avoid recording every tick when the sim
        # runs at 1-second resolution but the chart only needs 10s).
        def accumulate(self, state):
            t = float(state['global_time'])
            if self._times and (t - self._times[-1]) < self.config['sample_every']:
                return
            self._times.append(t)
            # self._series_a.append(_extract_a(state))
            # self._series_b.append(_extract_b(state))

        # ── 5. Sim-end render ───────────────────────────────────────
        # Called once at sim end. Return a self-contained HTML/SVG
        # string (no external CSS/JS deps so it works inside the
        # dashboard's offline-HTML report payload).
        def render(self) -> str:
            return (
                '<svg xmlns="http://www.w3.org/2000/svg" width="600" '
                'height="200"><text x="20" y="40">replace me</text></svg>'
            )


Conventions:
  * **One Step per chart** — easier to compose, reuse, and disable.
  * **No external assets** — render output must be self-contained so
    the dashboard's offline-HTML payload stays portable.
  * **Sample every N ticks** — accumulate cheaply; long sims with
    per-tick recording bloat the emitter's history table.
  * **Render is pure** — render() reads only self.* (no I/O), so the
    same accumulator state is reproducible from the recorded history.
"""
