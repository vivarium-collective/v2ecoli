"""BiRDTransportHours — seconds->hours time-base adapter for BiRDTransportProcess.

v2ecoli's whole-cell model steps in SECONDS. pbg-bioreactordesign's
:class:`BiRDTransportProcess` does its explicit-Euler gas-liquid transport math
in HOURS: ``dC = kLa[1/h] * (C* - C) * interval`` with ``interval`` treated as
hours (see ``pbg_bioreactordesign/processes.py::BiRDTransportProcess.update``).
process-bigraph passes a process its ``interval`` in global-time units — here
SECONDS — straight into that hours-based math, so the raw process transports
~3600x too fast and the dissolved-gas stores diverge/oscillate (which the
composite previously masked with a ``DEFAULT_TRANSPORT_INTERVAL = 0.01`` damping
hack, not a fix).

This thin v2ecoli-side subclass is the proper time-base bridge: it intercepts
``update`` and converts the incoming per-second interval to hours (``/ 3600``)
before delegating to the upstream transport update. Every input/output port,
config field, and the transport physics itself are inherited unchanged — only
the time unit of the interval is corrected.

This mirrors :class:`ReactorCellCoupler`'s own ``timestep / SECONDS_PER_HOUR``
conversion (it scales per-hour metabolic fluxes into per-step deltas), so after
this adapter BOTH the reactor transport deltas and the cell-consumption deltas
are accumulated on the same per-second basis — they net additively at the shared
``reactor.dissolved_o2`` / ``reactor.dissolved_co2`` stores correctly.
"""

from __future__ import annotations

from pbg_bioreactordesign import BiRDTransportProcess

# Convention shared with v2ecoli.steps.reactor_cell_coupler.SECONDS_PER_HOUR.
SECONDS_PER_HOUR: float = 3600.0


class BiRDTransportHours(BiRDTransportProcess):
    """``BiRDTransportProcess`` with its update interval read in SECONDS.

    The only behavioral change vs the upstream process: ``update`` receives a
    per-second ``interval`` (process-bigraph global-time units in v2ecoli) and
    converts it to hours before the upstream hours-based transport math runs.
    All ports and physics are inherited verbatim.
    """

    name = "BiRDTransportHours"

    def update(self, state, interval):
        # interval arrives in SECONDS (v2ecoli global time); the upstream
        # transport update expects HOURS.
        return super().update(state, interval / SECONDS_PER_HOUR)
