"""
=============
Cell Geometry
=============

Native cell-shape geometry deriver, ported from vEcoli's
``ecoli/processes/shape.py``.

Splits the whole-cell volume (from the mass listener) into periplasm and
cytoplasm compartments, and derives the outer surface area via 3D capsule
geometry (a cylinder capped with hemispheres). This exists for the mecillinam
candidate arm: the injected ``antibiotic_transport_odeint`` process divides
periplasmic/cytoplasmic molecule counts by ``state["volumes"]["periplasm"]`` /
``["cytoplasm"]`` and the boundary-permeability side of that chain reads
``boundary.outer_surface_area`` — neither of which the single-cell candidate
otherwise populates, since nothing else in v2ecoli writes them.

**Bridge-quirk guard**: vEcoli's ``Shape`` wires its ``cell_global`` port
directly to the agent's ``boundary`` store. In the candidate's vivarium
bridge, that would hand the WHOLE ``boundary`` store to the step — including
the media ``external`` dict of plain floats — which trips ``Shape``'s
all-items-are-``pint.Quantity`` assert (``shape.py`` ports_schema/next_update).
This step sidesteps that entirely: it reads the cell volume from THIS
candidate's OWN mass listener (``listeners.mass.volume``), never from
``boundary``, and only ever WRITES a single scalar (``outer_surface_area``)
into ``boundary`` — it never reads ``boundary`` at all.
"""

import math

from v2ecoli.library.ecoli_step import EcoliStep as Step
from v2ecoli.library.quantity_helpers import as_quantity
from v2ecoli.types.quantity import ureg as units

NAME = "cell-geometry"
TOPOLOGY = {
    "listeners": ("listeners",),
    "volumes": ("volumes",),
    "boundary": ("boundary",),
}

PI = math.pi

# vEcoli ecoli/processes/shape.py Shape.defaults["periplasm_fraction"].
PERIPLASM_FRACTION = 0.2

# 1 um**3 == 1e-15 L (equivalently, 1 um**3 == 1 fL) — used to convert the
# plain-float cell volume (L) into um**3 for the length/surface-area math,
# which vEcoli's shape.py performs in micron-based units.
L_PER_UM3 = 1e-15


def length_from_volume(volume_um3, width_um):
    """Cell length (um) from volume (um**3) and width (um), via 3D capsule
    geometry: V = (4/3)*pi*r**3 + pi*r**2*a, l = a + 2*r.

    Ported verbatim (modulo units) from vEcoli
    ``ecoli/processes/shape.py:length_from_volume``.
    """
    radius = width_um / 2
    cylinder_length = (volume_um3 - (4 / 3) * PI * radius**3) / (PI * radius**2)
    return cylinder_length + 2 * radius


def surface_area_from_length(length_um, width_um):
    """Outer surface area (um**2) from length + width, via 3D capsule
    geometry: SA = 4*pi*r**2 + 2*pi*r*a.

    Ported verbatim (modulo units) from vEcoli
    ``ecoli/processes/shape.py:surface_area_from_length``.
    """
    radius = width_um / 2
    cylinder_length = length_um - width_um
    return 4 * PI * radius**2 + 2 * PI * radius * cylinder_length


class CellGeometry(Step):
    """Split whole-cell volume into periplasm/cytoplasm; derive outer surface
    area from a fixed cell width.

    Ports:
      * ``listeners.mass.volume`` (in): whole-cell volume, pint
        Quantity[fL] — this candidate's OWN mass listener, never the shared
        ``boundary`` store.
      * ``volumes.periplasm`` / ``volumes.cytoplasm`` (out): pint
        Quantity[L] — matches what the injected
        ``antibiotic_transport_odeint`` divides molecule counts by.
      * ``boundary.outer_surface_area`` (out): pint Quantity[um**2].
    """

    name = NAME
    topology = TOPOLOGY

    config_schema = {
        # Fixed cell width (um), ported from shape.py Shape.defaults["width"]
        # (1.0 um). A config PARAMETER, not a port: deliberately not wired
        # to `boundary` (see module docstring on the bridge quirk).
        "width_um": {"_type": "float", "_default": 1.0},
    }

    def inputs(self):
        return {
            "listeners": {
                "mass": {
                    "volume": {"_type": "quantity[float,fL]", "_default": 0.0},
                },
            },
        }

    def outputs(self):
        return {
            "volumes": {
                "periplasm": {
                    "_type": "overwrite[quantity[float,L]]", "_default": 0.0},
                "cytoplasm": {
                    "_type": "overwrite[quantity[float,L]]", "_default": 0.0},
            },
            "boundary": {
                "outer_surface_area": {
                    "_type": "overwrite[quantity[float,um**2]]", "_default": 0.0},
            },
        }

    def initialize(self, config):
        self.width_um = self.parameters.get("width_um", 1.0)

    def compute(self, cell_volume_L, width_um=None):
        """Pure geometry math (plain floats, no pint) — testable in
        isolation. ``cell_volume_L``: whole-cell volume in liters. Returns
        ``{"periplasm": <L>, "cytoplasm": <L>, "outer_surface_area": <um**2>}``.
        """
        if width_um is None:
            width_um = self.width_um
        periplasm = cell_volume_L * PERIPLASM_FRACTION
        cytoplasm = cell_volume_L * (1 - PERIPLASM_FRACTION)
        volume_um3 = cell_volume_L / L_PER_UM3
        length_um = length_from_volume(volume_um3, width_um)
        outer_surface_area = surface_area_from_length(length_um, width_um)
        return {
            "periplasm": periplasm,
            "cytoplasm": cytoplasm,
            "outer_surface_area": outer_surface_area,
        }

    def update(self, states, interval=None):
        cell_volume = as_quantity(states["listeners"]["mass"]["volume"], units.fL)
        cell_volume_L = cell_volume.to(units.L).magnitude
        geometry = self.compute(cell_volume_L)
        return {
            "volumes": {
                "periplasm": geometry["periplasm"] * units.L,
                "cytoplasm": geometry["cytoplasm"] * units.L,
            },
            "boundary": {
                "outer_surface_area": geometry["outer_surface_area"] * units.um**2,
            },
        }
