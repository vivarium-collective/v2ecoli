"""Shape step — derive capsule cell geometry from mass.

A process-bigraph ``Step`` that turns the cell's *mass* into a 3D **capsule**
(spherocylinder) shape each timestep. At first, **width and density are fixed**
and only **length** changes:

    volume  = mass / density            (density fixed)
    radius  = width / 2                 (width fixed)
    length  ← volume                    (capsule:  V = πr²·L_cyl + 4/3·πr³)

It uses pbg-parsimony's ``Capsule`` for the geometry, so the shape it emits is
exactly what the parsimony 3D build consumes — i.e. the cell envelope grows with
mass over the simulation, and the state right before division (max mass) yields
the elongated, about-to-divide cell.

Wire ``mass_fg`` to the baseline's mass listener; feed ``shape`` to the build.
"""
from __future__ import annotations

import math

from process_bigraph import Step
from pbg_parsimony import Capsule

# Unit bridge: 1 g/mL == 1000 fg/fL  (fg = 1e-15 g, fL = 1e-15 L); fL ≈ µm³.
_G_PER_ML_TO_FG_PER_FL = 1000.0


def capsule_from_mass(mass_fg: float, width_um: float = 1.0,
                      density_g_per_ml: float = 1.1) -> dict:
    """Capsule geometry for a cell of ``mass_fg`` at fixed width + density.

    Width (cap radius = width/2) and density are fixed; **length derives from
    volume** (= mass/density) via the spherocylinder relation
    ``V = π·r²·L_cyl + 4/3·π·r³``.  We solve L_cyl directly (no minimum-elongation
    clamp) so length tracks volume from birth (≈sphere) through division
    (elongated rod).  ``length_um`` is the tip-to-tip length (cylinder + 2 caps).
    """
    volume_fl = max(0.0, float(mass_fg) / (density_g_per_ml * _G_PER_ML_TO_FG_PER_FL))
    r = (width_um / 2.0) * 1e4          # cap radius, µm → Å
    v = volume_fl * 1e12               # fL (µm³) → Å³
    lcyl = max(0.0, (v - (4.0 / 3.0) * math.pi * r ** 3) / (math.pi * r ** 2))
    half_len = lcyl / 2.0
    cap = Capsule(half_len=half_len, radius=r)  # the 3D capsule fed to the packer
    return {
        "mass_fg": float(mass_fg),
        "density_g_per_ml": density_g_per_ml,
        "volume_fl": volume_fl,
        "width_um": width_um,
        "radius_A": r,
        "half_len_A": half_len,
        "length_um": (lcyl + 2.0 * r) / 1e4,   # tip-to-tip
        "capsule": cap,
    }


class ShapeStep(Step):
    """Compute a capsule cell shape (length, width) from mass.

    Width and density are fixed config; length is derived from volume = mass /
    density. Output ``shape`` is the dict from :func:`capsule_from_mass`.
    """

    config_schema = {
        "width_um": {"_type": "float", "_default": 1.0},        # fixed cell width (diameter)
        "density_g_per_ml": {"_type": "float", "_default": 1.1},  # fixed cell density
    }

    def inputs(self):
        return {"mass_fg": "float"}

    def outputs(self):
        return {"shape": "any"}

    def update(self, state, interval=None):
        mass = float(state.get("mass_fg") or 0.0)
        return {"shape": capsule_from_mass(
            mass, self.config["width_um"], self.config["density_g_per_ml"])}
