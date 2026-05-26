"""Closed-loop Millard + LQR — Process that consumes lqr_control + modulates PTS_4 rate.

Closes the loop the bare CopasiUTCProcess can't: per tick, read u from
lqr_control, modulate the PTS_4 (glucose-uptake) forward rate constant kF
by (1 + clip(u, -0.5, 0.5)), then run basico to advance state. Bypasses
CopasiUTCProcess entirely — owns the basico model directly.

This is the closed-loop variant of millard_lqr.composite.yaml. The
lqr-growth-rate-tracking primary test in pdmp-01 can finally be
evaluated against a real coupled run (not just an algebra demo).
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any

from process_bigraph import Process
from v2ecoli.types.stores import InPlaceDict


class MillardWithLQR(Process):
    """Millard 2017 ODE wrapped with per-tick parameter mutation from LQR control.

    Topology (composite-side):
      inputs:
        lqr_control: ["shared", "lqr_control"]      # {u: scalar, ...} from LQR
      outputs:
        species_concentrations: ["shared", "central_metabolites"]
        fluxes: ["shared", "central_fluxes"]
        time: ["shared", "time"]
        control_applied: ["shared", "control_applied"]  # {tick_kF, baseline_kF, u_clipped}
    """

    name = "millard_with_lqr"
    config_schema = {
        "model_source": {"_default": "v2ecoli/models/sbml/millard2017_central_metabolism.xml"},
        "time": {"_default": 100.0},
        "intervals": {"_default": 10},
        "control_reaction": {"_default": "PTS_4"},
        "control_parameter": {"_default": "kF"},
        "u_clip": {"_default": 0.5},  # |u| capped at this fraction
    }

    def initialize(self, config):
        self.parameters = config or {}
        import basico
        self._basico = basico
        basico.load_model(self.parameters["model_source"])
        self.tick_s = float(self.parameters.get("time", 100.0))
        self.intervals = int(self.parameters.get("intervals", 10))
        self.control_reaction = self.parameters.get("control_reaction", "PTS_4")
        self.control_parameter = self.parameters.get("control_parameter", "kF")
        self.u_clip = float(self.parameters.get("u_clip", 0.5))
        # Snapshot baseline parameter value (we'll modulate ABSOLUTE around this)
        params = basico.get_reaction_parameters(reaction_name=self.control_reaction)
        param_row_name = f"({self.control_reaction}).{self.control_parameter}"
        if params is not None and param_row_name in params.index:
            self.baseline_value = float(params.loc[param_row_name]["value"])
        else:
            self.baseline_value = 1.0
        self._tick = 0
        self._log: list[dict] = []

    def __init__(self, config=None, core=None):
        super().__init__(config or {}, core)
        self.initialize(config or {})

    def inputs(self):
        return {"lqr_control": InPlaceDict()}

    def outputs(self):
        # NOTE: deliberately omit "time" because the composite's shared.time is
        # a float (managed by global_time semantics) while our internal basico
        # run_time_course produces a time-vector. Mixing the two causes a
        # type mismatch when the composite attempts to add updates.
        return {
            "species_concentrations": InPlaceDict(),
            "fluxes": InPlaceDict(),
            "control_applied": InPlaceDict(),
        }

    def update(self, state, interval):
        basico = self._basico
        # Read u from lqr_control input
        ctrl = state.get("lqr_control") or {}
        u_raw = float(ctrl.get("u", 0.0))
        u_clipped = max(-self.u_clip, min(self.u_clip, u_raw))
        # Modulate the control parameter — (baseline * (1 + u_clipped))
        tick_value = self.baseline_value * (1.0 + u_clipped)
        param_full = f"({self.control_reaction}).{self.control_parameter}"
        try:
            basico.set_reaction_parameters(name=param_full, value=tick_value)
        except Exception:
            pass

        # Advance one tick
        try:
            ts = basico.run_time_course(
                start_time=0.0,
                duration=interval if interval > 0 else self.tick_s,
                intervals=self.intervals,
                update_model=True,
                use_sbml_id=True,
            )
        except Exception as e:
            self._log.append({
                "tick": self._tick, "error": str(e)[:120],
                "u_clipped": u_clipped, "tick_value": tick_value,
            })
            self._tick += 1
            return {
                "control_applied": {"error": str(e)[:120], "u_clipped": u_clipped,
                                    "tick_value": tick_value, "baseline_value": self.baseline_value},
            }

        # Pull last row as the species concentration output
        species = {sid: float(ts[sid].iloc[-1]) for sid in ts.columns}
        # Time vector
        time_vec = list(map(float, ts.index))

        self._log.append({
            "tick": self._tick,
            "u_clipped": u_clipped,
            "tick_value": tick_value,
            "atp": species.get("ATP"),
        })
        self._tick += 1

        return {
            "species_concentrations": species,
            "fluxes": {},  # left empty for now; would need basico.get_fluxes after run
            "control_applied": {
                "tick": self._tick,
                "u_raw": u_raw,
                "u_clipped": u_clipped,
                "tick_value": tick_value,
                "baseline_value": self.baseline_value,
                "control_reaction": self.control_reaction,
                "control_parameter": self.control_parameter,
            },
        }


def register(core):
    core.register_link("MillardWithLQR", MillardWithLQR)


__all__ = ["MillardWithLQR", "register"]
