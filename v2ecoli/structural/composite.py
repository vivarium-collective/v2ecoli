"""The ``parsimony-ecoli`` composite: read a v2ecoli state → pack a 3D cell.

A one-shot Step pipeline (packing is a snapshot operation, not time-stepping):

    v2ecoli state ──counts+volume──▶ EcoliStructuralStep ──▶ pack.json + meta.json

The Step reads the molecular state (a saved snapshot or a live ``baseline``
run), selects species + maps them to real structures, and hands them to
pbg-parsimony's engine. The resulting pack + ingredient sidecar are what the
3D webapp renders.
"""
from __future__ import annotations
from typing import Any

from process_bigraph import Step
from pbg_superpowers.composite_generator import composite_generator


class EcoliStructuralStep(Step):
    """Pack a 3D E. coli cell from a v2ecoli molecular state via pbg-parsimony."""

    config_schema = {
        "out_dir": {"_type": "string", "_default": "out/ecoli3d"},
        "name": {"_type": "string", "_default": "ecoli_3d"},
        "top_n": {"_type": "integer", "_default": 40},
        "scale": {"_type": "float", "_default": 1.0},
        "state_source": {"_type": "string", "_default": "snapshot"},  # "snapshot" | "live"
        "proxy_lod": {"_type": "integer", "_default": 2},
    }

    def inputs(self):
        return {}

    def outputs(self):
        return {"pack": "any"}

    def update(self, state, interval=None):
        from v2ecoli.structural.build import build_model
        res = build_model(
            self.config["out_dir"], name=self.config["name"],
            top_n=self.config["top_n"], scale=self.config["scale"],
            state_source=self.config["state_source"], proxy_lod=self.config["proxy_lod"],
        )
        return {"pack": res}


@composite_generator(
    name="parsimony-ecoli",
    description="3D structural model of an E. coli cell: reads a v2ecoli molecular "
                "state and packs it into an interactive 3D scene with pbg-parsimony "
                "(parsimony engine). Output (pack.json + ingredient sidecar) renders "
                "in the bundled 3D webapp.",
    parameters={
        "top_n": {"type": "integer", "default": 40,
                  "description": "How many of the most-abundant protein monomers to include (AlphaFold-modelled), beyond the curated assemblies."},
        "scale": {"type": "number", "default": 1.0,
                  "description": "Abundance scale applied to copy numbers (1.0 = full abundance; lower = lighter/faster)."},
        "state_source": {"type": "string", "default": "snapshot",
                         "description": "'snapshot' (saved v2ecoli state, fast) or 'live' (run the baseline composite)."},
        "out_dir": {"type": "string", "default": "out/ecoli3d"},
    },
)
def parsimony_ecoli(core: Any = None, *, top_n: int = 40, scale: float = 1.0,
                    state_source: str = "snapshot", out_dir: str = "out/ecoli3d") -> dict:
    if core is None:
        from v2ecoli.core import build_core
        core = build_core()
    core.register_link("EcoliStructuralStep", EcoliStructuralStep)
    state = {
        "pack": {},
        "ecoli_structural": {
            "_type": "step",
            "address": "local:EcoliStructuralStep",
            "config": {"out_dir": out_dir, "top_n": top_n, "scale": scale,
                       "state_source": state_source},
            "inputs": {},
            "outputs": {"pack": ["pack"]},
        },
    }
    return {"state": state, "skip_initial_steps": False, "sequential_steps": True,
            "flow_order": ["ecoli_structural"]}
