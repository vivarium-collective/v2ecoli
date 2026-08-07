"""Geometry builders — place cells of any tier into a physical device.

Generalized from viva_munk's mother_machine/daughter_machine documents and
v2ecoli.colony.make_colony_document: the cell body comes from cell_factory,
so any tier runs in any geometry.
"""
from __future__ import annotations
import math

import numpy as np
from process_bigraph.emitter import emitter_from_wires
from viva_munk.processes.multibody import make_rng
from viva_munk.processes.remove_crossing import make_remove_crossing_process

from v2ecoli.colony_bench.tiers import cell_factory


def _emitter(emit_cells):
    wires = ({"agents": ["cells"], "time": ["global_time"]}
             if emit_cells else {"time": ["global_time"]})
    return emitter_from_wires(wires)


def _multibody(env_size, physics_interval, extra_config=None):
    config = {"env_size": env_size}
    if extra_config:
        config.update(extra_config)
    return {
        "_type": "process", "address": "local:PymunkProcess",
        "config": config, "interval": physics_interval,
        "inputs": {"segment_cells": ["cells"]},
        "outputs": {"segment_cells": ["cells"]},
    }


def free_colony(tier, *, n_cells=2, env_size=30.0, seed=0,
                cache_dir="out/cache", physics_interval=1.0,
                ecoli_interval=1.0, init_mass=None, jitter_per_second=1e-4,
                damping_per_second=0.5, emit_cells=False):
    rng = make_rng(seed)
    cells = {}
    for i in range(n_cells):
        x = env_size / 2 + rng.uniform(-5, 5)
        y = env_size / 2 + rng.uniform(-5, 5)
        angle = rng.uniform(0, 2 * np.pi)
        aid, body = cell_factory(
            tier, rng=rng, env_size=env_size, seed=seed + i, x=x, y=y,
            angle=angle, cache_dir=cache_dir, ecoli_interval=ecoli_interval,
            init_mass=init_mass,
        )
        cells[aid] = body
    return {
        "cells": cells,
        "multibody": _multibody(env_size, physics_interval, {
            "jitter_per_second": jitter_per_second,
            "damping_per_second": damping_per_second,
        }),
        "emitter": _emitter(emit_cells),
    }


def mother_machine(tier, *, n_channels=6, env_size=None, seed=0,
                   cache_dir="out/cache", channel_width=1.5,
                   spacer_thickness=0.3, channel_height=20.0,
                   physics_interval=30.0, ecoli_interval=1.0, init_mass=None,
                   emit_cells=False):
    cell_radius, cell_length = 0.5, 2.0
    if env_size is None:
        width = n_channels * (channel_width + spacer_thickness) + spacer_thickness + 2.0
        env_size = float(max(width, channel_height + 5.0))
    barriers, x = [], spacer_thickness
    for _ in range(n_channels + 1):
        barriers.append({"start": (x, 0), "end": (x, channel_height),
                         "thickness": spacer_thickness})
        x += channel_width + spacer_thickness
    rng = make_rng(seed)
    cells, x = {}, spacer_thickness + spacer_thickness / 2
    for i in range(n_channels):
        cx = x + channel_width / 2
        cy = cell_length / 2 + cell_radius + 0.5
        aid, body = cell_factory(
            tier, rng=rng, env_size=env_size, seed=seed + i, x=cx, y=cy,
            angle=math.pi / 2, length=cell_length, radius=cell_radius,
            cache_dir=cache_dir, ecoli_interval=ecoli_interval, init_mass=init_mass,
        )
        cells[aid] = body
        x += channel_width + spacer_thickness
    return {
        "cells": cells,
        "multibody": _multibody(env_size, physics_interval, {
            "elasticity": 0.1, "barriers": barriers,
            "wall_thickness": spacer_thickness,
        }),
        "remove_crossing": make_remove_crossing_process(
            crossing_y=channel_height, agents_key="cells"),
        "emitter": _emitter(emit_cells),
    }


def daughter_machine(tier, *, env_size=30.0, seed=0, cache_dir="out/cache",
                     flow_x=None, physics_interval=30.0, ecoli_interval=1.0,
                     init_mass=None, emit_cells=False):
    if flow_x is None:
        flow_x = env_size * 0.85
    rng = make_rng(seed)
    aid, body = cell_factory(
        tier, rng=rng, env_size=env_size, seed=seed,
        x=env_size / 2, y=env_size / 2, angle=0.0,
        cache_dir=cache_dir, ecoli_interval=ecoli_interval, init_mass=init_mass,
    )
    return {
        "cells": {aid: body},
        "multibody": _multibody(env_size, physics_interval, {"elasticity": 0.1}),
        "remove_crossing": make_remove_crossing_process(
            x_max=flow_x, agents_key="cells"),
        "emitter": _emitter(emit_cells),
    }
