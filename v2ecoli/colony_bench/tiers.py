"""Cell-tier factory — one cell body per model tier, uniform port contract.

Every tier returns (agent_id, cell_body) placeable by any geometry builder.
Port contract: id, location, angle, mass, length (+ volume/exchange for wcm),
and an `agents` division wire ['..','..',agents_key] matching what
GrowDivide / EcoliWCM._handle_division write {_remove,_add} to.
"""
from __future__ import annotations
from typing import Any

from viva_munk.processes.multibody import build_microbe
from viva_munk.processes.grow_divide import make_adder_grow_divide_process


def _simple(rng, env_size, seed, x, y, angle, length, radius, density,
            agents_key, **_):
    agent_id, body = build_microbe(
        rng, env_size=env_size, x=x, y=y, angle=angle,
        length=length, radius=radius, density=density,
        velocity=(0, 0), speed_range=(0, 0),
    )
    body["grow_divide"] = make_adder_grow_divide_process(
        agents_key=agents_key,
    )
    return agent_id, body


def _wcm(rng, env_size, seed, x, y, angle, length, radius, density,
         agents_key, cache_dir, ecoli_interval, init_mass, **_):
    agent_id, body = build_microbe(
        rng, env_size=env_size, x=x, y=y, angle=angle,
        length=length, radius=radius, density=density,
    )
    if init_mass is not None:
        body["mass"] = float(init_mass)
    body["ecoli"] = {
        "_type": "process",
        "address": "local:EcoliWCM",
        "config": {
            "cache_dir": cache_dir, "seed": seed,
            "transport": "local", "init_mass": init_mass, "env_size": env_size,
        },
        "interval": ecoli_interval,
        "inputs": {
            "local": ["local"], "agent_id": ["id"],
            "location": ["location"], "angle": ["angle"],
        },
        "outputs": {
            "mass": ["mass"], "length": ["length"], "volume": ["volume"],
            "exchange": ["exchange"], "agents": ["..", "..", agents_key],
        },
    }
    body.setdefault("local", {})
    body.setdefault("volume", 0.0)
    body.setdefault("exchange", {})
    return agent_id, body


_TIERS = {"simple": _simple, "wcm": _wcm}


def cell_factory(tier: str, *, rng, env_size: float, seed: int,
                 cache_dir: str = "out/cache", agents_key: str = "cells",
                 ecoli_interval: float = 1.0, init_mass: float | None = None,
                 x: float, y: float, angle: float,
                 length: float = 2.0, radius: float = 0.5,
                 density: float = 0.02) -> tuple[str, dict[str, Any]]:
    if tier not in _TIERS:
        raise ValueError(f"unknown tier: {tier!r} (have {sorted(_TIERS)})")
    return _TIERS[tier](
        rng, env_size, seed, x, y, angle, length, radius, density,
        agents_key, cache_dir=cache_dir, ecoli_interval=ecoli_interval,
        init_mass=init_mass,
    )
