"""Registry of modular comparison report cards.

A card is Callable[[CardContext], Section | list[Section]]. Register with the
@report_card("name") decorator; assign by name in the comparison manifest.
Cards are thin wrappers over the existing section functions and Chris Long's
on-main card library (v2ecoli.library.report_card).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

Section = dict  # {title, kind, html, anchor, verdict?}


@dataclass
class CardContext:
    config_name: str
    variant: int
    v2_dir: str
    ve_dir: str
    seeds: int
    gens: int
    per_obs: dict = field(default_factory=dict)
    plot_trajs: dict = field(default_factory=dict)
    v2_bounds: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


Card = Callable[[CardContext], "Section | list[Section]"]
REGISTRY: dict[str, Card] = {}


def report_card(name: str) -> Callable[[Card], Card]:
    def deco(fn: Card) -> Card:
        REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> Card:
    if name not in REGISTRY:
        raise KeyError(f"unknown report card {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name]


def all_names() -> list[str]:
    return sorted(REGISTRY)


def render(name: str, ctx: CardContext) -> list[Section]:
    out = get(name)(ctx)
    return out if isinstance(out, list) else [out]


# Built-in card modules (standard/statistical/parca/config_diff) are imported
# here once they exist (Tasks 4–5) so importing this package registers them.
from scripts._compare.report_cards import standard, statistical  # noqa: E402,F401
from scripts._compare.report_cards import parca, config_diff  # noqa: E402,F401
