"""Load investigation + study YAML as the single comparison spec.

The study YAML is the source of truth (no manifest JSON). A study's `name` is the
store/verdict/card key; its `condition` is the biological vEcoli condition the
ensemble simulates (these differ for disambiguated studies, e.g. `basal_4x4` runs
the `basal` condition with 4 seeds). The investigation YAML carries the shared
execution context (ParCa caches, vEcoli fork env, default cards).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
INVEST_ROOT = REPO / "workspace" / "investigations"
DEFAULT_INVEST = "v2ecoli-vecoli-comparison"
GRADED = {"standard", "statistical"}   # cards that produce a gating test
_DEFAULT_CARDS = ["config", "parca", "standard"]
_DEFAULT_V2_CACHE = "out/cache_full"
_DEFAULT_VE_CACHE = "out/compare_harness/vecoli_parca"


@dataclass
class StudySpec:
    name: str                 # store/verdict/card key (study identity)
    condition: str            # biological vEcoli condition (--condition)
    seeds: int
    gens: int
    cards: list
    invest_name: str
    v2_cache: str
    ve_cache: str
    fork: str                 # V2E_VECOLI_DIR value ("" if unset)
    study_path: str

    @property
    def graded_cards(self) -> list:
        return [c for c in self.cards if c in GRADED]


def _invest_dir(ref: str) -> Path:
    """Resolve an investigation NAME or a path (to investigation.yaml or its dir)."""
    p = Path(ref)
    if p.name == "investigation.yaml":
        return p.parent
    if p.is_dir() and (p / "investigation.yaml").exists():
        return p
    return INVEST_ROOT / ref


def _context(inv_dir: Path) -> dict:
    """Shared execution context from an investigation.yaml's `comparison` block."""
    path = inv_dir / "investigation.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    data = data or {}
    comp = data.get("comparison") or {}
    fork_env = comp.get("vecoli_dir_env", "V2E_VECOLI_DIR")
    return {
        "invest_name": data.get("name", inv_dir.name),
        "studies": data.get("studies", []),
        "v2_cache": comp.get("v2_cache", _DEFAULT_V2_CACHE),
        "ve_cache": comp.get("ve_cache", _DEFAULT_VE_CACHE),
        "fork": os.environ.get(fork_env, ""),
        "default_cards": (comp.get("defaults") or {}).get("cards") or list(_DEFAULT_CARDS),
        "inv_dir": inv_dir,
    }


def _spec_from_study(study_path: Path, ctx: dict) -> StudySpec:
    data = yaml.safe_load(study_path.read_text(encoding="utf-8")) or {}
    comp = data.get("comparison") or {}
    name = data.get("name") or study_path.parent.name
    condition = data.get("condition")
    if not condition:
        raise ValueError(f"{study_path}: study has no `condition` (the biological "
                         f"vEcoli condition to simulate)")
    raw_seeds = comp.get("seeds", 1)
    raw_gens = comp.get("generations", 1)
    seeds = int(raw_seeds) if raw_seeds is not None else 1
    gens = int(raw_gens) if raw_gens is not None else 1
    if seeds < 1 or gens < 1:
        raise ValueError(f"{study_path}: comparison.seeds/generations must be >= 1 "
                         f"(got seeds={seeds}, generations={gens})")
    return StudySpec(
        name=name,
        condition=condition,
        seeds=seeds,
        gens=gens,
        cards=list(comp.get("cards") or ctx["default_cards"]),
        invest_name=ctx["invest_name"],
        v2_cache=ctx["v2_cache"],
        ve_cache=ctx["ve_cache"],
        fork=ctx["fork"],
        study_path=str(study_path),
    )


def load_investigation(ref: str) -> tuple[dict, list]:
    """Return (context, [StudySpec, ...]) for an investigation name or path.

    Studies are loaded in the order listed in the investigation's `studies:`;
    a listed study whose study.yaml is missing is skipped.
    """
    inv_dir = _invest_dir(ref)
    if not (inv_dir / "investigation.yaml").exists():
        raise FileNotFoundError(f"investigation not found: {inv_dir}/investigation.yaml")
    ctx = _context(inv_dir)
    specs = []
    for sname in ctx["studies"]:
        sp = inv_dir / "studies" / sname / "study.yaml"
        if sp.exists():
            specs.append(_spec_from_study(sp, ctx))
    return ctx, specs


def load_study(ref: str) -> StudySpec:
    """Return a StudySpec for a study NAME (under the canonical investigation) or
    a path (to a study.yaml or its dir). The investigation `comparison` context
    (caches, fork) is resolved via the study's `investigation:` back-reference."""
    p = Path(ref)
    if p.name == "study.yaml":
        sp = p
    elif p.is_dir() and (p / "study.yaml").exists():
        sp = p / "study.yaml"
    else:
        sp = INVEST_ROOT / DEFAULT_INVEST / "studies" / ref / "study.yaml"
    if not sp.exists():
        raise FileNotFoundError(f"study not found: {sp}")
    data = yaml.safe_load(sp.read_text(encoding="utf-8")) or {}
    inv_dir = INVEST_ROOT / data.get("investigation", DEFAULT_INVEST)
    return _spec_from_study(sp, _context(inv_dir))
