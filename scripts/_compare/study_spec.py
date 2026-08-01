"""Load investigation + study YAML as the single comparison spec.

The study YAML is the source of truth (no manifest JSON). A study's `name` is the
store/verdict/card key; its `condition` is the biological vEcoli condition the
ensemble simulates (these differ for disambiguated studies, e.g. `basal_4x4` runs
the `basal` condition with 4 seeds). The investigation YAML carries the shared
execution context (ParCa caches, reference engine, default cards). A study's
`config` is a condition name OR a reference-config path -- a "swap" (e.g.
MetabolismRedux) is just a config path that drives both engines identically.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from scripts._compare.reference import ReferenceEngine

REPO = Path(__file__).resolve().parent.parent.parent
INVEST_ROOT = REPO / "workspace" / "investigations"
STUDIES_ROOT = REPO / "workspace" / "studies"
DEFAULT_INVEST = "whole-cell-model-comparison"
# Cards that GATE (pass/fail). The multi-seed `statistical` card (Welch t-test over
# >=4 seeds) is the gold standard for trajectory reproduction; `parca` gates the
# t=0 initial-state match. The single-seed `standard` card is DELIBERATELY NOT a
# gate: one seed's trajectory cannot separate a real port divergence from stochastic
# seed noise (v2ecoli's own 4-seed spread on succinate is ~8-9%, larger than the
# single-seed "mismatch" it was flagged for). `standard` is kept as an illustrative
# trajectory card only. See project memory `project_v2e_compare_singleseed_stochastic`.
GRADED = {"statistical", "parca"}   # cards that produce a gating test
_DEFAULT_CARDS = ["summary", "config", "parca", "standard", "statistical"]
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
    study_path: str
    config: str = ""           # a condition name, or a reference-config path
                                # driving a process swap on BOTH engines (e.g.
                                # metabolism_redux); a bare name = no swap
    reference: "ReferenceEngine | None" = None   # how to run the reference engine
    max_steps_per_gen: int = 15000  # per-generation tick budget; lower it for a
                                    # short-horizon run of an expensive swap (e.g.
                                    # MetabolismRedux solves an LP per tick)

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
    # Canonical model (post-#390): investigations reference top-level
    # workspace/studies/<slug>/ via `members:`. Fall back to the legacy
    # `studies:` key for any investigation not yet migrated.
    members = data.get("members") or data.get("studies") or []
    defaults = comp.get("defaults") or {}
    return {
        "invest_name": data.get("name", inv_dir.name),
        "members": members,
        "studies": members,
        "reference": ReferenceEngine.from_spec(comp.get("reference") or {}),
        "configs": comp.get("configs") or [],
        "v2_cache": comp.get("v2_cache", _DEFAULT_V2_CACHE),
        "ve_cache": comp.get("ve_cache", _DEFAULT_VE_CACHE),
        "defaults": defaults,
        "default_cards": defaults.get("cards") or list(_DEFAULT_CARDS),
        "inv_dir": inv_dir,
    }


def specs_from_configs(ctx: dict) -> list:
    """One StudySpec per `comparison.configs[]` entry -- a config is the unit."""
    defaults = ctx.get("defaults") or {}
    out = []
    for entry in ctx["configs"]:
        name = entry["name"]
        cfg = entry.get("config", name)
        out.append(StudySpec(
            name=name,
            condition=entry.get("condition", name),
            config=cfg,
            seeds=int(entry.get("seeds", defaults.get("seeds", 4))),
            gens=int(entry.get("gens", defaults.get("generations", defaults.get("gens", 1)))),
            cards=list(entry.get("cards") or defaults.get("cards") or list(_DEFAULT_CARDS)),
            invest_name=ctx["invest_name"],
            v2_cache=ctx["v2_cache"],
            ve_cache=ctx["ve_cache"],
            reference=ctx["reference"],
            study_path=str(REPO / "workspace" / "studies" / name / "study.yaml"),
            max_steps_per_gen=int(entry.get("max_steps_per_gen") or 15000),
        ))
    return out


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
    # `config` is the new name; `from_vecoli_config` is read for backward
    # compat with study.yaml content not yet migrated (see task-2 report).
    config = (data.get("config") or comp.get("config")
              or data.get("from_vecoli_config") or comp.get("from_vecoli_config")
              or name)
    return StudySpec(
        name=name,
        condition=condition,
        seeds=seeds,
        gens=gens,
        cards=list(comp.get("cards") or ctx["default_cards"]),
        invest_name=ctx["invest_name"],
        v2_cache=ctx["v2_cache"],
        ve_cache=ctx["ve_cache"],
        config=config,
        reference=ctx["reference"],
        study_path=str(study_path),
        max_steps_per_gen=int(comp.get("max_steps_per_gen") or 15000),
    )


def load_investigation(ref: str) -> tuple[dict, list]:
    """Return (context, [StudySpec, ...]) for an investigation name or path.

    Supports BOTH investigation schemas. If `comparison.configs[]` is present
    (the config-is-the-unit model, post-Task-6), specs are built directly via
    specs_from_configs() -- the same source run/init/scaffold already use, so
    render/run/scaffold/init all agree on one spec list. Otherwise (legacy
    investigations not yet migrated off `members:`), studies are loaded in
    the order listed in `members:` (legacy `studies:` for any un-migrated
    investigation); each is resolved from the canonical TOP-LEVEL
    workspace/studies/<slug>/study.yaml (Study Pipeline registry model,
    post-#390) -- NOT the legacy nested inv_dir/studies/<slug>/. A listed
    study whose study.yaml is missing is skipped.
    """
    inv_dir = _invest_dir(ref)
    if not (inv_dir / "investigation.yaml").exists():
        raise FileNotFoundError(f"investigation not found: {inv_dir}/investigation.yaml")
    ctx = _context(inv_dir)
    if ctx.get("configs"):
        return ctx, specs_from_configs(ctx)
    # Top-level studies live as a SIBLING of investigations/ under the same
    # workspace root (inv_dir is always <workspace>/investigations/<name>);
    # derive it from inv_dir rather than hardcoding REPO so a caller pointing
    # at an alternate/test workspace resolves studies within that workspace.
    studies_root = inv_dir.parent.parent / "studies"
    specs = []
    for sname in ctx["members"]:
        sp = studies_root / sname / "study.yaml"
        if sp.exists():
            specs.append(_spec_from_study(sp, ctx))
    return ctx, specs


def load_study(ref: str) -> StudySpec:
    """Return a StudySpec for a study NAME (under the canonical investigation) or
    a path (to a study.yaml or its dir). The investigation `comparison` context
    (caches, reference engine) is resolved via the study's `investigation:`
    back-reference."""
    p = Path(ref)
    if p.name == "study.yaml":
        sp = p
    elif p.is_dir() and (p / "study.yaml").exists():
        sp = p / "study.yaml"
    else:
        sp = STUDIES_ROOT / ref / "study.yaml"
    if not sp.exists():
        raise FileNotFoundError(f"study not found: {sp}")
    data = yaml.safe_load(sp.read_text(encoding="utf-8")) or {}
    inv_dir = INVEST_ROOT / data.get("investigation", DEFAULT_INVEST)
    return _spec_from_study(sp, _context(inv_dir))
