"""Composable baseline recipes — cascading study baselines.

Each recipe declares its parent recipe + the patches it adds. At runtime,
``build_recipe(name)`` walks the parent chain, accumulates patches, and
returns a recipe object whose ``build_doc()`` produces the composite
document for that study's baseline.

Two kinds of patches are supported:

1. **Bundle patches** — applied to the cached ParCa bundle BEFORE
   ``baseline()`` builds the composite document. Example: scale a
   translation-efficiency value, scale a delta_prob entry, flip a
   listener flag.

2. **Loop patches** — applied to the live Composite state BETWEEN
   ``comp.update()`` calls during simulation. Example: stochastic
   intrinsic-hydrolysis flux (DnaA-ATP → DnaA-ADP). This lets us
   prototype mechanisms without rewiring the composite's process graph.

Composite-wired patches (full Step integration) are a future extension
once the Step authoring stabilises. The two patch kinds here are
sufficient for the dnaa-investigation chain.

Why cached-bundle in-place mutation works
-----------------------------------------
``v2ecoli.core.load_cache_bundle`` is memoised by ``cache_dir`` via
``functools.lru_cache``. Mutating the returned bundle in place persists
to the next ``baseline()`` call. We clear the cache between recipe
invocations so each ``build_doc()`` call starts from a clean bundle.
"""
from __future__ import annotations

import functools
import importlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


# ── Patch primitives ────────────────────────────────────────────────────

def _patch_scale_te(bundle: dict, *, monomer_idx: int, factor: float) -> None:
    """Scale ecoli-polypeptide-initiation.translation_efficiencies[monomer_idx]."""
    pi = bundle['configs']['ecoli-polypeptide-initiation']
    pi['translation_efficiencies'][monomer_idx] *= factor


def _patch_scale_tf_fold_change(bundle: dict, *, tf_index: int, factor: float) -> None:
    """Scale all delta_prob.deltaV entries for the TF at column tf_index."""
    tf = bundle['configs']['ecoli-tf-binding']
    dp = tf['delta_prob']
    deltaV = np.asarray(dp['deltaV'])
    mask = (np.asarray(dp['deltaJ']) == tf_index)
    deltaV[mask] = deltaV[mask] * factor
    dp['deltaV'] = deltaV


def _patch_enable_heavy_tf_listener(bundle: dict) -> None:
    """Flip ecoli-tf-binding.emit_n_bound_TF_per_TU = True."""
    tf = bundle['configs']['ecoli-tf-binding']
    tf['emit_n_bound_TF_per_TU'] = True


BUNDLE_PATCH_DISPATCH: dict[str, Callable[..., None]] = {
    'scale_translation_efficiency': _patch_scale_te,
    'scale_tf_fold_change':         _patch_scale_tf_fold_change,
    'enable_heavy_tf_listener':     _patch_enable_heavy_tf_listener,
}


# ── Loop patches ────────────────────────────────────────────────────────

class LoopPatch:
    """Base class for run-loop patches applied between Composite updates.

    Subclasses implement ``init(comp)`` (called once after Composite is
    instantiated) and ``apply(comp, dt_s)`` (called after each
    ``comp.update(dt_s)`` invocation).
    """
    kind: str = 'abstract'

    def init(self, comp) -> None:  # noqa: ARG002
        pass

    def apply(self, comp, dt_s: float) -> dict:
        """Apply the patch effect. Returns an info dict for logging."""
        raise NotImplementedError


class DnaAIntrinsicHydrolysisLoop(LoopPatch):
    """Convert DnaA-ATP → DnaA-ADP at Boesen 2024 intrinsic rate (0.046/min default)."""
    kind = 'dnaa_intrinsic_hydrolysis_loop'

    def __init__(self, k_per_min: float = 0.046, seed: int = 0):
        self.k_per_s = float(k_per_min) / 60.0
        self.rng = np.random.RandomState(int(seed))
        self._atp_idx: Optional[int] = None
        self._adp_idx: Optional[int] = None
        self._cumulative = 0

    def init(self, comp) -> None:
        from wholecell.utils.random import stochasticRound  # noqa: F401  validate import
        bulk = comp.state['agents']['0']['bulk']
        ids = bulk['id']
        self._atp_idx = int(np.where(ids == 'MONOMER0-160[c]')[0][0])
        self._adp_idx = int(np.where(ids == 'MONOMER0-4565[c]')[0][0])

    def apply(self, comp, dt_s: float) -> dict:
        from wholecell.utils.random import stochasticRound
        bulk = comp.state['agents']['0']['bulk']
        n_atp = int(bulk['count'][self._atp_idx])
        expected = self.k_per_s * n_atp * dt_s
        n_hyd = int(stochasticRound(self.rng, np.asarray([expected]))[0])
        n_hyd = max(0, min(n_hyd, n_atp))
        if n_hyd > 0:
            bulk['count'][self._atp_idx] -= n_hyd
            bulk['count'][self._adp_idx] += n_hyd
            self._cumulative += n_hyd
        return {'n_hydrolyzed_step': n_hyd, 'cumulative': self._cumulative}


LOOP_PATCH_REGISTRY: dict[str, type[LoopPatch]] = {
    'dnaa_intrinsic_hydrolysis_loop': DnaAIntrinsicHydrolysisLoop,
}


# ── Recipe class + registry ─────────────────────────────────────────────

@dataclass
class Recipe:
    """A composable composite-baseline recipe."""
    name: str
    parent: Optional[str] = None
    bundle_patches: list[dict] = field(default_factory=list)
    loop_patches:   list[dict] = field(default_factory=list)
    description: str = ''

    def lineage(self) -> list[str]:
        """List of recipe names from root to this recipe."""
        out: list[str] = []
        cur: Optional[Recipe] = self
        while cur is not None:
            out.append(cur.name)
            cur = REGISTRY[cur.parent] if cur.parent else None
        return list(reversed(out))

    def all_bundle_patches(self) -> list[dict]:
        """Bundle patches from root to this recipe (in order)."""
        out: list[dict] = []
        for n in self.lineage():
            out.extend(REGISTRY[n].bundle_patches)
        return out

    def all_loop_patches(self) -> list[dict]:
        """Loop patches from root to this recipe (in order)."""
        out: list[dict] = []
        for n in self.lineage():
            out.extend(REGISTRY[n].loop_patches)
        return out

    def build_doc(self, *, core=None, seed: int = 0, cache_dir: str = 'out/cache'):
        """Apply accumulated bundle patches, then build the composite document."""
        # Clear the lru_cache so we start from a clean bundle every call.
        from v2ecoli.core import (
            _load_cache_bundle_cached, load_cache_bundle, build_core
        )
        from v2ecoli.composites.baseline import baseline
        _load_cache_bundle_cached.cache_clear()
        bundle = load_cache_bundle(cache_dir)
        for p in self.all_bundle_patches():
            kind = p['kind']
            fn = BUNDLE_PATCH_DISPATCH.get(kind)
            if fn is None:
                raise ValueError(f"Unknown bundle-patch kind: {kind!r}")
            args = {k: v for k, v in p.items() if k != 'kind'}
            fn(bundle, **args)
        if core is None:
            core = build_core()
        return baseline(core=core, seed=seed, cache_dir=cache_dir)

    def make_loop_patch_objects(self, *, seed: int = 0) -> list[LoopPatch]:
        """Instantiate all accumulated loop-patches as LoopPatch objects."""
        out: list[LoopPatch] = []
        for p in self.all_loop_patches():
            kind = p['kind']
            cls = LOOP_PATCH_REGISTRY.get(kind)
            if cls is None:
                raise ValueError(f"Unknown loop-patch kind: {kind!r}")
            args = {k: v for k, v in p.items() if k != 'kind'}
            args.setdefault('seed', seed)
            out.append(cls(**args))
        return out


REGISTRY: dict[str, Recipe] = {}


def register(recipe: Recipe) -> Recipe:
    if recipe.name in REGISTRY:
        raise ValueError(f"Recipe {recipe.name!r} already registered")
    REGISTRY[recipe.name] = recipe
    return recipe


def get_recipe(name: str) -> Recipe:
    if name not in REGISTRY:
        raise KeyError(f"Recipe {name!r} not registered. Known: {list(REGISTRY)}")
    return REGISTRY[name]


# ── dnaa investigation recipe chain ─────────────────────────────────────

register(Recipe(
    name='v2ecoli_baseline',
    parent=None,
    bundle_patches=[],
    loop_patches=[],
    description='Raw v2ecoli baseline (no patches). Equivalent to v2ecoli.composites.baseline.baseline.',
))

register(Recipe(
    name='dnaa_01g_calibrated',
    parent='v2ecoli_baseline',
    bundle_patches=[
        {'kind': 'enable_heavy_tf_listener'},
        # F-08 / dnaa-01g: TE multiplier on dnaA monomer (PD03831[c] @ idx 3861).
        {'kind': 'scale_translation_efficiency', 'monomer_idx': 3861, 'factor': 20.0},
        # F-02 / dnaa-01g: scale dnaA autorepression deltaV by 0.7.
        {'kind': 'scale_tf_fold_change', 'tf_index': 12, 'factor': 0.7},
    ],
    loop_patches=[],
    description='(TE=20×, fc=0.7) — the validated dnaA calibration.',
))

register(Recipe(
    name='dnaa_02_with_intrinsic_hydrolysis',
    parent='dnaa_01g_calibrated',
    bundle_patches=[],
    loop_patches=[
        # F-04 / dnaa-02: intrinsic hydrolysis at Boesen 0.046/min.
        # Insufficient on its own (0.987 ATP fraction) but the mechanism is correct.
        {'kind': 'dnaa_intrinsic_hydrolysis_loop', 'k_per_min': 0.046},
    ],
    description='dnaa-01g + intrinsic DnaA-ATP → DnaA-ADP hydrolysis at Boesen 0.046/min.',
))

register(Recipe(
    name='dnaa_02_with_extrinsic_target_rate',
    parent='dnaa_01g_calibrated',
    bundle_patches=[],
    loop_patches=[
        # F-04 / dnaa-02 sensitivity: 100× rate PASSES Boesen band. This is the
        # quantitative target for dnaa-05's extrinsic pathways (RIDA+DDAH+DARS combined).
        {'kind': 'dnaa_intrinsic_hydrolysis_loop', 'k_per_min': 4.6},
    ],
    description='dnaa-01g + hydrolysis at 100× intrinsic (4.6/min) — anticipated dnaa-05 net rate. Predicted to pass ATP-fraction band.',
))

register(Recipe(
    name='dnaa_03_with_box_binding',
    parent='dnaa_02_with_extrinsic_target_rate',
    bundle_patches=[],
    loop_patches=[],
    description='dnaa-02 (with extrinsic target rate) + DnaA-box cooperative binding. '
                'The cooperative-binding loop patch is TODO — currently identical to parent.',
))

register(Recipe(
    name='dnaa_04_with_dnaa_initiation_trigger',
    parent='dnaa_03_with_box_binding',
    bundle_patches=[],
    loop_patches=[],
    description='dnaa-03 + DnaA-occupancy-based replication initiation trigger '
                '(replaces chromosome_replication.py:244 mass-threshold heuristic). '
                'Implementation TODO — currently identical to parent.',
))

register(Recipe(
    name='dnaa_05_full_nucleotide_cycle',
    parent='dnaa_04_with_dnaa_initiation_trigger',
    bundle_patches=[],
    loop_patches=[],
    description='dnaa-04 + RIDA + DDAH + DARS extrinsic pathways. Replaces the lumped '
                '4.6/min rate from dnaa-02 with per-pathway machinery. TODO.',
))

register(Recipe(
    name='dnaa_06_with_seqa_sequestration',
    parent='dnaa_04_with_dnaa_initiation_trigger',
    bundle_patches=[],
    loop_patches=[],
    description='dnaa-04 + SeqA sequestration of newly-replicated oriCs. TODO.',
))


# ── Convenience factory wrappers (used as study.yaml `baseline.composite`) ──

def _make_recipe_factory(recipe_name: str):
    """Produce a function with the signature baseline() expects, so it can be
    referenced from study.yaml ``baseline.composite: v2ecoli.composites.baseline_recipes.X``."""
    def factory(core=None, *, seed: int = 0, cache_dir: str = 'out/cache'):
        return get_recipe(recipe_name).build_doc(core=core, seed=seed, cache_dir=cache_dir)
    factory.__name__ = recipe_name
    factory.__doc__ = REGISTRY[recipe_name].description
    return factory


# Auto-expose every registered recipe as a module-level function for easy
# referencing from study.yaml.
for _n in list(REGISTRY):
    globals()[_n] = _make_recipe_factory(_n)
