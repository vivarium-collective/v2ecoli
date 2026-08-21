"""Read a design-screen variant declaration and plan the builds it implies.

A genotype/expression design screen declares each grid point as a nested
parameter dict: a growth condition, per-gene translation multipliers for native
genes, and — for a heterologous construct — an expression level plus the
generation at which it switches on. vEcoli applies such a declaration by
mutating ``sim_data`` and leaving a deferred instruction in
``sim_data.internal_shift_dict`` for the later generations.

**v2ecoli cannot do that, and this module is the consequence.** The loader's
internal-shift branch (:mod:`v2ecoli.library.sim_data`) runs when a cache is
*built*, and ``baseline`` builds every cell — mother and daughter alike — from a
finished cache bundle, so nothing re-reads ``sim_data`` mid-run. A deferred
instruction is stored and never fires.

What v2ecoli *can* do is run a lineage's early generations against one cache and
its later generations against another, carrying the biological state across
(``scripts/run_condition_multigen_parquet.py`` dumps the state,
``scripts/extend_multigen_from_dill.py`` resumes it against a different
``--cache-dir``). A generation-indexed induction is therefore not a mutation
schedule but a **sequence of caches**, and reading the declaration means working
out which caches those are.

So this module **plans**; it does not apply. It is pure: no I/O, no ParCa, no
sim_data. That makes the whole grammar testable without a build, and it makes
the v1/v2 mechanism difference inspectable — you can print the plan and see
exactly how a declaration was interpreted.

Faithful grammar, different mechanism
-------------------------------------
The declaration is accepted **verbatim** in vEcoli's shape, so one grid-point
entry describes the same experiment to either engine and the two arms stay
comparable at the level of the config. What is *not* faithful is how it runs:
vEcoli mutates in-process at the induction generation; here each stage is a
separate cache and a separate resumed run. Given equal cache contents the two
are behaviourally equivalent, but the difference is real and is stated rather
than papered over — see :class:`DesignPlan`.

⚠ One consequence falls out of that and is easy to miss: a declaration carrying
both an induction and a knockout generation implies **three** stages, not two —
silent, then induced, then knocked out. Each stage is its own cache and its own
run invocation, because a resumed run reads its process configs once
(``extend_multigen_from_dill.py`` loads them before its generation loop) and so
cannot swap caches partway.

⚠ And a constraint the caller owns, not this module: every cache in a plan must
be built from the same ParCa state **and against the same code revision**. The
resume path passes a pre-loaded ``bundle`` and so skips ``load_cache_bundle``'s
staleness check, meaning a mismatched cache is accepted silently rather than
refused. For a long screen, pin the revision for the protocol's duration or
expect to rebuild partway and discard the earlier arm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


#: Declaration keys that carry a heterologous-construct induction. Both use the
#: same ``induction_gen`` / ``exp_trl_eff`` / ``knockout_gen`` shape; the second
#: adds per-target weight vectors under ``rel_adj``. A declaration may name
#: either, and each is read the same way apart from those vectors.
_NEW_GENE_KEYS = ("new_gene_shift", "new_gene_internal_shift_variable_strength")

#: Keys legal *inside* an induction block.
_BLOCK_KEYS = frozenset(
    {"condition", "induction_gen", "knockout_gen", "exp_trl_eff", "rel_adj"})

#: Keys legal at the top level of a composed (``strain_design``-shaped)
#: declaration.
_WRAPPER_KEYS = frozenset({"condition", "perturbations"}) | frozenset(_NEW_GENE_KEYS)

#: A declaration reaches a variant module already expanded to one grid point.
#: These keys belong to the *unexpanded* grid spec that ``parse_variants``
#: consumes (``op`` is popped; ``value``/``nested``/... are resolved), so seeing
#: one means a whole axis was handed over in place of a single point.
_GRID_SPEC_KEYS = frozenset(
    {"value", "nested", "linspace", "logspace", "arange", "geomspace"})


class DesignVariantError(ValueError):
    """A declaration could not be read. Raised in preference to guessing."""


@dataclass(frozen=True)
class NewGeneInduction:
    """The heterologous half of one stage: how hard the construct is driven.

    ``expression`` and ``translation_efficiency`` are the scalar levels; the two
    weight vectors distribute them across targets, one factor per new-gene RNA
    and one per monomer. ``None`` weights mean uniform.

    ⚠ ``translation_efficiency`` is a weight, not an achieved rate — the cached
    array is L1-normalised across every monomer, so only ratios survive. See
    :mod:`v2ecoli.perturbations.new_gene_cache`.
    """

    expression: float
    translation_efficiency: float
    rel_exp_adj: tuple[float, ...] | None = None
    rel_trl_eff_adj: tuple[float, ...] | None = None


@dataclass(frozen=True)
class CacheSpec:
    """Everything that distinguishes one cache in a plan from another.

    Two stages with equal ``CacheSpec``s need only one cache built; the planner
    does not deduplicate, because whether that is worth doing depends on build
    cost the planner cannot see.
    """

    label: str
    condition: str | None = None
    native_perturbations: Mapping[str, float] = field(default_factory=dict)
    new_gene: NewGeneInduction | None = None


@dataclass(frozen=True)
class Stage:
    """One contiguous run of generations against a single cache.

    ``first_generation`` is 1-based and inclusive, matching the declaration's own
    ``induction_gen`` convention. A stage runs until the next stage's
    ``first_generation``, or to the end of the lineage if it is the last.
    """

    first_generation: int
    cache: CacheSpec


@dataclass(frozen=True)
class DesignPlan:
    """The builds and runs one grid point implies.

    ``stages`` is ordered by ``first_generation`` and always non-empty: a
    declaration with no induction still yields a single stage covering the whole
    lineage, so a caller never has to special-case the unperturbed arm.

    ⚠ ``len(stages) > 1`` means the run is a **chain**: run the first stage,
    dump its final state, resume against the next stage's cache. That is the
    v1/v2 mechanism difference in one place.
    """

    stages: tuple[Stage, ...]

    @property
    def is_staged(self) -> bool:
        """True when this plan needs a state handoff rather than a single run."""
        return len(self.stages) > 1


def _weights(raw: Any, what: str) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise DesignVariantError(
            f"{what} must be a sequence of numbers, got {type(raw).__name__}")
    try:
        return tuple(float(x) for x in raw)
    except (TypeError, ValueError) as exc:
        raise DesignVariantError(f"{what} must contain only numbers: {exc}") from exc


def _native_perturbations(params: Mapping[str, Any]) -> dict[str, float]:
    """Read ``perturbations``: EcoCyc gene id -> multiplier (0 = knockout).

    Same vocabulary v2ecoli's ``translation_efficiency_override`` already
    accepts, so this is a read rather than a translation.
    """
    raw = params.get("perturbations")
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise DesignVariantError(
            "'perturbations' must be a mapping of {gene_id: multiplier}, got "
            f"{type(raw).__name__}")
    out: dict[str, float] = {}
    for gene, mult in raw.items():
        try:
            value = float(mult)
        except (TypeError, ValueError) as exc:
            raise DesignVariantError(
                f"perturbation for {gene!r} must be a number, got {mult!r}") from exc
        if value < 0:
            raise DesignVariantError(
                f"perturbation for {gene!r} must be non-negative, got {value}")
        out[str(gene)] = value
    return out


def _reject_grid_spec(params: Mapping[str, Any]) -> None:
    """Refuse an *unexpanded* grid spec handed over as if it were one point.

    ``parse_variants`` expands ``{"value": [...]}`` / ``{"nested": {...}}`` into
    one declaration per grid point and pops ``op`` on the way. So a declaration
    still carrying those has skipped expansion, and reading it would plan a
    single arm from a whole axis — silently, because the shapes are both
    mappings.
    """
    if "op" in params:
        raise DesignVariantError(
            "declaration carries an 'op' key, which parse_variants pops during "
            "expansion; this looks like an unexpanded grid spec rather than one "
            "grid point")
    for key, value in params.items():
        if isinstance(value, Mapping) and _GRID_SPEC_KEYS.intersection(value):
            spec = ", ".join(sorted(_GRID_SPEC_KEYS.intersection(value)))
            raise DesignVariantError(
                f"{key!r} carries grid-spec key(s) ({spec}); this looks like an "
                "unexpanded grid spec rather than one grid point")


def _reject_unknown_keys(params: Mapping[str, Any], allowed: frozenset,
                         what: str) -> None:
    """Refuse keys this reader does not act on.

    ⚠ A deliberate divergence from the reference, which ignores extras in
    silence. A misspelled key there yields an arm that builds, runs to
    completion and reports as a data point while carrying none of the
    perturbation its author wrote. On a screen that arm is indistinguishable
    from a real negative result.
    """
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise DesignVariantError(
            f"{what} names key(s) this reader does not act on: "
            f"{', '.join(repr(k) for k in unknown)}. Expected one of: "
            f"{', '.join(sorted(allowed))}")


def _induction_block(params: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str]:
    """Return the new-gene block and which key carried it.

    Two declaration shapes reach this reader, because vEcoli dispatches on the
    **single** key under ``variants:`` and hands ``apply_variant`` exactly the
    contents beneath it (``runscripts/create_variants.py:398-412``):

    * **composed** — the module is ``strain_design``, so the induction sits in a
      nested block beside ``condition`` / ``perturbations``;
    * **bare** — the module *is* an induction variant, so the declaration has no
      wrapper key and its own keys are the block's.

    ⚠ Reading only the composed shape is not a narrower reader, it is a wrong
    one: a bare declaration has no ``_NEW_GENE_KEYS`` key, so the induction is
    not partially read, it is **not seen at all** — and the plan comes back as a
    single unperturbed stage that looks entirely reasonable.

    ⚠ Both keys present is an error rather than a merge or a precedence rule:
    they set the same fields, so silently preferring one would make a
    declaration mean something its author did not write. The reference *does*
    accept it — two independent ``if`` blocks, and the second variant opens with
    ``sim_data.internal_shift_dict = {}``, so the first block is erased rather
    than merged. Refusing is a deliberate guard against a declaration whose
    first half is dead text; no config in the reference's own set declares both.
    """
    present = [k for k in _NEW_GENE_KEYS if params.get(k) is not None]
    bare = sorted(_BLOCK_KEYS.intersection(params) - {"condition"})

    if present and bare:
        raise DesignVariantError(
            f"declaration mixes a nested induction block ({', '.join(present)}) "
            f"with block-level key(s) ({', '.join(bare)}) at the top level; "
            "which one carries the induction would be a guess")
    if len(present) > 1:
        raise DesignVariantError(
            f"declaration names more than one induction block ({', '.join(present)}); "
            "they set the same fields, so which one wins would be arbitrary")

    if present:
        key = present[0]
        block = params[key]
        if not isinstance(block, Mapping):
            raise DesignVariantError(
                f"{key!r} must be a mapping, got {type(block).__name__}")
        _reject_unknown_keys(params, _WRAPPER_KEYS, "declaration")
        _reject_unknown_keys(block, _BLOCK_KEYS, f"{key!r}")
        return block, key

    if bare:
        _reject_unknown_keys(params, _BLOCK_KEYS, "declaration")
        return params, "declaration"

    _reject_unknown_keys(params, _WRAPPER_KEYS, "declaration")
    return None, ""


def _generation(raw: Any, what: str) -> int:
    """Read a 1-based generation index, refusing anything not already whole.

    ⚠ ``int()`` truncates, and the reference does not: its shift fires on
    ``generation >= induction_gen``, so ``2.7`` means generation **3** there and
    would mean generation **2** here. A silent off-by-one on the induction
    generation shifts the whole protocol by one cell cycle.
    """
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise DesignVariantError(
            f"{what} must be a whole number, got {type(raw).__name__}")
    if isinstance(raw, float) and not raw.is_integer():
        raise DesignVariantError(
            f"{what} must be a whole number, got {raw!r}; the reference fires on "
            f"'generation >= {raw!r}', which truncation would silently move")
    return int(raw)


def _condition(params: Mapping[str, Any], block: Mapping[str, Any] | None,
               key: str) -> str | None:
    """Resolve the growth condition, with the reference's precedence.

    ``strain_design`` applies a top-level ``condition`` first, then inherits it
    into the induction block **only if the block does not carry its own**
    (``ecoli/variants/strain_design.py:81-82``, *"inherit if not given"*), and
    the induction variant then applies whatever it ends up holding
    (``new_gene_internal_shift.py:161``). So a block-level condition is legal
    and **wins**.

    ⚠ This is not a corner case: no config in the reference's own set declares
    ``condition`` at the top level. Both CD2 screens declare it inside the
    induction block, where it carries the media axis of the grid — so reading
    only the top level drops that axis entirely and plans every arm in the
    default medium.

    The condition applies to **every** stage: the reference mutates ``sim_data``
    once at build time, not per generation.
    """
    resolved = params.get("condition")
    if block is not None and block.get("condition") is not None:
        resolved = block["condition"]
    if resolved is not None and not isinstance(resolved, str):
        raise DesignVariantError(
            f"'condition' must be a string, got {type(resolved).__name__}")
    if block is not None and resolved is None:
        # ``condition.apply_variant`` reads ``params["condition"]`` unguarded
        # (``ecoli/variants/condition.py:30``), so the reference raises KeyError
        # here rather than choosing a default. Defaulting would put the arm in a
        # medium nobody declared.
        raise DesignVariantError(
            f"{key} declares an induction but no growth condition, and the "
            "reference requires one (condition.apply_variant reads "
            "params['condition'] unguarded). Declare it on the block or at the "
            "top level rather than relying on a default")
    return resolved


def _induction(block: Mapping[str, Any], key: str,
               expression: float | None = None) -> NewGeneInduction:
    levels = block.get("exp_trl_eff")
    if not isinstance(levels, Mapping) or "exp" not in levels or "trl_eff" not in levels:
        raise DesignVariantError(
            f"{key}.exp_trl_eff must be a mapping with 'exp' and 'trl_eff'")
    rel = block.get("rel_adj") or {}
    if not isinstance(rel, Mapping):
        raise DesignVariantError(f"{key}.rel_adj must be a mapping")
    return NewGeneInduction(
        expression=float(levels["exp"]) if expression is None else expression,
        translation_efficiency=float(levels["trl_eff"]),
        rel_exp_adj=_weights(rel.get("rel_exp_adj_list"), f"{key}.rel_adj.rel_exp_adj_list"),
        rel_trl_eff_adj=_weights(rel.get("rel_trl_eff_adj_list"),
                                 f"{key}.rel_adj.rel_trl_eff_adj_list"),
    )


def plan_design_variant(params: Mapping[str, Any]) -> DesignPlan:
    """Read one grid point's declaration and return the builds it implies.

    Args:
        params: a variant declaration in vEcoli's shape. Recognised keys:

            ``condition``
                growth condition, applied to every stage.
            ``perturbations``
                ``{gene_id: multiplier}`` for native genes; ``0`` is a knockout,
                ``>1`` an overexpression. Applied to every stage — a native
                perturbation describes the chassis, not an event.
            ``new_gene_shift`` / ``new_gene_internal_shift_variable_strength``
                heterologous induction. ``induction_gen`` is the 1-based
                generation at which the construct switches on; ``exp_trl_eff``
                carries ``exp`` and ``trl_eff``; the second key additionally
                accepts ``rel_adj`` weight vectors. An optional ``knockout_gen``
                switches it back off, and must come after ``induction_gen``.

    Returns:
        A :class:`DesignPlan` whose stages are ordered and cover the lineage
        from generation 1.

    Raises:
        DesignVariantError: on any declaration this cannot read unambiguously.
            Preferred to a default, because a mis-read grid point produces a
            plausible-looking result for the wrong experiment.
    """
    if not isinstance(params, Mapping):
        raise DesignVariantError(
            f"declaration must be a mapping, got {type(params).__name__}")

    _reject_grid_spec(params)
    block, key = _induction_block(params)
    condition = _condition(params, block, key)
    native = _native_perturbations(params)

    def spec(label: str, new_gene: NewGeneInduction | None) -> CacheSpec:
        return CacheSpec(label=label, condition=condition,
                         native_perturbations=dict(native), new_gene=new_gene)

    if block is None:
        # No induction declared: one cache for the whole lineage. The chassis
        # perturbations still apply — they are not an event.
        return DesignPlan(stages=(Stage(1, spec("baseline", None)),))

    induction_gen = _generation(block.get("induction_gen", 1),
                                f"{key}.induction_gen")
    if induction_gen < 1:
        raise DesignVariantError(
            f"{key}.induction_gen is 1-based; got {induction_gen}")
    induced = _induction(block, key)

    stages: list[Stage] = []
    if induction_gen > 1:
        # ⚠ The generations before induction are NOT padding — they are the
        # within-lineage control, and a partially-installed construct is not
        # inert. Dropping them changes the experiment.
        stages.append(Stage(1, spec("uninduced", None)))
    stages.append(Stage(induction_gen, spec("induced", induced)))

    knockout_gen = block.get("knockout_gen")
    if knockout_gen is not None:
        knockout_gen = _generation(knockout_gen, f"{key}.knockout_gen")
        if knockout_gen <= induction_gen:
            raise DesignVariantError(
                f"{key}.knockout_gen ({knockout_gen}) must come after "
                f"induction_gen ({induction_gen})")
        # Expression to zero, efficiency and weights unchanged — matching the
        # reference, which switches the construct off without disturbing the
        # rest of the declaration.
        stages.append(Stage(knockout_gen,
                            spec("knocked_out", _induction(block, key, expression=0.0))))

    return DesignPlan(stages=tuple(stages))
